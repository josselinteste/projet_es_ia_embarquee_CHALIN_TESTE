import os, sys, types, importlib.util, traceback
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import torchvision, torchvision.transforms as T
import random

# === Chemins ===
script_dir = os.path.dirname(os.path.abspath(__file__))
attack_dir = os.path.join(script_dir, "attack")
print(f"[INFO] script_dir: {script_dir}")
print(f"[INFO] attack_dir: {attack_dir}")

# ------------------------------------------------------------
# 1️⃣  Créer un module "models.quantization" factice
# ------------------------------------------------------------
quant_mod = types.ModuleType("models.quantization")

class quan_Conv2d(nn.Conv2d): pass
class quan_Linear(nn.Linear): pass
class quan_Conv1d(nn.Conv1d): pass
def quantize(x): return x

quant_mod.quan_Conv2d = quan_Conv2d
quant_mod.quan_Linear = quan_Linear
quant_mod.quan_Conv1d = quan_Conv1d
quant_mod.quantize = quantize

sys.modules["models"] = types.ModuleType("models")
sys.modules["models.quantization"] = quant_mod

# ------------------------------------------------------------
# 2️⃣  Importer BFA et data_conversion depuis attack/
# ------------------------------------------------------------
def import_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

try:
    # Import BFA et data_conversion
    BFA_mod = import_module_from_path("BFA", os.path.join(attack_dir, "BFA.py"))
    DC_mod  = import_module_from_path("data_conversion", os.path.join(attack_dir, "data_conversion.py"))
    BFA = BFA_mod.BFA
    weight_conversion = getattr(DC_mod, "weight_conversion", None)
    print("[INFO] Imported BFA and data_conversion successfully.")

    # === Patch universel : rendre BFA compatible avec nn.Conv2d / nn.Linear ===
    import torch.nn as nn

    old_progressive_bit_search = BFA.progressive_bit_search  # si tu veux garder la référence

    import random
    import copy

    def new_progressive_bit_search(self, model, data, target, top_layers=3, top_weights=5):
        """
        True BFA with candidate evaluation: for each flip candidate (layer, weight index, bit),
        simulate it on the actual batch (data/target), measure real loss increase, choose the best
        candidate and apply it permanently. This prevents accuracy from increasing between flips.
        Params:
        top_layers: how many top layers (by mean abs grad) to consider
        top_weights: how many top-weight indices (by abs grad) per layer to consider
        """
        self.loss_dict = {}
        model.eval()
        model.zero_grad()

        # compute baseline loss and grads
        output = model(data)
        base_loss = float(self.criterion(output, target).item())
        base_output = output.detach()
        # backward to get grads used to pick candidates
        model.zero_grad()
        loss = self.criterion(output, target)
        loss.backward()

        # collect mean abs gradients per candidate layer
        layer_grads = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if getattr(module, "weight", None) is not None and module.weight.grad is not None:
                    layer_grads.append((name, float(module.weight.grad.abs().mean().item())))
        if not layer_grads:
            print("[WARN] Aucun module attaquable trouvé (loss_dict vide).")
            return []

        # sort layers by grad desc and keep top_layers
        layer_grads.sort(key=lambda x: x[1], reverse=True)
        layer_grads = layer_grads[:top_layers]

        best_candidate = None
        best_delta = -float("inf")

        # precompute grads per-layer to choose top weight indices
        for (name, _) in layer_grads:
            module = dict(model.named_modules())[name]
            grad_flat = module.weight.grad.view(-1).abs()
            # choose top weight indices (or all if small)
            k = min(top_weights, grad_flat.numel())
            if k <= 0:
                continue
            topk_vals, topk_idx = torch.topk(grad_flat, k)
            topk_idx = topk_idx.tolist()

            # ensure per-module set to avoid flipping same bit twice
            if not hasattr(module, "_flipped_bits"):
                module._flipped_bits = set()  # store (flat_idx, bit)

            # for each candidate weight index, evaluate flipping each bit
            N_bits = int(getattr(module, "N_bits", 8))
            if N_bits <= 1:
                N_bits = 8
            SCALE = 2**(N_bits-1) - 1
            mask_all = (1 << N_bits) - 1

            # helpers for quantization mapping (symmetric two's complement)
            def float_to_unsigned_int(x):
                signed = int(round(max(min(x, 1.0), -1.0) * SCALE))
                return signed & mask_all

            def unsigned_int_to_float(u):
                sign_bit = 1 << (N_bits - 1)
                if u & sign_bit:
                    signed = u - (1 << N_bits)
                else:
                    signed = u
                return float(signed) / SCALE

            for flat_idx in topk_idx:
                for bit in range(N_bits):
                    if (flat_idx, bit) in module._flipped_bits:
                        # skip already flipped bits for this module
                        continue

                    # simulate flip: compute candidate weight value
                    w_data = module.weight.data.view(-1)[flat_idx].item()
                    u_curr = float_to_unsigned_int(w_data)
                    u_cand = u_curr ^ (1 << bit)
                    w_cand = unsigned_int_to_float(u_cand)

                    # apply temporarily and measure loss on batch (use no_grad for assignment)
                    with torch.no_grad():
                        old_val = module.weight.data.view(-1)[flat_idx].item()
                        module.weight.data.view(-1)[flat_idx] = w_cand

                    # forward only to get new loss
                    with torch.no_grad():
                        out_cand = model(data)
                        loss_cand = float(self.criterion(out_cand, target).item())

                    # revert the weight
                    with torch.no_grad():
                        module.weight.data.view(-1)[flat_idx] = old_val

                    delta = loss_cand - base_loss
                    # keep best candidate by real loss increase
                    if delta > best_delta:
                        best_delta = delta
                        best_candidate = (name, flat_idx, bit, w_cand, loss_cand)

        if best_candidate is None:
            print("[WARN] No valid candidate found to increase loss.")
            return []

        # Apply the best candidate permanently
        name, flat_idx, bit, w_cand, loss_after = best_candidate
        module = dict(model.named_modules())[name]
        with torch.no_grad():
            module.weight.data.view(-1)[flat_idx] = w_cand
        # record as flipped
        if not hasattr(module, "_flipped_bits"):
            module._flipped_bits = set()
        module._flipped_bits.add((flat_idx, bit))

        print(f"[INFO] Applied best flip: layer={name}, idx={flat_idx}, bit={bit}, Δloss={best_delta:.6f} (loss {base_loss:.6f} -> {loss_after:.6f})")
        # update self.loss_dict (for logging / further logic)
        # recompute mean abs grads for layers (optional)
        self.loss_dict = {}
        model.zero_grad()
        out2 = model(data)
        (self.criterion(out2, target)).backward()
        for nm, mod in model.named_modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)) and getattr(mod, "weight", None) is not None and mod.weight.grad is not None:
                self.loss_dict[nm] = float(mod.weight.grad.abs().mean().item())

        # sort for convenience
        self.loss_dict = dict(sorted(self.loss_dict.items(), key=lambda x: x[1], reverse=True))
        return self.loss_dict

    # apply patch
    BFA.progressive_bit_search = new_progressive_bit_search
    print("[INFO] Patched BFA.progressive_bit_search: candidate-eval bit-flip (no accuracy rise).")





except Exception as e:
    print("Import failed:", e)
    traceback.print_exc()
    sys.exit(1)


except Exception as e:
    print("Import failed:", e)
    traceback.print_exc()
    sys.exit(1)

# ------------------------------------------------------------
# 3️⃣  Ton modèle d’origine
# ------------------------------------------------------------
def build_original_model(device):
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=4, padding=1, bias=True),
        nn.ReLU(), nn.BatchNorm2d(32),
        nn.Conv2d(32, 32, kernel_size=4, padding=1, bias=True),
        nn.ReLU(), nn.BatchNorm2d(32),
        nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),

        nn.Conv2d(32, 64, kernel_size=4, padding=1, bias=True),
        nn.ReLU(), nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, kernel_size=4, padding=1, bias=True),
        nn.ReLU(), nn.BatchNorm2d(64),
        nn.MaxPool2d(2, 2), nn.Dropout2d(0.3),

        nn.Conv2d(64, 128, kernel_size=4, padding=1, bias=True),
        nn.ReLU(), nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=1, padding=0, bias=True),
        nn.ReLU(), nn.BatchNorm2d(128), nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(128, 10)
    )
    return model.to(device)

# ------------------------------------------------------------
# 4️⃣  Patch pour compatibilité BFA
# ------------------------------------------------------------
def make_model_BFA_compatible(model, N_bits=8):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.N_bits = N_bits
            m.register_buffer("b_w", torch.tensor([2**i for i in range(N_bits)], dtype=torch.int16))
    print("[INFO] Model patched for BFA compatibility.")
    return model

# ------------------------------------------------------------
# 5️⃣  Dataset + évaluation
# ------------------------------------------------------------
def get_test_loader(batch_size=256):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3,(0.5,)*3)])
    ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)

def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct/total

# ------------------------------------------------------------
# 6️⃣  Main
# ------------------------------------------------------------
def apply_random_bit_flip(model):
    """Flip one random quantized bit in a random Conv2d/Linear layer (in-place)."""
    # collect candidate modules (with weights)
    candidates = [(name, m) for name, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear)) and getattr(m, "weight", None) is not None]
    if not candidates:
        return None
    name, module = random.choice(candidates)

    # flatten index
    numel = module.weight.data.numel()
    flat_idx = random.randrange(numel)

    # quant params
    N_bits = int(getattr(module, "N_bits", 8))
    if N_bits <= 1:
        N_bits = 8
    mask_all = (1 << N_bits) - 1
    sign_bit = 1 << (N_bits - 1)
    SCALE = 2**(N_bits-1) - 1

    def float_to_unsigned_int(x):
        signed = int(round(max(min(x, 1.0), -1.0) * SCALE))
        return signed & mask_all

    def unsigned_int_to_float(u):
        if u & sign_bit:
            signed = u - (1 << N_bits)
        else:
            signed = u
        return float(signed) / SCALE

    # current value
    with torch.no_grad():
        w_val = float(module.weight.data.view(-1)[flat_idx].item())
        u = float_to_unsigned_int(w_val)
        bit = random.randrange(N_bits)
        u_new = u ^ (1 << bit)
        w_new = unsigned_int_to_float(u_new)
        module.weight.data.view(-1)[flat_idx] = w_new

    return (name, flat_idx, bit, w_new)

def main(n_flips=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_original_model(device)
    weights_path = os.path.join(script_dir, "weights.pth")
    #weights_path = os.path.join(script_dir, "pruned_modele.pth")

    if os.path.isfile(weights_path):
        print("[INFO] Loading weights:", weights_path)
        state = torch.load(weights_path, map_location=device)
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        model.load_state_dict(state_dict, strict=True)
        print("[INFO] Weights loaded.")
    else:
        print("[WARN] No weights.pth found, using random init.")

    # --- sauvegarder les poids initiaux avant patch ---
    original_state = {k: v.clone().detach() for k, v in model.state_dict().items()}

    # patch BFA pour compatibilité
    model = make_model_BFA_compatible(model)

    if weight_conversion:
        try:
            weight_conversion(model)
        except Exception as e:
            print("[WARN] weight_conversion:", e)

    test_loader = get_test_loader()
    base_acc = evaluate_accuracy(model, test_loader, device)
    print(f"[INFO] Base accuracy: {base_acc*100:.2f}%")

    criterion = nn.CrossEntropyLoss()
    attacker = BFA(criterion=criterion, model=model, k_top=10)

    # --- BFA attack ---
    accuracies = [base_acc]
    grad_data, grad_target = next(iter(test_loader))
    grad_data, grad_target = grad_data[:8].to(device), grad_target[:8].to(device)

    for i in range(1, n_flips+1):
        print(f"\n--- BFA Flip {i}/{n_flips} ---")
        try:
            attacker.progressive_bit_search(model, grad_data, grad_target)
        except Exception as e:
            print("[ERROR] during BFA flip:", e)
            traceback.print_exc()
            break
        acc = evaluate_accuracy(model, test_loader, device)
        accuracies.append(acc)
        print(f"BFA Accuracy after {i} flips: {acc*100:.2f}%")

    torch.save(model.state_dict(), os.path.join(script_dir, "weights_attacked.pth"))

    # --- Random-flip baseline on fresh model ---
    model_rand = build_original_model(device)
    model_rand.load_state_dict(original_state, strict=True)  # charge poids initiaux
    model_rand = make_model_BFA_compatible(model_rand)
    if weight_conversion:
        try:
            weight_conversion(model_rand)
        except Exception as e:
            print("[WARN] weight_conversion (rand):", e)

    accuracies_rand = [base_acc]
    for i in range(1, n_flips+1):
        print(f"\n--- Random Flip {i}/{n_flips} ---")
        info = apply_random_bit_flip(model_rand)
        if info is None:
            print("[WARN] No module to flip on random baseline.")
            break
        name, idx, bit, wnew = info
        print(f"[RANDOM] flipped layer={name}, idx={idx}, bit={bit}")
        acc_r = evaluate_accuracy(model_rand, test_loader, device)
        accuracies_rand.append(acc_r)
        print(f"Random Accuracy after {i} flips: {acc_r*100:.2f}%")


    # --- summary print ---
    print("\n=== Accuracy summary BFA ===")
    for i, a in enumerate(accuracies):
        print(f"After {i} flips: {a*100:.2f}%")
    print("\n=== Accuracy summary Random ===")
    for i, a in enumerate(accuracies_rand):
        print(f"After {i} flips: {a*100:.2f}%")

    # --- plot both curves ---
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9,6))
    plt.plot(range(len(accuracies)), [a*100 for a in accuracies], marker='o', label='BFA (max-impact)')
    plt.plot(range(len(accuracies_rand)), [a*100 for a in accuracies_rand], marker='s', label='Random bit flips')
    plt.title("Accuracy vs Number of Bit Flips — BFA vs Random")
    plt.xlabel("Number of bit flips")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.xticks(range(max(len(accuracies), len(accuracies_rand))))
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Debug wrapper for deterministic startup & verbose errors ===
if __name__ == "__main__":
    try:
        print("[DEBUG] Starting main() now...")
        # Ajuste n_flips ici si besoin
        main(n_flips=20)
        print("[DEBUG] main() finished normally.")
    except Exception as e:
        # Affiche traceback complet afin de ne rien rater
        import traceback
        print("[FATAL] Exception in main():", e)
        traceback.print_exc()
        # Dump minimal environment info
        try:
            import torch
            print(f"[DEBUG] torch cuda available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}")
        except Exception:
            pass
        # Save error to log file for inspection
        with open(os.path.join(script_dir, "run_bfa_attack_error.log"), "w", encoding="utf-8") as f:
            f.write("Exception:\n")
            traceback.print_exc(file=f)
        raise
