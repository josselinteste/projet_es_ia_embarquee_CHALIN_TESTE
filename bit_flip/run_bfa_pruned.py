#!/usr/bin/env python3
# run_bfa_on_pruned.py
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
# 1️⃣  Créer un module "models.quantization" factice (comme avant)
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
    BFA_mod = import_module_from_path("BFA", os.path.join(attack_dir, "BFA.py"))
    DC_mod  = import_module_from_path("data_conversion", os.path.join(attack_dir, "data_conversion.py"))
    BFA = BFA_mod.BFA
    weight_conversion = getattr(DC_mod, "weight_conversion", None)
    print("[INFO] Imported BFA and data_conversion successfully.")
except Exception as e:
    print("Import failed:", e)
    traceback.print_exc()
    sys.exit(1)

# ------------------------------------------------------------
# 3️⃣  Model builder (must match architecture used to create state_dict)
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
# 4️⃣  Patch for BFA compatibility (buffers)
# ------------------------------------------------------------
def make_model_BFA_compatible(model, N_bits=8):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.N_bits = N_bits
            m.register_buffer("b_w", torch.tensor([2**i for i in range(N_bits)], dtype=torch.int16))
    return model

# ------------------------------------------------------------
# 5️⃣  Dataset + evaluation
# ------------------------------------------------------------
def get_test_loader(batch_size=256):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3,(0.5,)*3)])
    ds = torchvision.datasets.CIFAR10(root=os.path.join(script_dir,"data"), train=False, download=True, transform=tf)
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
# 6️⃣  Random single-bit flip (quantized) helper
# ------------------------------------------------------------
def apply_random_bit_flip(model):
    candidates = [(name, m) for name, m in model.named_modules()
                  if isinstance(m, (nn.Conv2d, nn.Linear)) and getattr(m, "weight", None) is not None]
    if not candidates:
        return None
    name, module = random.choice(candidates)
    numel = module.weight.data.numel()
    flat_idx = random.randrange(numel)

    N_bits = int(getattr(module, "N_bits", 8))
    if N_bits <= 1: N_bits = 8
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

    with torch.no_grad():
        w_val = float(module.weight.data.view(-1)[flat_idx].item())
        u = float_to_unsigned_int(w_val)
        bit = random.randrange(N_bits)
        u_new = u ^ (1 << bit)
        w_new = unsigned_int_to_float(u_new)
        module.weight.data.view(-1)[flat_idx] = w_new

    return (name, flat_idx, bit, w_new)

# ------------------------------------------------------------
# 7️⃣  BFA progressive_bit_search patch (candidate-eval version)
# ------------------------------------------------------------
# We assume BFA is imported: override its method to handle nn.Conv2d/Linear
old_progressive = BFA.progressive_bit_search

def new_progressive_bit_search(self, model, data, target, top_layers=3, top_weights=5):
    self.loss_dict = {}
    model.eval()
    model.zero_grad()

    output = model(data)
    base_loss = float(self.criterion(output, target).item())
    model.zero_grad()
    loss = self.criterion(output, target)
    loss.backward()

    layer_grads = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if getattr(module, "weight", None) is not None and module.weight.grad is not None:
                layer_grads.append((name, float(module.weight.grad.abs().mean().item())))
    if not layer_grads:
        return []

    layer_grads.sort(key=lambda x: x[1], reverse=True)
    layer_grads = layer_grads[:top_layers]

    best_candidate = None
    best_delta = -float("inf")

    for (name, _) in layer_grads:
        module = dict(model.named_modules())[name]
        grad_flat = module.weight.grad.view(-1).abs()
        k = min(top_weights, grad_flat.numel())
        if k <= 0:
            continue
        topk_vals, topk_idx = torch.topk(grad_flat, k)
        topk_idx = topk_idx.tolist()

        if not hasattr(module, "_flipped_bits"):
            module._flipped_bits = set()

        N_bits = int(getattr(module, "N_bits", 8))
        if N_bits <= 1: N_bits = 8
        SCALE = 2**(N_bits-1) - 1
        mask_all = (1 << N_bits) - 1

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
                    continue

                w_data = module.weight.data.view(-1)[flat_idx].item()
                u_curr = float_to_unsigned_int(w_data)
                u_cand = u_curr ^ (1 << bit)
                w_cand = unsigned_int_to_float(u_cand)

                with torch.no_grad():
                    old_val = module.weight.data.view(-1)[flat_idx].item()
                    module.weight.data.view(-1)[flat_idx] = w_cand

                with torch.no_grad():
                    out_cand = model(data)
                    loss_cand = float(self.criterion(out_cand, target).item())

                with torch.no_grad():
                    module.weight.data.view(-1)[flat_idx] = old_val

                delta = loss_cand - base_loss
                if delta > best_delta:
                    best_delta = delta
                    best_candidate = (name, flat_idx, bit, w_cand, loss_cand)

    if best_candidate is None:
        return []

    name, flat_idx, bit, w_cand, loss_after = best_candidate
    module = dict(model.named_modules())[name]
    with torch.no_grad():
        module.weight.data.view(-1)[flat_idx] = w_cand
    if not hasattr(module, "_flipped_bits"):
        module._flipped_bits = set()
    module._flipped_bits.add((flat_idx, bit))

    # recompute loss_dict for logging
    self.loss_dict = {}
    model.zero_grad()
    out2 = model(data)
    (self.criterion(out2, target)).backward()
    for nm, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)) and getattr(mod, "weight", None) is not None and mod.weight.grad is not None:
            self.loss_dict[nm] = float(mod.weight.grad.abs().mean().item())

    self.loss_dict = dict(sorted(self.loss_dict.items(), key=lambda x: x[1], reverse=True))
    return self.loss_dict

BFA.progressive_bit_search = new_progressive_bit_search

# ------------------------------------------------------------
# 8️⃣  Main adapted to accept full-model .pth or state_dict .pth
# ------------------------------------------------------------
def main(n_flips=20, weights_filename="pruned_modele.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(script_dir, weights_filename)

    if not os.path.isfile(weights_path):
        print("[ERROR] weights file not found:", weights_path)
        return

    # Try loading the file; it can be either a state_dict (dict) or a full nn.Module (Sequential)
    loaded = torch.load(weights_path, map_location=device)
    # If user saved the entire model, loaded is an nn.Module
    if isinstance(loaded, nn.Module):
        model = loaded
        print("[INFO] Loaded full model from .pth")
        # save a clean copy of state_dict before adding BFA buffers
        original_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
    elif isinstance(loaded, dict):
        # might be a dict that itself contains "state_dict"
        if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            state_dict = loaded["state_dict"]
        else:
            state_dict = loaded
        # build architecture then load state_dict
        model = build_original_model(device)
        model.load_state_dict(state_dict, strict=True)
        print("[INFO] Built model and loaded state_dict")
        original_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
    else:
        print("[ERROR] Unsupported .pth content type:", type(loaded))
        return

    # patch model for BFA
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

    # BFA attack loop
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

    # Random baseline: reload original model from original_state to ensure same start
    # Two cases: we have original_state dict (good) or not (then reload file)
    model_rand = None
    try:
        model_rand = build_original_model(device)
        model_rand.load_state_dict(original_state, strict=True)
        print("[INFO] Prepared fresh model for random baseline from saved original_state.")
    except Exception:
        # fallback: reload whole file
        print("[WARN] Could not load original_state into build_original_model(), reloading saved file.")
        loaded2 = torch.load(weights_path, map_location=device)
        if isinstance(loaded2, nn.Module):
            model_rand = loaded2
        elif isinstance(loaded2, dict):
            if "state_dict" in loaded2 and isinstance(loaded2["state_dict"], dict):
                st = loaded2["state_dict"]
            else:
                st = loaded2
            model_rand = build_original_model(device)
            model_rand.load_state_dict(st, strict=True)
        else:
            print("[ERROR] Cannot prepare random baseline model.")
            return

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

    # Summary + plot
    print("\n=== Accuracy summary BFA ===")
    for i, a in enumerate(accuracies):
        print(f"After {i} flips: {a*100:.2f}%")
    print("\n=== Accuracy summary Random ===")
    for i, a in enumerate(accuracies_rand):
        print(f"After {i} flips: {a*100:.2f}%")

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

# Debug wrapper
if __name__ == "__main__":
    try:
        print("[DEBUG] Starting main() now...")
        # change filename or n_flips if needed
        main(n_flips=20, weights_filename="pruned_modele.pth")
        print("[DEBUG] main() finished normally.")
    except Exception as e:
        import traceback
        print("[FATAL] Exception in main():", e)
        traceback.print_exc()
        try:
            import torch
            print(f"[DEBUG] torch cuda available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}")
        except Exception:
            pass
        with open(os.path.join(script_dir, "run_bfa_attack_error.log"), "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        raise
