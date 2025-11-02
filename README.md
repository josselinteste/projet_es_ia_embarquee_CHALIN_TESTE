# Déploiement d’un modèle DNN préentraîné sur le dataset CIFAR-10 pour microcontrôleur

Ce projet étudie le **déploiement d’un modèle de classification** sur une cible embarquée **STM32**.  
Il est divisé en deux parties principales :

---

## 🧠 Partie 1 — Conception du modèle

Cette première partie concerne la **conception et l’optimisation** du modèle via :
- l’élaboration d’une **méthode de pruning**,  
- la **sélection d’une architecture personnalisée**,  
- et l’**évaluation** de cette dernière sur le dataset **CIFAR-10**.

### Contenu :
- **`Pruning.pdf`** — rapport détaillant la méthode de pruning et les choix d’architecture du nouveau modèle ;  (dossier *Pruning*) 
- **`pruning_modele.ipynb`** — démonstration pratique de la méthode de pruning appliquée au modèle. (dossier *Pruning*) 
- **`entrainement_evaluation.ipynb`** - code d'entrainement et d'évaluation du modèle. 
- **`outils.ipynb`** - Ensemble de fonctions, pour la plupart absentes des autres notebooks, mais qui nous ont permis d’explorer la méthode de **pruning** et son évaluation.
- **`resultats_pruning.xlxs`** Tableau récapitulatif de l'ensemble des résultats obtenus pour les différentes méthodes de pruning (dossier *Pruning*) 
- **`resultats_pruning_bruts.txt`** Document enregistrants l'ensemble des résultats brutes pour les différentes méthodes de pruning (dossier *Pruning*) 
---

## ⚙️ Partie 2 — Déploiement sur microcontrôleur

La seconde partie porte sur le **déploiement du modèle conçu** sur la cible **STM32**.

### Contenu :
- **Fichiers de déploiement** pour le microcontrôleur ;  
- **Modèle converti** et prêt à être intégré sur la plateforme embarquée.

---

## 🧩 Objectif global

L’objectif du projet est de **proposer une méthodologie complète** allant de la conception d’un modèle de classification optimisé à son **déploiement sur une cible embarquée à ressources limitées**.

---
---
# VGG11_CIFAR10_simple - Analyse Architecturale Complète

## 📋 Vue d'ensemble

Cette analyse présente une architecture VGG-11 modifiée et optimisée pour la classification d'images CIFAR-10. Le modèle intègre des techniques modernes de régularisation tout en conservant la philosophie architecturale VGG classique. L'architecture VGG, introduite par Simonyan et Zisserman en 2014, a démontré l'efficacité des réseaux profonds utilisant exclusivement des filtres de petite taille (3×3), principe qui est conservé dans cette adaptation.

### Caractéristiques principales

- **8 couches convolutionnelles** organisées en 6 blocs distincts qui extraient progressivement des features de plus en plus abstraites
- **Classificateur dense** à 3 couches qui transforme les features extraites en prédictions de classes
- **~1.34 millions de paramètres** (optimisé pour CIFAR-10, bien inférieur aux 132M du VGG-11 original)
- **Régularisation moderne** : BatchNormalization pour la stabilité et SpatialDropout2D pour la robustesse

---

## 🏗️ Architecture

### Philosophie de conception

L'architecture s'inspire du paradigme VGG classique avec des améliorations modernes adaptées aux défis spécifiques du dataset CIFAR-10 :

- **Filtres uniformes** : Noyaux 3×3 exclusivement, permettant un empilement profond avec un nombre de paramètres réduit
- **Profondeur contrôlée** : 8 couches convolutionnelles, un compromis entre la capacité d'apprentissage et le risque de sur-apprentissage
- **Régularisation** : BatchNorm après chaque activation et SpatialDropout2D dans les blocs stratégiques
- **Adaptation** : Optimisée pour images 32×32 pixels avec un nombre de filtres maximal de 128 (vs 512 dans VGG original)

### Structure hiérarchique

Le modèle suit une structure pyramidale à 4 niveaux où les dimensions spatiales diminuent progressivement tandis que la profondeur des canaux augmente. Cette approche classique permet de capturer d'abord des détails fins avec une haute résolution, puis des concepts de plus en plus abstraits :

1. **Bloc 1** : Extraction des caractéristiques de bas niveau (32 filtres) - détection des contours, transitions de couleurs
2. **Bloc 2** : Consolidation des motifs élémentaires (32 filtres) avec première réduction spatiale
3. **Bloc 3** : Capture des motifs intermédiaires (64 filtres) - textures, formes géométriques simples
4. **Bloc 4** : Approfondissement des features intermédiaires (64 filtres) avec deuxième réduction spatiale
5. **Bloc 5** : Extraction des caractéristiques de haut niveau (128 filtres) - parties d'objets, motifs complexes
6. **Bloc 6** : Raffinement final (128 filtres) avec résolution spatiale minimale

---

## 📊 Détails de l'architecture

### Progression des caractéristiques

Le tableau ci-dessous montre l'évolution des dimensions et de la complexité à travers le réseau. On observe que le volume d'information croît initialement, puis décroît progressivement grâce aux opérations de max pooling :

| Bloc | Canaux entrée | Canaux sortie | Dimension spatiale | Volume total |
|------|---------------|---------------|--------------------|--------------| 
| 1 | 3 | 32 | 32×32 | 32,768 valeurs |
| 2 | 32 | 32 | 32×32 → 16×16 | 8,192 valeurs |
| 3 | 32 | 64 | 16×16 | 16,384 valeurs |
| 4 | 64 | 64 | 16×16 → 8×8 | 4,096 valeurs |
| 5 | 64 | 128 | 8×8 → 4×4 | 2,048 valeurs |
| 6 | 128 | 128 | 4×4 → 2×2 | 512 valeurs |

La progression montre une compensation intelligente : quand la résolution spatiale diminue de moitié, le nombre de canaux double, maintenant ainsi la capacité représentationnelle du réseau.

### Classificateur dense

Le classificateur adopte une architecture en pyramide inversée, contrastant avec la structure en entonnoir de la partie convolutionnelle. Cette expansion puis contraction permet de combiner richement les features extraites :

- **Flatten** : 512 dimensions (128 × 2 × 2), transformation du tenseur 3D en vecteur 1D
- **Dense 1** : 512 → 1024 (ReLU + Dropout 0.3) - expansion pour créer un espace de combinaisons riche
- **Dense 2** : 1024 → 512 (ReLU + Dropout 0.3) - synthèse et compression de l'information discriminante
- **Sortie** : 512 → 10 (Softmax) - projection vers les 10 classes avec distribution de probabilité

---

## 🔧 Innovations architecturales

### BatchNormalization stratégique

Intégrée après chaque activation ReLU dans tous les blocs convolutionnels, la BatchNormalization normalise la distribution des activations en ajustant leur moyenne à 0 et leur variance à 1. Cette technique apporte plusieurs bénéfices mesurables :

- **Stabiliser** la variance interne des activations, réduisant le phénomène d'Internal Covariate Shift
- **Accélérer** la convergence durant l'entraînement en permettant des learning rates plus élevés
- **Régulariser** implicitement le modèle grâce au bruit introduit par les statistiques de batch
- **Renforcer** la robustesse à l'initialisation des poids, facilitant l'expérimentation

La formule mathématique appliquée est : `x̂ = (x - μ_batch) / √(σ²_batch + ε)` suivie de `y = γ * x̂ + β`, où γ et β sont des paramètres apprenables qui permettent au réseau de retrouver la distribution originale si nécessaire.

### SpatialDropout2D

Amélioration significative par rapport au dropout classique, spécifiquement conçue pour les architectures convolutionnelles. Le SpatialDropout2D désactive aléatoirement des feature maps entières plutôt que des neurones individuels :

| Caractéristique | Dropout Classique | SpatialDropout2D |
|-----------------|-------------------|------------------|
| Unité de suppression | Neurones individuels | **Canaux complets** (25% avec taux 0.25) |
| Préservation spatiale | ❌ Non | ✅ Oui - maintient les corrélations spatiales |
| Corrélations locales | Perturbées par le bruit | Maintenues dans chaque feature map |
| Efficacité convolutionnelle | Limitée | Optimale pour les CNN |

Cette approche est plus efficace car les valeurs au sein d'une même feature map sont fortement corrélées spatialement (elles proviennent du même filtre). Désactiver des pixels aléatoires ne forcerait pas le réseau à développer des représentations robustes, alors que supprimer des canaux entiers oblige le réseau à ne pas dépendre excessivement de certaines features spécifiques.

### Séquence des opérations

**Séquence appliquée** : Conv2D → ReLU → [SpatialDropout2D] → BatchNorm → [MaxPool2D]

Cette séquence présente l'avantage de normaliser les activations après application du dropout, stabilisant ainsi la distribution des données d'entrée de la couche suivante. L'activation ReLU est appliquée avant la normalisation, ce qui permet de normaliser une distribution déjà filtrée par la non-linéarité.

---

## 📈 Distribution des paramètres

### Répartition par composant

Le tableau détaillé ci-dessous révèle la distribution complète des paramètres à travers l'architecture. On observe un déséquilibre notable vers les couches denses qui concentrent la majorité des paramètres :

| Composant | Paramètres | Pourcentage | Calcul détaillé |
|-----------|------------|-------------|-----------------|
| **Couches Convolutionnelles** | **287,008** | **21.4%** | |
| Conv2D (3→32) | 896 | 0.07% | 3×3×3×32 + 32 biais |
| Conv2D (32→32) | 9,248 | 0.69% | 3×3×32×32 + 32 biais |
| Conv2D (32→64) | 18,496 | 1.38% | 3×3×32×64 + 64 biais |
| Conv2D (64→64) | 36,928 | 2.75% | 3×3×64×64 + 64 biais |
| Conv2D (64→128) | 73,856 | 5.50% | 3×3×64×128 + 128 biais |
| Conv2D (128→128) | 147,584 | 11.0% | 3×3×128×128 + 128 biais |
| **BatchNormalization** | **896** | **0.07%** | |
| BN (32 canaux) × 2 | 128 | 0.01% | (γ + β) × 32 × 2 blocs |
| BN (64 canaux) × 2 | 256 | 0.02% | (γ + β) × 64 × 2 blocs |
| BN (128 canaux) × 2 | 512 | 0.04% | (γ + β) × 128 × 2 blocs |
| **Couches Denses** | **1,055,242** | **78.6%** | |
| Dense (512→1024) | 525,312 | 39.1% | 512×1024 + 1024 biais |
| Dense (1024→512) | 524,800 | 39.1% | 1024×512 + 512 biais |
| Dense (512→10) | 5,130 | 0.38% | 512×10 + 10 biais |
| **TOTAL** | **~1,343,146** | **100%** | |

### Analyse de l'efficacité

**Observations critiques** révélant les forces et faiblesses de l'architecture :

- **78.6%** des paramètres concentrés dans seulement 3 couches denses du classificateur
- Les deux premières couches denses contiennent à elles seules plus de 1 million de paramètres
- Cette concentration peut créer un risque de sur-apprentissage dans le classificateur
- Les couches convolutionnelles ne représentent que **21.4%** des paramètres mais effectuent l'essentiel du travail d'extraction de features
- La BatchNormalization ajoute un overhead paramétrique négligeable (0.07%) pour un bénéfice substantiel
- Meilleur équilibre que certaines architectures CNN basiques où les couches denses peuvent représenter >90% des paramètres

---

## 🆚 Comparaison avec VGG-11 Original

### Adaptations pour CIFAR-10

Le tableau comparatif suivant illustre les modifications majeures apportées pour adapter VGG-11 au contexte spécifique de CIFAR-10 :

| Aspect | VGG-11 Original | Version CIFAR-10 | Justification |
|--------|-----------------|------------------|---------------|
| Résolution d'entrée | 224×224 | 32×32 | CIFAR-10 natif |
| Nombre de couches conv | 8 | 8 | Maintenu identique |
| Filtres maximum | 512 | 128 | Réduit de 75% |
| Première couche dense | 25,088 entrées | 512 entrées | Adapté à la résolution |
| Taille classificateur | 25K→4K→4K→1K | 512→1K→512→10 | Simplifié pour 10 classes |
| BatchNormalization | ❌ Non | ✅ Oui | Technique moderne |
| SpatialDropout | ❌ Non | ✅ Oui | Régularisation spatiale |
| Paramètres totaux | 132M | 1.34M | Réduction de 99% |

### Justifications des modifications

**Réduction de la complexité des filtres** - La limitation à 128 filtres maximum (vs 512 dans VGG original) s'explique par plusieurs facteurs convergents :

- **Résolution d'entrée réduite** : Avec des images 32×32 au lieu de 224×224, il y a simplement moins d'information spatiale à encoder
- **Complexité du dataset** : CIFAR-10 avec 10 classes est considérablement plus simple qu'ImageNet avec 1000 classes
- **Prévention du sur-apprentissage** : Une capacité excessive conduirait à mémoriser les données d'entraînement plutôt qu'à généraliser

**Intégration de la régularisation moderne** - L'ajout de BatchNorm et SpatialDropout répond aux standards actuels d'entraînement :

- **Stabilité d'entraînement** : BatchNorm réduit drastiquement la sensibilité à l'initialisation des poids
- **Généralisation** : SpatialDropout prévient le sur-apprentissage en forçant la redondance des features
- **Performance** : Ces techniques permettent d'atteindre la convergence plus rapidement avec moins d'époques

---

## 📊 Performances attendues

### Estimation théorique

Basé sur l'architecture et les techniques employées, ainsi que sur les résultats typiques d'architectures similaires sur CIFAR-10 :

| Métrique | Estimation | Explication |
|----------|------------|-------------|
| Accuracy CIFAR-10 | 88-93% | Architecture VGG + régularisation moderne |
| Époques de convergence | 50-80 | BatchNorm accélère mais dataset complexe |
| Temps d'entraînement | Modéré | ~5-10 min/epoch sur GPU moderne |
| Stabilité | Élevée | BatchNorm assure une convergence stable |
| Généralisation | Bonne | Régularisation triple (Spatial+Batch+Drop) |
| Robustesse | Élevée | Architecture VGG éprouvée depuis 2014 |

### Benchmarking architectural

Positionnement du modèle par rapport aux architectures de référence sur CIFAR-10. Ce tableau permet de situer les performances attendues dans le contexte plus large de l'état de l'art :

| Architecture | Paramètres | CIFAR-10 Acc. | Complexité | Commentaire |
|--------------|------------|---------------|------------|-------------|
| LeNet-5 | 60K | ~70% | Faible | Trop simple pour CIFAR-10 |
| AlexNet Adapté | 2M | ~85% | Moyenne | Performance correcte mais datée |
| **VGG-11 CIFAR-10** | **1.34M** | **~90%** | **Moyenne** | **Bon compromis** |
| ResNet-20 | 270K | ~92% | Moyenne | Skip connections efficaces |
| DenseNet-40 | 1M | ~94% | Élevée | Dense connections avancées |
| EfficientNet-B0 | 5M | ~95% | Élevée | Architecture state-of-the-art |

Le modèle se positionne comme une baseline solide : pas le plus performant, mais un excellent équilibre entre simplicité, performances et ressources nécessaires.

---

## ⚠️ Limitations et défis

### Limitations architecturales

Malgré ses qualités, l'architecture présente plusieurs limitations inhérentes à sa conception séquentielle pure :

- **Absence de skip connections** : Contrairement aux architectures ResNet qui utilisent des connexions résiduelles, ce modèle peut souffrir de problèmes de gradient dans les couches profondes
- **Pooling agressif** : Quatre opérations de max pooling réduisent l'image de 32×32 à 2×2, entraînant une perte potentielle d'information spatiale fine qui pourrait être discriminante
- **Architecture séquentielle** : Pas de parallélisation des branches comme dans Inception, limitant la diversité des features à chaque niveau
- **Classificateur dense dominant** : 78.6% des paramètres concentrés dans 3 couches denses peut créer un goulot d'étranglement et un risque de sur-apprentissage localisé

### Défis computationnels

Plusieurs aspects de l'architecture posent des défis pratiques lors de l'entraînement et du déploiement :

- **Mémoire** : Les feature maps volumineuses des premières couches (32×32×32) nécessitent une mémoire GPU substantielle, surtout avec des batchs de grande taille
- **BatchNorm** : Dépendance à la taille du batch pour des statistiques fiables ; performance peut se dégrader avec des batchs très petits (<16)
- **Régularisation** : Équilibrage délicat entre dropout et BatchNorm nécessaire ; trop de régularisation peut sous-fitter, pas assez peut sur-fitter
- **Temps d'inférence** : Les couches denses massives ralentissent l'inférence comparé à des architectures plus modernes avec Global Average Pooling

---

## 🎓 Cas d'usage recommandés

### ✅ Idéal pour

Cette architecture brille dans plusieurs contextes spécifiques où ses caractéristiques sont particulièrement adaptées :

- **Prototypage rapide** : Architecture simple à implémenter, comprendre et déboguer ; excellente pour tester rapidement des idées
- **Enseignement** : Illustre parfaitement les concepts fondamentaux de CNN (convolution, pooling, normalisation, dropout)
- **Baseline** : Point de comparaison solide et reproductible pour expérimenter d'autres architectures ou techniques
- **Datasets similaires** : Images de faible/moyenne résolution (32×32 à 64×64) avec un nombre modéré de classes (10-100)
- **Ressources limitées** : 1.34M paramètres permet l'entraînement sur GPU grand public (GTX 1660, RTX 3060) avec batchs raisonnables

### ❌ Moins adapté pour

Certains contextes nécessitent des architectures plus spécialisées où ce modèle serait sous-optimal :

- **Résolution élevée** : Images >128×128 nécessiteraient plus de couches et de filtres, augmentant drastiquement les paramètres
- **Très nombreuses classes** : Au-delà de 100 classes, le classificateur dense deviendrait démesurément lourd
- **Détection fine** : La perte d'information spatiale via 4 poolings rend difficile la localisation précise d'objets
- **Production critique** : Architectures state-of-the-art (EfficientNet, Vision Transformer) offrent de meilleures performances
- **Inférence temps réel** : MobileNet ou SqueezeNet sont significativement plus rapides avec moins de paramètres

---

## 📚 Conclusion

### Synthèse des forces

Ce modèle représente une **adaptation moderne et réussie** du paradigme VGG pour CIFAR-10, avec plusieurs points forts identifiés :

✅ **Architecture éprouvée et stable** - Basée sur VGG, une architecture qui a fait ses preuves depuis 2014  
✅ **Régularisation multi-niveaux efficace** - Combinaison de SpatialDropout2D, BatchNorm et Dropout classique  
✅ **Complexité paramétrique raisonnable** - 1.34M paramètres offre un bon compromis capacité/généralisation  
✅ **Potentiel de performance élevé** - Estimation de ~90% d'accuracy sur CIFAR-10  
✅ **Excellent pour l'apprentissage** - Code simple et maintenable, concepts clairement illustrés  

### Axes de contexte

L'architecture se positionne comme un **excellent compromis entre simplicité conceptuelle et performance pratique**. Elle est particulièrement adaptée pour l'enseignement des concepts de deep learning et le prototypage rapide, tout en maintenant un potentiel de performance suffisant pour des applications réelles non critiques.

Le déséquilibre paramétrique vers le classificateur dense (78.6%) suggère une opportunité d'optimisation via des techniques comme le Global Average Pooling, qui pourrait réduire drastiquement le nombre de paramètres tout en maintenant voire améliorant les performances.

Cette architecture constitue une **baseline solide et reproductible** pour expérimenter avec différentes techniques de régularisation, stratégies d'entraînement, et modifications architecturales sur le dataset CIFAR-10.



*Document généré pour analyse architecturale détaillée - VGG11_CIFAR10_simple*
