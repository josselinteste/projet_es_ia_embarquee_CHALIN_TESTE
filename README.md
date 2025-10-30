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
- **`Pruning.pdf`** — rapport détaillant la méthode de pruning et les choix d’architecture du nouveau modèle ;  
- **pruning_modele.ipynb** — démonstration pratique de la méthode de pruning appliquée au modèle.
- **entrainement_evaluation.ipynb** - code d'entrainement et d'évaluation du modèle
- **outils.ipynb** - Ensemble de fonctions, pour la plupart absentes des autres notebooks, mais qui nous ont permis d’explorer la méthode de **pruning** et son évaluation.
- **resultats_pruning.xlxs** Tableau récapitulatif de l'ensemble des résultats obtenus pour les différentes méthodes de pruning
- **resultats_pruning_bruts.txt** Document enregistrants l'ensemble des résultats brutes pour les différentes méthodes de pruning
---

## ⚙️ Partie 2 — Déploiement sur microcontrôleur

La seconde partie porte sur le **déploiement du modèle conçu** sur la cible **STM32**.

### Contenu :
- **Fichiers de déploiement** pour le microcontrôleur ;  
- **Modèle converti** et prêt à être intégré sur la plateforme embarquée.

---

## 🧩 Objectif global

L’objectif du projet est de **proposer une méthodologie complète** allant de la conception d’un modèle de classification optimisé à son **déploiement sur une cible embarquée à ressources limitées**.
