
# Plan

## Orateur 1. Introduction et Baseline (Eugenie)

**Slide 1 - Introduction et Objectifs**

- Rappel de l'objectif général : implémenter du flow matching sur un jeu de données 2D simple
- Choisir un probability path simple entre bruit et données
- Cas utilisé : Optimal Transport Path (interpolation linéaire entre les deux distributions)
- Comparaison avec RealNVP

**Slide 2 - Rappels sur les Flows Normalisants Discrets et RealNVP**

- Rappel général sur les flows normalisants discrets (RealNVP) et leur fonctionnement

## Orateur 2. Flows Normalisants Continus (Lucas Duport)

**Slide 3 - Flow Discret vs Flow Continu**

- Au lieu d'apprendre la position dans l'espace latent, on apprend la vitesse de déplacement
- Rappel de l'ODE

**Slide 4 - L'Intuition du Flow Matching**

- Apprentissage d'un champ de vitesse conditionnel plutôt qu'une transformation bijective discrète
- Objectif : "pousser" un nuage de points (bruit) vers un autre nuage de points (données)
- Avantage : pas de contrainte d'architecture pour le réseau v_θ (pas besoin d'être inversible)

## Orateur 3. Théorie du Flow Matching (Arthur)

**Slide 5 - Formalisation de l'Objectif Flow Matching**

- Minimiser la différence entre le champ de vitesse appris et le champ de vitesse réel
- Problème : le champ de vitesse réel est inconnu
- Solution : utiliser le Conditional Flow Matching

**Slide 6 - Le Conditional Flow Matching**

Démonstration si nécessaire

## Orateur 4. Implémentation (Flavien)

**Slide 7 - Optimal Transport Path**

- Chemin simple entre bruit et données : interpolation linéaire
- Formule : x_t = (1-t)x_0 + tx_1
- Champ de vitesse réel : u_t = x_1 - x_0 (constante)
- Le réseau apprend à prédire cette constante

**Slide 8 - Échantillonnage (Inférence)**

- Résolution de l'ODE avec la méthode d'Euler
- En partant du bruit, intégrer la dynamique discrète pour obtenir des échantillons

## Orateur 5. Implémentation (Code) (Lucas Juanico)

**Slide 9 - Implémentation (Code)**

**Slide 10 - Exemple avec UNet**

## Orateur 6. Résultats et Comparaison (Baptiste)

**Slide 11 - Résultats et Comparaison**

- **RealNVP** : plus rapide à l'inférence (simple forward pass)
- **Flow Matching** : plus puissant mais plus lent à l'inférence (résolution d'ODE)
- **Avantage FM** : plus rapide à l'entraînement, convergence plus rapide (MSE sur ligne droite vs transformation complexe pour RealNVP)

