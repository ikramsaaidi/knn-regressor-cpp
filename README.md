<div align="center">

# 🤖 KNN Regressor — From Scratch in C++

**Implémentation d'un algorithme K-Nearest Neighbors Regressor from scratch en C++**

*Mini-projet universitaire — Machine Learning & Programmation Orientée Objet*

![C++](https://img.shields.io/badge/C++-17-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)
![POO](https://img.shields.io/badge/POO-Héritage%20%26%20Polymorphisme-6366F1?style=for-the-badge)
![ML](https://img.shields.io/badge/Machine%20Learning-From%20Scratch-22C55E?style=for-the-badge)
![Licence MIT](https://img.shields.io/badge/Licence-MIT-F59E0B?style=for-the-badge)

</div>

---

## 📌 Description

Ce projet implémente un **KNN Regressor (K-Nearest Neighbors)** entièrement **from scratch en C++17**, sans aucune bibliothèque externe de Machine Learning. L'algorithme est intégré dans une mini-bibliothèque ML structurée autour d'une **architecture orientée objet** avec héritage et polymorphisme.

> 💡 **Exemple :** Prédire le prix d'un appartement (en k€) à partir de sa surface et de son nombre de pièces, en trouvant les K voisins les plus proches dans le dataset d'entraînement.

---

## 🏗️ Architecture du projet

```
KNNRegressor/
├── include/
│   ├── MLModel.h           ← Classe abstraite de base (interface commune)
│   ├── SupervisedModel.h   ← Modèles supervisés (héritage MLModel)
│   ├── Regressor.h         ← Classe de régression (score R², MSE)
│   ├── KNNRegressor.h      ← KNN Regressor (héritage Regressor)
│   └── Matrix.h            ← Classe Matrix (stockage des données)
├── src/
│   ├── SupervisedModel.cpp
│   ├── Regressor.cpp
│   ├── KNNRegressor.cpp    ← Implémentation principale
│   └── Matrix.cpp
├── main.cpp                ← Programme principal + tests
├── CMakeLists.txt          ← Configuration CMake
├── how_to_run.txt          ← Instructions de compilation
└── README.md               ← Ce fichier
```

### Arbre d'héritage

```
MLModel  (classe abstraite)
└── SupervisedModel
    └── Regressor
        └── KNNRegressor  ✓ (notre implémentation)
```

---

## ⚙️ Algorithme KNN Regressor

### Principe

Le KNN Regressor prédit la valeur d'un point en calculant la **moyenne des K voisins les plus proches** dans l'espace des features.

### Métriques de distance supportées

| Métrique | Formule |
|---------|---------|
| **Euclidienne** (défaut) | `d(x,y) = √(Σ(xi - yi)²)` |
| **Manhattan** | `d(x,y) = Σ|xi - yi|` |

### Étapes de l'algorithme

```
1. fit(X_train, y_train)   → mémoriser les données d'entraînement
2. Pour chaque point x_test :
   a. Calculer la distance avec tous les points d'entraînement
   b. Sélectionner les K plus proches voisins
   c. Retourner la moyenne de leurs valeurs y
3. score(X_test, y_test)   → calculer le R²
```

### Score R²

```
R² = 1 - (SS_res / SS_tot)
   = 1 - Σ(yi - ŷi)² / Σ(yi - ȳ)²
```

---

## 🚀 Compilation et exécution

### Prérequis

- **g++** avec support C++17 (`g++ --version`)
- Ou **CMake** ≥ 3.10

### Méthode 1 — Compilation directe (g++)

```bash
# Cloner le repo
git clone https://github.com/ikramsaaidi/knn-regressor-cpp.git
cd knn-regressor-cpp

# Compiler
g++ -std=c++17 -I include main.cpp src/Matrix.cpp src/SupervisedModel.cpp src/Regressor.cpp src/KNNRegressor.cpp -o KNNRegressor

# Exécuter
./KNNRegressor        # Linux / macOS
KNNRegressor.exe      # Windows
```

### Méthode 2 — CMake

```bash
mkdir build && cd build
cmake ..
cmake --build .
./knn_regressor
```

---

## 📊 Résultats

Le programme principal teste le modèle sur un dataset immobilier (surface + nb pièces → prix en k€) :

```
=== KNN Regressor ===

Modele entraine  K = 3

Resultats sur le jeu de test :
------------------------------------
  i  |  Reel (ke)  |  Predit (ke)
------------------------------------
  0  |    175.00   |    176.67
  1  |    270.00   |    266.67
  2  |    320.00   |    320.00
------------------------------------

Score R2 = 0.9961

Influence du parametre K :
-------------------
  K  |     R2
-------------------
  1  |   0.9983
  2  |   0.9973
  3  |   0.9961
  4  |   0.9942
  5  |   0.9908
-------------------

Test polymorphisme (MLModel*) :
  Prediction point 0 : 176.67 ke
```

---

## 🧩 Concepts POO appliqués

| Concept | Application |
|---------|-------------|
| **Abstraction** | Classe abstraite `MLModel` avec méthodes virtuelles pures (`fit`, `predict`, `score`) |
| **Héritage** | Chaîne `MLModel → SupervisedModel → Regressor → KNNRegressor` |
| **Polymorphisme** | `MLModel* ptr = new KNNRegressor(3)` — appel via pointeur de base |
| **Encapsulation** | Attributs privés (`k_`, `X_train_`, `y_train_`), accesseurs `getK()` / `setK()` |
| **Modularité** | Séparation stricte `.h` / `.cpp`, un fichier par classe |

---

## 🛠️ Technologies utilisées

- **C++17** — Langage principal
- **STL** — `vector`, `algorithm`, `cmath`, `utility`
- **CMake** — Système de build
- **g++** — Compilateur

---

## 👥 Auteur

Réalisé par **Ikram** — 2025-2026

---

## 📄 Licence

Ce projet est distribué sous licence **MIT**. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

<div align="center">

**KNN Regressor C++** — *Implémenter pour vraiment comprendre*

</div>
