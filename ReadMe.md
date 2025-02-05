# 📌 Prédiction du Risque de Crédit avec Azure ML et Optuna

Ce projet a pour objectif de **prédire le risque de défaut de crédit** à partir de données financières et démographiques, en utilisant **Azure ML**, **Optuna** pour l'optimisation des hyperparamètres, et **MLflow** pour le suivi des expérimentations sur Azure Cloud.

---

## 🎯 **1. Contexte du Projet**
### 📌 **1.1. Objectif du Dataset**
- **Dataset utilisé** : [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit).
- **Objectif** : Prédire si un client fera défaut dans les **2 prochaines années** (la variable cible est : `SeriousDlqin2yrs`).
- **Problématique** :
  - **Données déséquilibrées**
  - **Présence de valeurs aberrantes**
  - **Données manquantes** (`MonthlyIncome`, `NumberOfDependents`).
  - **Interprétabilité**

---

## 🔥 **2. Approche Méthodologique**
Ce projet s’articule autour de **trois grandes étapes** :

### ✅ **2.1 Analyse Exploratoire et Prétraitement**
- **Nettoyage des données** : Gestion des valeurs manquantes (remplissage par médiane, imputations).
- **Transformation des variables** : Log-transformation pour réduire l’asymétrie.
- **Détection des outliers** : Méthodes basées sur la distribution (boxplots, kurtosis).

---

### 🤖 **2.2 Optimisation des Modèles**
Nous avons testé plusieurs modèles :
- **Régression Logistique** : Modèle classique et interprétable en classification binaire. Il sert de baseline et permet de comprendre l'impact des variables sur la probabilité de défaut.
- **Random Forest** : Algorithme robuste et efficace, particulièrement en présence de bruit et de données hétérogènes. Grâce à son approche ensembliste, il réduit le risque de sur-apprentissage et capture des interactions complexes entre les variables.
- **XGBoost** : Algorithmes de boosting puissants, particulièrement adaptés aux jeux de données déséquilibrés. Leur capacité à pondérer les erreurs et à ajuster dynamiquement l'importance des classes permet d'améliorer la performance prédictive, notamment sur la minorité des cas de détresse financière.
- **LightGBM** : Alternative performante à XGBoost, LightGBM est optimisé pour les grands volumes de données et offre une meilleure rapidité d'entraînement. Suite à l'utilisation `d'AutoML` sur Azure, ce modèle a obtenu d'excellentes performances, ce qui m'a motivé son intégration dans notre comparaison.

#### 🔎 **Pourquoi utiliser Optuna pour l'optimisation des hyperparamètres ?**
- **Optimisation bayésienne** `L'estimation de densité de Parzen` avec Optuna est plus flexibles que les modèles gaussiens classiques.
- **Exploration efficace du paramètre espace**.

Les **hyperparamètres optimisés** :
- **XGBoost** : `n_estimators`, `learning_rate`, `max_depth`
- **LightGBM** : `n_estimators`, `learning_rate`
- **RandomForest** : `n_estimators`, `max_depth`
- **Logistic Regression** : `C`

---

### 📈 **2.3 Évaluation des Résultats**
Nous avons suivi **trois métriques clés** :
1. **AUC-ROC** : Mesure la capacité à discriminer entre bons et mauvais clients.
2. **F1-score** : Prend en compte le déséquilibre des classes.
3. **Gini** : 

Les **modèles de boosting** (XGBoost, LightGBM) offrent **les meilleurs scores AUC et donc Gini**.

---

## 📊 **3. Analyse des Résultats**
### 📌 **3.1 Importance des Variables (SHAP & AzureML Interpret)**
- **`RevolvingUtilizationOfUnsecuredLines`** : Indicateur clé du défaut.
- **`DebtRatio`** : Plus élevé chez les clients en défaut.
- **`MonthlyIncome`** : Impacte directement la solvabilité.

### 📌 **3.2 Courbes de Précision-Rappel**
- **Optimisation du seuil de décision** pour équilibrer **précision vs rappel**.

---

## 🏆 **4. Soumission sur Kaggle**
Meilleur résultat obtenue :
- **private** : 0.86472
- **public** : 0.85805


