# Prédiction du Risque de Crédit avec Azure ML, Optuna et MLflow

Ce projet vise à **prédire le risque de défaut de crédit** à partir de données financières et démographiques, en utilisant **Azure Machine Learning** pour l'entraînement et le déploiement, **Optuna** pour l’optimisation des hyperparamètres, et **MLflow** pour le suivi des expérimentations sur Azure Cloud.

---

## **1. Contexte du Projet**

### **1.1. Objectif du Dataset**
- **Source** : [Give Me Some Credit (Kaggle)](https://www.kaggle.com/c/GiveMeSomeCredit).
- **Objectif** : Prédire si un client fera défaut dans les **deux prochaines années** (variable cible : `SeriousDlqin2yrs`).
- **Enjeux** :
  - Fort **déséquilibre des classes** (faible taux de défaut).
  - **Présence de valeurs aberrantes**.
  - **Données manquantes** (`MonthlyIncome`, `NumberOfDependents`).
  - **Exigence d’interprétabilité** pour un usage en contexte réel.

---

## **2. Approche Méthodologique**

Le projet se structure autour de **trois grandes étapes** :

### **2.1 Analyse Exploratoire et Prétraitement**
- **Nettoyage des données** : Imputation des valeurs manquantes (médiane, règles métiers).
- **Transformation des variables** : Log-transformations pour réduire la skewness.
- **Détection des outliers** : Analyse par distribution (boxplots, kurtosis, z-scores).

---

### **2.2 Modélisation et Optimisation**

Plusieurs modèles de classification ont été testés :

- **Régression Logistique** : Modèle de référence simple et interprétable, idéal pour évaluer l’impact des variables.
- **Random Forest** : Modèle ensembliste robuste face au bruit et aux données hétérogènes.
- **XGBoost** : Algorithme de boosting performant, adapté aux données déséquilibrées grâce à sa gestion fine des pondérations.
- **LightGBM** : Variante optimisée de boosting, plus rapide et efficace sur de grands jeux de données. Ce modèle a montré d'excellents résultats via AutoML sur Azure.

#### **Pourquoi Optuna pour l’optimisation ?**
- **Optimisation bayésienne** (Tree-structured Parzen Estimator) : meilleure exploration de l’espace des hyperparamètres que la recherche aléatoire ou grid search.
- **Recherche adaptative** : ajustement dynamique basé sur les essais précédents.

#### **Hyperparamètres optimisés :**
- **XGBoost** : `n_estimators`, `learning_rate`, `max_depth`
- **LightGBM** : `n_estimators`, `learning_rate`
- **Random Forest** : `n_estimators`, `max_depth`
- **Régression Logistique** : `C`

---

### **2.3 Évaluation des Modèles**

Trois métriques principales ont été utilisées :
1. **AUC-ROC** : Mesure la capacité de discrimination du modèle.
2. **F1-score** : Intéressant pour les données déséquilibrées.
3. **Gini** : Interprétation du pouvoir discriminant (2×AUC − 1).

Les modèles **boostés** (XGBoost, LightGBM) ont obtenu les **meilleurs scores AUC et Gini**, confirmant leur pertinence sur ce problème.

---

## **3. Analyse des Résultats**

### **3.1 Importance des Variables (via SHAP & AzureML Interpret)**
- **`RevolvingUtilizationOfUnsecuredLines`** : Utilisation excessive des crédits renouvelables → facteur prédictif fort.
- **`DebtRatio`** : Plus élevé chez les emprunteurs en difficulté.
- **`MonthlyIncome`** : Corrélé à la capacité de remboursement.

### **3.2 Seuil de Décision**
- Ajustement du **seuil de classification** via les **courbes précision-rappel**, pour un meilleur compromis entre faux positifs et faux négatifs.

---

## **4. Résultats Kaggle**
- **Score privé** : 0.86472  
- **Score public** : 0.85805
