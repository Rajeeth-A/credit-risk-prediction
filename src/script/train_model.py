import os
import argparse
import optuna
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import argparse
import mlflow
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
# import shap

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
# from interpret.ext.blackbox import TabularExplainer
# from azureml.interpret import ExplanationClient

# ======================== Optimisation des hyperparamètres ========================
def optimiser_modele(trial, X_train, y_train, X_test, y_test, model_type):
    """Optimize model hyperparameters using Optuna based on the selected model type"""

    if model_type == "LightGBM":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.5),
        }
        model = LGBMClassifier(**params)

    elif model_type == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.5),
        }
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")

    elif model_type == "RandomForest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
        }
        model = RandomForestClassifier(**params)

    elif model_type == "LogisticRegression":
        params = {
            "C": trial.suggest_loguniform("C", 0.01, 10),
        }
        model = LogisticRegression(**params, max_iter=500)

    else:
        raise ValueError(f"Modèle non pris en charge : {model_type}")

    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, model.predict(X_test))

    trial.set_user_attr("F1-score", f1)

    return auc

# ======================== Sauvegarde des graphiques ========================
def plot_courbe_precision_rappel(y_test, y_pred_proba, filename):
    """Save the Precision-Recall curve as an image"""
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label="Courbe Précision-Rappel")
    plt.xlabel("Rappel")
    plt.ylabel("Précision")
    plt.title("Courbe Précision-Rappel")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()

def plot_confusion_matrix(y_test, y_pred, filename):
    """Save the confusion matrix as an image"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Défaut", "Défaut"], yticklabels=["Non-Défaut", "Défaut"])
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de Confusion")
    plt.savefig(filename)
    plt.close()

def plot_optuna(study, model_type):
    """Generate and save a 2D (for logistic) or 3D (other models) plot of Optuna hyperparameter tuning"""

    df_results = study.trials_dataframe()
    df_results = df_results[df_results.state == "COMPLETE"]

    if model_type == "LogisticRegression":
        filename = "logistic_auc_2d.html"
        if "params_C" in df_results.columns:
            fig = px.scatter(
                df_results, 
                x="params_C", 
                y="value", 
                title="Impact de C sur l'AUC (Logistic Regression)",
                labels={"params_C": "C", "value": "AUC"},
                color="value", 
                color_continuous_scale="viridis"
            )
            fig.write_html(filename)
            return filename

    elif model_type in ["LightGBM", "XGBoost"]:
        filename = f"{model_type.lower()}_auc_3d.html"
        if "params_n_estimators" in df_results.columns and "params_learning_rate" in df_results.columns:
            fig = px.scatter_3d(
                df_results,
                x="params_n_estimators",
                y="params_learning_rate",
                z="value",
                color="value",
                color_continuous_scale="viridis",
                labels={
                    "params_n_estimators": "n_estimators",
                    "params_learning_rate": "learning_rate",
                    "value": "AUC"
                },
                title=f"Impact des Hyperparamètres sur l'AUC pour {model_type}"
            )
            fig.write_html(filename)
            return filename

    elif model_type == "RandomForest":
        filename = "randomforest_auc_3d.html"
        if "params_n_estimators" in df_results.columns and "params_max_depth" in df_results.columns:
            fig = px.scatter_3d(
                df_results,
                x="params_n_estimators",
                y="params_max_depth",
                z="value",
                color="value",
                color_continuous_scale="viridis",
                labels={
                    "params_n_estimators": "n_estimators",
                    "params_max_depth": "Max depth",
                    "value": "AUC"
                },
                title="Impact des Hyperparamètres sur l'AUC pour RandomForest"
            )
            fig.write_html(filename)
            return filename

    return None

# ======================== Exécution principale ========================
def main():
    """ Main function of the script"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, help="Modèle à optimiser")
    parser.add_argument("--data", type=str, required=True, help="Chemin vers les données")
    parser.add_argument("--registered_model_name", type=str, required=True, help="Nom du modèle à enregistrer")
    parser.add_argument("--n_trials", type=int, default=20, help="Nombre d'itérations Optuna")
    args = parser.parse_args()

    df = pd.read_csv(args.data, header=0)
    print(df.columns)
    if "age_cat" in df.columns:
        df = df.drop(columns=["age_cat"])
    print(df.columns)
    y = df.pop("SeriousDlqin2yrs").values
    X = df.values
    feature_names=df.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optimiser_modele(trial, X_train, y_train, X_test, y_test, args.model_type), n_trials=args.n_trials)

    best_params = study.best_params
    best_auc = study.best_value
    best_f1 = study.best_trial.user_attrs["F1-score"]

    print("\n======================= Résultat =======================")
    print(f"✅ Meilleurs paramètres : {best_params}")
    print(f"✅ Meilleur AUC : {best_auc}")
    print(f"✅ Meilleur F1-score : {best_f1}")
    print("========================================================")

    model_class = {"LightGBM": LGBMClassifier, "XGBoost": XGBClassifier, "RandomForest": RandomForestClassifier, "LogisticRegression": LogisticRegression}
    best_model = model_class[args.model_type](**best_params)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    plot_courbe_precision_rappel(y_test, y_pred_proba, "precision_recall_curve.png")
    plot_confusion_matrix(y_test, y_pred, "confusion_matrix.png")
    optuna_plot = plot_optuna(study, args.model_type)

    # explainer = TabularExplainer(best_model, X, features=feature_names, classes=["Non-Défaut", "Défaut"])
    # explanation = explainer.explain_global(X)

    # # 🔹 Enregistrer l'explication dans Azure ML
    # explanation_client = ExplanationClient.from_run(mlflow.active_run())
    # explanation_client.upload_model_explanation(explanation, comment="Explication SHAP du modèle")

    # # 🔹 Graphique des features importantes
    # shap_values = explainer.explain_local(X[:100])  # Prend 100 lignes pour éviter d’être trop lourd
    # plt.figure(figsize=(8, 6))
    # shap.summary_plot(shap_values.local_importance_values, X[:100], feature_names=feature_names, show=False)
    # plt.savefig("feature_importance.png")
    # mlflow.log_artifact("feature_importance.png")
    # plt.close()

    # # 🔹 Graphique de l'impact des features
    # plt.figure(figsize=(8, 6))
    # shap.summary_plot(shap_values.local_importance_values, X[:100], feature_names=feature_names, plot_type="bar", show=False)
    # plt.savefig("feature_impact.png")
    # mlflow.log_artifact("feature_impact.png")
    # plt.close()

    mlflow.start_run()
    mlflow.log_params(best_params)
    mlflow.log_metric("best_auc", best_auc)
    mlflow.log_metric("best_f1_score", best_f1)
    mlflow.sklearn.log_model(best_model, args.registered_model_name)

    mlflow.log_artifact("precision_recall_curve.png")
    mlflow.log_artifact("confusion_matrix.png")
    if optuna_plot:
        mlflow.log_artifact(optuna_plot)

    mlflow.end_run()

if __name__ == "__main__":
    main()
