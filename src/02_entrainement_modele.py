import json
import os
from pathlib import Path
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment
from azure.ai.ml.sweep import Choice, Uniform, MedianStoppingPolicy
from azure.identity import DefaultAzureCredential

### -------------------------------------------
### 🔹 Chargement de la configuration depuis un fichier JSON
### -------------------------------------------
config_path = Path(__file__).resolve().parent.parent / "src/config" / "config_train.json"

if not config_path.exists():
    raise FileNotFoundError(f"Le fichier de configuration n'existe pas : {config_path}")

with open(config_path, "r", encoding="utf-8") as config_file_train:
    config = json.load(config_file_train)

### Extraction des paramètres de configuration
SUBSCRIPTION = config["SUBSCRIPTION"]
RESOURCE_GROUP = config["RESOURCE_GROUP"]
WS_NAME = config["WS_NAME"]
DATA_PATH = config["DATA_PATH"]
COMPUTE_CLUSTER = config["COMPUTE_CLUSTER"]
AML_ENVIRONMENT = config["AML_ENVIRONMENT"]

### -------------------------------------------
### 🔹 Connexion à Azure Machine Learning Workspace
### -------------------------------------------
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WS_NAME,
)

print(f"✅ Connexion réussie à Azure ML Workspace : {WS_NAME} ✅")
print(SUBSCRIPTION, RESOURCE_GROUP, WS_NAME)

### Vérification de la connexion au workspace
ws = ml_client.workspaces.get(WS_NAME)
print(ws.location, ":", ws.resource_group)

### -------------------------------------------
### 🔹 Configuration de l'environnement Azure ML
### -------------------------------------------
dependencies_dir = "./dependencies"
custom_env_name = "aml-scikit-learn"

custom_job_env = Environment(
    name=custom_env_name,
    description="Custom environment for Credit Card Defaults job",
    tags={"scikit-learn": "1.0.2"},
    conda_file=os.path.join(dependencies_dir, "conda.yaml"),
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
)

### Création ou mise à jour de l'environnement dans Azure ML
custom_job_env = ml_client.environments.create_or_update(custom_job_env)
print(f"L'environnement {custom_job_env.name} est configuré dans le workspace avec la version {custom_job_env.version}.")

### -------------------------------------------
### 🔹 Définition du job Optuna pour l'optimisation des hyperparamètres
### -------------------------------------------
train_src_dir = "./src"
registered_model_name = "credit_defaults_model"

optuna_job = command(
    inputs=dict(
        model_type="LightGBM",  ### Modèle choisi parmi "LightGBM", "XGBoost", "RandomForest", "LogisticRegression"
        data=Input(type="uri_file", path=DATA_PATH),
        registered_model_name=registered_model_name,
        n_trials=50,
    ),
    code="./src/",  ### Répertoire contenant le script d'entraînement
    command="python train_model.py --model_type ${{inputs.model_type}} --data ${{inputs.data}} --registered_model_name ${{inputs.registered_model_name}} --n_trials ${{inputs.n_trials}}",
    compute=COMPUTE_CLUSTER,
    environment=AML_ENVIRONMENT,
    display_name="optuna_credit_default_LightGBM2",
)

print(optuna_job)

### -------------------------------------------
### 🔹 Exécution des jobs sur Azure ML
### -------------------------------------------
print("Envoie des travaux sur Azure...")
ml_client.jobs.create_or_update(optuna_job)  ### Exécution du tuning avec Optuna
print("✅ Jobs envoyés avec succès ✅")


### -------------------------------------------
### 🔹 Définition du Hyperparameter Sweep Job (Tuning automatique)
### -------------------------------------------
### sweep_job = train_job.sweep(
###     compute=compute_cluster,
###     sampling_algorithm="random",
###     primary_metric="test-auc",
###     goal="Maximize",
### )

### sweep_job.set_limits(max_total_trials=20, max_concurrent_trials=10, timeout=7200)
### sweep_job.early_termination = MedianStoppingPolicy(delay_evaluation=5, evaluation_interval=2)

### ### ml_client.jobs.create_or_update(sweep_job)   ### Exécuter le sweep pour ajuster les hyperparamètres

