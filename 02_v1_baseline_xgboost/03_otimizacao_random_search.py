"""
Fragmento de estudo (V1): busca de hiperparametros com RandomizedSearchCV.

Este arquivo mostra apenas a etapa de configuracao da busca.
"""
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import xgboost as xgb

# ETAPA 1 - Definir espaco de busca.
PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [300, 500, 800],
    "model__max_depth": [6, 7, 8, 10],
    "model__learning_rate": [0.05, 0.08, 0.1, 0.12, 0.15],
    "model__subsample": [0.7, 0.8, 0.9],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9],
}


def montar_busca_random(preprocessor="passthrough", device="cuda"):
    """
    Cria o objeto RandomizedSearchCV.

    Observacao: em ambiente sem GPU, usar device='cpu'.
    """
    # ETAPA 2 - Pipeline com SMOTE dentro da CV para evitar vazamento.
    xgb_base = xgb.XGBClassifier(
        device=device,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
    )
    cv_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(sampling_strategy=0.15, random_state=42)),
            ("model", xgb_base),
        ]
    )

    # ETAPA 3 - Definir estrategia de validacao.
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # ETAPA 4 - Criar busca aleatoria.
    search = RandomizedSearchCV(
        estimator=cv_pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=50,
        scoring="f1",
        cv=cv_strategy,
        random_state=42,
        n_jobs=1,
        verbose=2,
    )
    return search


if __name__ == "__main__":
    print("[*] Configurando RandomizedSearchCV da V1...")
    search = montar_busca_random()
    print("    OK: busca criada com 50 combinacoes e CV estratificada.")
