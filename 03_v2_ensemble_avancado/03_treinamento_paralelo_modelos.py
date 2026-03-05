"""
Fragmento de estudo (V2): configuracao dos tres modelos base.

Este arquivo mostra apenas a criacao dos estimadores do ensemble.
"""
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb


def criar_modelos_base(prefer_gpu=True):
    """Retorna os modelos base usados no stacking da V2."""
    # ETAPA 1 - XGBoost (bom equilibrio geral em tabular).
    xgb_model = xgb.XGBClassifier(
        device="cuda" if prefer_gpu else "cpu",
        tree_method="hist",
        n_estimators=10000,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        early_stopping_rounds=100,
        eval_metric="logloss",
        random_state=42,
    )

    # ETAPA 2 - LightGBM (eficiente em CPU e rapido para iterar).
    lgb_model = lgb.LGBMClassifier(
        n_estimators=10000,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=20,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )

    # ETAPA 3 - CatBoost (forte em dados categoricos e tabulares).
    cb_model = CatBoostClassifier(
        task_type="GPU" if prefer_gpu else "CPU",
        iterations=10000,
        depth=7,
        learning_rate=0.1,
        l2_leaf_reg=3.0,
        random_state=42,
        verbose=0,
        early_stopping_rounds=100,
        eval_metric="Logloss",
    )

    return {"xgb": xgb_model, "lgb": lgb_model, "cb": cb_model}


if __name__ == "__main__":
    print("[*] Criando modelos base da V2...")
    modelos = criar_modelos_base(prefer_gpu=True)
    print(f"    OK: modelos disponiveis -> {list(modelos.keys())}")
