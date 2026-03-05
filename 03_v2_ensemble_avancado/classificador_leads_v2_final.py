"""
Classificador de Leads V2 — Ensemble Avançado (XGBoost + LightGBM + CatBoost)

Treina 3 modelos gradient boosting individualmente e combina com Stacking.
Usa feature engineering avançado com interações polinomiais.

Uso:
    python 03_v2_ensemble_avancado/classificador_leads_v2_final.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')
import time as _time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    LEADS_PARQUET, MODELO_V2,
    CATEGORICAL_FEATURES_V2, NUMERIC_FEATURES_V2, BINARY_FEATURES_V2, TARGET
)
from src.features import criar_features_avancadas
from src.utils import avaliar_modelo


def criar_xgb_classifier(device='cuda', **kwargs):
    """Cria XGBoost com configuração base e fallback simples para CPU."""
    return xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        tree_method='hist',
        device=device,
        max_bin=512,
        **kwargs
    )


def main():
    print("=" * 70)
    print(" CLASSIFICADOR DE LEADS V2 — ENSEMBLE AVANÇADO")
    print("=" * 70)
    print("\n  Modelos: XGBoost + LightGBM + CatBoost + Stacking")
    print("  Features: Interações + Polinomiais + Binning avançado\n")

    # =====================================================================
    # 1. CARREGAMENTO
    # =====================================================================
    print("[*] 1. Carregando base de dados...")
    try:
        df = pd.read_parquet(LEADS_PARQUET)
    except FileNotFoundError:
        print(f"[!] Erro: '{LEADS_PARQUET}' não encontrado. Rode o gerador primeiro.")
        return

    # ETAPA 1.1 - Remover colunas legadas para evitar vazamento de informacao.
    colunas_legadas = ['Score_Oculto_Probabilidade', 'Fator_Sorte']
    df = df.drop(columns=[c for c in colunas_legadas if c in df.columns], errors='ignore')

    print(f"    {len(df)} registros | {len(df.columns)} colunas")
    print(f"    Não Compra: {(df[TARGET] == 0).sum()} ({(df[TARGET] == 0).mean()*100:.1f}%)")
    print(f"    Compra:     {(df[TARGET] == 1).sum()} ({(df[TARGET] == 1).mean()*100:.1f}%)")

    # =====================================================================
    # 2. FEATURE ENGINEERING AVANÇADO
    # =====================================================================
    print("\n[*] 2. Feature engineering avançado (v2)...")
    df = criar_features_avancadas(df)
    print("    Features: Interações, Polinomiais, Faixas, Flags")

    # =====================================================================
    # 3. SELEÇÃO DE FEATURES
    # =====================================================================
    print("\n[*] 3. Selecionando features...")

    categorical_features = CATEGORICAL_FEATURES_V2
    numeric_features = NUMERIC_FEATURES_V2
    binary_features = BINARY_FEATURES_V2
    features = categorical_features + numeric_features + binary_features
    print(f"    Total: {len(features)} features ({len(categorical_features)} cat + {len(numeric_features)} num + {len(binary_features)} bin)")

    X = df[features]
    y = df[TARGET]

    # =====================================================================
    # 4. PRÉ-PROCESSAMENTO
    # =====================================================================
    print("\n[*] 4. Pipeline de pré-processamento...")

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', 'passthrough', binary_features)
        ])

    # =====================================================================
    # 5. DIVISÃO 80/20
    # =====================================================================
    print("\n[*] 5. Dividindo dados (80% treino / 20% teste)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train, test_size=0.10, random_state=42, stratify=y_train
    )
    print(f"    Treino: {X_train_main.shape[0]} / Val: {X_val.shape[0]} / Teste: {X_test.shape[0]}")

    # =====================================================================
    # 6. PRÉ-PROCESSAMENTO + SMOTE
    # =====================================================================
    print("\n[*] 6. Pré-processamento + SMOTE (0.15)...")

    X_train_processed = preprocessor.fit_transform(X_train_main)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    smote = SMOTE(sampling_strategy=0.15, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train_main)
    print(f"    SMOTE: {(y_train_resampled == 0).sum()} neg / {(y_train_resampled == 1).sum()} pos")

    # =====================================================================
    # 7. TREINAMENTO DOS 3 MODELOS
    # =====================================================================
    _inicio_total = _time.time()
    resultados = []

    # ----- 7a. XGBoost (GPU) -----
    print("\n" + "=" * 70)
    print(" MODELO 1: XGBoost (GPU)")
    print("=" * 70)
    _t = _time.time()

    xgb_device = 'cuda'
    xgb_model = criar_xgb_classifier(
        device=xgb_device,
        n_estimators=10000, max_depth=7, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
        gamma=0.2, reg_alpha=0.1, reg_lambda=0.5, scale_pos_weight=1,
        early_stopping_rounds=100
    )
    try:
        xgb_model.fit(
            X_train_resampled, y_train_resampled,
            eval_set=[(X_val_processed, y_val)], verbose=0
        )
    except Exception as exc:
        print(f"    ⚠️ GPU indisponível no XGBoost ({exc.__class__.__name__}), usando CPU...")
        xgb_device = 'cpu'
        xgb_model = criar_xgb_classifier(
            device=xgb_device,
            n_estimators=10000, max_depth=7, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
            gamma=0.2, reg_alpha=0.1, reg_lambda=0.5, scale_pos_weight=1,
            early_stopping_rounds=100
        )
        xgb_model.fit(
            X_train_resampled, y_train_resampled,
            eval_set=[(X_val_processed, y_val)], verbose=0
        )

    xgb_prob = xgb_model.predict_proba(X_test_processed)[:, 1]
    res_xgb = avaliar_modelo('XGBoost', y_test, xgb_prob)
    resultados.append(res_xgb)
    print(f"    Device: {xgb_device.upper()} | Early stop: {xgb_model.best_iteration + 1} árvores | AUC-ROC: {res_xgb['auc_roc']:.4f} | F1: {res_xgb['f1']:.4f} | {_time.time() - _t:.1f}s")

    # ----- 7b. LightGBM (CPU) -----
    print("\n" + "=" * 70)
    print(" MODELO 2: LightGBM (CPU)")
    print("=" * 70)
    _t = _time.time()

    lgb_model = lgb.LGBMClassifier(
        n_estimators=10000, max_depth=7, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=0.5, scale_pos_weight=1,
        random_state=42, verbose=-1, n_jobs=4
    )
    lgb_model.fit(X_train_resampled, y_train_resampled,
                  eval_set=[(X_val_processed, y_val)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])

    lgb_prob = lgb_model.predict_proba(X_test_processed)[:, 1]
    res_lgb = avaliar_modelo('LightGBM', y_test, lgb_prob)
    resultados.append(res_lgb)
    print(f"    Early stop: {lgb_model.best_iteration_} árvores | AUC-ROC: {res_lgb['auc_roc']:.4f} | F1: {res_lgb['f1']:.4f} | {_time.time() - _t:.1f}s")

    # ----- 7c. CatBoost -----
    print("\n" + "=" * 70)
    print(" MODELO 3: CatBoost")
    print("=" * 70)
    _t = _time.time()

    cb_device = 'GPU'
    try:
        cb_model = CatBoostClassifier(
            iterations=10000, depth=7, learning_rate=0.1, l2_leaf_reg=3.0,
            scale_pos_weight=1, random_state=42, verbose=0,
            task_type='GPU', early_stopping_rounds=100, eval_metric='Logloss'
        )
        cb_model.fit(X_train_resampled, y_train_resampled,
                     eval_set=(X_val_processed, y_val), verbose=0)
    except Exception as exc:
        print(f"    ⚠️ GPU indisponível no CatBoost ({exc.__class__.__name__}), usando CPU...")
        cb_device = 'CPU'
        cb_model = CatBoostClassifier(
            iterations=10000, depth=7, learning_rate=0.1, l2_leaf_reg=3.0,
            scale_pos_weight=1, random_state=42, verbose=0,
            task_type='CPU', thread_count=4, early_stopping_rounds=100, eval_metric='Logloss'
        )
        cb_model.fit(X_train_resampled, y_train_resampled,
                     eval_set=(X_val_processed, y_val), verbose=0)

    cb_prob = cb_model.predict_proba(X_test_processed)[:, 1]
    res_cb = avaliar_modelo('CatBoost', y_test, cb_prob)
    resultados.append(res_cb)
    print(f"    Device: {cb_device} | Early stop: {cb_model.best_iteration_} árvores | AUC-ROC: {res_cb['auc_roc']:.4f} | F1: {res_cb['f1']:.4f} | {_time.time() - _t:.1f}s")

    # =====================================================================
    # 8. STACKING ENSEMBLE
    # =====================================================================
    print("\n" + "=" * 70)
    print(" MODELO 4: STACKING ENSEMBLE")
    print("=" * 70)
    _t = _time.time()

    X_meta_train = np.column_stack([
        xgb_model.predict_proba(X_val_processed)[:, 1],
        lgb_model.predict_proba(X_val_processed)[:, 1],
        cb_model.predict_proba(X_val_processed)[:, 1]
    ])
    X_meta_test = np.column_stack([xgb_prob, lgb_prob, cb_prob])

    meta_learner = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    meta_learner.fit(X_meta_train, y_val)

    stack_prob = meta_learner.predict_proba(X_meta_test)[:, 1]
    res_stack = avaliar_modelo('Stacking', y_test, stack_prob)
    resultados.append(res_stack)

    coefs = meta_learner.coef_[0]
    pesos = np.abs(coefs) / np.abs(coefs).sum() * 100
    print(f"    Pesos: XGBoost={pesos[0]:.1f}% | LightGBM={pesos[1]:.1f}% | CatBoost={pesos[2]:.1f}%")
    print(f"    AUC-ROC: {res_stack['auc_roc']:.4f} | F1: {res_stack['f1']:.4f} | {_time.time() - _t:.1f}s")

    _tempo_total = _time.time() - _inicio_total

    # =====================================================================
    # 9. COMPARATIVO
    # =====================================================================
    print("\n" + "=" * 70)
    print(" COMPARATIVO DE TODOS OS MODELOS")
    print("=" * 70)

    print(f"\n{'Modelo':<15} {'AUC-ROC':>8} {'F1':>8} {'Acc':>8} {'BalAcc':>8} {'AUC-PR':>8} {'Threshold':>10}")
    print("-" * 70)
    for r in sorted(resultados, key=lambda x: x['auc_roc'], reverse=True):
        print(f"{r['nome']:<15} {r['auc_roc']:>8.4f} {r['f1']:>8.4f} {r['accuracy']*100:>7.2f}% {r['balanced_accuracy']*100:>7.2f}% {r['auc_pr']:>8.4f} {r['threshold']:>10.4f}")

    print(f"\n    ⏱️  Tempo total: {_tempo_total:.1f}s ({_tempo_total/60:.1f} min)")

    # =====================================================================
    # 10. MELHOR MODELO
    # =====================================================================
    melhor = max(resultados, key=lambda x: x['auc_roc'])
    print(f"\n{'=' * 70}")
    print(f" MELHOR: {melhor['nome']} (AUC-ROC: {melhor['auc_roc']:.4f})")
    print(f"{'=' * 70}")

    print(f"\n--- Classification Report (Threshold {melhor['threshold']:.4f}) ---")
    print(classification_report(y_test, melhor['y_pred']))

    cm = confusion_matrix(y_test, melhor['y_pred'])
    print("Matriz de Confusão:")
    print(pd.DataFrame(cm,
        columns=["Previsto: Ñ Compra", "Previsto: Compra"],
        index=["Real: Ñ Compra", "Real: Compra"]
    ))

    # =====================================================================
    # 11. ANÁLISE DE THRESHOLDS
    # =====================================================================
    print(f"\n{'=' * 70}")
    print(f" ANÁLISE DE THRESHOLDS ({melhor['nome']})")
    print(f"{'=' * 70}")

    print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Acc':>8} {'Leads':>15}")
    print("-" * 62)
    for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        pred = (melhor['y_prob'] >= t).astype(int)
        if pred.sum() == 0:
            continue
        prec = (y_test[pred == 1] == 1).mean()
        rec = (y_test[melhor['y_prob'] >= t] == 1).sum() / (y_test == 1).sum()
        f1_t = 2 * prec * rec / (prec + rec + 1e-10)
        print(f"{t:>10.2f} {prec*100:>9.1f}% {rec*100:>7.1f}% {f1_t:>8.4f} {accuracy_score(y_test, pred)*100:>7.1f}% {pred.sum():>15,}")

    # =====================================================================
    # 12. FEATURE IMPORTANCE
    # =====================================================================
    print(f"\n{'=' * 70}")
    print(f" TOP 15 FEATURES (XGBoost)")
    print(f"{'=' * 70}")

    cat_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    all_names = np.concatenate([numeric_features, cat_names, binary_features])
    fi = pd.DataFrame({'Feature': all_names, 'Importance': xgb_model.feature_importances_})
    print(fi.sort_values('Importance', ascending=False).head(15).to_string(index=False))

    # =====================================================================
    # 13. SALVAR
    # =====================================================================
    print(f"\n[*] Salvando ensemble em '{MODELO_V2}'...")
    joblib.dump({
        'preprocessor': preprocessor,
        'xgb_model': xgb_model,
        'lgb_model': lgb_model,
        'cb_model': cb_model,
        'meta_learner': meta_learner,
        'melhor_modelo': melhor['nome'],
        'threshold': melhor['threshold'],
        'feature_names': features,
        'categorical_features': categorical_features,
        'numeric_features': numeric_features,
        'binary_features': binary_features,
        'metricas': {
            'accuracy': melhor['accuracy'],
            'balanced_accuracy': melhor['balanced_accuracy'],
            'f1_score': melhor['f1'],
            'auc_roc': melhor['auc_roc'],
            'auc_pr': melhor['auc_pr'],
        }
    }, MODELO_V2)
    print("[+] Ensemble salvo com sucesso!")

    print(f"\n{'=' * 70}")
    print(f" TREINAMENTO V2 CONCLUÍDO!")
    print(f"{'=' * 70}")
    print(f"\n  Para usar o ensemble:")
    print(f"    modelo = joblib.load('{MODELO_V2}')")
    print(f"    X_proc = modelo['preprocessor'].transform(X_novo)")
    print(f"    prob_xgb = modelo['xgb_model'].predict_proba(X_proc)[:, 1]")
    print(f"    prob_lgb = modelo['lgb_model'].predict_proba(X_proc)[:, 1]")
    print(f"    prob_cb  = modelo['cb_model'].predict_proba(X_proc)[:, 1]")
    print(f"    X_meta = np.column_stack([prob_xgb, prob_lgb, prob_cb])")
    print(f"    previsoes = modelo['meta_learner'].predict(X_meta)")
    print()


if __name__ == "__main__":
    main()
