"""
Classificador de Leads V1 — XGBoost com GPU (CUDA)

Treina um modelo XGBoost para classificar leads como compradores ou não.
Usa feature engineering, SMOTE para balanceamento, RandomizedSearchCV para
otimização de hiperparâmetros, e early stopping.

Uso:
    python 02_v1_baseline_xgboost/classificador_leads_v1_final.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_auc_score, precision_recall_curve, auc
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')
import time as _time

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    LEADS_PARQUET, MODELO_V1,
    CATEGORICAL_FEATURES_BASE, NUMERIC_FEATURES_BASE, BINARY_FEATURES_BASE, TARGET
)
from src.features import criar_features_base
from src.utils import encontrar_threshold_otimo


def criar_xgb_classifier(device='cuda', **kwargs):
    """Cria XGBoost com configuração base e possibilidade de fallback CPU."""
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
    print(" CLASSIFICADOR DE LEADS V1 — XGBoost (GPU)")
    print("=" * 70)
    print("\nIniciando treinamento do modelo de Classificação de Leads...\n")

    # =====================================================================
    # 1. CARREGAMENTO DOS DADOS
    # =====================================================================
    print("[*] 1. Carregando base de dados...")
    try:
        df = pd.read_parquet(LEADS_PARQUET)
    except FileNotFoundError:
        print(f"[!] Erro: Arquivo '{LEADS_PARQUET}' não encontrado. Rode o gerador_leads_ml.py primeiro.")
        return

    # ETAPA 1.1 - Remover colunas que podem gerar leakage do alvo.
    colunas_legadas = ['Score_Oculto_Probabilidade', 'Fator_Sorte']
    df = df.drop(columns=[c for c in colunas_legadas if c in df.columns], errors='ignore')

    print(f"    Base carregada com {len(df)} registros e {len(df.columns)} colunas.")
    print(f"    Distribuição da classe alvo:")
    print(f"      -> Não Compra (0): {(df[TARGET] == 0).sum()} ({(df[TARGET] == 0).mean()*100:.1f}%)")
    print(f"      -> Compra (1):     {(df[TARGET] == 1).sum()} ({(df[TARGET] == 1).mean()*100:.1f}%)")

    # =====================================================================
    # 2. ENGENHARIA DE FEATURES
    # =====================================================================
    print("\n[*] 2. Criando features derivadas dos dados reais...")
    df = criar_features_base(df)
    print("    Features criadas: Idade, Tem_Email_Corp, Score_Engajamento, Faixa_Salarial")

    # =====================================================================
    # 3. SELEÇÃO DE FEATURES
    # =====================================================================
    print("\n[*] 3. Selecionando colunas e removendo dados sensíveis/leakage...")

    categorical_features = CATEGORICAL_FEATURES_BASE
    numeric_features = NUMERIC_FEATURES_BASE
    binary_features = BINARY_FEATURES_BASE
    features = categorical_features + numeric_features + binary_features
    print(f"    Total de features selecionadas: {len(features)}")

    X = df[features]
    y = df[TARGET]

    # =====================================================================
    # 4. PRÉ-PROCESSAMENTO
    # =====================================================================
    print("\n[*] 4. Configurando pipeline de pré-processamento...")

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', 'passthrough', binary_features)
        ])

    # =====================================================================
    # 5. DIVISÃO EM TREINO E TESTE (80/20)
    # =====================================================================
    print("\n[*] 5. Dividindo dados em Treino (80%) e Teste (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Treino: {X_train.shape[0]} amostras / Teste: {X_test.shape[0]} amostras")

    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train, test_size=0.10, random_state=42, stratify=y_train
    )
    print(f"    Treino efetivo: {X_train_main.shape[0]} / Validação (early stop): {X_val.shape[0]}")

    # =====================================================================
    # 6. OTIMIZAÇÃO DE HIPERPARÂMETROS (SEM LEAKAGE)
    # =====================================================================
    print("\n[*] 6. Otimizando hiperparâmetros com RandomizedSearchCV (F1 como métrica)...")
    print("    SMOTE e pré-processamento serão aplicados dentro de cada fold (sem vazamento).")

    param_distributions = {
        'model__n_estimators': [300, 500, 800],
        'model__max_depth': [6, 7, 8, 10],
        'model__learning_rate': [0.05, 0.08, 0.1, 0.12, 0.15],
        'model__subsample': [0.7, 0.8, 0.9],
        'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'model__min_child_weight': [1, 3, 5, 7],
        'model__gamma': [0, 0.1, 0.2, 0.3],
        'model__reg_alpha': [0, 0.01, 0.1, 0.5],
        'model__reg_lambda': [0.5, 1.0, 1.5, 2.0],
        'model__scale_pos_weight': [1, 3, 5]
    }

    xgb_device = 'cuda'
    xgb_base = criar_xgb_classifier(device=xgb_device)
    cv_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(sampling_strategy=0.15, random_state=42)),
        ('model', xgb_base)
    ])

    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=cv_pipeline, param_distributions=param_distributions,
        n_iter=50, scoring='f1', cv=cv_strategy, random_state=42,
        n_jobs=1, verbose=2
    )

    total_fits = 50 * cv_strategy.get_n_splits()
    print(f"    Total de fits a executar: {total_fits}\n")

    _inicio_search = _time.time()
    try:
        print("    Tentando busca com GPU (CUDA)...")
        search.fit(X_train_main, y_train_main)
    except Exception as exc:
        print(f"    ⚠️ GPU indisponível no XGBoost ({exc.__class__.__name__}). Reexecutando em CPU...")
        xgb_device = 'cpu'
        xgb_base = criar_xgb_classifier(device=xgb_device)
        cv_pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(sampling_strategy=0.15, random_state=42)),
            ('model', xgb_base)
        ])
        search = RandomizedSearchCV(
            estimator=cv_pipeline, param_distributions=param_distributions,
            n_iter=50, scoring='f1', cv=cv_strategy, random_state=42,
            n_jobs=1, verbose=2
        )
        search.fit(X_train_main, y_train_main)

    _tempo_search = _time.time() - _inicio_search
    print(f"\n    ✅ Busca concluída em {_tempo_search:.1f}s ({_tempo_search/60:.1f} min)")

    best_params_raw = search.best_params_.copy()
    best_params = {k.replace('model__', ''): v for k, v in best_params_raw.items()}

    print(f"\n    Melhores hiperparâmetros encontrados:")
    for param, valor in best_params.items():
        print(f"      {param}: {valor}")
    print(f"    Melhor F1-Score (CV): {search.best_score_:.4f}")

    # =====================================================================
    # 7. RETREINAMENTO COM EARLY STOPPING
    # =====================================================================
    print("\n[*] 7. Pré-processando dados e aplicando SMOTE para treino final...")
    X_train_processed = preprocessor.fit_transform(X_train_main)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    smote = SMOTE(sampling_strategy=0.15, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train_main)

    print(f"    Antes do SMOTE:  Classe 0 = {(y_train_main == 0).sum()}, Classe 1 = {(y_train_main == 1).sum()}")
    print(f"    Depois do SMOTE: Classe 0 = {(y_train_resampled == 0).sum()}, Classe 1 = {(y_train_resampled == 1).sum()}")
    print("\n    Retreinando modelo final com Early Stopping (até 10000 épocas)...")

    best_params['n_estimators'] = 10000

    best_model = criar_xgb_classifier(
        device=xgb_device, **best_params, early_stopping_rounds=100
    )

    try:
        best_model.fit(
            X_train_resampled, y_train_resampled,
            eval_set=[(X_val_processed, y_val)], verbose=200
        )
    except Exception as exc:
        if xgb_device != 'cuda':
            raise
        print(f"    ⚠️ GPU indisponível no treino final ({exc.__class__.__name__}). Reexecutando em CPU...")
        xgb_device = 'cpu'
        best_model = criar_xgb_classifier(
            device=xgb_device, **best_params, early_stopping_rounds=100
        )
        best_model.fit(
            X_train_resampled, y_train_resampled,
            eval_set=[(X_val_processed, y_val)], verbose=200
        )

    n_arvores = best_model.best_iteration + 1
    print(f"\n    ✅ Early stopping: {n_arvores} árvores | Logloss: {best_model.best_score:.4f}")

    # =====================================================================
    # 8. THRESHOLD OTIMIZADO (F1)
    # =====================================================================
    print("\n[*] 8. Otimizando threshold de decisão...")

    y_prob = best_model.predict_proba(X_test_processed)[:, 1]
    threshold_otimo, melhor_f1 = encontrar_threshold_otimo(y_test, y_prob)

    print(f"    Threshold padrão: 0.50")
    print(f"    Threshold ótimo (F1): {threshold_otimo:.4f} | F1 = {melhor_f1:.4f}")

    y_pred_otimizado = (y_prob >= threshold_otimo).astype(int)
    y_pred_padrao = best_model.predict(X_test_processed)

    # =====================================================================
    # 9. AVALIAÇÃO
    # =====================================================================
    print("\n" + "=" * 70)
    print(" RESULTADOS")
    print("=" * 70)

    print("\n--- THRESHOLD PADRÃO (0.50) ---")
    print(classification_report(y_test, y_pred_padrao))

    print(f"--- THRESHOLD OTIMIZADO F1 ({threshold_otimo:.4f}) ---")
    print(classification_report(y_test, y_pred_otimizado))

    cm = confusion_matrix(y_test, y_pred_otimizado)
    print("Matriz de Confusão:")
    print(pd.DataFrame(cm,
        columns=["Previsto: Ñ Compra", "Previsto: Compra"],
        index=["Real: Ñ Compra", "Real: Compra"]
    ))

    acc = accuracy_score(y_test, y_pred_otimizado)
    f1 = f1_score(y_test, y_pred_otimizado)
    roc_auc = roc_auc_score(y_test, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(rec_curve, prec_curve)

    print(f"\n  Acurácia:    {acc*100:.2f}%")
    print(f"  F1 (Classe 1): {f1:.4f}")
    print(f"  AUC-ROC:     {roc_auc:.4f}")
    print(f"  AUC-PR:      {pr_auc:.4f}")

    # =====================================================================
    # 10. FEATURE IMPORTANCE
    # =====================================================================
    print("\n[*] 10. Top 15 Features:")
    cat_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    all_names = np.concatenate([numeric_features, cat_names, binary_features])
    fi = pd.DataFrame({'Feature': all_names, 'Importance': best_model.feature_importances_})
    print(fi.sort_values('Importance', ascending=False).head(15).to_string(index=False))

    # =====================================================================
    # 11. SALVAR
    # =====================================================================
    print(f"\n[*] 11. Salvando modelo em '{MODELO_V1}'...")
    joblib.dump({
        'preprocessor': preprocessor,
        'model': best_model,
        'threshold': threshold_otimo,
        'feature_names': features,
        'categorical_features': categorical_features,
        'numeric_features': numeric_features,
        'binary_features': binary_features,
        'metricas': {'accuracy': acc, 'f1_score': f1, 'auc_roc': roc_auc, 'auc_pr': pr_auc}
    }, MODELO_V1)
    print(f"[+] Modelo salvo com sucesso!")

    print(f"\n{'=' * 70}")
    print(f" TREINAMENTO V1 CONCLUÍDO!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
