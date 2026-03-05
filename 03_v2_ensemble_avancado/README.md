# Modulo 03 - V2 Ensemble Avancado

Evolucao da baseline com feature engineering extra e combinacao de modelos.

## Objetivos

- Criar features avancadas (interacoes, polinomiais, faixas).
- Treinar tres modelos base (XGBoost, LightGBM, CatBoost).
- Combinar as predicoes com `LogisticRegression` (stacking).
- Salvar pipeline completo de inferencia.

## Ordem recomendada

1. Ler contexto:
   - [`01_evolucao_para_v2.md`](01_evolucao_para_v2.md)
2. Estudar partes didaticas:
   - [`02_feature_engineering_v2.py`](02_feature_engineering_v2.py)
   - [`03_treinamento_paralelo_modelos.py`](03_treinamento_paralelo_modelos.py)
   - [`04_stacking_e_meta_learning.py`](04_stacking_e_meta_learning.py)
3. Executar treino final:
   - `python 03_v2_ensemble_avancado/classificador_leads_v2_final.py`

## Entrada

- `data/leads_gerados.parquet` (modulo 01)

## Saida

- `models/modelo_ensemble_leads_v2.pkl`

## Bibliotecas usadas neste modulo

| Biblioteca | Uso | Documentacao |
|---|---|---|
| `pandas` | carga de dados e relatorios tabulares | https://pandas.pydata.org/docs/ |
| `numpy` | combinacao de probabilidades e vetores meta | https://numpy.org/doc/ |
| `scikit-learn` | preprocessamento, logistic regression e metricas | https://scikit-learn.org/stable/documentation.html |
| `imbalanced-learn` | SMOTE no treino | https://imbalanced-learn.org/stable/ |
| `xgboost` | base learner 1 | https://xgboost.readthedocs.io/en/stable/ |
| `lightgbm` | base learner 2 | https://lightgbm.readthedocs.io/en/latest/ |
| `catboost` | base learner 3 | https://catboost.ai/docs/ |
| `joblib` | persistencia do ensemble final | https://joblib.readthedocs.io/en/latest/ |

## O que validar ao terminar

- O comparativo de modelos aparece no terminal.
- O modelo vencedor e identificado.
- O arquivo `modelo_ensemble_leads_v2.pkl` inclui:
  - `preprocessor`
  - modelos base (`xgb_model`, `lgb_model`, `cb_model`)
  - `meta_learner`
  - `threshold`

## Proximo passo

Executar a V3 para comparar abordagem de arvores vs rede neural tabular.
