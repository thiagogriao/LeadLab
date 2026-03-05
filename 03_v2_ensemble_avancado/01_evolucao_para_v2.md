# Modulo 03 - Evolucao para V2 (Ensemble)

Aqui a baseline vira um sistema de modelos combinados.

## Objetivo didatico

Evoluir da pergunta:

"Qual o melhor modelo isolado?"

para:

"Qual combinacao de modelos maximiza qualidade sem perder controle?"

## O que muda da V1 para a V2

1. Entra feature engineering avancado.
2. Saem buscas extensas de hiperparametros.
3. Entram tres modelos base treinados em paralelo.
4. Um meta-modelo decide a predicao final (stacking).

## Componentes principais

### `02_feature_engineering_v2.py`

Mostra criacao de features extras:

- interacoes
- termos polinomiais
- faixas categoricas auxiliares

### `03_treinamento_paralelo_modelos.py`

Mostra treino dos modelos base:

- XGBoost
- LightGBM
- CatBoost

### `04_stacking_e_meta_learning.py`

Mostra como combinar saidas de probabilidade em um meta-classificador.

## Bibliotecas deste guia

| Biblioteca | Uso principal | Documentacao |
|---|---|---|
| `pandas` | carga de dados e comparativos tabulares | https://pandas.pydata.org/docs/ |
| `numpy` | empilhamento de probabilidades e vetores meta | https://numpy.org/doc/ |
| `scikit-learn` | preprocessamento, meta learner e metricas | https://scikit-learn.org/stable/documentation.html |
| `imbalanced-learn` | balanceamento via SMOTE | https://imbalanced-learn.org/stable/ |
| `xgboost` | base learner 1 | https://xgboost.readthedocs.io/en/stable/ |
| `lightgbm` | base learner 2 | https://lightgbm.readthedocs.io/en/latest/ |
| `catboost` | base learner 3 | https://catboost.ai/docs/ |
| `joblib` | serializacao do pipeline ensemble | https://joblib.readthedocs.io/en/latest/ |

## Roteiro de estudo pratico

1. Rodar V1 e guardar metricas de referencia.
2. Rodar V2 e comparar:
   - AUC-ROC
   - F1
   - custo de treino
3. Inspecionar pesos do meta-modelo para entender contribuicao de cada base learner.

## Execucao final

```bash
python 03_v2_ensemble_avancado/classificador_leads_v2_final.py
```

## Checkpoint antes de seguir

- Artefato `models/modelo_ensemble_leads_v2.pkl` gerado.
- Melhor modelo reportado no comparativo.
- Threshold e metricas persistidos para inferencia.
