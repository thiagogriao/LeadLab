# Modulo 02 - V1 Baseline com XGBoost

Este modulo define o primeiro modelo de referencia do projeto.

## Objetivo didatico

Criar uma baseline confiavel para responder:

"Qual performance minima qualquer versao futura precisa superar?"

## Por que comecar com XGBoost

- Funciona muito bem em dados tabulares.
- Tem boa relacao entre performance e custo computacional.
- Permite analise de importancia de features.

## Conceitos praticados

1. Pipeline tabular com `ColumnTransformer`.
2. Transformacao numerica (`StandardScaler`) e categorica (`OneHotEncoder`).
3. Balanceamento de classe com `SMOTE`.
4. Busca de hiperparametros com `RandomizedSearchCV`.
5. Ajuste de `threshold` orientado a F1.

## Bibliotecas deste guia

| Biblioteca | Uso principal | Documentacao |
|---|---|---|
| `pandas` | leitura da base e analise tabular | https://pandas.pydata.org/docs/ |
| `numpy` | operacoes em arrays e vetores | https://numpy.org/doc/ |
| `scikit-learn` | split, preprocessamento, pipeline e metricas | https://scikit-learn.org/stable/documentation.html |
| `imbalanced-learn` | SMOTE para classe minoritaria | https://imbalanced-learn.org/stable/ |
| `xgboost` | classificador principal da baseline | https://xgboost.readthedocs.io/en/stable/ |
| `joblib` | persistencia do modelo treinado | https://joblib.readthedocs.io/en/latest/ |

## Roteiro passo a passo

### Passo 1 - Revisar preprocessamento

Arquivo: `02_estrutura_dados_pre_proc.py`

Foco:

- como separar features por tipo
- como evitar vazamento com pipeline

### Passo 2 - Revisar otimizacao

Arquivo: `03_otimizacao_random_search.py`

Foco:

- como definir espaco de busca
- como validar por CV estratificada

### Passo 3 - Rodar treino final

```bash
python 02_v1_baseline_xgboost/classificador_leads_v1_final.py
```

## Saida esperada

- `models/modelo_xgboost_leads_v1.pkl`
- metricas no terminal (F1, AUC-ROC, AUC-PR)
- threshold otimizado salvo no artefato

## Checkpoint antes de seguir

- Pipeline salvo abre com `joblib.load`.
- Modelo consegue prever no conjunto de teste sem erro.
- Resultado da V1 esta documentado para comparar com a V2.
