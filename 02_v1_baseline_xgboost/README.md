# Modulo 02 - V1 Baseline XGBoost

Primeira versao de modelo supervisionado do projeto.  
Aqui voce estabelece a baseline que as proximas versoes precisam superar.

## Objetivos

- Montar pipeline com preprocessamento tabular.
- Tratar desbalanceamento com SMOTE.
- Otimizar hiperparametros com `RandomizedSearchCV`.
- Salvar modelo e threshold de decisao.

## Ordem recomendada

1. Ler contexto teorico:
   - [`01_conceitos_xgboost_v1.md`](01_conceitos_xgboost_v1.md)
2. Estudar scripts didaticos:
   - [`02_estrutura_dados_pre_proc.py`](02_estrutura_dados_pre_proc.py)
   - [`03_otimizacao_random_search.py`](03_otimizacao_random_search.py)
3. Executar treino final:
   - `python 02_v1_baseline_xgboost/classificador_leads_v1_final.py`

## Entrada

- `data/leads_gerados.parquet` (gerado no modulo 01)

## Saida

- `models/modelo_xgboost_leads_v1.pkl`

## Bibliotecas usadas neste modulo

| Biblioteca | Uso | Documentacao |
|---|---|---|
| `pandas` | carga do dataset e analises auxiliares | https://pandas.pydata.org/docs/ |
| `numpy` | operacoes numericas e arrays | https://numpy.org/doc/ |
| `scikit-learn` | split, pipeline, preprocessamento e metricas | https://scikit-learn.org/stable/documentation.html |
| `imbalanced-learn` | SMOTE dentro do pipeline | https://imbalanced-learn.org/stable/ |
| `xgboost` | classificador principal da V1 | https://xgboost.readthedocs.io/en/stable/ |
| `joblib` | salvar artefato final `.pkl` | https://joblib.readthedocs.io/en/latest/ |

## O que validar ao terminar

- O script imprime metricas finais (F1, AUC-ROC, AUC-PR).
- O `threshold_otimo` e salvo junto com o modelo.
- O arquivo `.pkl` existe na pasta `models/`.

## Proximo passo

Comparar os resultados da V1 com o modulo 03 (ensemble avancado).
