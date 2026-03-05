# Guia Passo a Passo (Revisita Rapida)

Use este guia quando quiser revisar o projeto do zero sem pensar na ordem.

## Documentacao rapida das libs principais

| Biblioteca | Link oficial |
|---|---|
| `pandas` | https://pandas.pydata.org/docs/ |
| `numpy` | https://numpy.org/doc/ |
| `scikit-learn` | https://scikit-learn.org/stable/documentation.html |
| `imbalanced-learn` | https://imbalanced-learn.org/stable/ |
| `xgboost` | https://xgboost.readthedocs.io/en/stable/ |
| `lightgbm` | https://lightgbm.readthedocs.io/en/latest/ |
| `catboost` | https://catboost.ai/docs/ |
| `torch` (PyTorch) | https://pytorch.org/docs/stable/index.html |
| `streamlit` | https://docs.streamlit.io/ |
| `plotly` | https://plotly.com/python/ |
| `pyarrow` | https://arrow.apache.org/docs/python/ |
| `faker` | https://faker.readthedocs.io/ |
| `joblib` | https://joblib.readthedocs.io/en/latest/ |

## Etapa 0 - Ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Etapa 1 - Dados

```bash
python 01_geracao_de_dados/02_gerador_leads_ml.py
```

Validar:

- `data/leads_gerados.parquet`
- `data/clientes_gerados.parquet`

## Etapa 2 - Baseline (V1)

```bash
python 02_v1_baseline_xgboost/classificador_leads_v1_final.py
```

Validar:

- `models/modelo_xgboost_leads_v1.pkl`

## Etapa 3 - Ensemble (V2)

```bash
python 03_v2_ensemble_avancado/classificador_leads_v2_final.py
```

Validar:

- `models/modelo_ensemble_leads_v2.pkl`

## Etapa 4 - Deep Learning (V3)

```bash
python 04_v3_deep_learning_pytorch/classificador_leads_v3_final.py
```

Validar:

- `models/modelo_dl_leads_v3.pth`
- `models/modelo_dl_leads_v3_pipeline.pkl`

## Etapa 5 - App

```bash
streamlit run 05_visualizacao_streamlit/app.py
```

## Etapa 6 - Teste rapido de sanidade

```bash
python -m unittest tests/test_pipeline_sanity.py
```

Se `python` nao existir no sistema, use `python3`.
