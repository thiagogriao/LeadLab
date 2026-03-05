# LeadLab

Laboratorio didatico de Machine Learning para classificacao de propensao de compra de leads.

Objetivo do projeto: permitir que voce revisite o fluxo completo de ML (dados -> baseline -> ensemble -> deep learning -> app) com scripts separados por etapa e material de apoio.

## Estrutura do repositorio

| Pasta | Objetivo |
|---|---|
| [`01_geracao_de_dados`](01_geracao_de_dados/) | Geracao de dados sinteticos, conversao SQL -> Parquet e estudo visual |
| [`02_v1_baseline_xgboost`](02_v1_baseline_xgboost/) | Baseline com XGBoost + pipeline sklearn + otimizacao |
| [`03_v2_ensemble_avancado`](03_v2_ensemble_avancado/) | Feature engineering avancado + ensemble (XGB/LGBM/CatBoost + stacking) |
| [`04_v3_deep_learning_pytorch`](04_v3_deep_learning_pytorch/) | Modelo tabular em PyTorch com embeddings |
| [`05_visualizacao_streamlit`](05_visualizacao_streamlit/) | Interface para inferencia individual e em lote |
| [`src`](src/) | Configuracoes compartilhadas, features e utilitarios |
| [`tests`](tests/) | Testes de sanidade do pipeline |
| [`ceps`](ceps/) | Base de CEPs e instrucoes de conversao |

## Pre-requisitos

- Python 3.10+ (recomendado)
- `pip`
- Opcional: GPU CUDA para acelerar V1/V2/V3

## Bibliotecas do projeto

| Biblioteca | Papel no projeto | Documentacao |
|---|---|---|
| `pandas` | manipulacao tabular | https://pandas.pydata.org/docs/ |
| `numpy` | operacoes numericas | https://numpy.org/doc/ |
| `scikit-learn` | preprocessamento, metricas e modelos auxiliares | https://scikit-learn.org/stable/documentation.html |
| `imbalanced-learn` | tratamento de desbalanceamento (SMOTE) | https://imbalanced-learn.org/stable/ |
| `xgboost` | baseline V1 e parte da V2 | https://xgboost.readthedocs.io/en/stable/ |
| `lightgbm` | modelo base da V2 | https://lightgbm.readthedocs.io/en/latest/ |
| `catboost` | modelo base da V2 | https://catboost.ai/docs/ |
| `torch` (PyTorch) | modelo V3 (deep learning) | https://pytorch.org/docs/stable/index.html |
| `streamlit` | interface de inferencia | https://docs.streamlit.io/ |
| `plotly` | visualizacoes interativas no app | https://plotly.com/python/ |
| `matplotlib`/`seaborn` | graficos de analise exploratoria | https://matplotlib.org/stable/users/index.html / https://seaborn.pydata.org/ |
| `pyarrow` | leitura/escrita parquet | https://arrow.apache.org/docs/python/ |
| `faker` | geracao de dados sinteticos | https://faker.readthedocs.io/ |
| `joblib` | serializacao de pipelines/modelos | https://joblib.readthedocs.io/en/latest/ |

## Setup rapido

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

## Trilha de execucao (passo a passo)

### 1. Preparar base de CEPs (apenas se precisar regenerar)

Os scripts de geracao usam [`ceps/base_ceps_otimizada.parquet`](ceps/base_ceps_otimizada.parquet).  
Se voce tiver os dumps SQL (`cidade.sql` e `logradouro.sql`), rode:

```bash
python 01_geracao_de_dados/03_conversor_sql_parquet.py
```

### 2. Gerar dados sinteticos

```bash
python 01_geracao_de_dados/02_gerador_leads_ml.py
```

Artefatos esperados:
- `data/leads_gerados.parquet`
- `data/clientes_gerados.parquet`
- `data/ceps_validos.parquet`

Configuracoes uteis por variavel de ambiente:
- `LEADLAB_MODO_NEGOCIO=b2b|b2c`
- `LEADLAB_ESCOPO_GEOGRAFICO=regional|nacional`

### 3. (Opcional) Rodar estudo visual de clientes

```bash
python 01_geracao_de_dados/04_estudo_visual_clientes.py
```

Saidas em `data/estudos/` (heatmap, elbow, silhouette, PCA etc).

### 4. Treinar baseline V1 (XGBoost)

```bash
python 02_v1_baseline_xgboost/classificador_leads_v1_final.py
```

Artefato:
- `models/modelo_xgboost_leads_v1.pkl`

### 5. Treinar V2 (Ensemble avancado)

```bash
python 03_v2_ensemble_avancado/classificador_leads_v2_final.py
```

Artefato:
- `models/modelo_ensemble_leads_v2.pkl`

### 6. Treinar V3 (Deep Learning com PyTorch)

```bash
python 04_v3_deep_learning_pytorch/classificador_leads_v3_final.py
```

Artefatos:
- `models/modelo_dl_leads_v3.pth`
- `models/modelo_dl_leads_v3_pipeline.pkl`

### 7. Subir interface Streamlit

```bash
streamlit run 05_visualizacao_streamlit/app.py
```

## Validacao minima

```bash
python -m unittest tests/test_pipeline_sanity.py
```

## Ordem recomendada para estudo do codigo

1. [`src/config.py`](src/config.py) para entender caminhos e features padrao.
2. [`src/features.py`](src/features.py) para entender o feature engineering.
3. Modulo 01 (geracao de dados) antes de treinar qualquer modelo.
4. V1 -> V2 -> V3 para acompanhar a evolucao tecnica.
5. Streamlit por ultimo para ver inferencia e produto final.

## Troubleshooting rapido

- Erro de arquivo ausente em `data/`: rode primeiro `02_gerador_leads_ml.py`.
- Erro de modelo ausente no Streamlit: treine a versao escolhida antes de abrir o app.
- Treino muito lento: execute com GPU (quando disponivel) ou reduza volumetria em `src/config.py` para estudos locais.

## Documentacao por modulo

Cada pasta principal agora possui um `README.md` proprio com:
- objetivo do modulo
- scripts na ordem certa
- entradas, saidas e checkpoints de estudo

Comece por [`01_geracao_de_dados/README.md`](01_geracao_de_dados/README.md).

Para revisao rapida, use [`GUIA_PASSO_A_PASSO.md`](GUIA_PASSO_A_PASSO.md).
# LeadLab
# LeadLab
