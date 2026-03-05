# Modulo 05 - Visualizacao com Streamlit

Interface para usar os modelos treinados em cenarios de atendimento individual e lote.

## Objetivos

- Carregar modelo V1, V2 ou V3.
- Simular inferencia de um lead em tempo real.
- Processar lista em lote e exportar CSV enriquecido.

## Pre-requisito

Treinar pelo menos um modelo antes de abrir o app:

- `models/modelo_xgboost_leads_v1.pkl` ou
- `models/modelo_ensemble_leads_v2.pkl` ou
- `models/modelo_dl_leads_v3.pth` + `models/modelo_dl_leads_v3_pipeline.pkl`

## Execucao

```bash
streamlit run 05_visualizacao_streamlit/app.py
```

## Fluxo dentro do app

1. Selecionar o motor de IA na barra lateral.
2. Usar aba de simulacao individual para teste rapido.
3. Usar aba de lote para processar CSV/Parquet e baixar resultado.

## Arquivo recomendado para teste

- `data/leads_gerados.parquet` (removendo ou ignorando colunas alvo quando necessario).

## Bibliotecas usadas neste modulo

| Biblioteca | Uso | Documentacao |
|---|---|---|
| `streamlit` | interface web e fluxo de upload/download | https://docs.streamlit.io/ |
| `pandas` | leitura e escrita de CSV/Parquet | https://pandas.pydata.org/docs/ |
| `numpy` | classificacao vetorizada por threshold | https://numpy.org/doc/ |
| `plotly` | gauge e visualizacoes interativas | https://plotly.com/python/ |
| `joblib` | carregamento de modelos V1/V2 e pipeline V3 | https://joblib.readthedocs.io/en/latest/ |
| `torch` (PyTorch) | inferencia do modelo V3 | https://pytorch.org/docs/stable/index.html |
