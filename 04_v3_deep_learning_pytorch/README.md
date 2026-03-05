# Modulo 04 - V3 Deep Learning (PyTorch)

Versao com rede neural tabular usando embeddings para categoricas.

## Objetivos

- Migrar de OneHot para embeddings aprendidos.
- Preparar datasets e dataloaders para treino em batches.
- Treinar MLP tabular com `BCEWithLogitsLoss` e `AdamW`.
- Salvar pesos e pipeline de pre-processamento para inferencia.

## Ordem recomendada

1. Ler contexto:
   - [`01_evolucao_para_v3.md`](01_evolucao_para_v3.md)
2. Estudar partes didaticas:
   - [`02_preparo_dados_pytorch.py`](02_preparo_dados_pytorch.py)
   - [`03_dataset_e_dataloaders.py`](03_dataset_e_dataloaders.py)
   - [`04_arquitetura_mlp.py`](04_arquitetura_mlp.py)
   - [`05_loop_de_treinamento.py`](05_loop_de_treinamento.py)
3. Executar treino final:
   - `python 04_v3_deep_learning_pytorch/classificador_leads_v3_final.py`

## Entrada

- `data/leads_gerados.parquet` (modulo 01)

## Saidas

- `models/modelo_dl_leads_v3.pth`
- `models/modelo_dl_leads_v3_pipeline.pkl`

## Bibliotecas usadas neste modulo

| Biblioteca | Uso | Documentacao |
|---|---|---|
| `pandas` | leitura da base e preparacao inicial | https://pandas.pydata.org/docs/ |
| `numpy` | arrays e manipulacao numerica | https://numpy.org/doc/ |
| `scikit-learn` | split, encoding/scaling e metricas | https://scikit-learn.org/stable/documentation.html |
| `torch` (PyTorch) | rede neural, treino e inferencia | https://pytorch.org/docs/stable/index.html |
| `joblib` | salvar pipeline de preprocessamento e threshold | https://joblib.readthedocs.io/en/latest/ |

## O que validar ao terminar

- O script detecta `CUDA` ou `CPU`.
- O treinamento executa com early stopping.
- O `threshold` otimizado e salvo no pipeline `.pkl`.

## Observacao pratica

Sem GPU, o treino pode levar significativamente mais tempo.  
Para estudos locais rapidos, reduza temporariamente a volumetria em `src/config.py`.
