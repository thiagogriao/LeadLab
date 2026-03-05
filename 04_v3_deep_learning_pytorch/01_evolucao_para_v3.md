# Modulo 04 - Evolucao para V3 (Deep Learning)

Este modulo troca modelos baseados em arvores por rede neural tabular em PyTorch.

## Objetivo didatico

Aprender quando vale migrar para Deep Learning em dados tabulares e quais custos essa migracao traz.

## Principal diferenca tecnica

- V1/V2: categorias via OneHot.
- V3: categorias via embeddings aprendidos durante treino.

Embeddings ajudam a representar similaridade entre categorias em espaco vetorial continuo.

## Vantagens praticas da V3

1. Pipeline mais flexivel para evolucoes de arquitetura.
2. Treino em batches via `Dataset` + `DataLoader`.
3. Boa escalabilidade quando ha GPU e alto volume de dados.

## Trade-offs da V3

1. Menor interpretabilidade nativa que modelos de arvore.
2. Dependencia maior de hardware para bom tempo de treino.
3. Mais parametros e mais pontos de ajuste.

## Bibliotecas deste guia

| Biblioteca | Uso principal | Documentacao |
|---|---|---|
| `pandas` | leitura da base e preparacao inicial | https://pandas.pydata.org/docs/ |
| `numpy` | manipulacao de arrays | https://numpy.org/doc/ |
| `scikit-learn` | split, encoding/scaling e metricas | https://scikit-learn.org/stable/documentation.html |
| `torch` (PyTorch) | definicao de rede, treino e inferencia | https://pytorch.org/docs/stable/index.html |
| `joblib` | salvar pipeline de preprocessamento | https://joblib.readthedocs.io/en/latest/ |

## Roteiro de estudo pratico

### Passo 1 - Preprocessamento para DL

Arquivo: `02_preparo_dados_pytorch.py`

Foco:

- `LabelEncoder` para categoricas
- `StandardScaler` para numericas
- definicao de tamanho de embeddings

### Passo 2 - Dados em batches

Arquivo: `03_dataset_e_dataloaders.py`

Foco:

- classe `Dataset`
- uso de `DataLoader` com batch size

### Passo 3 - Arquitetura

Arquivo: `04_arquitetura_mlp.py`

Foco:

- camadas de embedding
- bloco denso (MLP)
- normalizacao e dropout

### Passo 4 - Treino

Arquivo: `05_loop_de_treinamento.py`

Foco:

- funcao de perda com `pos_weight`
- scheduler de learning rate
- early stopping

## Execucao final

```bash
python 04_v3_deep_learning_pytorch/classificador_leads_v3_final.py
```

## Checkpoint final

- `models/modelo_dl_leads_v3.pth` existe.
- `models/modelo_dl_leads_v3_pipeline.pkl` existe.
- threshold otimizado foi salvo e pode ser usado no Streamlit.
