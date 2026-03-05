# Modulo 01 - Fundamentos de Dados Sinteticos

Este modulo cria o dataset que alimenta todas as versoes de modelo do projeto.

## Objetivo didatico

Entender como construir uma base sintetica com:

- volume alto
- desbalanceamento controlado
- regras de negocio escondidas (que o modelo precisa descobrir)

## O que voce vai praticar

1. Geracao de dados tabulares com `Faker`.
2. Enriquecimento geografico com CEPs reais.
3. Exportacao em Parquet para treino eficiente.
4. Analise exploratoria visual com clustering e PCA.

## Bibliotecas deste guia

| Biblioteca | Uso principal | Documentacao |
|---|---|---|
| `pandas` | montagem e exportacao de dataframes | https://pandas.pydata.org/docs/ |
| `numpy` | calculos numericos e distribuicoes | https://numpy.org/doc/ |
| `faker` | criacao de dados sinteticos | https://faker.readthedocs.io/ |
| `pyarrow` | leitura e escrita parquet | https://arrow.apache.org/docs/python/ |
| `scikit-learn` | KMeans, PCA e silhouette | https://scikit-learn.org/stable/documentation.html |
| `matplotlib` | graficos estaticos | https://matplotlib.org/stable/users/index.html |
| `seaborn` | visualizacoes estatisticas | https://seaborn.pydata.org/ |

## Roteiro passo a passo

### Passo 1 - Garantir base de CEPs

Se necessario, reconstrua:

```bash
python 01_geracao_de_dados/03_conversor_sql_parquet.py
```

Saida esperada: `ceps/base_ceps_otimizada.parquet`.

### Passo 2 - Gerar leads e clientes

```bash
python 01_geracao_de_dados/02_gerador_leads_ml.py
```

Arquivos esperados:

- `data/leads_gerados.parquet`
- `data/clientes_gerados.parquet`
- `data/ceps_validos.parquet`

### Passo 3 - Explorar a base visualmente (opcional, recomendado)

```bash
python 01_geracao_de_dados/04_estudo_visual_clientes.py
```

Graficos esperados: pasta `data/estudos/`.

## Como ler os scripts deste modulo

- `02_gerador_leads_ml.py`: gera distribuicoes, score oculto e target.
- `03_conversor_sql_parquet.py`: transforma dumps SQL em parquet leve.
- `04_estudo_visual_clientes.py`: analise nao supervisionada dos clientes convertidos.

## Checkpoint antes de seguir

- Existe coluna `Status_Venda` no parquet de leads.
- Conversao final esta perto da faixa esperada (base desbalanceada).
- Artefatos de dados estao na pasta `data/`.

Com isso, voce pode seguir para o modulo de baseline (`02_v1_baseline_xgboost`).
