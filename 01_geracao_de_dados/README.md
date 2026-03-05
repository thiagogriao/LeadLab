# Modulo 01 - Geracao de Dados

Este modulo constroi a base sintetica usada por todo o restante do projeto.

## Objetivos

- Gerar leads e clientes com padroes realistas.
- Preparar cache de CEPs para enriquecer dados geograficos.
- Produzir estudos visuais para exploracao de perfis.

## Ordem recomendada

1. Ler fundamentos:
   - [`01_fundamentos_dados_sinteticos.md`](01_fundamentos_dados_sinteticos.md)
2. (Opcional) Regenerar a base de CEPs a partir de dumps SQL:
   - `python 01_geracao_de_dados/03_conversor_sql_parquet.py`
3. Gerar base sintetica principal:
   - `python 01_geracao_de_dados/02_gerador_leads_ml.py`
4. Rodar estudo visual:
   - `python 01_geracao_de_dados/04_estudo_visual_clientes.py`

## Entradas

- [`ceps/base_ceps_otimizada.parquet`](../ceps/base_ceps_otimizada.parquet)
- (Opcional para reconstrucao) `ceps/cidade.sql` e `ceps/logradouro.sql`

## Saidas esperadas

- `data/leads_gerados.parquet`
- `data/clientes_gerados.parquet`
- `data/ceps_validos.parquet`
- graficos em `data/estudos/`

## Bibliotecas usadas neste modulo

| Biblioteca | Uso | Documentacao |
|---|---|---|
| `pandas` | montagem e exportacao das bases | https://pandas.pydata.org/docs/ |
| `numpy` | distribuicoes e calculos numericos | https://numpy.org/doc/ |
| `faker` | geracao de dados sinteticos pt-BR | https://faker.readthedocs.io/ |
| `pyarrow` | escrita/leitura de parquet | https://arrow.apache.org/docs/python/ |
| `scikit-learn` | KMeans, PCA e silhouette no estudo visual | https://scikit-learn.org/stable/documentation.html |
| `matplotlib` | graficos estaticos de analise | https://matplotlib.org/stable/users/index.html |
| `seaborn` | visualizacao estatistica | https://seaborn.pydata.org/ |

## Variaveis de ambiente uteis

- `LEADLAB_MODO_NEGOCIO=b2b|b2c`
- `LEADLAB_ESCOPO_GEOGRAFICO=regional|nacional`

Exemplo:

```bash
LEADLAB_MODO_NEGOCIO=b2c LEADLAB_ESCOPO_GEOGRAFICO=nacional \
python 01_geracao_de_dados/02_gerador_leads_ml.py
```

## Checkpoint de estudo

Antes de ir para o Modulo 02, confirme:

- Os arquivos parquet foram gerados em `data/`.
- A coluna alvo `Status_Venda` existe na base de leads.
- O modulo visual gerou os PNGs em `data/estudos/`.
