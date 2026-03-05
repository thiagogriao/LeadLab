# Base de CEPs

Pasta com base geografica usada para enriquecer os dados sinteticos.

## Arquivos esperados

| Arquivo | Descricao |
|---|---|
| `cidade.sql` | Dump SQL com cidades brasileiras |
| `logradouro.sql` | Dump SQL com logradouros e CEP |
| `base_ceps_otimizada.parquet` | Saida consolidada usada pelos modulos de dados |

## Como regenerar `base_ceps_otimizada.parquet`

Da raiz do projeto:

```bash
python 01_geracao_de_dados/03_conversor_sql_parquet.py
```

O script:

1. Le os dumps SQL de `ceps/`.
2. Realiza o join por `id_cidade`.
3. Exporta parquet final com colunas:
   - `CEP`
   - `Logradouro`
   - `Bairro`
   - `Cidade`
   - `Estado`

## Observacao

Se os arquivos SQL nao estiverem na pasta `ceps/`, a reconstrucao nao sera executada.
