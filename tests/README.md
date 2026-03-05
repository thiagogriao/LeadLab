# tests - Validacao de Sanidade

Testes minimos para garantir que pipeline e utilitarios principais continuam funcionais.

## Arquivo atual

- [`test_pipeline_sanity.py`](test_pipeline_sanity.py)

## Como rodar

```bash
python -m unittest tests/test_pipeline_sanity.py
```

## O que e validado

- Colunas esperadas apos feature engineering.
- Transformacao de pre-processador sem `NaN` ou valores infinitos.
- Saida valida das funcoes utilitarias de metricas.
