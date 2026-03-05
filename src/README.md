# src - Componentes Compartilhados

Esta pasta centraliza codigo reutilizado pelos modulos de treino e inferencia.

## Arquivos

- [`config.py`](config.py): caminhos de dados/modelos e listas de features.
- [`features.py`](features.py): funcoes de feature engineering (`base` e `avancada`).
- [`utils.py`](utils.py): metodos de avaliacao e busca de threshold.

## Como usar no estudo

1. Leia `config.py` para entender onde cada script le/escreve artefatos.
2. Leia `features.py` antes de V1/V2/V3 para entender as colunas derivadas.
3. Leia `utils.py` para entender como as metricas e threshold sao calculados.
