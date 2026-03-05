"""
Fragmento de estudo (V1): estrutura do pre-processamento.

Este arquivo isola apenas a etapa de transformacao de colunas.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ETAPA 1 - Definir o tipo de cada coluna.
CATEGORICAL_FEATURES = ["Genero", "Estado_Civil", "Setor"]
NUMERIC_FEATURES = ["Idade", "Salario", "Score_Engajamento"]
BINARY_FEATURES = ["Tem_Email_Corp", "Abriu_Email"]


def montar_preprocessador():
    """Monta o ColumnTransformer usado pela V1 e V2."""
    # ETAPA 2 - Pipeline numerico.
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    # ETAPA 3 - Pipeline categorico.
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    # ETAPA 4 - Unificar transformacoes em um unico objeto.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )
    return preprocessor


if __name__ == "__main__":
    print("[*] Montando preprocessor da V1...")
    preprocessor = montar_preprocessador()
    print("    OK: ColumnTransformer criado com etapas num/cat/bin.")
