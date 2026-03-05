"""
Fragmento de estudo (V3): preparo de dados para embeddings.

Converte categoricas para indices e normaliza numericas para treino em rede neural.
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preparar_dados_dl(df, categorical_cols, numerical_cols):
    """
    ETAPA 1 - LabelEncoder em categoricas.
    ETAPA 2 - Calcular tamanho dos embeddings.
    ETAPA 3 - StandardScaler em numericas.
    """
    print("[*] Preparando dados para PyTorch...")
    label_encoders = {}
    embedding_sizes = []

    # ETAPA 1 - Categóricas para indices inteiros.
    df_cat = np.zeros((len(df), len(categorical_cols)), dtype=np.int64)
    for i, col in enumerate(categorical_cols):
        le = LabelEncoder()
        valores = df[col].astype(str).fillna("Desconhecido").values
        df_cat[:, i] = le.fit_transform(valores)
        label_encoders[col] = le

        # ETAPA 2 - Regra simples para dimensao do embedding.
        num_classes = len(le.classes_)
        emb_dim = min(50, max(2, num_classes // 2))
        embedding_sizes.append((num_classes, emb_dim))

    # ETAPA 3 - Numéricas normalizadas.
    scaler = StandardScaler()
    df_num = scaler.fit_transform(df[numerical_cols].fillna(0))

    return df_cat, df_num, label_encoders, scaler, embedding_sizes
