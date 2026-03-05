"""
Fragmento de estudo (V2): feature engineering adicional.

Este arquivo mostra como criar colunas derivadas antes do treino do ensemble.
"""
import pandas as pd


def aplicar_engenharia_features_magicas(df):
    """Cria features de faixa e interacao de negocio."""
    print("[*] Aplicando feature engineering didatico da V2...")
    df = df.copy()

    # ETAPA 1 - Faixas de idade para reduzir ruido de valores isolados.
    df["Faixa_Etaria"] = pd.cut(
        df["Idade"],
        bins=[17, 24, 34, 44, 54, 100],
        labels=["18-24", "25-34", "35-44", "45-54", "55+"],
    )

    # ETAPA 2 - Faixas de engajamento por quantis (mais robusto a escala).
    if df["Score_Engajamento"].nunique() >= 3:
        df["Nivel_Engajamento"] = pd.qcut(
            df["Score_Engajamento"],
            q=3,
            labels=["Baixo", "Medio", "Alto"],
            duplicates="drop",
        )
    else:
        # Fallback para bases pequenas ou quase constantes.
        df["Nivel_Engajamento"] = "Medio"

    # ETAPA 3 - Interacoes entre comportamento e capacidade financeira.
    df["Intensidade_Navegacao"] = df["Paginas_Visitadas"] * df["Tempo_Site_Min"]
    df["Poder_Engajamento"] = (df["Salario"] / 1000.0) * df["Score_Engajamento"]
    df["Salario_por_Minuto_Site"] = df["Salario"] / (df["Tempo_Site_Min"] + 1.0)

    return df


if __name__ == "__main__":
    print("Fragmento carregado. Chame aplicar_engenharia_features_magicas(df).")
