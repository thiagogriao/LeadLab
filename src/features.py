"""
Feature Engineering — funções compartilhadas entre V1 e V2 do classificador.
"""
import pandas as pd
import numpy as np
from datetime import datetime


def criar_features_base(df):
    """
    Cria features derivadas dos dados reais (usada por V1 e V2).
    Inclui: Idade, Email Corp, Score Engajamento, Faixa Salarial, Interações.
    """
    hoje = datetime.now()

    # 1. Idade a partir de Data_Nascimento
    df['Data_Nascimento'] = pd.to_datetime(df['Data_Nascimento'], errors='coerce')
    df['Idade'] = df['Data_Nascimento'].apply(
        lambda x: hoje.year - x.year - ((hoje.month, hoje.day) < (x.month, x.day)) if pd.notnull(x) else np.nan
    )
    df['Idade'] = df['Idade'].fillna(df['Idade'].median())

    # 2. Flag: Tem Email Corporativo
    df['Tem_Email_Corp'] = (df['Email_Corporativo'].notna() & (df['Email_Corporativo'] != "")).astype(int)

    # 3. Score de Engajamento Digital
    df['Score_Engajamento'] = (
        df['Paginas_Visitadas'] * 0.5 +
        df['Tempo_Site_Min'] * 0.3 +
        df['Abriu_Email'] * 5 +
        df['Clicou_Email'] * 10
    )

    # 4. Faixa Salarial (binning)
    df['Faixa_Salarial'] = pd.cut(
        df['Salario'],
        bins=[0, 3000, 6000, 10000, 20000, 45001],
        labels=['Ate_3k', '3k_6k', '6k_10k', '10k_20k', 'Acima_20k']
    ).astype(str)

    # 5. Feature de Interação: Engajamento × Clicou Email
    df['Engajamento_x_Clicou'] = df['Score_Engajamento'] * df['Clicou_Email']

    # 6. Feature de Interação: Páginas × Tempo (intensidade de navegação)
    df['Intensidade_Navegacao'] = df['Paginas_Visitadas'] * df['Tempo_Site_Min']

    # 7. Salário normalizado (0 a 1)
    df['Salario_Norm'] = (df['Salario'] - df['Salario'].min()) / (df['Salario'].max() - df['Salario'].min() + 1e-10)

    return df


def criar_features_avancadas(df):
    """
    Feature engineering avançado V2: polinomiais, faixas de idade, flags.
    Chama criar_features_base() + adiciona features extras.
    """
    df = criar_features_base(df)

    # 8. Features Polinomiais
    df['Engajamento_Quadrado'] = df['Score_Engajamento'] ** 2
    df['Salario_x_Engajamento'] = df['Salario_Norm'] * df['Score_Engajamento']
    df['Idade_x_Salario'] = df['Idade'] * df['Salario_Norm']

    # 9. Faixa de Idade
    df['Faixa_Idade'] = pd.cut(
        df['Idade'],
        bins=[0, 25, 35, 45, 55, 100],
        labels=['Jovem', 'Adulto_Jovem', 'Adulto', 'Maduro', 'Senior']
    ).astype(str)

    # 10. Flags binárias de alto valor
    df['Engajamento_Alto'] = (df['Score_Engajamento'] > df['Score_Engajamento'].quantile(0.75)).astype(int)
    df['Salario_Alto'] = (df['Salario'] > 20000).astype(int)

    return df
