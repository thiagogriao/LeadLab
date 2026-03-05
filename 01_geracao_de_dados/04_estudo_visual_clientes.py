"""
Estudo Visual de Clientes — Segmentação com Scikit-learn

Análise exploratória visual dos clientes do LeadLab usando clustering (KMeans),
redução dimensional (PCA) e visualizações ricas com matplotlib/seaborn.

Uso:
    python estudo_visual_clientes.py

Gráficos gerados em data/estudos/:
    01_heatmap_correlacao.png
    02_distribuicao_features.png
    03_elbow_method.png
    04_silhouette_scores.png
    05_pca_2d_clusters.png
    06_pca_3d_clusters.png
    07_perfil_clusters.png
    08_boxplots_clusters.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sem GUI para salvar gráficos
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CLIENTES_PARQUET, ESTUDOS_DIR, NUMERIC_FEATURES_V2, BINARY_FEATURES_V2
from src.features import criar_features_avancadas

warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURAÇÕES DE ESTILO
# =====================================================================
sns.set_theme(style='whitegrid', palette='viridis', font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

CORES_CLUSTERS = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6',
                  '#1abc9c', '#e67e22', '#34495e']


# =====================================================================
# FUNÇÕES AUXILIARES
# =====================================================================
def salvar_grafico(fig, nome, diretorio=ESTUDOS_DIR):
    """Salva o gráfico como PNG em alta resolução."""
    caminho = os.path.join(diretorio, nome)
    fig.savefig(caminho, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ Salvo: {caminho}")


# =====================================================================
# 1. HEATMAP DE CORRELAÇÃO
# =====================================================================
def gerar_heatmap_correlacao(df_num, features_nomes):
    """Gera heatmap de correlação entre features numéricas."""
    print("\n📊 [1/8] Heatmap de Correlação...")

    corr = df_num[features_nomes].corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
        center=0, linewidths=0.5, square=True,
        cbar_kws={'shrink': 0.8, 'label': 'Correlação'},
        ax=ax
    )
    ax.set_title('Correlação entre Features Numéricas dos Clientes', fontsize=16, fontweight='bold')
    salvar_grafico(fig, '01_heatmap_correlacao.png')


# =====================================================================
# 2. DISTRIBUIÇÃO POR FEATURE
# =====================================================================
def gerar_distribuicoes(df_num, features_principais):
    """Gera histogramas das principais features numéricas."""
    print("📊 [2/8] Distribuição por Feature...")

    n_features = len(features_principais)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(features_principais):
        sns.histplot(df_num[feat], bins=50, kde=True, ax=axes[i],
                     color=CORES_CLUSTERS[i % len(CORES_CLUSTERS)], alpha=0.7)
        axes[i].set_title(feat.replace('_', ' '), fontweight='bold')
        axes[i].set_xlabel('')

    # Remove eixos extras
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Distribuição das Features dos Clientes', fontsize=18, fontweight='bold', y=1.02)
    fig.tight_layout()
    salvar_grafico(fig, '02_distribuicao_features.png')


# =====================================================================
# 3. MÉTODO DO COTOVELO (ELBOW)
# =====================================================================
def gerar_elbow(X_scaled, max_k=10):
    """Elbow method para encontrar K ótimo."""
    print("📊 [3/8] Método do Cotovelo (Elbow)...")

    inertias = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(K_range, inertias, 'o-', color='#3498db', linewidth=2.5, markersize=8)
    ax.fill_between(K_range, inertias, alpha=0.1, color='#3498db')

    # Encontrar o "cotovelo" — maior variação na segunda derivada
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    elbow_k = np.argmax(np.abs(diffs2)) + 3  # +3 porque começa em 2 e perdeu 2 com diffs

    ax.axvline(x=elbow_k, color='#e74c3c', linestyle='--', linewidth=2, label=f'Cotovelo sugerido: K={elbow_k}')
    ax.set_xlabel('Número de Clusters (K)', fontsize=13)
    ax.set_ylabel('Inércia (Soma das Distâncias)', fontsize=13)
    ax.set_title('Método do Cotovelo — Determinação do K Ideal', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_xticks(list(K_range))
    salvar_grafico(fig, '03_elbow_method.png')

    return elbow_k


# =====================================================================
# 4. SILHOUETTE SCORE
# =====================================================================
def gerar_silhouette(X_scaled, max_k=10):
    """Calcula e plota Silhouette Score para cada K."""
    print("📊 [4/8] Silhouette Scores...")

    silhouettes = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels, sample_size=min(10000, len(X_scaled)))
        silhouettes.append(sil)

    best_k = list(K_range)[np.argmax(silhouettes)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(K_range, silhouettes, color=CORES_CLUSTERS[:len(K_range)], alpha=0.8, edgecolor='white')
    bars[np.argmax(silhouettes)].set_edgecolor('#e74c3c')
    bars[np.argmax(silhouettes)].set_linewidth(3)

    ax.set_xlabel('Número de Clusters (K)', fontsize=13)
    ax.set_ylabel('Silhouette Score', fontsize=13)
    ax.set_title(f'Silhouette Score por K — Melhor K={best_k} ({max(silhouettes):.4f})',
                 fontsize=16, fontweight='bold')
    ax.set_xticks(list(K_range))
    salvar_grafico(fig, '04_silhouette_scores.png')

    return best_k


# =====================================================================
# 5. PCA 2D — SCATTER DOS CLUSTERS
# =====================================================================
def gerar_pca_2d(X_pca_2d, labels, k):
    """Scatter plot 2D dos clusters via PCA."""
    print("📊 [5/8] PCA 2D — Clusters...")

    fig, ax = plt.subplots(figsize=(12, 8))

    for cluster_id in range(k):
        mask = labels == cluster_id
        ax.scatter(
            X_pca_2d[mask, 0], X_pca_2d[mask, 1],
            c=CORES_CLUSTERS[cluster_id], label=f'Cluster {cluster_id}',
            alpha=0.4, s=15, edgecolors='none'
        )

    ax.set_xlabel('PC1 (Componente Principal 1)', fontsize=13)
    ax.set_ylabel('PC2 (Componente Principal 2)', fontsize=13)
    ax.set_title(f'Segmentação de Clientes — PCA 2D (K={k})', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    salvar_grafico(fig, '05_pca_2d_clusters.png')


# =====================================================================
# 6. PCA 3D — SCATTER DOS CLUSTERS
# =====================================================================
def gerar_pca_3d(X_pca_3d, labels, k):
    """Scatter plot 3D dos clusters via PCA."""
    print("📊 [6/8] PCA 3D — Clusters...")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    for cluster_id in range(k):
        mask = labels == cluster_id
        ax.scatter(
            X_pca_3d[mask, 0], X_pca_3d[mask, 1], X_pca_3d[mask, 2],
            c=CORES_CLUSTERS[cluster_id], label=f'Cluster {cluster_id}',
            alpha=0.4, s=10
        )

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(f'Segmentação 3D de Clientes (K={k})', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.view_init(elev=25, azim=45)
    salvar_grafico(fig, '06_pca_3d_clusters.png')


# =====================================================================
# 7. PERFIL DOS CLUSTERS (BARRAS AGRUPADAS)
# =====================================================================
def gerar_perfil_clusters(df_clientes, labels, features_perfil, k):
    """Barras agrupadas com médias normalizadas por cluster."""
    print("📊 [7/8] Perfil dos Clusters...")

    df_temp = df_clientes[features_perfil].copy()
    df_temp['Cluster'] = labels

    # Normalizar para 0-1 para comparação visual
    for col in features_perfil:
        min_val = df_temp[col].min()
        max_val = df_temp[col].max()
        df_temp[col] = (df_temp[col] - min_val) / (max_val - min_val + 1e-10)

    medias = df_temp.groupby('Cluster')[features_perfil].mean()

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(features_perfil))
    width = 0.8 / k

    for cluster_id in range(k):
        offset = (cluster_id - k / 2 + 0.5) * width
        bars = ax.bar(x + offset, medias.loc[cluster_id], width,
                      label=f'Cluster {cluster_id}',
                      color=CORES_CLUSTERS[cluster_id], alpha=0.85,
                      edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Features', fontsize=13)
    ax.set_ylabel('Média Normalizada (0-1)', fontsize=13)
    ax.set_title(f'Perfil Médio dos {k} Clusters de Clientes', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', '\n') for f in features_perfil], fontsize=9, rotation=30, ha='right')
    ax.legend(fontsize=11, ncols=k, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    salvar_grafico(fig, '07_perfil_clusters.png')


# =====================================================================
# 8. BOXPLOTS POR CLUSTER
# =====================================================================
def gerar_boxplots(df_clientes, labels, features_box, k):
    """Boxplots das principais features por cluster."""
    print("📊 [8/8] Boxplots por Cluster...")

    n_features = len(features_box)
    fig, axes = plt.subplots(1, n_features, figsize=(6 * n_features, 7))

    df_temp = df_clientes[features_box].copy()
    df_temp['Cluster'] = labels.astype(str)

    for i, feat in enumerate(features_box):
        ax = axes[i] if n_features > 1 else axes
        palette = {str(c): CORES_CLUSTERS[c] for c in range(k)}
        sns.boxplot(data=df_temp, x='Cluster', y=feat, palette=palette,
                    order=[str(c) for c in range(k)],
                    ax=ax, fliersize=1, linewidth=1.2)
        ax.set_title(feat.replace('_', ' '), fontsize=14, fontweight='bold')
        ax.set_xlabel('Cluster')

    fig.suptitle(f'Distribuição por Cluster (K={k})', fontsize=18, fontweight='bold', y=1.02)
    fig.tight_layout()
    salvar_grafico(fig, '08_boxplots_clusters.png')


# =====================================================================
# RELATÓRIO TEXTUAL DOS CLUSTERS
# =====================================================================
def imprimir_relatorio(df_clientes, labels, features_perfil, k):
    """Imprime resumo estatístico de cada cluster."""
    df_temp = df_clientes[features_perfil].copy()
    df_temp['Cluster'] = labels

    print("\n" + "=" * 70)
    print("📋 RELATÓRIO DE SEGMENTAÇÃO DOS CLIENTES")
    print("=" * 70)
    print(f"\n🔢 Total de clientes analisados: {len(df_clientes):,}")
    print(f"📊 Número de clusters (K): {k}")

    for cluster_id in range(k):
        cluster_mask = df_temp['Cluster'] == cluster_id
        n_cluster = cluster_mask.sum()
        pct = n_cluster / len(df_temp) * 100

        print(f"\n{'─' * 50}")
        print(f"🏷️  CLUSTER {cluster_id} — {n_cluster:,} clientes ({pct:.1f}%)")
        print(f"{'─' * 50}")

        for feat in features_perfil:
            media = df_temp.loc[cluster_mask, feat].mean()
            mediana = df_temp.loc[cluster_mask, feat].median()
            print(f"  {feat:30s}  μ={media:10.2f}  med={mediana:10.2f}")

    print(f"\n{'=' * 70}")
    print("✅ Estudo concluído! Gráficos salvos em data/estudos/")
    print(f"{'=' * 70}\n")


# =====================================================================
# PIPELINE PRINCIPAL
# =====================================================================
def main():
    print("=" * 70)
    print("🔬 LeadLab — Estudo Visual de Clientes com Scikit-learn")
    print("=" * 70)

    # 1. Carregar dados
    print("\n📂 Carregando clientes...")
    df = pd.read_parquet(CLIENTES_PARQUET)
    print(f"   → {len(df):,} clientes carregados")

    # 2. Feature Engineering
    print("⚙️  Aplicando feature engineering...")
    df = criar_features_avancadas(df)

    # 3. Selecionar features para clustering
    features_numericas = [f for f in NUMERIC_FEATURES_V2 if f in df.columns]
    features_binarias = [f for f in BINARY_FEATURES_V2 if f in df.columns]
    features_clustering = features_numericas + features_binarias

    print(f"   → {len(features_clustering)} features selecionadas para clustering")

    # 4. Preparar dados
    df_cluster = df[features_clustering].copy()
    df_cluster = df_cluster.fillna(df_cluster.median())

    # 5. Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    # 6. PCA (2D e 3D)
    print("🔄 Aplicando PCA...")
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    print(f"   → PCA 2D: variância explicada = {pca_2d.explained_variance_ratio_.sum():.2%}")

    pca_3d = PCA(n_components=3, random_state=42)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
    print(f"   → PCA 3D: variância explicada = {pca_3d.explained_variance_ratio_.sum():.2%}")

    # 7. Gerar visualizações iniciais
    gerar_heatmap_correlacao(df_cluster, features_clustering)
    features_dist = ['Salario', 'Idade', 'Score_Engajamento', 'Paginas_Visitadas',
                     'Tempo_Site_Min', 'Intensidade_Navegacao']
    features_dist = [f for f in features_dist if f in df_cluster.columns]
    gerar_distribuicoes(df_cluster, features_dist)

    # 8. Encontrar K ótimo
    elbow_k = gerar_elbow(X_scaled, max_k=10)
    silhouette_k = gerar_silhouette(X_scaled, max_k=10)

    # Usar o K sugerido pelo Silhouette (mais confiável)
    k_otimo = silhouette_k
    print(f"\n🎯 K escolhido: {k_otimo} (Elbow={elbow_k}, Silhouette={silhouette_k})")

    # 9. KMeans final com K ótimo
    print(f"🔄 Treinando KMeans com K={k_otimo}...")
    kmeans = KMeans(n_clusters=k_otimo, random_state=42, n_init=20, max_iter=500)
    labels = kmeans.fit_predict(X_scaled)

    # 10. Gerar gráficos dos clusters
    gerar_pca_2d(X_pca_2d, labels, k_otimo)
    gerar_pca_3d(X_pca_3d, labels, k_otimo)

    features_perfil = ['Salario', 'Idade', 'Score_Engajamento', 'Paginas_Visitadas',
                       'Tempo_Site_Min', 'Intensidade_Navegacao', 'Salario_Norm']
    features_perfil = [f for f in features_perfil if f in df_cluster.columns]

    gerar_perfil_clusters(df_cluster, labels, features_perfil, k_otimo)

    features_box = ['Salario', 'Score_Engajamento', 'Idade']
    features_box = [f for f in features_box if f in df_cluster.columns]
    gerar_boxplots(df_cluster, labels, features_box, k_otimo)

    # 11. Relatório final
    imprimir_relatorio(df_cluster, labels, features_perfil, k_otimo)


if __name__ == "__main__":
    main()
