# -*- coding: utf-8 -*-
"""
SCRIPT 5: ANÁLISE DE CLUSTERS (K-MEANS) E PCA
============================================
- Clusterização K-means (K=3)
- Análise de Componentes Principais (PCA)
- Visualização Espacial com correção de legendas
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. PREPARAR DADOS
df = pd.read_csv('dados/base_final.csv')

# =============================================================================
# 1. DEFINIÇÃO DAS VARIÁVEIS
# =============================================================================

#  SOMENTE VARIÁVEIS ESTRUTURAIS (ENTRAM NO CLUSTER)
vars_treino = [
    'QT_MAT_FUND_AF',      # Porte
    'TP_LOCALIZACAO',      # Geografia
    'QT_TABLET_ALUNO',     # Tecnologia
    'IN_SALA_LEITURA'      # Infra pedagógica
]

#  SOMENTE PARA INTERPRETAÇÃO (NÃO ENTRAM NO MODELO)
vars_validacao = [
    'IDEB_2023',
    'MEDIA_INSE',
    'QT_DOC_FUND_AF'
]

# Base final
df_agrupamento = df_final_clean[vars_treino + vars_validacao].dropna().copy()

# Separação
X_cluster = df_agrupamento[vars_treino]
X_validacao = df_agrupamento[vars_validacao]

# =============================================================================
# 2. PADRONIZAÇÃO (Z-SCORE)
# =============================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# =============================================================================
# 3. DEFINIÇÃO DO K ÓTIMO
# =============================================================================
sns.set_style("white")
plt.rcParams['font.family'] = 'Arial'

inercia = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    inercia.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

# Gráfico de validação
fig, ax1 = plt.subplots(figsize=(10, 5))
cores = sns.color_palette("viridis", 2)

lns1 = ax1.plot(K_range, inercia, marker='o', color=cores[0], label='Inércia')
ax1.set_xlabel('Número de Clusters (k)')
ax1.set_ylabel('Inércia')

ax2 = ax1.twinx()
lns2 = ax2.plot(K_range, silhouettes, marker='o', color=cores[1], label='Silhouette')
ax2.set_ylabel('Índice Silhouette')

lns = lns1 + lns2
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc='upper right')

sns.despine()
plt.tight_layout()
plt.savefig('figura_cluster_validacao.png', dpi=300)
plt.show()

# Melhor K (Silhouette)
k_ideal = K_range[np.argmax(silhouettes)]
print(f"\nMelhor k sugerido (Silhouette): {k_ideal}")

# =============================================================================
# 4. MODELO FINAL (PARCIMÔNIA: K=3)
# =============================================================================
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=50)
df_agrupamento['cluster'] = kmeans_final.fit_predict(X_scaled)

# Nomear clusters (interpretável)
nomes_cluster = {
    0: "Urbano Referência",
    1: "Rural Eficiente",
    2: "Urbano Vulnerável"
}

df_agrupamento['perfil'] = df_agrupamento['cluster'].map(nomes_cluster)

# =============================================================================
# 5. MATRIZ DE CENTROIDES (INTERPRETAÇÃO REAL)
# =============================================================================
tabela_centroides = df_agrupamento.groupby('perfil').agg({
    'IDEB_2023': 'mean',
    'MEDIA_INSE': 'mean',
    'QT_MAT_FUND_AF': 'mean',
    'QT_DOC_FUND_AF': 'mean',
    'TP_LOCALIZACAO': 'mean',
    'QT_TABLET_ALUNO': 'mean',
    'IN_SALA_LEITURA': 'mean'
}).round(3)

tabela_centroides['Contagem'] = df_agrupamento['perfil'].value_counts()

print("\n" + "="*80)
print("MATRIZ DE CENTROIDES (VALORES REAIS)")
print("="*80)
print(tabela_centroides)

# =============================================================================
# 6. GRÁFICO DE PERFIS (NORMALIZADO + VALORES REAIS)
# =============================================================================
resumo_real = df_agrupamento.groupby('perfil').agg({
    'IDEB_2023': 'mean',
    'MEDIA_INSE': 'mean',
    'QT_MAT_FUND_AF': 'mean',
    'QT_TABLET_ALUNO': 'mean',
    'IN_SALA_LEITURA': 'mean'
})

resumo_real.columns = ['IDEB', 'INSE', 'Porte', 'Tech', 'Infra']

scaler_mm = MinMaxScaler()
resumo_scaled = pd.DataFrame(
    scaler_mm.fit_transform(resumo_real),
    columns=resumo_real.columns,
    index=resumo_real.index
)

fig, ax = plt.subplots(figsize=(14, 7))

resumo_scaled.T.plot(
    kind='bar',
    ax=ax,
    colormap='viridis',
    edgecolor='black',
    linewidth=0.5,
    width=0.8
)

# Rótulos reais
for i, p in enumerate(ax.patches):
    cluster_idx = i // len(resumo_real.columns)
    col_idx = i % len(resumo_real.columns)
    valor_real = resumo_real.iloc[cluster_idx, col_idx]

    label = f'{valor_real:.1f}' if valor_real > 1 else f'{valor_real:.2%}'

    ax.annotate(label,
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold')

plt.ylabel('Escala Normalizada (0–1)')
plt.xticks(rotation=0, fontsize=10)
plt.ylim(0, 1.15)

labels_cluster = ['Cluster 0 (Urbano Ref.)', 'Cluster 1 (Rural Eficiente)', 'Cluster 2 (Urbano Crítico)']
plt.legend(labels_cluster, title='Perfil do Cluster', loc='upper left', bbox_to_anchor=(1, 1), frameon=False)


sns.despine()
plt.tight_layout()
plt.savefig('figura_perfis_cluster.png', dpi=300)
plt.show()

# =============================================================================
# 7. PCA (VISUALIZAÇÃO COM VARIÁVEIS DE VALIDAÇÃO)
# =============================================================================
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#  Variáveis explicativas (agora incluindo desempenho)
vars_pca = [
    'IDEB_2023',
    'MEDIA_INSE',
    'QT_DOC_FUND_AF',
    'QT_MAT_FUND_AF',
    'TP_LOCALIZACAO',
    'QT_TABLET_ALUNO',
    'IN_SALA_LEITURA'
]

# Base alinhada
df_pca_in = df_agrupamento[vars_pca].dropna().copy()

#  garantir alinhamento com cluster
df_pca_in['perfil'] = df_agrupamento.loc[df_pca_in.index, 'perfil']
cores_perfil = {"Urbano Referência": '#440154', "Rural Eficiente": '#21918c', "Urbano Vulnerável": '#fde725'}

# Padronização
X_scaled = StandardScaler().fit_transform(df_pca_in[vars_pca])

# PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(coords, columns=['CP1', 'CP2'])
df_pca['perfil'] = df_pca_in['perfil'].values

# Plot
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_pca,
    x='CP1',
    y='CP2',
    hue='perfil',
    palette=cores_perfil,
    alpha=0.3,
    s=25
)

# Centroides
centroids = df_pca.groupby('perfil')[['CP1', 'CP2']].mean()

for idx, row in centroids.iterrows():
    plt.scatter(row['CP1'], row['CP2'],
                marker='X', s=400,
                edgecolor='black', linewidth=1.5)

# Variância explicada
var_exp = pca.explained_variance_ratio_ * 100

plt.xlabel(f'Dimensão de Variância 1 ({var_exp[0]:.1f}%)')
plt.ylabel(f'Dimensão de Variância 2 ({var_exp[1]:.1f}%)')

sns.despine()
plt.tight_layout()
plt.savefig('figura_pca_cluster_validacao.png', dpi=300)
plt.show()