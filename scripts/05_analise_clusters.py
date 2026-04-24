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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. PREPARAR DADOS
df = pd.read_csv('dados/base_final.csv')
colunas_pca = ['IDEB_2023', 'MEDIA_INSE', 'QT_MAT_FUND_AF', 'QT_DOC_FUND_AF',
               'URBANA', 'QT_TABLET_ALUNO', 'IN_SALA_LEITURA']

# 2. CLUSTERIZAÇÃO (K=3)
# Utilizando as dimensões estratégicas para o treino do agrupamento
vars_treino = ['QT_MAT_FUND_AF', 'URBANA', 'QT_TABLET_ALUNO', 'IN_SALA_LEITURA']
scaler = StandardScaler()
X_scaled_cluster = scaler.fit_transform(df[vars_treino])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled_cluster)
df['cluster'] = kmeans.labels_

# 3. PCA E VISUALIZAÇÃO (SEU CÓDIGO ATUALIZADO)
X_scaled_pca = scaler.fit_transform(df[colunas_pca])
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled_pca)

df_pca_plot = pd.DataFrame(coords, columns=['CP1', 'CP2'])
df_pca_plot['cluster'] = df['cluster']
label_map = {0: "Urbano Ref.", 1: "Rural Eficiente", 2: "Urbano Crítico"}
df_pca_plot['Perfil'] = df_pca_plot['cluster'].map(label_map)
cores_perfil = {"Urbano Ref.": '#440154', "Rural Eficiente": '#21918c', "Urbano Crítico": '#fde725'}

# Início da Plotagem
sns.set_style("white")
plt.rcParams['font.family'] = 'Arial'
fig, ax = plt.subplots(figsize=(12, 8))

sns.scatterplot(data=df_pca_plot, x='CP1', y='CP2', hue='Perfil',
                palette=cores_perfil, alpha=0.2, s=25, edgecolor=None, ax=ax)

# Centroides e Rótulos
centroids = df_pca_plot.groupby('cluster')[['CP1', 'CP2']].mean().reset_index()
for i, row in centroids.iterrows():
    perfil_nome = label_map[row['cluster']]
    ax.scatter(row['CP1'], row['CP2'], marker='X', s=500, color=cores_perfil[perfil_nome],
               edgecolor='black', linewidth=1.5, zorder=10)
    ax.annotate(perfil_nome, (row['CP1'], row['CP2']), xytext=(0, 18), 
                textcoords='offset points', ha='center', va='bottom', 
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9), zorder=11)

# Ajuste Fino da Legenda
leg = ax.legend(title='Perfil Escolar', loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
leg.get_title().set_fontweight('bold')
for lh in leg.legend_handles:
    lh.set_alpha(1.0)
    if hasattr(lh, 'set_sizes'): lh.set_sizes([100])
    else: lh.set_markersize(10)

# Títulos e Eixos
var_exp = pca.explained_variance_ratio_ * 100
ax.set_xlabel(f'Dimensão de Variância 1 ({var_exp[0]:.1f}% explicada)', fontweight='bold', labelpad=12)
ax.set_ylabel(f'Dimensão de Variância 2 ({var_exp[1]:.1f}% explicada)', fontweight='bold', labelpad=12)

sns.despine()
plt.tight_layout()
plt.savefig('outputs/figuras/figura6_pca_espacial.png', dpi=300, bbox_inches='tight')
plt.show()