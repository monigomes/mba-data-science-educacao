# -*- coding: utf-8 -*-
"""
SCRIPT 5: ANÁLISE DE CLUSTERS (K-MEANS)
========================================
- Determinação do número ótimo de clusters
- Aplicação do K-means
- Caracterização dos clusters
- Figuras: método do cotovelo, dispersão INSE x IDEB
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway
import sys
sys.path.append('..')
from utils.config import configurar_estilo, CORES_VIRIDIS

# Carregar dados
df_final_clean = pd.read_csv('dados/base_final.csv')

# =============================================================================
# 1. PREPARAR DADOS PARA CLUSTERIZAÇÃO
# =============================================================================
variaveis_cluster = [
    'MEDIA_INSE', 'IN_AGUA_POTAVEL', 'IN_ENERGIA_REDE_PUBLICA', 'IN_ESGOTO_REDE_PUBLICA',
    'IN_BANDA_LARGA', 'IN_QUADRA_ESPORTES', 'IN_REFEITORIO', 'IN_SALA_LEITURA',
    'IN_LABORATORIO_INFORMATICA', 'IN_LABORATORIO_CIENCIAS', 'IN_SALA_MULTIUSO',
    'IN_EQUIP_LOUSA_DIGITAL', 'IN_EQUIP_MULTIMIDIA', 'QT_DESKTOP_ALUNO',
    'QT_COMP_PORTATIL_ALUNO', 'QT_TABLET_ALUNO', 'QT_MAT_FUND_AF', 'QT_DOC_FUND_AF',
    'URBANA'
]

df_cluster = df_final_clean[variaveis_cluster].dropna().copy()
print(f"Observações para clusterização: {len(df_cluster)}")

# Padronizar
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(df_cluster)

# =============================================================================
# 2. DETERMINAR NÚMERO ÓTIMO DE CLUSTERS
# =============================================================================
inercia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled)
    inercia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_cluster_scaled, kmeans.labels_))

melhor_k = K_range[np.argmax(silhouette_scores)]
print(f"\nMelhor número de clusters (silhouette): {melhor_k}")
print(f"Silhouette score: {max(silhouette_scores):.4f}")

# =============================================================================
# 3. FIGURA: MÉTODO DO COTOVELO E SILHOUETTE
# =============================================================================
configurar_estilo()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico do cotovelo
axes[0].plot(K_range, inercia, 'o-', color=CORES_VIRIDIS[0], linewidth=2, markersize=6)
axes[0].set_xlabel('Número de Clusters (k)')
axes[0].set_ylabel('Inércia')
axes[0].scatter(melhor_k, inercia[melhor_k-2], color='red', s=100, zorder=5,
                edgecolor='black', label=f'k={melhor_k} (escolhido)')
axes[0].legend(frameon=False)

# Índice silhouette
axes[1].plot(K_range, silhouette_scores, 'o-', color=CORES_VIRIDIS[1], linewidth=2, markersize=6)
axes[1].set_xlabel('Número de Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].scatter(melhor_k, max(silhouette_scores), color='red', s=100, zorder=5,
                edgecolor='black', label=f'k={melhor_k} (melhor)')
axes[1].legend(frameon=False)

plt.tight_layout()
plt.savefig('outputs/figuras/figura5_metodo_cotovelo_silhouette.pdf', bbox_inches='tight', dpi=300)
plt.savefig('outputs/figuras/figura5_metodo_cotovelo_silhouette.png', bbox_inches='tight', dpi=300)
plt.show()

# =============================================================================
# 4. APLICAR K-MEANS COM K=3
# =============================================================================
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df_cluster['cluster'] = kmeans_final.fit_predict(X_cluster_scaled)

# Adicionar cluster ao dataframe original
df_final_clean['cluster'] = np.nan
df_final_clean.loc[df_cluster.index, 'cluster'] = df_cluster['cluster']
df_final_clean['cluster'] = df_final_clean['cluster'].astype('Int64')

print(f"\nDistribuição dos clusters:")
print(df_final_clean['cluster'].value_counts().sort_index())

# =============================================================================
# 5. RESUMO DOS CLUSTERS
# =============================================================================
resumo_clusters = pd.DataFrame({
    'Cluster': range(3),
    'Nº Escolas': df_final_clean['cluster'].value_counts().sort_index().values,
    'IDEB Médio': df_final_clean.groupby('cluster')['IDEB_2023'].mean().values,
    'INSE Médio': df_final_clean.groupby('cluster')['MEDIA_INSE'].mean().values,
    'Matrículas (média)': df_final_clean.groupby('cluster')['QT_MAT_FUND_AF'].mean().values,
    'Docentes (média)': df_final_clean.groupby('cluster')['QT_DOC_FUND_AF'].mean().values,
    '% Urbana': df_final_clean.groupby('cluster')['URBANA'].mean().values * 100
})

print("\n" + "="*80)
print("RESUMO DOS CLUSTERS")
print("="*80)
print(resumo_clusters.round(2))

# Salvar
resumo_clusters.to_csv('outputs/tabelas/resumo_clusters.csv', index=False)

# =============================================================================
# 6. FIGURA: DISPERSÃO INSE x IDEB POR CLUSTER
# =============================================================================
plt.figure(figsize=(10, 6))
cores_clusters = ['#FF0000', '#0000FF', '#FFFF00']  # vermelho, azul, amarelo

for i in range(3):
    subset = df_final_clean[df_final_clean['cluster'] == i]
    plt.scatter(subset['MEDIA_INSE'], subset['IDEB_2023'],
                c=cores_clusters[i], label=f'Cluster {i}', alpha=0.6, s=30,
                edgecolor='black', linewidth=0.3)

plt.xlabel('INSE', fontsize=11)
plt.ylabel('IDEB 2023', fontsize=11)
plt.legend(frameon=False, fontsize=9)

ax = plt.gca()
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('outputs/figuras/figura6_clusters_inse_ideb.pdf', bbox_inches='tight', dpi=300)
plt.savefig('outputs/figuras/figura6_clusters_inse_ideb.png', bbox_inches='tight', dpi=300)
plt.show()

# =============================================================================
# 7. ANOVA PARA COMPARAR CLUSTERS
# =============================================================================
print("\n" + "="*60)
print("ANOVA - DIFERENÇAS ENTRE CLUSTERS")
print("="*60)

for var in ['IDEB_2023', 'MEDIA_INSE', 'QT_MAT_FUND_AF', 'QT_DOC_FUND_AF']:
    grupos = [df_final_clean[df_final_clean['cluster'] == i][var].dropna() for i in range(3)]
    f_stat, p_val = f_oneway(*grupos)
    print(f"{var}: F={f_stat:.2f}, p-valor={p_val:.4f}")