# -*- coding: utf-8 -*-
"""
SCRIPT 6: GERAÇÃO DE TODAS AS FIGURAS DO TCC
==============================================
- Figura 1: Matriz de correlação (já no script 2)
- Figura 2: Diagnóstico dos resíduos (já no script 3)
- Figura 3: Comparação de coeficientes (já no script 4)
- Figura 4: Valores reais vs preditos
- Figura 5: Método do cotovelo e silhouette (já no script 5)
- Figura 6: Dispersão INSE x IDEB por cluster (já no script 5)
- Figura 7: Características dos clusters - variáveis contínuas
- Figura 8: Características dos clusters - variáveis binárias
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('..')
from utils.config import configurar_estilo, CORES_VIRIDIS

# Carregar dados
df_final_clean = pd.read_csv('dados/base_final.csv')

configurar_estilo()

# =============================================================================
# FIGURA 4: VALORES REAIS vs PREDITOS (OLS e RIDGE)
# =============================================================================
print("\nGerando Figura 4...")

# Preparar dados para Ridge
x_vars = [
    'MEDIA_INSE', 'IN_AGUA_POTAVEL', 'IN_ENERGIA_REDE_PUBLICA', 'IN_ESGOTO_REDE_PUBLICA',
    'IN_BANDA_LARGA', 'IN_QUADRA_ESPORTES', 'IN_REFEITORIO', 'IN_SALA_LEITURA',
    'IN_LABORATORIO_INFORMATICA', 'IN_LABORATORIO_CIENCIAS', 'IN_SALA_MULTIUSO',
    'IN_EQUIP_LOUSA_DIGITAL', 'IN_EQUIP_MULTIMIDIA', 'QT_DESKTOP_ALUNO',
    'QT_COMP_PORTATIL_ALUNO', 'QT_TABLET_ALUNO', 'QT_MAT_FUND_AF', 'QT_DOC_FUND_AF',
    'URBANA'
]

X = df_final_clean[x_vars].copy()
y = df_final_clean['IDEB_2023'].copy()

# Ridge
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ridge_model = Ridge(alpha=138.95)  # valor do seu melhor alpha
ridge_model.fit(X_scaled, y)
y_pred_ridge = ridge_model.predict(X_scaled)

# OLS
X_const = sm.add_constant(X)
model_ols = sm.OLS(y, X_const).fit()

# Figura
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# OLS
axes[0].scatter(y, model_ols.fittedvalues, alpha=0.4, s=8,
                color=CORES_VIRIDIS[0], edgecolor='black', linewidth=0.2)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r-', linewidth=1.5)
axes[0].set_xlabel('IDEB Real', fontsize=11)
axes[0].set_ylabel('IDEB Predito (OLS)', fontsize=11)
axes[0].text(0.05, 0.95, f'R² = {model_ols.rsquared:.3f}',
             transform=axes[0].transAxes, fontsize=11, verticalalignment='top')

# Ridge
axes[1].scatter(y, y_pred_ridge, alpha=0.4, s=8,
                color=CORES_VIRIDIS[1], edgecolor='black', linewidth=0.2)
axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r-', linewidth=1.5)
axes[1].set_xlabel('IDEB Real', fontsize=11)
axes[1].set_ylabel('IDEB Predito (Ridge)', fontsize=11)
axes[1].text(0.05, 0.95, f'R² = {0.420:.3f}',
             transform=axes[1].transAxes, fontsize=11, verticalalignment='top')

plt.tight_layout()
plt.savefig('outputs/figuras/figura4_real_vs_predito.pdf', bbox_inches='tight', dpi=300)
plt.savefig('outputs/figuras/figura4_real_vs_predito.png', bbox_inches='tight', dpi=300)
plt.show()

# =============================================================================
# FIGURA 7: CARACTERÍSTICAS DOS CLUSTERS - VARIÁVEIS CONTÍNUAS
# =============================================================================
print("Gerando Figura 7...")

variaveis_continuas = [
    'MEDIA_INSE', 'QT_MAT_FUND_AF', 'QT_DOC_FUND_AF',
    'QT_DESKTOP_ALUNO', 'QT_COMP_PORTATIL_ALUNO', 'QT_TABLET_ALUNO'
]

cluster_means_cont = df_final_clean.groupby('cluster')[variaveis_continuas].mean()

nomes_pt = {
    'MEDIA_INSE': 'INSE',
    'QT_MAT_FUND_AF': 'Matrículas',
    'QT_DOC_FUND_AF': 'Docentes',
    'QT_DESKTOP_ALUNO': 'Desktops',
    'QT_COMP_PORTATIL_ALUNO': 'Portáteis',
    'QT_TABLET_ALUNO': 'Tablets'
}
cluster_means_cont = cluster_means_cont.rename(columns=nomes_pt)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (var, ax) in enumerate(zip(cluster_means_cont.columns, axes)):
    values = cluster_means_cont[var].values
    x_pos = np.arange(len(values))
    bars = ax.bar(x_pos, values, color=CORES_VIRIDIS, edgecolor='black', linewidth=0.5)

    for j, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Cluster {j}' for j in range(len(values))])
    ax.set_ylabel('Média', fontsize=10)
    ax.set_title(var, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('outputs/figuras/figura7_clusters_continuas.pdf', bbox_inches='tight', dpi=300)
plt.savefig('outputs/figuras/figura7_clusters_continuas.png', bbox_inches='tight', dpi=300)
plt.show()

# =============================================================================
# FIGURA 8: CARACTERÍSTICAS DOS CLUSTERS - VARIÁVEIS BINÁRIAS
# =============================================================================
print("Gerando Figura 8...")

variaveis_binarias = [
    'IN_AGUA_POTAVEL', 'IN_ENERGIA_REDE_PUBLICA', 'IN_ESGOTO_REDE_PUBLICA',
    'IN_BANDA_LARGA', 'IN_QUADRA_ESPORTES', 'IN_REFEITORIO',
    'IN_SALA_LEITURA', 'IN_LABORATORIO_INFORMATICA', 'IN_LABORATORIO_CIENCIAS',
    'IN_SALA_MULTIUSO', 'IN_EQUIP_LOUSA_DIGITAL', 'IN_EQUIP_MULTIMIDIA',
    'URBANA'
]

cluster_means_bin = df_final_clean.groupby('cluster')[variaveis_binarias].mean() * 100

nomes_pt_bin = {
    'IN_AGUA_POTAVEL': 'Água',
    'IN_ENERGIA_REDE_PUBLICA': 'Energia',
    'IN_ESGOTO_REDE_PUBLICA': 'Esgoto',
    'IN_BANDA_LARGA': 'Banda Larga',
    'IN_QUADRA_ESPORTES': 'Quadra',
    'IN_REFEITORIO': 'Refeitório',
    'IN_SALA_LEITURA': 'Sala Leitura',
    'IN_LABORATORIO_INFORMATICA': 'Lab. Info',
    'IN_LABORATORIO_CIENCIAS': 'Lab. Ciências',
    'IN_SALA_MULTIUSO': 'Sala Multiuso',
    'IN_EQUIP_LOUSA_DIGITAL': 'Lousa Digital',
    'IN_EQUIP_MULTIMIDIA': 'Multimídia',
    'URBANA': 'Urbana'
}
cluster_means_bin = cluster_means_bin.rename(columns=nomes_pt_bin)

fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(cluster_means_bin.columns))
width = 0.25

for i in range(3):
    offset = (i - 1) * width
    values = cluster_means_bin.iloc[i].values
    bars = ax.bar(x + offset, values, width, label=f'Cluster {i}',
                  color=['#FF0000', '#0000FF', '#FFFF00'][i],
                  edgecolor='black', linewidth=0.5)

    for j, (bar, val) in enumerate(zip(bars, values)):
        if val > 5:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                   f'{val:.0f}%', ha='center', va='bottom', fontsize=7)

ax.set_xlabel('Variáveis de Infraestrutura', fontsize=12)
ax.set_ylabel('Proporção (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(cluster_means_bin.columns, rotation=45, ha='right', fontsize=9)
ax.set_ylim(0, 105)
ax.legend(frameon=False, fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('outputs/figuras/figura8_clusters_binarias.pdf', bbox_inches='tight', dpi=300)
plt.savefig('outputs/figuras/figura8_clusters_binarias.png', bbox_inches='tight', dpi=300)
plt.show()

print("\n✅ Todas as figuras foram geradas e salvas em outputs/figuras/")dd