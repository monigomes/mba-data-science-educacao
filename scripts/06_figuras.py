# -*- coding: utf-8 -*-
"""
SCRIPT 6: GERAÇÃO DE FIGURAS FINAIS (REPOSITÓRIO E TCC)
======================================================
- Figura 1: Matriz de Correlação
- Figura 5: Validação de Clusters (Cotovelo e Silhouette)
- Figura: Ridge Trace Plot (Encolhimento de Betas)
- Figura 3: Importância dos Coeficientes
- Figuras 7/8: Perfis dos Clusters (Centroides)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.stats as stats

# 1. CONFIGURAÇÕES DE ESTILO (PADRÃO ESALQ)
sns.set_style("white")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
CORES_V = sns.color_palette("viridis", 3)

# Carregar dados
df = pd.read_csv('dados/base_final.csv')
x_vars = ['MEDIA_INSE', 'URBANA', 'QT_TABLET_ALUNO', 'IN_SALA_LEITURA', 
          'QT_MAT_FUND_AF', 'IN_LABORATORIO_CIENCIAS', 'IN_EQUIP_LOUSA_DIGITAL']

# Preparar dados padronizados para os modelos
X = df[x_vars]
y = df['IDEB_2023']
X_scaled = StandardScaler().fit_transform(X)

# =============================================================================
# FIGURA 1: MATRIZ DE CORRELAÇÃO (PADRÃO USP)
# =============================================================================
plt.figure(figsize=(10, 8))
corr = df[x_vars + ['IDEB_2023']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('outputs/figuras/figura1_correlacao_usp.png', dpi=300)

# =============================================================================
# FIGURA 5: VALIDAÇÃO DO NÚMERO DE CLUSTERS (SEU CÓDIGO)
# =============================================================================
inercia, silhouettes = [], []
K_range = range(2, 11)
X_cluster = StandardScaler().fit_transform(df[['QT_MAT_FUND_AF', 'URBANA', 'QT_TABLET_ALUNO', 'IN_SALA_LEITURA']])

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_cluster)
    inercia.append(km.inertia_)
    silhouettes.append(silhouette_score(X_cluster, km.labels_))

fig, ax1 = plt.subplots(figsize=(10, 5))
lns1 = ax1.plot(K_range, inercia, marker='o', color=CORES_V[0], label='Inércia (Cotovelo)')
ax1.set_xlabel('Número de Clusters (k)')
ax1.set_ylabel('Inércia')

ax2 = ax1.twinx()
lns2 = ax2.plot(K_range, silhouettes, marker='o', color=CORES_V[1], label='Índice Silhouette')
ax2.set_ylabel('Índice Silhouette')

lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper right', frameon=True)
sns.despine(top=True, right=False)
plt.tight_layout()
plt.savefig('outputs/figuras/figura5_validacao_k.svg')

# =============================================================================
# RIDGE TRACE PLOT (SUGESTÃO ADICIONAL)
# =============================================================================
alphas = np.logspace(-2, 6, 100)
coefs = []
for a in alphas:
    ridge = Ridge(alpha=a).fit(X_scaled, y)
    coefs.append(ridge.coef_)

plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('Alpha (λ) - Escala Log')
plt.ylabel('Coeficientes (β)')
plt.title('Ridge Trace: Encolhimento dos Coeficientes')
sns.despine()
plt.savefig('outputs/figuras/ridge_trace_plot.png', dpi=300)

# =============================================================================
# FIGURA 3: IMPORTÂNCIA DAS VARIÁVEIS
# =============================================================================
# Treinar modelos para comparação (usando dados padronizados)
from sklearn.linear_model import LinearRegression
ols = LinearRegression().fit(X_scaled, y)
ridge = Ridge(alpha=100).fit(X_scaled, y) # Exemplo de alpha

importancia = pd.DataFrame({
    'Variável': x_vars,
    'OLS': ols.coef_,
    'Ridge': ridge.coef_
}).sort_values(by='Ridge', ascending=True)

plt.figure(figsize=(10, 8))
y_pos = np.arange(len(importancia))
plt.barh(y_pos - 0.2, importancia['OLS'], 0.4, label='OLS', color=CORES_V[0], alpha=0.7)
plt.barh(y_pos + 0.2, importancia['Ridge'], 0.4, label='Ridge', color=CORES_V[1], alpha=0.7)
plt.yticks(y_pos, importancia['Variável'])
plt.xlabel('Coeficiente Padronizado (Beta)')
plt.legend(frameon=False)
sns.despine()
plt.tight_layout()
plt.savefig('outputs/figuras/figura3_importancia_juliano.png', dpi=300)

plt.show()