# -*- coding: utf-8 -*-
"""
SCRIPT 2: ANÁLISE DESCRITIVA E MATRIZ DE CORRELAÇÃO
======================================================
- Estatísticas descritivas
- Matriz de correlação
- Heatmap da correlação
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')
from utils.config import configurar_estilo

# Carregar dados
df_final_clean = pd.read_csv('dados/base_final.csv')

# =============================================================================
# 1. ESTATÍSTICAS DESCRITIVAS
# =============================================================================
print("\n" + "="*80)
print("ESTATÍSTICAS DESCRITIVAS")
print("="*80)

# Lista de variáveis para análise
x_variables = [
    "MEDIA_INSE", "IN_AGUA_POTAVEL", "IN_ENERGIA_REDE_PUBLICA", "IN_ESGOTO_REDE_PUBLICA",
    "IN_BANDA_LARGA", "IN_QUADRA_ESPORTES", "IN_REFEITORIO", "IN_SALA_LEITURA",
    "IN_LABORATORIO_INFORMATICA", "IN_LABORATORIO_CIENCIAS", "IN_SALA_MULTIUSO",
    "IN_EQUIP_LOUSA_DIGITAL", "IN_EQUIP_MULTIMIDIA", "QT_DESKTOP_ALUNO",
    "QT_COMP_PORTATIL_ALUNO", "QT_TABLET_ALUNO", "QT_MAT_FUND_AF", "QT_DOC_FUND_AF"
]
y_variable = "IDEB_2023"
vars_analise = [y_variable] + x_variables

# Estatísticas descritivas
desc_stats = df_final_clean[vars_analise].describe(percentiles=[.25, .5, .75]).T
desc_stats['variancia'] = df_final_clean[vars_analise].var()
desc_stats['coef_variacao'] = (desc_stats['std'] / desc_stats['mean']) * 100
print("\n", desc_stats.round(4))

# Assimetria e curtose
print("\n" + "-"*50)
print("MEDIDAS DE FORMA (Assimetria e Curtose)")
print("-"*50)
for var in vars_analise:
    skewness = df_final_clean[var].skew()
    kurtosis = df_final_clean[var].kurtosis()
    print(f"{var}: Assimetria={skewness:.4f}, Curtose={kurtosis:.4f}")

# =============================================================================
# 2. MATRIZ DE CORRELAÇÃO
# =============================================================================
print("\n" + "="*80)
print("MATRIZ DE CORRELAÇÃO DE PEARSON")
print("="*80)

df_analise = df_final_clean[vars_analise].copy()
corr_matrix = df_analise.corr()
print("\n", corr_matrix.round(4))

# Correlações com a variável dependente
print("\n" + "-"*50)
print(f"CORRELAÇÕES COM {y_variable}")
print("-"*50)
corr_with_y = corr_matrix[y_variable].drop(y_variable).sort_values(ascending=False)
for var, corr in corr_with_y.items():
    print(f"{var}: {corr:.4f}")

# =============================================================================
# 3. HEATMAP DA CORRELAÇÃO
# =============================================================================
configurar_estilo()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f',
            cmap='RdBu_r', center=0, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlação - Variáveis do Estudo', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figuras/matriz_correlacao.png', dpi=300, bbox_inches='tight')
plt.savefig('outputs/figuras/matriz_correlacao.pdf', bbox_inches='tight')
plt.show()