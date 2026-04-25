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
# 2. MATRIZ DE CORRELAÇÃO RESUMIDA
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("\n" + "=" * 80)
print("MATRIZ DE CORRELAÇÃO DE PEARSON - VARIÁVEIS SELECIONADAS")
print("=" * 80)

# -----------------------------------------------------------------------------
# 1. SELEÇÃO DAS VARIÁVEIS ESSENCIAIS
# -----------------------------------------------------------------------------
variaveis_essenciais = [
    'IDEB_2023',
    'MEDIA_INSE',
    'QT_MAT_FUND_AF',
    'QT_DOC_FUND_AF',
    'TP_LOCALIZACAO',
    'IN_EQUIP_LOUSA_DIGITAL',
    'QT_TABLET_ALUNO',
    'IN_LABORATORIO_INFORMATICA',
    'IN_SALA_LEITURA',
    'IN_SALA_MULTIUSO',
    'IN_LABORATORIO_CIENCIAS'
]

df_subset = df_final_clean[variaveis_essenciais].dropna().copy()

# -----------------------------------------------------------------------------
# 2. MATRIZ DE CORRELAÇÃO
# -----------------------------------------------------------------------------
corr_matrix_essencial = df_subset.corr()

# -----------------------------------------------------------------------------
# 3. HEATMAP (COM VALORES)
# -----------------------------------------------------------------------------
sns.set_style("white")
sns.set_context("paper", font_scale=1.2)

plt.figure(figsize=(10, 8))

mask = np.triu(np.ones_like(corr_matrix_essencial, dtype=bool))

sns.heatmap(
    corr_matrix_essencial,
    mask=mask,
    annot=True,
    fmt='.3f',
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0,
    cbar_kws={
        "shrink": 0.8,
        "label": "Coeficiente de Correlação de Pearson",
        "ticks": [-1.0, -0.5, 0, 0.5, 1.0]
    },
    annot_kws={"size": 8}
)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# -----------------------------------------------------------------------------
# 4. SALVAR (SEM DUPLICAÇÃO)
# -----------------------------------------------------------------------------
plt.savefig('matriz_correlacao_essencial.png', dpi=300, bbox_inches='tight')
plt.savefig('matriz_correlacao_essencial.pdf', format='pdf', bbox_inches='tight')
plt.savefig('matriz_correlacao_essencial.svg', format='svg', bbox_inches='tight')

plt.show()

# -----------------------------------------------------------------------------
# 5. CORRELAÇÕES DIRETAS COM IDEB
# -----------------------------------------------------------------------------
print("\n" + "-" * 50)
print("CORRELAÇÕES COM IDEB_2023")
print("-" * 50)

print(
    corr_matrix_essencial['IDEB_2023']
    .sort_values(ascending=False)
    .round(4)
)

# =============================================================================
# 3. MATRIZ DE CORRELAÇÃO COMPLETA (SEM RÓTULOS)
# =============================================================================

sns.set_style("white")
sns.set_context("paper", font_scale=1.2)

# -----------------------------------------------------------------------------
# MATRIZ COMPLETA (OU PODE USAR A RESUMIDA)
# -----------------------------------------------------------------------------
corr_matrix = df_final_clean[variaveis_essenciais].dropna().corr()

plt.figure(figsize=(10, 8))

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

ax = sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=False,
    cmap='RdBu_r',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={
        "shrink": 0.8,
        "label": "Coeficiente de Correlação de Pearson ($r$)",
        "ticks": [-1.0, -0.5, 0, 0.5, 1.0]
    }
)

# Remover bordas externas
for spine in ax.spines.values():
    spine.set_visible(False)

plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)

plt.tight_layout()

# -----------------------------------------------------------------------------
# SALVAR
# -----------------------------------------------------------------------------
plt.savefig('matriz_correlacao_clean.png', dpi=300, bbox_inches='tight')
plt.savefig('matriz_correlacao_clean.pdf', format='pdf', bbox_inches='tight')
plt.savefig('matriz_correlacao_clean.svg', format='svg', bbox_inches='tight')

plt.show()