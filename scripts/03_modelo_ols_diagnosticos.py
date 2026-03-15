# -*- coding: utf-8 -*-
"""
SCRIPT 3: MODELO OLS E DIAGNÓSTICOS
=====================================
- Regressão OLS
- Testes de heterocedasticidade
- Erros robustos HC3
- VIF
- Diagnóstico de resíduos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import sys
sys.path.append('..')
from utils.config import configurar_estilo

# Carregar dados
df_final_clean = pd.read_csv('dados/base_final.csv')

# =============================================================================
# 1. PREPARAR DADOS
# =============================================================================
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
X_const = sm.add_constant(X)

# =============================================================================
# 2. MODELO OLS
# =============================================================================
print("\n" + "="*80)
print("MODELO OLS (MÍNIMOS QUADRADOS ORDINÁRIOS)")
print("="*80)

model = sm.OLS(y, X_const).fit()
print(model.summary())

# =============================================================================
# 3. VIF (FATOR DE INFLAÇÃO DA VARIÂNCIA)
# =============================================================================
print("\n" + "="*80)
print("VIF - FATOR DE INFLAÇÃO DA VARIÂNCIA")
print("="*80)

vif_data = pd.DataFrame()
vif_data['variável'] = X_const.columns
vif_data['VIF'] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
vif_data = vif_data.sort_values('VIF', ascending=False)
print(vif_data.round(4))

# =============================================================================
# 4. TESTES DE HETEROCEDASTICIDADE
# =============================================================================
print("\n" + "="*80)
print("TESTES DE HETEROCEDASTICIDADE")
print("="*80)

residuos = model.resid

# Breusch-Pagan
bp_test = het_breuschpagan(residuos, X_const)
print(f"Breusch-Pagan: LM={bp_test[0]:.4f}, p-valor={bp_test[1]:.4f}")

# White
white_test = het_white(residuos, X_const)
print(f"White: LM={white_test[0]:.4f}, p-valor={white_test[1]:.4f}")

# =============================================================================
# 5. MODELO COM ERROS ROBUSTOS HC3
# =============================================================================
if bp_test[1] < 0.05:
    print("\n" + "="*80)
    print("MODELO COM ERROS-PADRÃO ROBUSTOS (HC3)")
    print("="*80)
    model_robust = sm.OLS(y, X_const).fit(cov_type='HC3')
    print(model_robust.summary())

# =============================================================================
# 6. DIAGNÓSTICO GRÁFICO DOS RESÍDUOS
# =============================================================================
configurar_estilo()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histograma
axes[0, 0].hist(residuos, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].set_title('Histograma dos Resíduos')
axes[0, 0].set_xlabel('Resíduos')
axes[0, 0].set_ylabel('Frequência')

# Q-Q plot
stats.probplot(residuos, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot')

# Resíduos vs Valores Ajustados
axes[1, 0].scatter(model.fittedvalues, residuos, alpha=0.5, color='steelblue')
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_title('Resíduos vs Valores Ajustados')
axes[1, 0].set_xlabel('Valores Ajustados')
axes[1, 0].set_ylabel('Resíduos')

# Boxplot
axes[1, 1].boxplot(residuos, vert=False)
axes[1, 1].set_title('Boxplot dos Resíduos')
axes[1, 1].set_xlabel('Resíduos')

plt.tight_layout()
plt.savefig('outputs/figuras/diagnostico_residuos.png', dpi=300, bbox_inches='tight')
plt.savefig('outputs/figuras/diagnostico_residuos.pdf', bbox_inches='tight')
plt.show()

# =============================================================================
# 7. VALIDAÇÃO CRUZADA
# =============================================================================
print("\n" + "="*80)
print("VALIDAÇÃO CRUZADA K-FOLD (5 FOLDS)")
print("="*80)

lr = LinearRegression()
scores = cross_val_score(lr, X, y, cv=5, scoring='r2')
print(f"R² médio: {scores.mean():.4f} (±{scores.std():.4f})")