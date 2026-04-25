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
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan


# Carregar dados
df_final_clean = pd.read_csv('dados/base_final.csv')

# =============================================================================
# 1. DEFINIR VARIÁVEIS
# =============================================================================
x_vars = [
    'MEDIA_INSE',
    'IN_AGUA_POTAVEL',
    'IN_ENERGIA_REDE_PUBLICA',
    'IN_ESGOTO_REDE_PUBLICA',
    'IN_BANDA_LARGA',
    'IN_QUADRA_ESPORTES',
    'IN_REFEITORIO',
    'IN_SALA_LEITURA',
    'IN_LABORATORIO_INFORMATICA',
    'IN_LABORATORIO_CIENCIAS',
    'IN_SALA_MULTIUSO',
    'IN_EQUIP_LOUSA_DIGITAL',
    'IN_EQUIP_MULTIMIDIA',
    'QT_DESKTOP_ALUNO',
    'QT_COMP_PORTATIL_ALUNO',
    'QT_TABLET_ALUNO',
    'QT_MAT_FUND_AF',
    'QT_DOC_FUND_AF',
    'TP_LOCALIZACAO'
]

X = df_final_clean[x_vars].copy()
y = df_final_clean['IDEB_2023'].copy()

# =============================================================================
# 2. SPLIT TREINO / TESTE
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================================================================
# 3. OLS
# =============================================================================
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

model_ols = sm.OLS(y_train, X_train_const).fit()

print("\n" + "="*60)
print("MODELO OLS")
print("="*60)
print(model_ols.summary())

# =============================================================================
# 4. VIF
# =============================================================================
vif_data = pd.DataFrame()
vif_data['variável'] = X_train_const.columns
vif_data['VIF'] = [
    variance_inflation_factor(X_train_const.values, i)
    for i in range(X_train_const.shape[1])
]

print("\nVIF")
print(vif_data.sort_values('VIF', ascending=False).round(4))

# =============================================================================
# 5. TESTES DE HETEROCEDASTICIDADE
# =============================================================================
print("\n" + "="*80)
print("TESTES DE HETEROCEDASTICIDADE")
print("="*80)

from statsmodels.stats.diagnostic import het_breuschpagan, het_white

residuos = model_ols.resid

# Breusch-Pagan
bp_test = het_breuschpagan(residuos, X_train_const)
print(f"Breusch-Pagan: LM={bp_test[0]:.4f}, p-valor={bp_test[1]:.4f}")

# White
white_test = het_white(residuos, X_train_const)
print(f"White: LM={white_test[0]:.4f}, p-valor={white_test[1]:.4f}")

# =============================================================================
# 6. OLS ROBUSTO (HC3)
# =============================================================================
model_ols_robust = sm.OLS(y_train, X_train_const).fit(cov_type='HC3')

print("\n" + "="*60)
print("OLS ROBUSTO (HC3)")
print("="*60)
print(model_ols_robust.summary())

# =============================================================================
# 7. MÉTRICAS OLS (TESTE)
# =============================================================================
y_pred_ols = model_ols_robust.predict(X_test_const)

r2_ols = r2_score(y_test, y_pred_ols)
mae_ols = mean_absolute_error(y_test, y_pred_ols)
rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))

print("\nOLS (TESTE)")
print(f"R²: {r2_ols:.4f}")
print(f"MAE: {mae_ols:.4f}")
print(f"RMSE: {rmse_ols:.4f}")

# =============================================================================
# 8. RIDGE (SEM DATA LEAKAGE)
# =============================================================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

alphas = np.logspace(-3, 3, 50)

ridge_cv = RidgeCV(alphas=alphas, scoring='r2', cv=5)
ridge_cv.fit(X_train_scaled, y_train)

best_alpha = ridge_cv.alpha_

print("\nMelhor alpha:", round(best_alpha, 4))

ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_train_scaled, y_train)

# Coeficientes
ridge_coefs = pd.DataFrame({
    'variável': X.columns,
    'coef_ridge': ridge_model.coef_,
    'abs_coef': np.abs(ridge_model.coef_)
}).sort_values('abs_coef', ascending=False)

print("\nCoeficientes Ridge:")
print(ridge_coefs.round(4))


# =============================================================================
# FIGURA  - DIAGNÓSTICO DOS RESÍDUOS
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson

residuos = y_test - y_pred_ols
y_pred = y_pred_ols

# -----------------------------------------------------------------------------
# CONFIGURAÇÕES DE ESTILO 
# -----------------------------------------------------------------------------
sns.set_style("white")
plt.rcParams['axes.grid'] = False
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# -----------------------------------------------------------------------------
# FIGURA
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# -----------------------------------------------------------------------------
# HISTOGRAMA + NORMAL
# -----------------------------------------------------------------------------
axes[0, 0].hist(residuos, bins=30, density=True, alpha=0.7,
                color=sns.color_palette("viridis", 1)[0],
                edgecolor='black', linewidth=0.5)

x = np.linspace(residuos.min(), residuos.max(), 100)
axes[0, 0].plot(x, stats.norm.pdf(x, residuos.mean(), residuos.std()),
                'r-', lw=2, label='Distribuição Normal')

axes[0, 0].axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

axes[0, 0].set_xlabel('Resíduos')
axes[0, 0].set_ylabel('Densidade')
axes[0, 0].legend(frameon=False)

# -----------------------------------------------------------------------------
# Q-Q PLOT
# -----------------------------------------------------------------------------
stats.probplot(residuos, dist="norm", plot=axes[0, 1])

axes[0, 1].get_lines()[0].set_color(sns.color_palette("viridis", 1)[0])
axes[0, 1].get_lines()[1].set_color('red')

axes[0, 1].set_xlabel('Quantis Teóricos')
axes[0, 1].set_ylabel('Quantis Observados')

# -----------------------------------------------------------------------------
# RESÍDUOS VS AJUSTADOS
# -----------------------------------------------------------------------------
axes[1, 0].scatter(y_pred, residuos, alpha=0.5, s=15,
                  color=sns.color_palette("viridis", 1)[0],
                  edgecolor='black', linewidth=0.2)

axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1.5)

axes[1, 0].set_xlabel('Valores Ajustados (IDEB previsto)')
axes[1, 0].set_ylabel('Resíduos')

# -----------------------------------------------------------------------------
# BOXPLOT
# -----------------------------------------------------------------------------
boxplot = axes[1, 1].boxplot(residuos, vert=False, patch_artist=True,
                            widths=0.6, showmeans=True)

for patch in boxplot['boxes']:
    patch.set_facecolor(sns.color_palette("viridis", 1)[0])
    patch.set_edgecolor('black')

for median in boxplot['medians']:
    median.set_color('red')

# -----------------------------------------------------------------------------
# FINALIZAÇÃO
# -----------------------------------------------------------------------------
for ax in axes.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()

plt.savefig('figura2_diagnostico_residuos.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figura2_diagnostico_residuos.png', bbox_inches='tight', dpi=300)

print("✅ Figura salva")
plt.show()