# -*- coding: utf-8 -*-
"""
SCRIPT 4: REGRESSÃO RIDGE
==========================
- Regressão Ridge com validação cruzada
- Comparação com OLS
- Coeficientes padronizados
- Figuras comparativas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
import sys
sys.path.append('..')
from utils.config import configurar_estilo, CORES_VIRIDIS

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

# Padronizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================================================================
# 2. REGRESSÃO RIDGE COM VALIDAÇÃO CRUZADA
# =============================================================================
print("\n" + "="*80)
print("REGRESSÃO RIDGE")
print("="*80)

alphas = np.logspace(-3, 3, 50)
ridge_cv = RidgeCV(alphas=alphas, scoring='r2', cv=5)
ridge_cv.fit(X_scaled, y)

best_alpha = ridge_cv.alpha_
print(f"Melhor alpha: {best_alpha:.4f}")
print(f"Melhor R² CV: {ridge_cv.best_score_:.4f}")

# Modelo final
ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_scaled, y)

# =============================================================================
# 3. COEFICIENTES
# =============================================================================
ridge_coefs = pd.DataFrame({
    'variável': X.columns,
    'coef_ridge': ridge_model.coef_,
    'abs_coef': np.abs(ridge_model.coef_)
}).sort_values('abs_coef', ascending=False)
print("\nCoeficientes Ridge:")
print(ridge_coefs.round(4))

# =============================================================================
# 4. MÉTRICAS
# =============================================================================
y_pred_ridge = ridge_model.predict(X_scaled)
r2_ridge = r2_score(y, y_pred_ridge)
mae_ridge = mean_absolute_error(y, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y, y_pred_ridge))

print("\n" + "="*50)
print("MÉTRICAS DO MODELO RIDGE")
print("="*50)
print(f"R²: {r2_ridge:.4f}")
print(f"MAE: {mae_ridge:.4f}")
print(f"RMSE: {rmse_ridge:.4f}")
print(f"Erro percentual: {(mae_ridge / y.mean())*100:.1f}%")

# =============================================================================
# 5. FIGURA: COMPARAÇÃO DE COEFICIENTES
# =============================================================================
configurar_estilo()

# Coeficientes do OLS
X_const = sm.add_constant(X)
model_ols = sm.OLS(y, X_const).fit()

# DataFrame de comparação
coef_comparison = pd.DataFrame({
    'variável': X.columns,
    'coef_ols': model_ols.params[1:].values,
    'coef_ridge': ridge_model.coef_
})
coef_comparison['abs_ols'] = np.abs(coef_comparison['coef_ols'])
coef_comparison = coef_comparison.sort_values('abs_ols', ascending=True)

# Gráfico
fig, ax = plt.subplots(figsize=(10, 8))
y_pos = np.arange(len(coef_comparison))
height = 0.35

ax.barh(y_pos - height/2, coef_comparison['coef_ols'], height,
        label='OLS', color=CORES_VIRIDIS[0], alpha=0.8, edgecolor='black', linewidth=0.3)
ax.barh(y_pos + height/2, coef_comparison['coef_ridge'], height,
        label=f'Ridge (λ={best_alpha:.2f})', color=CORES_VIRIDIS[1], alpha=0.8, edgecolor='black', linewidth=0.3)

ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(coef_comparison['variável'], fontsize=9)
ax.set_xlabel('Coeficiente', fontsize=11)
ax.legend(loc='lower right', frameon=False)

# Adicionar valores
for i, (v_ols, v_ridge) in enumerate(zip(coef_comparison['coef_ols'], coef_comparison['coef_ridge'])):
    if abs(v_ols) > 0.02:
        ax.text(v_ols + 0.01, i - height/2, f'{v_ols:.3f}', va='center', fontsize=7)
    if abs(v_ridge) > 0.02:
        ax.text(v_ridge + 0.01, i + height/2, f'{v_ridge:.3f}', va='center', fontsize=7)

plt.tight_layout()
plt.savefig('outputs/figuras/figura3_comparacao_coeficientes.pdf', bbox_inches='tight', dpi=300)
plt.savefig('outputs/figuras/figura3_comparacao_coeficientes.png', bbox_inches='tight', dpi=300)
plt.show()