# -*- coding: utf-8 -*-
""" SCRIPT 4: REGRESSÃO RIDGE (STRICT VALIDATION) """
import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



# Carregar dados
df = pd.read_csv('dados/base_final.csv')

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
# 3. RIDGE (SEM DATA LEAKAGE)
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
# 4. MÉTRICAS RIDGE (TESTE)
# =============================================================================
y_pred_ridge = ridge_model.predict(X_test_scaled)

r2_ridge = r2_score(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

print("\nRIDGE (TESTE)")
print(f"R²: {r2_ridge:.4f}")
print(f"MAE: {mae_ridge:.4f}")
print(f"RMSE: {rmse_ridge:.4f}")

# =============================================================================
# 5. COMPARAÇÃO FINAL
# =============================================================================
print("\n" + "="*60)
print("COMPARAÇÃO FINAL (TESTE)")
print("="*60)

print(f"R² OLS: {r2_ols:.4f}")
print(f"R² Ridge: {r2_ridge:.4f}")
print(f"Δ R²: {r2_ols - r2_ridge:.4f}")

print(f"\nMAE OLS: {mae_ols:.4f}")
print(f"MAE Ridge: {mae_ridge:.4f}")

print(f"\nRMSE OLS: {rmse_ols:.4f}")
print(f"RMSE Ridge: {rmse_ridge:.4f}")
