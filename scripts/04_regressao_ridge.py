# -*- coding: utf-8 -*-
""" SCRIPT 4: REGRESSÃO RIDGE (STRICT VALIDATION) """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Carregar dados
df = pd.read_csv('dados/base_final.csv')
x_vars = ['MEDIA_INSE', 'IN_AGUA_POTAVEL', 'IN_ENERGIA_REDE_PUBLICA', 'IN_ESGOTO_REDE_PUBLICA',
          'IN_BANDA_LARGA', 'IN_QUADRA_ESPORTES', 'IN_REFEITORIO', 'IN_SALA_LEITURA',
          'IN_LABORATORIO_INFORMATICA', 'IN_LABORATORIO_CIENCIAS', 'IN_SALA_MULTIUSO',
          'IN_EQUIP_LOUSA_DIGITAL', 'IN_EQUIP_MULTIMIDIA', 'QT_DESKTOP_ALUNO',
          'QT_COMP_PORTATIL_ALUNO', 'QT_TABLET_ALUNO', 'QT_MAT_FUND_AF', 'QT_DOC_FUND_AF', 'URBANA']

X = df[x_vars]
y = df['IDEB_2023']

# 1. DIVISÃO TREINO E TESTE (Evita Data Leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. PADRONIZAÇÃO (Fit apenas no Treino)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Aplica a regra do treino no teste

# 3. RIDGE COM VALIDAÇÃO CRUZADA
alphas = np.logspace(-3, 3, 50)
ridge_cv = RidgeCV(alphas=alphas, scoring='r2', cv=5).fit(X_train_scaled, y_train)

# 4. MODELO FINAL E MÉTRICAS
ridge_model = Ridge(alpha=ridge_cv.alpha_).fit(X_train_scaled, y_train)
y_pred = ridge_model.predict(X_test_scaled)

print(f"Melhor Alpha: {ridge_cv.alpha_:.4f}")
print(f"R² Teste: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")