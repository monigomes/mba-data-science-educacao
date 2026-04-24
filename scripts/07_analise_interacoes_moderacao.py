# -*- coding: utf-8 -*-
"""
SCRIPT 7: ANÁLISE DE INTERAÇÃO E MODERAÇÃO (GAIOLA DIGITAL)
==========================================================
- Testa se a Infraestrutura Pedagógica (Sala de Leitura) modera 
  o efeito dos Tablets no desempenho escolar.
- Modelo: OLS com Erros-Padrão Robustos (HC3).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Carregar dados tratados
df = pd.read_csv('dados/base_final.csv')

# =============================================================================
# 1. PREPARAÇÃO DA INTERAÇÃO
# =============================================================================
# Criando o termo de interação: Quantidade de Tablets * Presença de Sala de Leitura
df['INTERACAO_TABLET_SALA'] = df['QT_TABLET_ALUNO'] * df['IN_SALA_LEITURA']

# Lista de variáveis independentes (incluindo a interação)
x_vars_interacao = [
    'MEDIA_INSE', 'IN_AGUA_POTAVEL', 'IN_ENERGIA_REDE_PUBLICA', 'IN_ESGOTO_REDE_PUBLICA',
    'IN_BANDA_LARGA', 'IN_QUADRA_ESPORTES', 'IN_REFEITORIO', 'IN_SALA_LEITURA',
    'IN_LABORATORIO_INFORMATICA', 'IN_LABORATORIO_CIENCIAS', 'IN_SALA_MULTIUSO',
    'IN_EQUIP_LOUSA_DIGITAL', 'IN_EQUIP_MULTIMIDIA', 'QT_DESKTOP_ALUNO',
    'QT_COMP_PORTATIL_ALUNO', 'QT_TABLET_ALUNO', 'QT_MAT_FUND_AF', 'QT_DOC_FUND_AF', 
    'URBANA', 'INTERACAO_TABLET_SALA'
]

X = df[x_vars_interacao]
X = sm.add_constant(X)
y = df['IDEB_2023']

# =============================================================================
# 2. ESTIMAÇÃO DO MODELO ROBUSTO (HC3)
# =============================================================================
# O uso de HC3 é essencial devido à heterocedasticidade confirmada no Script 03
model_interact = sm.OLS(y, X).fit(cov_type='HC3')

print("\n" + "="*60)
print("RESULTADOS DO MODELO COM INTERAÇÃO (MODERAÇÃO)")
print("="*60)
print(model_interact.summary())

# =============================================================================
# 3. ANÁLISE DE SIGNIFICÂNCIA E AJUSTE
# =============================================================================
p_valor_interacao = model_interact.pvalues['INTERACAO_TABLET_SALA']
r2_interacao = model_interact.rsquared

print(f"\nR-quadrado com interação: {r2_interacao:.4f}")
print(f"P-valor do termo de interação: {p_valor_interacao:.4f}")

if p_valor_interacao > 0.05:
    print("\nCONCLUSÃO: O termo de interação NÃO é estatisticamente significativo (p > 0.05).")
    print("Isso indica que a presença de Sala de Leitura não altera o efeito dos tablets.")
else:
    print("\nCONCLUSÃO: O termo de interação é significativo.")

# =============================================================================
# 4. VISUALIZAÇÃO DO EFEITO (OPCIONAL)
# =============================================================================
plt.figure(figsize=(10, 6))
sns.regplot(x='QT_TABLET_ALUNO', y='IDEB_2023', data=df[df['IN_SALA_LEITURA'] == 1], 
            scatter_kws={'alpha':0.2}, label='Com Sala de Leitura', color='green')
sns.regplot(x='QT_TABLET_ALUNO', y='IDEB_2023', data=df[df['IN_SALA_LEITURA'] == 0], 
            scatter_kws={'alpha':0.2}, label='Sem Sala de Leitura', color='red')

plt.title('Efeito dos Tablets no IDEB moderado pela Sala de Leitura')
plt.xlabel('Quantidade de Tablets por Aluno')
plt.ylabel('IDEB 2023')
plt.legend()
plt.savefig('outputs/figuras/analise_interacao_moderacao.png', dpi=300)
plt.show()