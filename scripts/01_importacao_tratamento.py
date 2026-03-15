# -*- coding: utf-8 -*-
"""
SCRIPT 1: IMPORTAÇÃO E TRATAMENTO DOS DADOS
============================================
- Importa bases do IDEB, INSE e Censo Escolar
- Filtra apenas escolas de SP
- Realiza merge das bases
- Trata valores inconsistentes
"""

import pandas as pd
import numpy as np
import os
from google.colab import drive

# Montar Google Drive (se estiver no Colab)
if 'COLAB_GPU' in os.environ:
    drive.mount('/content/drive')

# =============================================================================
# CONFIGURAÇÃO DOS CAMINHOS
# =============================================================================
# ATENÇÃO: Ajuste os caminhos conforme sua estrutura de pastas
BASE_PATH = '/content/drive/MyDrive/TCC - MBA USP/'

CAMINHO_IDEB = BASE_PATH + 'divulgacao_anos_finais_escolas_2023/divulgacao_anos_finais_escolas_2023.xlsx'
CAMINHO_INSE = BASE_PATH + 'INSE_2023/INSE_2023_escolas.xlsx'
CAMINHO_CENSO = BASE_PATH + 'microdados_ed_basica_2023.csv'

# =============================================================================
# 1. IMPORTAR E TRATAR IDEB
# =============================================================================
print("\n" + "="*60)
print("IMPORTANDO BASE DO IDEB")
print("="*60)

# IDEB tem cabeçalho na linha 9
df_raw = pd.read_excel(CAMINHO_IDEB, header=None)
df_ideb = df_raw.iloc[10:].copy()
df_ideb.columns = df_raw.iloc[9]
df_ideb.reset_index(drop=True, inplace=True)

# Filtrar SP
df_ideb_sp = df_ideb[df_ideb["SG_UF"] == "SP"].copy()

# Selecionar colunas
df_ideb_sp = df_ideb_sp[["ID_ESCOLA", "NO_ESCOLA", "CO_MUNICIPIO", "NO_MUNICIPIO",
                          "REDE", "VL_OBSERVADO_2023"]].copy()
df_ideb_sp = df_ideb_sp.rename(columns={"VL_OBSERVADO_2023": "IDEB_2023"})

# Converter tipos
df_ideb_sp["ID_ESCOLA"] = pd.to_numeric(df_ideb_sp["ID_ESCOLA"], errors="coerce").astype("Int64")
df_ideb_sp["CO_MUNICIPIO"] = pd.to_numeric(df_ideb_sp["CO_MUNICIPIO"], errors="coerce").astype("Int64")
df_ideb_sp["IDEB_2023"] = pd.to_numeric(df_ideb_sp["IDEB_2023"], errors="coerce")

# Remover missings
df_ideb_sp_clean = df_ideb_sp.dropna().copy()
print(f"IDEB SP: {len(df_ideb_sp_clean)} escolas")

# =============================================================================
# 2. IMPORTAR E TRATAR INSE
# =============================================================================
print("\n" + "="*60)
print("IMPORTANDO BASE DO INSE")
print("="*60)

df_inse = pd.read_excel(CAMINHO_INSE)
df_inse_sp = df_inse[df_inse["SG_UF"] == "SP"].copy()
df_inse_sp = df_inse_sp[["ID_ESCOLA", "MEDIA_INSE"]].copy()

print(f"INSE SP: {len(df_inse_sp)} escolas")

# =============================================================================
# 3. IMPORTAR E TRATAR CENSO ESCOLAR
# =============================================================================
print("\n" + "="*60)
print("IMPORTANDO CENSO ESCOLAR")
print("="*60)

df_censo = pd.read_csv(CAMINHO_CENSO, sep=";", encoding="latin1", low_memory=False)
df_censo_sp = df_censo[df_censo["SG_UF"] == "SP"].copy()

# Selecionar colunas
colunas_censo = [
    "CO_ENTIDADE", "IN_AGUA_POTAVEL", "IN_ENERGIA_REDE_PUBLICA", "IN_ESGOTO_REDE_PUBLICA",
    "IN_BANDA_LARGA", "IN_QUADRA_ESPORTES", "IN_REFEITORIO", "IN_SALA_LEITURA",
    "IN_LABORATORIO_INFORMATICA", "IN_LABORATORIO_CIENCIAS", "IN_SALA_MULTIUSO",
    "IN_EQUIP_LOUSA_DIGITAL", "IN_EQUIP_MULTIMIDIA", "QT_DESKTOP_ALUNO",
    "QT_COMP_PORTATIL_ALUNO", "QT_TABLET_ALUNO", "QT_MAT_FUND_AF", "QT_DOC_FUND_AF",
    "TP_LOCALIZACAO"
]
df_censo_sp = df_censo_sp[colunas_censo].copy()
df_censo_sp = df_censo_sp.rename(columns={"CO_ENTIDADE": "ID_ESCOLA"})

print(f"Censo SP: {len(df_censo_sp)} escolas")

# =============================================================================
# 4. MERGE DAS BASES
# =============================================================================
print("\n" + "="*60)
print("REALIZANDO MERGE DAS BASES")
print("="*60)

# Converter ID_ESCOLA para string em todas as bases
for df in [df_ideb_sp_clean, df_inse_sp, df_censo_sp]:
    df['ID_ESCOLA'] = df['ID_ESCOLA'].astype(str)

# Merge
df_temp = pd.merge(df_ideb_sp_clean, df_inse_sp, on='ID_ESCOLA', how='inner')
df_final = pd.merge(df_temp, df_censo_sp, on='ID_ESCOLA', how='inner')

print(f"Total de escolas após merge: {len(df_final)}")

# =============================================================================
# 5. TRATAMENTO DE VALORES INCONSISTENTES
# =============================================================================
print("\n" + "="*60)
print("TRATAMENTO DE VALORES INCONSISTENTES")
print("="*60)

# Substituir 88888 por NaN
df_final['QT_COMP_PORTATIL_ALUNO'] = df_final['QT_COMP_PORTATIL_ALUNO'].replace(88888, np.nan)

# Remover linhas com NaN
df_final_clean = df_final.dropna().copy()

print(f"Escolas após limpeza: {len(df_final_clean)}")
print(f"Escolas removidas: {len(df_final) - len(df_final_clean)}")

# =============================================================================
# 6. RECODIFICAR LOCALIZAÇÃO (DUMMY)
# =============================================================================
# TP_LOCALIZACAO: 1 = urbana, 2 = rural
df_final_clean['URBANA'] = (df_final_clean['TP_LOCALIZACAO'] == 1).astype(int)

# =============================================================================
# 7. SALVAR BASE FINAL
# =============================================================================
df_final_clean.to_csv('dados/base_final.csv', index=False, encoding='utf-8')
print("\n✅ Base salva em: dados/base_final.csv")