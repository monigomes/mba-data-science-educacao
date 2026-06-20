# -*- coding: utf-8 -*-
"""Determinantes do desempenho escolar: 
integração de regressão e agrupamentos em escolas 
públicas paulistas - MBA Data Science e Analytics - TCC



## SETUP E REPRODUTIBILIDADE
"""


import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA

SEED = 42
rng = np.random.default_rng(SEED)

"""##CAMINHOS DAS TRÊS BASES"""

from pathlib import Path

# Caminhos relativos ao repositório.
# Coloque os arquivos brutos nas pastas indicadas em dados/raw/.
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "dados" / "raw"
FIGURES_DIR = ROOT / "figuras"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Mantém compatibilidade com os paths "figuras/..." definidos no módulo de gráficos.
os.chdir(ROOT)

PATH_CENSO = DATA_RAW / "censo" / "microdados_ed_basica_2023.csv"
PATH_IDEB  = DATA_RAW / "ideb" / "divulgacao_anos_finais_escolas_2023.xlsx"
PATH_INSE  = DATA_RAW / "inse" / "INSE_2023_escolas.xlsx"

df_raw = pd.read_excel(PATH_IDEB, header=None)
ideb_df = df_raw.iloc[10:].copy()
ideb_df.columns = df_raw.iloc[9]
ideb_df.reset_index(drop=True, inplace=True)
COL_IDEB_CHAVE = "ID_ESCOLA"
COL_IDEB_VALOR = "VL_OBSERVADO_2023"
COL_INSE_CHAVE = "ID_ESCOLA"
COL_INSE_VALOR = "MEDIA_INSE"

"""##CARGA E INTEGRAÇÃO DAS BASES"""

INFRA_BIN = [
    "IN_AGUA_POTAVEL", "IN_ENERGIA_REDE_PUBLICA", "IN_ESGOTO_REDE_PUBLICA",
    "IN_BANDA_LARGA", "IN_QUADRA_ESPORTES", "IN_REFEITORIO", "IN_SALA_LEITURA",
    "IN_LABORATORIO_INFORMATICA", "IN_LABORATORIO_CIENCIAS", "IN_SALA_MULTIUSO",
    "IN_EQUIP_LOUSA_DIGITAL", "IN_EQUIP_MULTIMIDIA",
]
QT_TEC = ["QT_DESKTOP_ALUNO", "QT_COMP_PORTATIL_ALUNO", "QT_TABLET_ALUNO"]
PORTE  = ["QT_MAT_FUND_AF", "QT_DOC_FUND_AF"]
EXTRA  = ["IN_BIBLIOTECA", "IN_BIBLIOTECA_SALA_LEITURA",      # robustez biblioteca
          "IN_INTERNET_APRENDIZAGEM", "IN_INTERNET_ALUNOS",   # nuance tecnológica
          "QT_TUR_FUND_AF"]                                   # tamanho de turma
IDENT  = ["CO_ENTIDADE", "SG_UF", "TP_DEPENDENCIA", "TP_SITUACAO_FUNCIONAMENTO",
          "TP_LOCALIZACAO", "CO_ORGAO_REGIONAL", "CO_MUNICIPIO"]

usecols = IDENT + INFRA_BIN + QT_TEC + PORTE + EXTRA

censo = pd.read_csv(PATH_CENSO, sep=";", encoding="latin-1",
                    usecols=usecols, low_memory=False)

ideb = ideb_df
inse = pd.read_excel(PATH_INSE)

ideb = ideb[[COL_IDEB_CHAVE, COL_IDEB_VALOR]].rename(
    columns={COL_IDEB_CHAVE: "CO_ENTIDADE", COL_IDEB_VALOR: "IDEB_2023"})
inse = inse[[COL_INSE_CHAVE, COL_INSE_VALOR]].rename(
    columns={COL_INSE_CHAVE: "CO_ENTIDADE", COL_INSE_VALOR: "MEDIA_INSE"})

# Garante numérico (algumas bases trazem vírgula decimal)
for d, c in [(ideb, "IDEB_2023"), (inse, "MEDIA_INSE")]:
    d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", ".", regex=False),
                         errors="coerce")

"""## CONSTRUÇÃO DA AMOSTRA + VIÉS DE SELEÇÃO"""

# Etapa 1->2: SP, rede estadual, em atividade, com anos finais
base = censo[(censo.SG_UF == "SP") & (censo.TP_DEPENDENCIA == 2) &
             (censo.TP_SITUACAO_FUNCIONAMENTO == 1) & (censo.QT_MAT_FUND_AF > 0)].copy()
print("Etapa 2 — rede estadual c/ anos finais:", len(base))

# Dummy de localização: 1 = urbana, 0 = rural  (Censo: 1=urbana, 2=rural)
base["TP_LOCALIZACAO"] = (base["TP_LOCALIZACAO"] == 1).astype(int)

for d in (ideb, inse):
    d["CO_ENTIDADE"] = pd.to_numeric(d["CO_ENTIDADE"], errors="coerce").astype("Int64")
base["CO_ENTIDADE"] = base["CO_ENTIDADE"].astype("Int64")

# Junta IDEB e INSE
base = base.merge(ideb, on="CO_ENTIDADE", how="left").merge(inse, on="CO_ENTIDADE", how="left")

# Marca quem seria excluído ANTES de excluir (para o teste de viés)
infra_tec_cols = QT_TEC  # "sem dados de infraestrutura tecnológica"
base["falta_ideb"]  = base["IDEB_2023"].isna()
base["falta_infra"] = base[infra_tec_cols].isna().any(axis=1)
base["excluida"]    = base["falta_ideb"] | base["falta_infra"]

# --- Teste de viés de seleção: excluídas vs. retidas em observáveis -------------
def compara_grupos(df, grupo, variaveis):
    linhas = []
    for v in variaveis:
        a = df.loc[~df[grupo], v].dropna()   # retidas
        b = df.loc[df[grupo], v].dropna()    # excluídas
        if len(a) > 1 and len(b) > 1:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            linhas.append({"variavel": v, "media_retidas": a.mean(),
                           "media_excluidas": b.mean(), "dif": a.mean()-b.mean(),
                           "p_valor": p})
    return pd.DataFrame(linhas)

obs_vars = ["MEDIA_INSE", "QT_MAT_FUND_AF", "QT_DOC_FUND_AF",
            "TP_LOCALIZACAO", "QT_TABLET_ALUNO", "IN_SALA_LEITURA"]
viés = compara_grupos(base, "excluida", obs_vars)
print("\n[Viés de seleção] excluídas vs. retidas:\n", viés.round(3))
# Interpretação: se as diferenças forem pequenas/não significativas, o viés é baixo.

# Amostra final (listwise deletion)  [B2] - CÓDIGO CORRIGIDO:
todas_as_vars = ["IDEB_2023", "MEDIA_INSE"] + INFRA_BIN + QT_TEC + PORTE + ["TP_LOCALIZACAO"]
df = base.loc[~base["excluida"]].dropna(subset=todas_as_vars).copy()
print("\nAmostra final validada:", len(df), f"({len(df)/len(base)*100:.1f}% da rede)")

# Descobrindo o motivo exato das 409 exclusões para a Tabela 1
excluidas_ideb = base["falta_ideb"].sum()

# Escolas que TINHAM IDEB, mas caíram por falta de infraestrutura ou INSE
excluidas_outros = len(base) - len(df) - excluidas_ideb

print(f"Excluídas por falta de IDEB (Valor A): {excluidas_ideb}")
print(f"Excluídas por falta de Infra/INSE (Valor B): {excluidas_outros}")
print(f"Total de exclusões: {excluidas_ideb + excluidas_outros}") # Tem que dar 409

"""## DEFINIÇÃO DAS VARIÁVEIS DO MODELO"""

PREDITORES = (["MEDIA_INSE"] + INFRA_BIN + QT_TEC +
              ["QT_MAT_FUND_AF", "QT_DOC_FUND_AF", "TP_LOCALIZACAO"])
ALVO = "IDEB_2023"

X = df[PREDITORES].astype(float)
y = df[ALVO].astype(float)

"""## ESTATÍSTICAS DESCRITIVAS"""

desc = df[[ALVO, "MEDIA_INSE"] + PORTE + QT_TEC].describe().T
desc["variancia"] = desc["std"] ** 2
desc["CV_%"]      = (desc["std"] / desc["mean"] * 100).round(2)
desc["assimetria"] = df[[ALVO, "MEDIA_INSE"] + PORTE + QT_TEC].skew()
desc["curtose"]    = df[[ALVO, "MEDIA_INSE"] + PORTE + QT_TEC].kurt()  # excesso de curtose (Fisher)
print(desc[["mean","std","variancia","min","max","CV_%","assimetria","curtose"]].round(2))

# 1. Definir quais são as variáveis binárias (INFRA_BIN + binárias do EXTRA)
vars_binarias = INFRA_BIN + [
    "IN_BIBLIOTECA", "IN_BIBLIOTECA_SALA_LEITURA",
    "IN_INTERNET_APRENDIZAGEM", "IN_INTERNET_ALUNOS", "TP_LOCALIZACAO"
]

# 2. Garantir que vamos calcular apenas para as colunas que estão no df final
vars_binarias = [var for var in vars_binarias if var in df.columns]

# 3. Criar um DataFrame para armazenar os resultados
estatisticas_bin = pd.DataFrame(index=vars_binarias)

# 4. Calcular Média (Proporção), Desvio Padrão e CV
media = df[vars_binarias].mean()
desvio = df[vars_binarias].std()

estatisticas_bin["Distribuição (%)"] = media * 100
estatisticas_bin["Desvio Padrão"] = desvio
estatisticas_bin["CV"] = desvio / media

# 5. Formatar para exibição acadêmica (arredondando as casas decimais)
tabela_bin_formatada = estatisticas_bin.copy()
tabela_bin_formatada["Distribuição (%)"] = tabela_bin_formatada["Distribuição (%)"].map("{:.1f}%".format)
tabela_bin_formatada["Desvio Padrão"] = tabela_bin_formatada["Desvio Padrão"].map("{:.2f}".format)
tabela_bin_formatada["CV"] = tabela_bin_formatada["CV"].map("{:.2f}".format)

print("Estatísticas Descritivas - Variáveis Binárias:")
print("-" * 55)
print(tabela_bin_formatada)

"""##MATRIZ DE CORRELAÇÃO"""

corr = df[[ALVO, "MEDIA_INSE", "QT_MAT_FUND_AF", "QT_DOC_FUND_AF",
           "QT_TABLET_ALUNO", "IN_SALA_LEITURA", "IN_LABORATORIO_INFORMATICA"]].corr()
print(corr.round(3))
print("\nColinearidade-chave r(matrículas,docentes):",
      round(df.QT_MAT_FUND_AF.corr(df.QT_DOC_FUND_AF), 3))

"""##REGRESSÃO OLS + ERROS-PADRÃO ROBUSTOS HC3"""

def tidy(model, nome):
    return pd.DataFrame({
        "coef": model.params, "se": model.bse,
        "z_ou_t": model.tvalues, "p": model.pvalues
    }).round(4).rename_axis("variavel").reset_index().assign(modelo=nome)

formula = f"{ALVO} ~ " + " + ".join(PREDITORES)
ols_hc3 = smf.ols(formula, data=df).fit(cov_type="HC3")
print(ols_hc3.summary())
print("\nR² ajustado:", round(ols_hc3.rsquared_adj, 4))

"""##DIAGNÓSTICOS DOS PRESSUPOSTOS"""

# Breusch-Pagan (heterocedasticidade)
exog = ols_hc3.model.exog  # Extrai a matriz limpa diretamente do modelo
bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(ols_hc3.resid, exog)
print(f"Breusch-Pagan: LM={bp_lm:.2f}  p={bp_p:.4g}  -> {'heteroced.' if bp_p<0.05 else 'homoced.'}")

# Durbin-Watson (independência dos resíduos)
print("Durbin-Watson:", round(durbin_watson(ols_hc3.resid), 3))

# Shapiro-Wilk (normalidade) — n>5000 não suportado; subamostra se preciso
res = ols_hc3.resid
res_s = res if len(res) <= 5000 else res.sample(5000, random_state=SEED)
sw_w, sw_p = stats.shapiro(res_s)
print(f"Shapiro-Wilk: W={sw_w:.4f}  p={sw_p:.4g}  (TCL garante normalidade assintótica, n={len(df)})")

# VIF (multicolinearidade)
exog_vif = ols_hc3.model.exog
nomes_vif = ols_hc3.model.exog_names

vif = pd.DataFrame({
    "variavel": nomes_vif,
    "VIF": [variance_inflation_factor(exog_vif, i) for i in range(exog_vif.shape[1])]
}).query("variavel != 'Intercept'").sort_values("VIF", ascending=False)

print("\nVIF:\n", vif.round(2).to_string(index=False))

"""## ERRO-PADRÃO AGRUPADO POR DIRETORIA DE ENSINO"""

ols_clu = smf.ols(formula, data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["CO_ORGAO_REGIONAL"]})
n_clusters = df["CO_ORGAO_REGIONAL"].nunique()
print(f"Erros-padrão agrupados por Diretoria de Ensino ({n_clusters} agrupamentos)")

# Compara significância HC3 vs. agrupado (o ponto: algum coef. perde significância?)
comp = (tidy(ols_hc3, "HC3")[["variavel", "coef", "p"]]
        .merge(tidy(ols_clu, "Cluster")[["variavel", "se", "p"]],
               on="variavel", suffixes=("_HC3", "_cluster")))
comp["muda_signif_5%"] = (comp["p_HC3"] < .05) != (comp["p_cluster"] < .05)
print(comp.round(4).to_string(index=False))

"""##RIDGE EM PIPELINE + VALIDAÇÃO CRUZADA"""

# CRÍTICO: o StandardScaler e a seleção de lambda ficam DENTRO de cada fold.
alphas = np.logspace(-3, 3, 100)
pipe = Pipeline([("scaler", StandardScaler()),
                 ("ridge", RidgeCV(alphas=alphas))])

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
r2_cv  = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
mae_cv = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error")
rmse_cv = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error")
print(f"Validação cruzada (sem vazamento):")
print(f"  R²   = {r2_cv.mean():.4f} (dp {r2_cv.std():.4f})  | por fold: {np.round(r2_cv,3)}")
print(f"  MAE  = {mae_cv.mean():.4f}  (~{mae_cv.mean()/y.mean()*100:.2f}% da média)")
print(f"  RMSE = {rmse_cv.mean():.4f}")

# Lambda final (ajustado em toda a base, só para reportar)
pipe.fit(X, y)
print("  lambda (alpha) selecionado:", round(pipe.named_steps["ridge"].alpha_, 2))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score


def fig_ridge_summary(X, y, preditores, alpha_sel, ols_hc3=None,
                      alphas=np.logspace(-3, 3, 100),
                      path="figuras/14_ridge_summary.png"):
    # espaço padronizado (betas comparáveis — igual ao bloco coef_std)
    Xz = StandardScaler().fit_transform(X)
    yz = StandardScaler().fit_transform(np.asarray(y).reshape(-1, 1)).ravel()

    ols   = LinearRegression().fit(Xz, yz)
    ridge = Ridge(alpha=alpha_sel).fit(Xz, yz)
    caminho = np.array([Ridge(alpha=a).fit(Xz, yz).coef_ for a in alphas])

    # "summary" comparativo (Ridge não tem inferência clássica)
    tab = pd.DataFrame({"variavel": preditores,
                        "beta_OLS": ols.coef_, "beta_Ridge": ridge.coef_})
    tab["encolhimento_%"] = (1 - tab.beta_Ridge.abs()
                             / tab.beta_OLS.abs().replace(0, np.nan)) * 100
    if ols_hc3 is not None:                       # p só do OLS/HC3 (inferência)
        tab["p_HC3"] = tab.variavel.map(ols_hc3.pvalues.to_dict())
    print(tab.round(4).to_string(index=False))

    r2_ols   = r2_score(yz, ols.predict(Xz))
    r2_ridge = r2_score(yz, ridge.predict(Xz))
    print(f"\nlambda selecionado = {alpha_sel:.4g}")
    print(f"R² in-sample  OLS = {r2_ols:.6f} | Ridge = {r2_ridge:.6f} "
          f"| diferença = {r2_ols - r2_ridge:.2e}")
    print(f"encolhimento médio dos |beta| = {tab['encolhimento_%'].mean():.2f}%")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.8))

    # (A) ridge trace: coeficientes x lambda; lambda escolhido marcado
    cores = plt.cm.viridis(np.linspace(0, 0.9, len(preditores)))
    for j in range(len(preditores)):
        axA.plot(alphas, caminho[:, j], color=cores[j], lw=1.3)
    axA.axvline(alpha_sel, color=GUIA, ls="--", lw=1.2)   # lambda escolhido
    axA.set_xscale("log")
    axA.set_xlabel("λ (alpha) — escala log")
    axA.set_ylabel("Coeficiente padronizado")
    _letra_painel(axA, "A")

    # (B) beta OLS x beta Ridge: caem sobre y=x -> quase idênticos
    axB.scatter(ols.coef_, ridge.coef_, s=30, color=TEAL, edgecolors="none")
    lim = [min(ols.coef_.min(), ridge.coef_.min()),
           max(ols.coef_.max(), ridge.coef_.max())]
    axB.plot(lim, lim, color=GUIA, ls="--", lw=1.3, label="β OLS = β Ridge")
    axB.set_xlabel("β padronizado — OLS")
    axB.set_ylabel("β padronizado — Ridge")
    axB.legend(loc="lower right")
    _letra_painel(axB, "B")

    fig.tight_layout()
    _salvar(fig, path)
    plt.close(fig)
    return path

"""##COEFICIENTES PADRONIZADOS: OLS vs RIDGE"""

# Padroniza X e y na base completa (uso DESCRITIVO, não de generalização)
Xz = StandardScaler().fit_transform(X)
yz = StandardScaler().fit_transform(y.values.reshape(-1, 1)).ravel()

ols_std = LinearRegression().fit(Xz, yz)
ridge_std = Ridge(alpha=pipe.named_steps["ridge"].alpha_).fit(Xz, yz)

coef_std = pd.DataFrame({
    "variavel": PREDITORES,
    "beta_OLS_padr": ols_std.coef_,
    "beta_Ridge_padr": ridge_std.coef_,
}).assign(abs_ols=lambda d: d.beta_OLS_padr.abs()).sort_values("abs_ols", ascending=False)
print(coef_std.drop(columns="abs_ols").round(3).to_string(index=False))
# Leia magnitude (efeito) pelos padronizados; leia inferência (p) pelo HC3/cluster.

"""##K-MEANS: K ÓTIMO (COTOVELO + SILHOUETTE)"""

VARS_CLUSTER = ["QT_MAT_FUND_AF", "TP_LOCALIZACAO", "QT_TABLET_ALUNO", "IN_SALA_LEITURA"]
# IDEB e INSE ficam FORA do treino (evita vazamento; permite validação a posteriori)
Xclu = StandardScaler().fit_transform(df[VARS_CLUSTER].astype(float))

inercias, silhs = [], []
ks = range(2, 11)
for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Xclu)
    inercias.append(km.inertia_)
    silhs.append(silhouette_score(Xclu, km.labels_))
for k, i, s in zip(ks, inercias, silhs):
    print(f"k={k}  inércia={i:8.1f}  silhouette={s:.4f}")

K = 3
km = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit(Xclu)
df["cluster"] = km.labels_

"""##ESTABILIDADE DOS CLUSTERS

ARI (Adjusted Rand Index) entre seeds + boostrap do grupo rural)
"""

# (a) ARI entre 10 inicializações distintas: ~1.0 = partição estável
labels_seeds = [KMeans(K, n_init=20, random_state=s).fit_predict(Xclu) for s in range(10)]
aris = [adjusted_rand_score(labels_seeds[0], l) for l in labels_seeds[1:]]
print(f"ARI entre seeds: média={np.mean(aris):.3f}  mín={np.min(aris):.3f}")

# (b) Estabilidade do cluster rural via bootstrap (índice de Jaccard, ~Hennig 2007)
#     O cluster "rural" é o de menor % urbana.
rural_id = df.groupby("cluster")["TP_LOCALIZACAO"].mean().idxmin()
membros_rural = set(df.index[df.cluster == rural_id])
jaccards = []
for _ in range(100):
    idx = rng.choice(df.index, size=len(df), replace=True)
    Xb = StandardScaler().fit_transform(df.loc[idx, VARS_CLUSTER].astype(float))
    lb = KMeans(K, n_init=10, random_state=SEED).fit_predict(Xb)
    tmp = pd.Series(lb, index=idx)
    # identifica o cluster rural no bootstrap
    rid = tmp.groupby(tmp).apply(lambda g: df.loc[g.index, "TP_LOCALIZACAO"].mean()).idxmin()
    boot_rural = set(idx[lb == rid])
    inter = len(membros_rural & boot_rural); uni = len(membros_rural | boot_rural)
    if uni: jaccards.append(inter/uni)
print(f"Jaccard médio do cluster rural (bootstrap): {np.mean(jaccards):.3f} "
      f"(>0,75 = estável; >0,60 = padrão real)")

"""##ROBUSTEZ PARA DADOS MISTOS: K-PROTOTYPES"""


# Duas das 4 variáveis são binárias -> K-means (euclidiano) não é o ideal.
# k-prototypes (Huang, 1998) trata contínuas + categóricas corretamente.
try:
    from kmodes.kprototypes import KPrototypes
    Xmix = df[VARS_CLUSTER].astype(float).copy()
    # padroniza só as contínuas; mantém binárias como categóricas
    cont = ["QT_MAT_FUND_AF", "QT_TABLET_ALUNO"]
    Xmix[cont] = StandardScaler().fit_transform(Xmix[cont])
    cat_idx = [VARS_CLUSTER.index("TP_LOCALIZACAO"), VARS_CLUSTER.index("IN_SALA_LEITURA")]
    kp = KPrototypes(n_clusters=K, random_state=SEED, n_init=10)
    lab_kp = kp.fit_predict(Xmix.values, categorical=cat_idx)
    print("ARI K-means x k-prototypes:", round(adjusted_rand_score(df.cluster, lab_kp), 3),
          "(próximo de 1 = tipologia robusta à escolha do algoritmo)")
except ImportError:
    print("Instale: pip install kmodes")

from kmodes.kprototypes import KPrototypes
from sklearn.metrics import adjusted_rand_score

Xmix = df[VARS_CLUSTER].astype(float).copy()
cont = ["QT_MAT_FUND_AF", "QT_TABLET_ALUNO"]
Xmix[cont] = StandardScaler().fit_transform(Xmix[cont])
cat_idx = [VARS_CLUSTER.index("TP_LOCALIZACAO"), VARS_CLUSTER.index("IN_SALA_LEITURA")]

for g in [None, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
    kp = KPrototypes(n_clusters=K, random_state=SEED, n_init=10, gamma=g)
    lab = kp.fit_predict(Xmix.values, categorical=cat_idx)
    ari = adjusted_rand_score(df["cluster"], lab)
    print(f"gamma={str(g):>5}  ARI vs K-means={ari:+.3f}  gamma_usado={kp.gamma:.3f}")

# ====== GOWER + CLUSTERING HIERÁRQUICO (curiosidade / robustez) ======
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Mesmas variáveis do cluster original
CONT = ["QT_MAT_FUND_AF", "QT_TABLET_ALUNO"]   # contínuas
CAT  = ["TP_LOCALIZACAO", "IN_SALA_LEITURA"]   # binárias

Xc = df[CONT].astype(float).to_numpy()
Xk = df[CAT].astype(float).to_numpy()

def gower_matrix(X_cont, X_cat):
    """Dissimilaridade de Gower (1971): |dx|/range p/ contínuas,
    simple matching p/ categóricas, média das p variáveis."""
    rng_ = X_cont.max(0) - X_cont.min(0)
    rng_[rng_ == 0] = 1.0
    n = X_cont.shape[0]
    p = X_cont.shape[1] + X_cat.shape[1]
    D = np.zeros((n, n), dtype=np.float32)
    for j in range(X_cont.shape[1]):
        col = X_cont[:, j][:, None]
        D += np.abs(col - col.T) / rng_[j]
    for j in range(X_cat.shape[1]):
        col = X_cat[:, j][:, None]
        D += (col != col.T).astype(np.float32)
    return D / p

G = gower_matrix(Xc, Xk)   # ~3421x3421 float32 (~94 MB), OK no Colab

# Seleção de K por silhueta SOBRE a matriz de Gower (metric precomputed)
print("Seleção de K (Gower + average linkage):")
for k in range(2, 9):
    lab = AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="average"
        # se der erro de versão antiga do sklearn, troque metric= por affinity=
    ).fit_predict(G)
    s = silhouette_score(G, lab, metric="precomputed")
    print(f"  k={k}  silhouette={s:.4f}")

# Partição final com o mesmo K que você usou no K-means
K_GOWER = 3
lab_gower = AgglomerativeClustering(
    n_clusters=K_GOWER, metric="precomputed", linkage="average"
).fit_predict(G)
df["cluster_gower"] = lab_gower

# Concordância com o K-means original (e, se quiser, com o k-prototypes)
print("\nARI K-means x Gower/hierárquico:",
      round(adjusted_rand_score(df["cluster"], df["cluster_gower"]), 3),
      "(perto de 1 = mesma estrutura; perto de 0 = discordam)")

# Perfil dos clusters de Gower (mesma lógica do seu perfil original)
perfil_g = df.groupby("cluster_gower").agg(
    N=("CO_ENTIDADE", "size"),
    IDEB=("IDEB_2023", "mean"), INSE=("MEDIA_INSE", "mean"),
    Matriculas=("QT_MAT_FUND_AF", "mean"), Tablets=("QT_TABLET_ALUNO", "mean"),
    SalaLeitura=("IN_SALA_LEITURA", "mean"), Urbana=("TP_LOCALIZACAO", "mean"),
).round(2)
print("\nPerfil (Gower):\n", perfil_g)

"""##PERFIL DOS CLUSTERS + ANOVA (treino x exógenas)"""

perfil = df.groupby("cluster").agg(
    N=("CO_ENTIDADE", "size"),
    IDEB=("IDEB_2023", "mean"), INSE=("MEDIA_INSE", "mean"),
    Matriculas=("QT_MAT_FUND_AF", "mean"), Tablets=("QT_TABLET_ALUNO", "mean"),
    SalaLeitura=("IN_SALA_LEITURA", "mean"), Urbana=("TP_LOCALIZACAO", "mean"),
).round(2)
print(perfil)

# ANOVA — SEPARANDO explicitamente os dois tipos de variável:
TREINO   = ["QT_MAT_FUND_AF", "IN_SALA_LEITURA"]   # usadas no cluster -> F alto é esperado
EXOGENAS = ["IDEB_2023", "MEDIA_INSE"]             # fora do cluster -> validação REAL
print("\n--- Variáveis de TREINO (confirmam coerência do agrupamento; NÃO são achado) ---")
for v in TREINO:
    g = [df.loc[df.cluster == c, v] for c in sorted(df.cluster.unique())]
    f, p = stats.f_oneway(*g)
    print(f"  {v:20s} F={f:12.2f}  p={p:.4g}")
print("--- Variáveis EXÓGENAS (validação a posteriori — AQUI está o resultado) ---")
for v in EXOGENAS:
    g = [df.loc[df.cluster == c, v] for c in sorted(df.cluster.unique())]
    f, p = stats.f_oneway(*g)
    print(f"  {v:20s} F={f:12.2f}  p={p:.4g}")

"""##REGRESSÃO DENTRO DE CADA CLUSTER (efeitos heterogêneos)"""

# Mostra que os efeitos médios da regressão variam por perfil -> a "integração" vira resultado.
foco = ["MEDIA_INSE", "QT_TABLET_ALUNO", "IN_SALA_LEITURA"]
linhas = []
for c in sorted(df.cluster.unique()):
    sub = df[df.cluster == c]
    if len(sub) > len(PREDITORES) + 5:
        m = smf.ols(formula, data=sub).fit(cov_type="HC3")
        linhas.append({"cluster": c, "N": len(sub),
                       **{f"beta_{v}": round(m.params.get(v, np.nan), 4) for v in foco}})
print(pd.DataFrame(linhas).to_string(index=False))

"""##PCA - VALIDAÇÃO VISUAL DA SEGREGAÇÃO DOS PERFIS"""

pca = PCA(n_components=2, random_state=SEED).fit(Xclu)
print("Variância explicada:", np.round(pca.explained_variance_ratio_, 3),
      "| acumulada:", round(pca.explained_variance_ratio_.sum(), 3))
# coords = pca.transform(Xclu)  # use para o scatter colorido por df.cluster

"""##ROBUSTEZ - INDICADOR DE LEITURA COMBINADO"""

sem_sala = df[df.IN_SALA_LEITURA == 0]
print("Cobertura sala de leitura:", round(df.IN_SALA_LEITURA.mean()*100, 1), "%")
print("Cobertura indicador combinado:", round(df.IN_BIBLIOTECA_SALA_LEITURA.mean()*100, 1), "%")
print("Entre as SEM sala de leitura, % com biblioteca:",
      round(sem_sala.IN_BIBLIOTECA.mean()*100, 1), "%")

# Reespecifica o modelo trocando sala de leitura pelo combinado e compara
PRED_ALT = ["IN_BIBLIOTECA_SALA_LEITURA" if p == "IN_SALA_LEITURA" else p for p in PREDITORES]
f_alt = f"{ALVO} ~ " + " + ".join(PRED_ALT)
m_alt = smf.ols(f_alt, data=df).fit(cov_type="HC3")
print("Coef. do espaço de leitura — sala:", round(ols_hc3.params["IN_SALA_LEITURA"], 4),
      "| combinado:", round(m_alt.params["IN_BIBLIOTECA_SALA_LEITURA"], 4))

"""##NUANCE DO PARADOXO - INTERNET COM FINALIDADE PEDAGÓGICA"""

# Inclui uso pedagógico da conectividade e a interação tablet x uso pedagógico.
form_tec = (f"{ALVO} ~ " + " + ".join(PREDITORES) +
            " + IN_INTERNET_APRENDIZAGEM + IN_INTERNET_ALUNOS"
            " + QT_TABLET_ALUNO:IN_INTERNET_APRENDIZAGEM")
m_tec = smf.ols(form_tec, data=df).fit(cov_type="HC3")
for v in ["QT_TABLET_ALUNO", "IN_INTERNET_APRENDIZAGEM", "IN_INTERNET_ALUNOS",
          "QT_TABLET_ALUNO:IN_INTERNET_APRENDIZAGEM"]:
    if v in m_tec.params:
        print(f"  {v:42s} beta={m_tec.params[v]:+.5f}  p={m_tec.pvalues[v]:.4g}")
# ATENÇÃO: cobertura alta (~96%) reduz a variância -> pode não dar significância.
# Se não der, o achado honesto é: "universalização do acesso já não é o gargalo".

"""##ESPECIFICAÇÃO ALTERNATIVA - TAMANHO DE TURMA"""

# Substitui as duas contagens colineares (matrículas+docentes) por
# matrículas + alunos-por-turma (menos colinear e pedagogicamente interpretável).
df["ALUNO_POR_TURMA"] = df.QT_MAT_FUND_AF / df.QT_TUR_FUND_AF.replace(0, np.nan)
PRED_ESCALA = [p for p in PREDITORES if p != "QT_DOC_FUND_AF"] + ["ALUNO_POR_TURMA"]
print("r(matrículas,docentes)=", round(df.QT_MAT_FUND_AF.corr(df.QT_DOC_FUND_AF), 3),
      "| r(matrículas,aluno/turma)=", round(df.QT_MAT_FUND_AF.corr(df.ALUNO_POR_TURMA), 3))
m_esc = smf.ols(f"{ALVO} ~ " + " + ".join(PRED_ESCALA), data=df.dropna(subset=["ALUNO_POR_TURMA"])
                ).fit(cov_type="HC3")
print("R² ajustado (especificação alternativa):", round(m_esc.rsquared_adj, 4))

# %% ============================================================================
# 20. NOTAS DE REPRODUTIBILIDADE
# ===============================================================================
# - SEED fixo em KMeans, KFold, PCA, bootstrap e k-prototypes.
# - Padronização SEMPRE dentro do Pipeline na validação cruzada (sem vazamento).
# - IDEB e INSE mantidos fora do treino do cluster (sem vazamento de alvo).
# - Salve este script + requirements.txt + versões no repositório GitHub.
# - Para a banca: reporte HC3 e cluster lado a lado; padronizados p/ efeito; CV p/ generalização.

"""## Gráficos"""

# -*- coding: utf-8 -*-
"""
================================================================================
TCC — Módulo de visualização (figuras no padrão do trabalho) — VERSÃO ALINHADA
AO SCRIPT FINAL (N = 3.422) E ÀS NORMAS DE FORMATAÇÃO DE GRÁFICOS DO MANUAL
MBA USP/Esalq.
================================================================================
PRINCÍPIO DESTA VERSÃO: as figuras NÃO recalculam o que o pipeline já calculou.
Quando um resultado já existe na sessão (inércias, silhuetas, rótulos de Gower e
k-prototypes), ele é PASSADO como argumento e apenas plotado — garantindo que a
imagem coincida exatamente com a tabela/texto. Só há recálculo como fallback, e
nesse caso com os MESMOS hiperparâmetros do pipeline (ex.: n_init=20).

AJUSTES DE DESIGN (Normas de Formatação — Gráficos):
- Linhas de grade ......... REMOVIDAS (axes.grid = False).
- Borda ................... REMOVIDAS as bordas superior e direita (spines).
- Preenchimento ........... fundo estritamente BRANCO (figura, eixos e arquivo).
- Título do gráfico ....... REMOVIDO da imagem (vai na legenda, no Word).
- Eixos (x e y) ........... linha sólida PRETA, largura 1,5 pt.
- Fonte ("Fonte: ...") .... REMOVIDA da imagem (vai abaixo da figura, no Word).
- Cores (paleta Viridis) .. #440154 (roxo), #21918c (verde-azulado),
                            #5ec962 (verde claro). Exceção: matriz de correlação
                            usa paleta divergente RdBu (sinal/intensidade de r).
- Painéis múltiplos ....... LETRA MAIÚSCULA no canto superior esquerdo.

REQUISITOS (rode o pipeline antes; estes objetos precisam existir):
    df, base, ols_hc3, pipe, X, y, PREDITORES, VARS_CLUSTER, jaccards,
    inercias, silhs            (da célula "K-MEANS: K ÓTIMO")
    lab_kp                     (da célula k-prototypes; opcional)
    df["cluster_gower"]        (da célula Gower; opcional)
PACOTES: numpy, pandas, matplotlib, statsmodels, scikit-learn, scipy
        (opcionais) kmodes [concordância]; geopandas, geobr [mapa]
================================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Ellipse
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import adjusted_rand_score, silhouette_score, r2_score

os.makedirs("figuras", exist_ok=True)
SEED = 42

# %% ---------------------------------------------------------------------------
# PALETA VIRIDIS (norma do manual) E CONVENÇÃO DE CLUSTERS (Tabela 7)
# 0 = Urbano Referência | 1 = Rural Eficiente | 2 = Urbano Crítico
# -----------------------------------------------------------------------------
ROXO = "#440154"                 # viridis - roxo
TEAL = "#21918c"                 # viridis - verde-azulado
VERDE = "#5ec962"                # viridis - verde claro
AMARELO_ESVERDEADO = "#DCE319"   # viridis - amarelo-esverdeado
GUIA = "#4D4D4D"                 # cinza neutro: APENAS linhas-guia (zero, referência)

CLUSTER_COLORS = {0: ROXO, 1: TEAL, 2: AMARELO_ESVERDEADO}
CLUSTER_NOMES = {0: "Urbano Referência", 1: "Rural Eficiente", 2: "Urbano Crítico"}
PALETA = [ROXO, TEAL, VERDE]

# %% ---------------------------------------------------------------------------
# ESTILO GLOBAL — IMPÕE TODAS AS REGRAS DO MANUAL DE UMA SÓ VEZ
# -----------------------------------------------------------------------------
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "font.size": 11,
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.grid": False,
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "legend.frameon": False,
})


# %% ---------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def _fonte(fig):
    """A fonte da figura NÃO entra na imagem (norma do manual): vai no Word."""
    return None


def _letra_painel(ax, letra):
    """Identifica painéis múltiplos com letra MAIÚSCULA no canto superior esq."""
    ax.text(0.02, 0.98, letra, transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color="black")


def _salvar(fig, path):
    """Salva PNG (300 dpi, fundo branco, bbox=tight) e a versão vetorial em SVG."""
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".svg"), format="svg")


def _virg(x, casas=3):
    """Formata número com vírgula decimal (padrão ABNT/PT-BR)."""
    return f"{x:.{casas}f}".replace(".", ",")


def canonizar_clusters(df, col="cluster"):
    """Renomeia os rótulos do K-means para o padrão FIXO do script final:
    0 = Urbano Referência, 1 = Rural Eficiente, 2 = Urbano Crítico."""
    perf = df.groupby(col).agg(urb=("TP_LOCALIZACAO", "mean"),
                               sala=("IN_SALA_LEITURA", "mean"))
    rural = perf["urb"].idxmin()
    critico = perf.drop(rural)["sala"].idxmin()
    ref = [c for c in perf.index if c not in (rural, critico)][0]
    mapa = {ref: 0, rural: 1, critico: 2}
    df[col] = df[col].map(mapa)
    return df


# %% ===========================================================================
# 1. RESÍDUOS x VALORES AJUSTADOS (o "funil" -> justifica HC3)
# =============================================================================
def fig_residuos_ajustados(model, path="figuras/01_residuos_ajustados.png"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(model.fittedvalues, model.resid, s=8, alpha=0.35, color=TEAL,
               edgecolors="none")
    ax.axhline(0, color=GUIA, lw=1.3)
    ax.set_xlabel("Valores ajustados (IDEB predito)")
    ax.set_ylabel("Resíduos")
    _salvar(fig, path)
    plt.close(fig)
    return path


# %% ===========================================================================
# 2. DIAGNÓSTICO DE RESÍDUOS (histograma + curva normal + Q-Q)
# A inferência se sustenta pelo TCL (n grande), não pela normalidade estrita.
# =============================================================================
def fig_residuos(model, path="figuras/02_residuos.png"):
    res = np.asarray(model.resid)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # (A) histograma de densidade + curva normal teórica
    ax1.hist(res, bins=40, density=True, color=TEAL, alpha=0.55, edgecolor="white")
    xs = np.linspace(res.min(), res.max(), 200)
    ax1.plot(xs, stats.norm.pdf(xs, res.mean(), res.std()),
             color=ROXO, lw=2, label="Distribuição normal teórica")
    ax1.set_xlabel("Resíduos")
    ax1.set_ylabel("Densidade")
    ax1.legend(loc="upper right")
    _letra_painel(ax1, "A")

    # (B) Q-Q plot
    stats.probplot(res, dist="norm", plot=ax2)
    ax2.set_title("")
    ax2.get_lines()[0].set(marker="o", ms=3, alpha=0.4,
                           markerfacecolor=TEAL, markeredgecolor="none")
    ax2.get_lines()[1].set(color=ROXO, lw=2)
    ax2.set_xlabel("Quantis teóricos")
    ax2.set_ylabel("Quantis observados")
    _letra_painel(ax2, "B")

    # Shapiro-Wilk (n>5000 -> usa amostra de 5000) — anotação, não título
    amostra = res if len(res) <= 5000 else pd.Series(res).sample(5000, random_state=SEED)
    w, p = stats.shapiro(amostra)
    p_txt = "p < 0,001" if p < 0.001 else f"p = {_virg(p)}"
    ax2.text(0.97, 0.03,
             f"Shapiro-Wilk: W = {_virg(w, 4)}; {p_txt}\n"
             f"(TCL garante normalidade assintótica, n = {len(res)})",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=8,
             color="black",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC"))
    fig.tight_layout()
    _salvar(fig, path)
    plt.close(fig)
    return path


# %% ===========================================================================
# 3. VALORES REAIS vs. PREDITOS (OLS e Ridge)
# =============================================================================
def fig_real_vs_predito(df, ols_model, pipe, X, y, alvo="IDEB_2023",
                        path="figuras/03_real_vs_predito.png"):
    pred_ols = np.asarray(ols_model.fittedvalues)
    pred_ridge = np.asarray(pipe.predict(X))
    yv = np.asarray(y)
    r2_ols = r2_score(yv, pred_ols)
    r2_ridge = r2_score(yv, pred_ridge)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    paineis = [(axes[0], pred_ols, r2_ols, ROXO, "OLS", "A"),
               (axes[1], pred_ridge, r2_ridge, TEAL, "Ridge", "B")]
    for ax, pred, r2, cor, nome, letra in paineis:
        ax.scatter(yv, pred, s=8, alpha=0.30, color=cor, edgecolors="none")
        lim = [min(yv.min(), pred.min()), max(yv.max(), pred.max())]
        ax.plot(lim, lim, color=GUIA, ls="--", lw=1.3,
                label="Predição perfeita (y = x)")
        ax.set_xlabel("IDEB real")
        ax.legend(loc="lower right")
        _letra_painel(ax, letra)
        ax.text(0.03, 0.88, f"{nome}\nR² = {_virg(r2)}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9, color="black")
    axes[0].set_ylabel("IDEB predito")
    fig.tight_layout()
    _salvar(fig, path)
    plt.close(fig)
    print(f"R² in-sample: OLS={r2_ols:.4f} | Ridge={r2_ridge:.4f}")
    return path


# %% ===========================================================================
# 4. ESCOLHA DE k: COTOVELO (INÉRCIA) + SILHUETA
# CORRIGIDO: reaproveita 'inercias'/'silhuetas' do pipeline (ks=range(2,11)).
# Fallback recalcula com n_init=20 (idêntico ao pipeline) — nunca mais diverge.
# =============================================================================
def fig_silhueta_inercia(df, vars_cluster, ks=range(2, 11), k_escolhido=3,
                         seed=SEED, inercias=None, silhuetas=None,
                         path="figuras/04_silhueta_inercia.png"):
    ks = list(ks)

    if inercias is None or silhuetas is None:
        X = StandardScaler().fit_transform(df[vars_cluster].astype(float))
        inercias, silhuetas = [], []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=seed, n_init=20).fit(X)
            inercias.append(km.inertia_)
            silhuetas.append(silhouette_score(X, km.labels_))

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax2 = ax1.twinx()
    l1, = ax1.plot(ks, inercias, "-o", color=ROXO, lw=1.8, ms=6,
                   label="Inércia (método do cotovelo)")
    ax1.set_xlabel("Número de clusters (k)")
    ax1.set_ylabel("Inércia")
    l2, = ax2.plot(ks, silhuetas, "-o", color=TEAL, lw=1.8, ms=6,
                   label="Coeficiente de silhueta médio")
    ax2.set_ylabel("Coeficiente de silhueta médio")
    ax1.axvline(k_escolhido, color=GUIA, ls="--", lw=1.2, zorder=0)
    if k_escolhido in ks:
        i = ks.index(k_escolhido)
        ax2.scatter([k_escolhido], [silhuetas[i]], s=90, color=VERDE,
                    edgecolor="black", zorder=5)

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(True)
    ax1.spines["bottom"].set_visible(True)
    for lado in ("top", "left", "bottom"):
        ax2.spines[lado].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("black")
    ax2.spines["right"].set_linewidth(1.5)
    ax2.tick_params(axis="x", length=0)
    ax1.set_xticks(ks)
    ax1.legend(handles=[l1, l2], loc="upper right")
    fig.tight_layout()
    _salvar(fig, path)
    plt.close(fig)
    return path


# %% ===========================================================================
# 5. CONCORDÂNCIA ENTRE ALGORITMOS (PCA 2D; painéis A/B/C, ARI anotado)
# CORRIGIDO: aceita lab_gower/lab_kp já calculados (passe-os na execução).
# =============================================================================
def fig_concordancia_algoritmos(df, vars_cluster, lab_gower=None, lab_kp=None,
                                path="figuras/05_concordancia_algoritmos.png"):
    X = StandardScaler().fit_transform(df[vars_cluster].astype(float))
    lab_km = df["cluster"].to_numpy()

    if lab_gower is None:
        cont = [vars_cluster.index("QT_MAT_FUND_AF"), vars_cluster.index("QT_TABLET_ALUNO")]
        cat = [vars_cluster.index("TP_LOCALIZACAO"), vars_cluster.index("IN_SALA_LEITURA")]
        Xc = df[vars_cluster].astype(float).to_numpy()
        rng_ = Xc[:, cont].max(0) - Xc[:, cont].min(0)
        rng_[rng_ == 0] = 1.0
        n = len(df)
        p = len(vars_cluster)
        D = np.zeros((n, n), dtype=np.float32)
        for k, j in enumerate(cont):
            c = Xc[:, j][:, None]
            D += np.abs(c - c.T) / rng_[k]
        for j in cat:
            c = Xc[:, j][:, None]
            D += (c != c.T).astype(np.float32)
        D /= p
        lab_gower = AgglomerativeClustering(n_clusters=3, metric="precomputed",
                                            linkage="average").fit_predict(D)
    lab_gower = np.asarray(lab_gower)

    if lab_kp is None:
        try:
            from kmodes.kprototypes import KPrototypes
            Xmix = df[vars_cluster].astype(float).copy()
            conts = ["QT_MAT_FUND_AF", "QT_TABLET_ALUNO"]
            Xmix[conts] = StandardScaler().fit_transform(Xmix[conts])
            cat_idx = [vars_cluster.index("TP_LOCALIZACAO"),
                       vars_cluster.index("IN_SALA_LEITURA")]
            lab_kp = KPrototypes(n_clusters=3, random_state=SEED,
                                 n_init=10).fit_predict(Xmix.values, categorical=cat_idx)
        except Exception:
            lab_kp = None
    if lab_kp is not None:
        lab_kp = np.asarray(lab_kp)

    coords = PCA(n_components=2, random_state=SEED).fit_transform(X)
    paineis = [("K-means", lab_km), ("Gower + hierárquico", lab_gower)]
    if lab_kp is not None:
        paineis.append(("k-prototypes", lab_kp))

    fig, axes = plt.subplots(1, len(paineis), figsize=(5.0 * len(paineis), 4.6),
                             sharex=True, sharey=True)
    letras = ["A", "B", "C", "D"]
    for idx, (ax, (nome, lab)) in enumerate(zip(np.atleast_1d(axes), paineis)):
        for k, g in enumerate(np.unique(lab)):
            m = lab == g
            ax.scatter(coords[m, 0], coords[m, 1], s=8, alpha=0.45,
                       edgecolors="none", color=PALETA[k % len(PALETA)])
        _letra_painel(ax, letras[idx])
        if nome != "K-means":
            ari = adjusted_rand_score(lab_km, lab)
            txt = f"{nome}\nARI vs. K-means = {_virg(ari, 2)}"
        else:
            txt = f"{nome}\n(partição de referência)"
        ax.text(0.97, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color="black")
        ax.set_xlabel("Componente principal 1")
    np.atleast_1d(axes)[0].set_ylabel("Componente principal 2")
    fig.tight_layout()
    _salvar(fig, path)
    plt.close(fig)
    return path


# %% ===========================================================================
# 6. COEFICIENTES: HC3 vs ERRO-PADRÃO AGRUPADO (robustez da inferência)
# =============================================================================
def fig_coef_hc3_vs_cluster(df, preditores, alvo="IDEB_2023",
                            grupos="CO_ORGAO_REGIONAL",
                            path="figuras/06_coef_hc3_vs_cluster.png"):
    z = df.copy()
    cols = preditores + [alvo]
    z[cols] = StandardScaler().fit_transform(z[cols].astype(float))
    f = f"{alvo} ~ " + " + ".join(preditores)
    m_hc3 = smf.ols(f, data=z).fit(cov_type="HC3")
    m_clu = smf.ols(f, data=z).fit(cov_type="cluster", cov_kwds={"groups": df[grupos]})

    base_ord = m_hc3.params.drop("Intercept")
    ordem = base_ord.reindex(base_ord.abs().sort_values().index).index
    yv = np.arange(len(ordem))

    fig, ax = plt.subplots(figsize=(7.5, 8))
    for m, off, cor, lab in [(m_hc3, -0.16, ROXO, "HC3"),
                             (m_clu, 0.16, TEAL, "Agrupado (Diretoria de Ensino)")]:
        ci = m.conf_int().loc[ordem]
        ax.errorbar(m.params.loc[ordem], yv + off,
                    xerr=[m.params.loc[ordem] - ci[0], ci[1] - m.params.loc[ordem]],
                    fmt="o", ms=5, color=cor, lw=1.4, capsize=2, label=lab)
    ax.axvline(0, color=GUIA, ls="--", lw=1)
    ax.set_yticks(yv)
    ax.set_yticklabels(ordem, fontsize=9)
    ax.set_xlabel("Coeficiente padronizado (IC 95%)")
    ax.legend(loc="lower right")
    _salvar(fig, path)
    plt.close(fig)
    return path


# %% ===========================================================================
# 7. RADAR POR CLUSTER (dimensões normalizadas 0–1)
# =============================================================================
def fig_radar_clusters(df, path="figuras/07_radar_clusters.png"):
    dims = ["IDEB_2023", "MEDIA_INSE", "QT_MAT_FUND_AF",
            "QT_TABLET_ALUNO", "IN_SALA_LEITURA", "TP_LOCALIZACAO"]
    rotulos = ["IDEB", "INSE", "Matrículas", "Tablets", "Sala leitura", "Urbana"]
    medias = df.groupby("cluster")[dims].mean()
    norm = (medias - medias.min()) / (medias.max() - medias.min())
    ang = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    ang += ang[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    ax.set_facecolor("white")
    for c in sorted(df.cluster.unique()):
        v = norm.loc[c].tolist()
        v += v[:1]
        ax.plot(ang, v, color=CLUSTER_COLORS[c], lw=2, label=CLUSTER_NOMES[c])
        ax.fill(ang, v, color=CLUSTER_COLORS[c], alpha=0.12)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(rotulos)
    ax.set_yticklabels([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.34, 1.12))
    _salvar(fig, path)
    plt.close(fig)
    return path


# %% ===========================================================================
# 8. COEFICIENTES POR CLUSTER (efeitos heterogêneos — exploratório)
# =============================================================================
def fig_coef_por_cluster(df, preditores, foco, alvo="IDEB_2023",
                         path="figuras/08_coef_por_cluster.png"):
    z = df.copy()
    z[preditores + [alvo]] = StandardScaler().fit_transform(z[preditores + [alvo]].astype(float))
    f = f"{alvo} ~ " + " + ".join(preditores)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for c in sorted(df.cluster.unique()):
        sub = z[df.cluster == c]
        if len(sub) <= len(preditores) + 5:
            continue
        m = smf.ols(f, data=sub).fit(cov_type="HC3")
        ci = m.conf_int()
        for j, v in enumerate(foco):
            if v in m.params:
                ax.errorbar(m.params[v], j + (c - 1) * 0.18,
                            xerr=[[m.params[v] - ci.loc[v, 0]], [ci.loc[v, 1] - m.params[v]]],
                            fmt="o", ms=6, color=CLUSTER_COLORS[c], capsize=2)
    ax.axvline(0, color=GUIA, ls="--", lw=1)
    ax.set_yticks(range(len(foco)))
    ax.set_yticklabels(foco)
    ax.set_xlabel("Coeficiente padronizado por cluster (IC 95%)")
    ax.legend(handles=[Patch(color=CLUSTER_COLORS[c], label=CLUSTER_NOMES[c])
                       for c in sorted(df.cluster.unique())], loc="best")
    _salvar(fig, path)
    plt.close(fig)
    return path


# %% ===========================================================================
# 9. INTERAÇÃO TABLET x USO PEDAGÓGICO DA INTERNET
# =============================================================================
def fig_interacao_tablet_internet(df, preditores, alvo="IDEB_2023",
                                  path="figuras/09_interacao.png"):
    f = (f"{alvo} ~ " + " + ".join(preditores) +
         " + IN_INTERNET_APRENDIZAGEM + QT_TABLET_ALUNO:IN_INTERNET_APRENDIZAGEM")
    m = smf.ols(f, data=df).fit(cov_type="HC3")
    grid_t = np.linspace(df.QT_TABLET_ALUNO.quantile(.02),
                         df.QT_TABLET_ALUNO.quantile(.98), 50)
    base_ = {p: (df[p].mean() if df[p].nunique() > 2 else df[p].mode()[0]) for p in preditores}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for uso, cor, lab in [(0, ROXO, "Sem uso pedagógico da internet"),
                          (1, TEAL, "Com uso pedagógico da internet")]:
        g = pd.DataFrame([{**base_, "QT_TABLET_ALUNO": t,
                           "IN_INTERNET_APRENDIZAGEM": uso} for t in grid_t])
        ax.plot(grid_t, m.predict(g), color=cor, lw=2.2, label=lab)
    ax.set_xlabel("Tablets por aluno")
    ax.set_ylabel("IDEB predito")
    ax.legend(loc="best")
    _salvar(fig, path)
    plt.close(fig)
    return path


# %% ===========================================================================
# 10. BOXPLOTS EXCLUÍDAS vs RETIDAS (viés de seleção)
# =============================================================================
def fig_excluidas_retidas(base, variaveis, path="figuras/10_excluidas_retidas.png"):
    n = len(variaveis)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 4.2))
    letras = [chr(65 + i) for i in range(n)]
    for i, (ax, v) in enumerate(zip(np.atleast_1d(axes), variaveis)):
        dados = [base.loc[~base.excluida, v].dropna(), base.loc[base.excluida, v].dropna()]
        bp = ax.boxplot(dados, tick_labels=["Retidas", "Excluídas"], patch_artist=True,
                        widths=0.6, showfliers=False)
        for patch, cor in zip(bp["boxes"], [ROXO, TEAL]):
            patch.set_facecolor(cor)
            patch.set_alpha(0.65)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_xlabel(v, fontsize=10)
        _letra_painel(ax, letras[i])
    _salvar(fig, path)
    plt.close(fig)
    return path


# %% ===========================================================================
# 11. HISTOGRAMA DO JACCARD (bootstrap) — estabilidade do cluster rural
# =============================================================================
def fig_jaccard(jaccards, path="figuras/11_jaccard_rural.png"):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(jaccards, bins=20, color=TEAL, alpha=0.85, edgecolor="white")
    ax.axvline(0.75, color=ROXO, ls="--", lw=1.5, label="estável (> 0,75)")
    ax.axvline(0.60, color=VERDE, ls=":", lw=1.8, label="padrão real (> 0,60)")
    ax.axvline(np.mean(jaccards), color="black", lw=1.6,
               label=f"média = {_virg(np.mean(jaccards))}")
    ax.set_xlabel("Índice de Jaccard (reamostragem do cluster Rural Eficiente)")
    ax.set_ylabel("Frequência")
    ax.legend(loc="upper left")
    _salvar(fig, path)
    plt.close(fig)
    return path


# %% ===========================================================================
# 12. VIOLINO DO IDEB POR CLUSTER
# =============================================================================
def fig_ideb_por_cluster(df, path="figuras/12_ideb_por_cluster.png"):
    cs = sorted(df.cluster.unique())
    dados = [df.loc[df.cluster == c, "IDEB_2023"] for c in cs]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vp = ax.violinplot(dados, showmeans=True, showextrema=False)
    for b, c in zip(vp["bodies"], cs):
        b.set_facecolor(CLUSTER_COLORS[c])
        b.set_alpha(0.55)
        b.set_edgecolor("black")
    vp["cmeans"].set_color("black")
    for i, (c, serie) in enumerate(zip(cs, dados), start=1):
        media = serie.mean()
        topo = serie.max()
        ax.text(i, topo + 0.05, f"média = {_virg(media, 2)}\n(n = {len(serie)})",
                ha="center", va="bottom", fontsize=9, color="black")
    ax.set_xticks(range(1, len(cs) + 1))
    ax.set_xticklabels([CLUSTER_NOMES[c] for c in cs], rotation=10)
    ax.set_ylabel("IDEB 2023")
    ax.margins(y=0.12)
    fig.tight_layout()
    _salvar(fig, path)
    plt.close(fig)
    return path


# %% ===========================================================================
# 13. MAPA COROPLÉTICO — resíduo médio por município (governança / FDE)
# =============================================================================
def fig_mapa_residuos(df, model, nivel="municipio",
                      path="figuras/13_mapa_residuos.png",
                      shapefile_de=None, col_muni_de=None):
    try:
        import geopandas as gpd  # noqa: F401
        import geobr
    except Exception as e:
        print("Mapa não gerado — instale: pip install geopandas geobr")
        print("Detalhe:", e)
        return None

    df = df.copy()
    df["residuo"] = model.resid
    if nivel == "municipio":
        geo = geobr.read_municipality(code_muni="SP", year=2022)
        geo["code_muni"] = geo["code_muni"].astype("int64")
        agg = df.groupby("CO_MUNICIPIO")["residuo"].mean().reset_index()
        gdf = geo.merge(agg, left_on="code_muni", right_on="CO_MUNICIPIO", how="left")
    else:
        if shapefile_de is not None:
            agg = df.groupby("CO_ORGAO_REGIONAL")["residuo"].mean().reset_index()
            gdf = shapefile_de.copy().merge(agg, on="CO_ORGAO_REGIONAL", how="left")
        else:
            geo = geobr.read_municipality(code_muni="SP", year=2022)
            geo["code_muni"] = geo["code_muni"].astype("int64")
            geo["DE"] = geo["code_muni"].map(col_muni_de)
            geo = geo.dissolve(by="DE")
            agg = df.groupby("CO_ORGAO_REGIONAL")["residuo"].mean()
            gdf = geo.join(agg.rename("residuo"))

    fig, ax = plt.subplots(figsize=(8, 8))
    gdf.plot(column="residuo", cmap="viridis",
             linewidth=0.2, edgecolor="white", legend=True, ax=ax,
             missing_kwds={"color": "#EEEEEE", "label": "sem dados"})
    ax.axis("off")
    _salvar(fig, path)
    plt.close(fig)
    return path


# %% ===========================================================================
# 14. RIDGE SUMMARY (trace + dispersão beta OLS vs Ridge)
# =============================================================================
def fig_ridge_summary(X, y, preditores, alpha_sel, ols_hc3=None,
                      alphas=np.logspace(-3, 3, 100),
                      path="figuras/14_ridge_summary.png"):
    Xz = StandardScaler().fit_transform(X)
    yz = StandardScaler().fit_transform(np.asarray(y).reshape(-1, 1)).ravel()
    ols = LinearRegression().fit(Xz, yz)
    ridge = Ridge(alpha=alpha_sel).fit(Xz, yz)
    caminho = np.array([Ridge(alpha=a).fit(Xz, yz).coef_ for a in alphas])

    tab = pd.DataFrame({"variavel": preditores,
                        "beta_OLS": ols.coef_, "beta_Ridge": ridge.coef_})
    tab["encolhimento_%"] = (1 - tab.beta_Ridge.abs()
                             / tab.beta_OLS.abs().replace(0, np.nan)) * 100
    if ols_hc3 is not None:
        tab["p_HC3"] = tab.variavel.map(ols_hc3.pvalues.to_dict())
    print(tab.round(4).to_string(index=False))

    r2_ols = r2_score(yz, ols.predict(Xz))
    r2_ridge = r2_score(yz, ridge.predict(Xz))
    print(f"\nlambda selecionado = {alpha_sel:.4g}")
    print(f"R² in-sample OLS = {r2_ols:.6f} | Ridge = {r2_ridge:.6f} "
          f"| diferença = {r2_ols - r2_ridge:.2e}")
    print(f"encolhimento médio dos |beta| = {tab['encolhimento_%'].mean():.2f}%")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.8))
    cores = plt.cm.viridis(np.linspace(0, 0.9, len(preditores)))
    for j in range(len(preditores)):
        axA.plot(alphas, caminho[:, j], color=cores[j], lw=1.3)
    axA.axvline(alpha_sel, color=GUIA, ls="--", lw=1.2)
    axA.set_xscale("log")
    axA.set_xlabel("λ (alpha) — escala log")
    axA.set_ylabel("Coeficiente padronizado")
    _letra_painel(axA, "A")

    axB.scatter(ols.coef_, ridge.coef_, s=30, color=TEAL, edgecolors="none")
    lim = [min(ols.coef_.min(), ridge.coef_.min()),
           max(ols.coef_.max(), ridge.coef_.max())]
    axB.plot(lim, lim, color=GUIA, ls="--", lw=1.3, label="β OLS = β Ridge")
    axB.set_xlabel("β padronizado — OLS")
    axB.set_ylabel("β padronizado — Ridge")
    axB.legend(loc="lower right")
    _letra_painel(axB, "B")
    fig.tight_layout()
    _salvar(fig, path)
    plt.close(fig)
    return path


# %% ===========================================================================
# 15. PCA — PERFIS COM ELIPSES DE CONFIANÇA
# =============================================================================
def fig_pca_perfis(df, vars_cluster, seed=SEED, path="figuras/15_pca_perfis.png"):
    X = StandardScaler().fit_transform(df[vars_cluster].astype(float))
    pca = PCA(n_components=2, random_state=seed).fit(X)
    coords = pca.transform(X)
    var = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    for c in sorted(df.cluster.unique()):
        m = df.cluster.to_numpy() == c
        pts = coords[m]
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.30,
                   color=CLUSTER_COLORS[c], edgecolors="none")
        mu = pts.mean(axis=0)
        cov = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(*vecs[:, ::-1][0]))
        w, h = 2 * np.sqrt(vals * 5.991)   # qui-quadrado(2 gl, 95%) = 5,991
        ax.add_patch(Ellipse(mu, w, h, angle=ang, facecolor=CLUSTER_COLORS[c],
                             alpha=0.12, edgecolor=CLUSTER_COLORS[c], lw=1.6))
        ax.scatter(*mu, marker="X", s=240, color=CLUSTER_COLORS[c],
                   edgecolors="white", linewidths=1.8, zorder=5)
        ax.annotate(CLUSTER_NOMES[c], mu, fontsize=10, fontweight="bold",
                    color="black", ha="center", va="center",
                    xytext=(0, 16), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=CLUSTER_COLORS[c], alpha=0.85))
    ax.set_xlabel(f"Componente principal 1 ({_virg(var[0], 1)}% da variância)")
    ax.set_ylabel(f"Componente principal 2 ({_virg(var[1], 1)}% da variância)")
    ax.text(0.97, 0.03, f"PCA: {_virg(var.sum(), 1)}% da variância acumulada",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="black")
    ax.legend(handles=[Patch(color=CLUSTER_COLORS[c], label=CLUSTER_NOMES[c])
                       for c in sorted(df.cluster.unique())], loc="upper right")
    fig.tight_layout()
    _salvar(fig, path)
    plt.close(fig)
    print(f"Variância explicada: PC1={var[0]:.1f}% PC2={var[1]:.1f}% "
          f"acumulada={var.sum():.1f}%")
    return path


# %% ===========================================================================
# 16. CONCORDÂNCIA — MATRIZ DE ARI + CONTINGÊNCIA
# CORRIGIDO: aceita lab_gower/lab_kp já calculados (passe-os na execução).
# =============================================================================
def fig_concordancia_matriz(df, vars_cluster, lab_gower=None, lab_kp=None,
                            path="figuras/16_concordancia_matriz.png"):
    from matplotlib.colors import ListedColormap, BoundaryNorm

    lab_km = df["cluster"].to_numpy()

    if lab_gower is None:
        Xc = df[vars_cluster].astype(float).to_numpy()
        cont = [vars_cluster.index("QT_MAT_FUND_AF"), vars_cluster.index("QT_TABLET_ALUNO")]
        cat = [vars_cluster.index("TP_LOCALIZACAO"), vars_cluster.index("IN_SALA_LEITURA")]
        rng_ = Xc[:, cont].max(0) - Xc[:, cont].min(0)
        rng_[rng_ == 0] = 1.0
        n = len(df)
        D = np.zeros((n, n), np.float32)
        for k, j in enumerate(cont):
            c = Xc[:, j][:, None]
            D += np.abs(c - c.T) / rng_[k]
        for j in cat:
            c = Xc[:, j][:, None]
            D += (c != c.T).astype(np.float32)
        D /= len(vars_cluster)
        lab_gower = AgglomerativeClustering(3, metric="precomputed",
                                            linkage="average").fit_predict(D)
    lab_gower = np.asarray(lab_gower)

    if lab_kp is None:
        try:
            from kmodes.kprototypes import KPrototypes
            Xmix = df[vars_cluster].astype(float).copy()
            cc = ["QT_MAT_FUND_AF", "QT_TABLET_ALUNO"]
            Xmix[cc] = StandardScaler().fit_transform(Xmix[cc])
            ci = [vars_cluster.index("TP_LOCALIZACAO"), vars_cluster.index("IN_SALA_LEITURA")]
            lab_kp = KPrototypes(n_clusters=3, random_state=SEED,
                                 n_init=10).fit_predict(Xmix.values, categorical=ci)
        except Exception:
            lab_kp = None
    if lab_kp is not None:
        lab_kp = np.asarray(lab_kp)

    metodos = {"K-means": lab_km, "Gower": lab_gower}
    if lab_kp is not None and not np.all(lab_kp == -1):
        metodos["k-prototypes"] = lab_kp
    nomes = list(metodos)
    M = np.array([[adjusted_rand_score(metodos[a], metodos[b]) for b in nomes]
                  for a in nomes])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # (A) Matriz de ARI — cores discretas do módulo (sem mapa contínuo)
    cmap_ari = ListedColormap([ROXO, "#BBBBBB", TEAL])
    norm_ari = BoundaryNorm([-0.2, 0.2, 0.7, 1.0], cmap_ari.N)
    ax1.imshow(M, cmap=cmap_ari, norm=norm_ari)
    ax1.set_xticks(range(len(nomes)))
    ax1.set_xticklabels(nomes, rotation=15)
    ax1.set_yticks(range(len(nomes)))
    ax1.set_yticklabels(nomes)
    for i in range(len(nomes)):
        for j in range(len(nomes)):
            ax1.text(j, i, _virg(M[i, j], 2), ha="center", va="center",
                     fontsize=12, fontweight="bold", color="white")
    for s in ax1.spines.values():
        s.set_visible(False)
    ax1.tick_params(length=0)
    ax1.text(0.0, 1.04, "A", transform=ax1.transAxes, ha="left", va="bottom",
             fontsize=12, fontweight="bold", color="black")
    ax1.legend(handles=[Patch(color=TEAL, label="Concordância alta (ARI ≈ 1)"),
                        Patch(color=ROXO, label="Discordância (ARI ≈ 0)")],
               loc="upper left", bbox_to_anchor=(1.02, 1.0))

    # (B) Contingência K-means x Gower (diagonal = mesma partição)
    ct = pd.crosstab(pd.Series(lab_km, name="K-means"),
                     pd.Series(lab_gower, name="Gower"))
    ax2.imshow(ct.values, cmap="Blues")
    ax2.set_xticks(range(ct.shape[1]))
    ax2.set_xticklabels(ct.columns)
    ax2.set_yticks(range(ct.shape[0]))
    ax2.set_yticklabels(ct.index)
    ax2.set_xlabel("Cluster (Gower)")
    ax2.set_ylabel("Cluster (K-means)")
    vmax = ct.values.max()
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.values[i, j]
            ax2.text(j, i, f"{v}", ha="center", va="center", fontweight="bold",
                     color="white" if v > vmax * 0.5 else "black")
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.tick_params(length=0)
    ax2.text(0.0, 1.04, "B", transform=ax2.transAxes, ha="left", va="bottom",
             fontsize=12, fontweight="bold", color="black")
    fig.tight_layout()
    _salvar(fig, path)
    plt.close(fig)
    return path


# %% ===========================================================================
# 17. MATRIZ DE CORRELAÇÃO DE PEARSON  (NOVA)
# Exceção à paleta Viridis (já prevista no cabeçalho): paleta divergente RdBu,
# pois representa sinal/intensidade de r. Valores via df[...].corr() —
# determinístico, sem reamostragem -> SEMPRE coincide com o texto/tabela.
# =============================================================================
def fig_matriz_correlacao(df, variaveis=None, rotulos=None,
                          path="figuras/17_matriz_correlacao.png"):
    from matplotlib.colors import TwoSlopeNorm

    if variaveis is None:
        variaveis = ["IDEB_2023", "MEDIA_INSE", "QT_MAT_FUND_AF", "QT_DOC_FUND_AF",
                     "TP_LOCALIZACAO", "IN_EQUIP_LOUSA_DIGITAL", "QT_TABLET_ALUNO",
                     "IN_LABORATORIO_INFORMATICA", "IN_SALA_LEITURA",
                     "IN_LABORATORIO_CIENCIAS"]
    variaveis = [v for v in variaveis if v in df.columns]
    rotulos = rotulos if rotulos is not None else variaveis

    corr = df[variaveis].astype(float).corr()      # Pearson exato (determinístico)
    M = corr.to_numpy()
    n = len(variaveis)

    mask = np.triu(np.ones_like(M, dtype=bool), k=1)   # mascara triângulo superior
    Mm = np.ma.masked_array(M, mask=mask)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("white")
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im = ax.imshow(Mm, cmap=cmap, norm=norm)

    for i in range(n):
        for j in range(n):
            if not mask[i, j]:
                val = M[i, j]
                cor_txt = "white" if abs(val) >= 0.55 else "black"
                ax.text(j, i, _virg(val, 2), ha="center", va="center",
                        fontsize=8, color=cor_txt)

    ax.set_xticks(range(n))
    ax.set_xticklabels(rotulos, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(rotulos, fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Coeficiente de correlação de Pearson (r)", fontsize=10)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    _salvar(fig, path)
    plt.close(fig)
    return path


# %% ===========================================================================
# EXECUÇÃO
# Pré-requisitos na sessão (rode o pipeline antes):
#   df, base, ols_hc3, pipe, X, y, PREDITORES, VARS_CLUSTER, jaccards,
#   inercias, silhs  (célula "K-MEANS: K ÓTIMO")
#   lab_kp           (célula k-prototypes — opcional)
#   df["cluster_gower"] (célula Gower — opcional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    df = canonizar_clusters(df)                  # 1º: alinha rótulos à Tabela 7

    # ---- objetos opcionais: usa se existirem, senão deixa o fallback agir -----
    _lab_gower = df["cluster_gower"].to_numpy() if "cluster_gower" in df.columns else None
    _lab_kp = lab_kp if "lab_kp" in dir() else None
    _inercias = inercias if "inercias" in dir() else None
    _silhs = silhs if "silhs" in dir() else None

    fig_residuos_ajustados(ols_hc3, path="figuras/01_residuos_ajustados.png")
    fig_residuos(ols_hc3, path="figuras/02_residuos.png")
    fig_real_vs_predito(df, ols_hc3, pipe, X, y, path="figuras/03_real_vs_predito.png")

    # reaproveita inércias/silhuetas do pipeline -> bate com a tabela do texto
    fig_silhueta_inercia(df, VARS_CLUSTER,
                         inercias=_inercias, silhuetas=_silhs,
                         path="figuras/04_silhueta_inercia.png")

    # reaproveita rótulos já computados; nada de recriar Gower/k-prototypes
    fig_concordancia_algoritmos(df, VARS_CLUSTER,
                                lab_gower=_lab_gower, lab_kp=_lab_kp,
                                path="figuras/05_concordancia_algoritmos.png")

    fig_coef_hc3_vs_cluster(df, PREDITORES, path="figuras/06_coef_hc3_vs_cluster.png")
    fig_radar_clusters(df, path="figuras/07_radar_clusters.png")
    fig_coef_por_cluster(df, PREDITORES,
                         foco=["MEDIA_INSE", "QT_TABLET_ALUNO", "IN_SALA_LEITURA"],
                         path="figuras/08_coef_por_cluster.png")
    fig_interacao_tablet_internet(df, PREDITORES, path="figuras/09_interacao.png")
    fig_excluidas_retidas(base,
                          ["MEDIA_INSE", "QT_MAT_FUND_AF", "QT_DOC_FUND_AF",
                           "TP_LOCALIZACAO", "QT_TABLET_ALUNO", "IN_SALA_LEITURA"],
                          path="figuras/10_excluidas_retidas.png")
    fig_jaccard(jaccards, path="figuras/11_jaccard_rural.png")
    fig_ideb_por_cluster(df, path="figuras/12_ideb_por_cluster.png")
    fig_mapa_residuos(df, ols_hc3, nivel="municipio", path="figuras/13_mapa_residuos.png")

    fig_ridge_summary(X, y, PREDITORES,
                      alpha_sel=pipe.named_steps["ridge"].alpha_,
                      ols_hc3=ols_hc3, path="figuras/14_ridge_summary.png")

    fig_pca_perfis(df, VARS_CLUSTER, path="figuras/15_pca_perfis.png")

    fig_concordancia_matriz(df, VARS_CLUSTER,
                            lab_gower=_lab_gower, lab_kp=_lab_kp,
                            path="figuras/16_concordancia_matriz.png")

    # NOVA — matriz de correlação (valores exatos do df.corr())
    fig_matriz_correlacao(df, path="figuras/17_matriz_correlacao.png")
