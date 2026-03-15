# MBA em Data Science - Educação (USP/ESALQ)

## Determinantes do desempenho escolar: integração de regressão e agrupamentos em escolas públicas paulistas

**Autora:** Monica Gomes de Souza Silva  
**Orientador:** Prof. Dr. Tadeu Alcides Marques

### Sobre o Projeto

Este repositório contém todos os códigos Python utilizados na análise do Trabalho de Conclusão de Curso do MBA em Data Science & Analytics da USP/ESALQ. O estudo investiga os fatores associados ao desempenho escolar medido pelo Índice de Desenvolvimento da Educação Básica (IDEB) nas escolas públicas do Estado de São Paulo, utilizando dados de 2023 do Censo Escolar, IDEB e INSE.

### Principais Achados

- O nível socioeconômico (INSE) é o principal determinante do IDEB
- Evidências do "paradoxo da tecnologia" (laboratórios de informática associados a menor desempenho)
- Escolas rurais apresentam desempenho superior quando controlado o INSE
- Identificação de três perfis distintos de escolas via análise de clusters

### Estrutura do Repositório
## Estrutura do Repositório

- `scripts/`
  - `01_importacao_tratamento.py` - Importação e merge das bases
  - `02_analise_descritiva.py` - Estatísticas descritivas e correlações
  - `03_modelo_ols_diagnosticos.py` - Regressão OLS e testes
  - `04_regressao_ridge.py` - Regressão Ridge
  - `05_analise_clusters.py` - Análise de clusters (K-means)
  - `06_figuras.py` - Geração de todas as figuras
- `utils/`
  - `config.py` - Configurações de estilo
- `dados/` - Instruções para acesso aos dados
- `outputs/`
  - `figuras/` - Figuras geradas
  - `tabelas/` - Tabelas exportadas
- `requirements.txt` - Dependências do projeto

### Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/monicagomes/mba-data-science-educacao.git
   cd mba-data-science-educacao
