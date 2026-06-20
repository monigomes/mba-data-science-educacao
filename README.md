# MBA em Data Science e Analytics -  USP/ESALQ

## Determinantes do desempenho escolar: integração de regressão e agrupamentos em escolas públicas paulistas

**Autora:** Monica Gomes de Souza Silva  
**Orientador:** Prof. Dr. Fábio Lima  

---

## Sobre o projeto

Este repositório reúne os códigos Python e as figuras utilizados no Trabalho de Conclusão de Curso do MBA em Data Science & Analytics da USP/ESALQ.

O estudo investiga fatores associados ao desempenho escolar, medido pelo Índice de Desenvolvimento da Educação Básica (IDEB) de 2023, em escolas públicas estaduais do Estado de São Paulo. Para isso, foram integradas bases oficiais do Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira (INEP), incluindo:

- IDEB 2023;
- Censo Escolar 2023;
- Índice de Nível Socioeconômico das Escolas (INSE) 2023.

A análise utiliza técnicas de regressão linear múltipla, erros-padrão robustos, regressão Ridge, análise de agrupamentos, validação por métodos alternativos de clusterização e visualizações exploratórias.

---

## Principais achados

- O nível socioeconômico (INSE) apresentou-se como o principal preditor do IDEB 2023.
- Foram encontradas evidências do chamado "paradoxo da tecnologia", em que a presença ou maior disponibilidade de determinados recursos tecnológicos não se associou automaticamente a melhor desempenho escolar.
- Escolas rurais apresentaram desempenho superior ao esperado em alguns modelos, especialmente quando controladas as demais características.
- A análise de agrupamentos identificou três perfis estruturais de escolas:
  - Urbano Referência;
  - Rural Eficiente;
  - Urbano Crítico.

---

## Estrutura do repositório

```text
.
├── figuras/
│   ├── 01_residuos_ajustados.png
│   ├── 02_residuos.png
│   ├── 03_real_vs_predito.png
│   ├── ...
│   └── 17_matriz_correlacao.png
│
├── scripts/
│   └── script_completo_do_estudo.py
│
├── requirements.txt
├── CITATION.cff
└── README.md
````

### Descrição das pastas e arquivos

* `figuras/`: contém os gráficos finais gerados a partir das análises.
* `scripts/`: contém o script Python principal utilizado para importação, tratamento, modelagem estatística, análise de agrupamentos e geração das figuras.
* `requirements.txt`: lista as bibliotecas Python necessárias para executar o projeto.
* `CITATION.cff`: arquivo de citação do repositório.
* `README.md`: documentação geral do projeto.

---

## Bases de dados

As bases de dados não estão incluídas diretamente neste repositório. Para reproduzir a análise, é necessário baixar os arquivos originais nas fontes oficiais do INEP.

### Onde encontrar os dados

**Censo Escolar 2023**
https://www.gov.br/inep/pt-br/areas-de-atuacao/pesquisas-estatisticas-e-indicadores/censo-escolar/resultados

**IDEB 2023**
https://www.gov.br/inep/pt-br/areas-de-atuacao/pesquisas-estatisticas-e-indicadores/ideb/resultados

**INSE 2023**
https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/indicadores-educacionais/nivel-socioeconomico

Após o download, os caminhos dos arquivos devem ser ajustados no início do script, na seção de definição dos caminhos das bases.

---

## Como executar

### 1. Clone o repositório

```bash
git clone https://github.com/monicagomes/mba-data-science-educacao.git
cd mba-data-science-educacao
```

### 2. Crie um ambiente virtual

No Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

No macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Baixe as bases oficiais

Baixe as bases do Censo Escolar, IDEB e INSE nos links indicados acima.

Em seguida, ajuste no script os caminhos correspondentes aos arquivos baixados.

### 5. Execute o script principal

```bash
python script_completo_do_estudo.py
```

---

## Saídas esperadas

A execução do script gera análises relacionadas a:

* composição e saneamento da amostra;
* estatísticas descritivas;
* matriz de correlação;
* regressão OLS com erros-padrão robustos;
* diagnóstico dos pressupostos do modelo;
* regressão Ridge e validação cruzada;
* análise de agrupamentos por K-means;
* testes de robustez dos clusters;
* análise de componentes principais;
* geração das figuras finais.

As figuras finais estão armazenadas na pasta `figuras/`.

---

## Reprodutibilidade

O projeto utiliza semente fixa (`SEED = 42`) nos procedimentos que envolvem aleatoriedade, incluindo validação cruzada, K-means, PCA e bootstrap.

As análises foram desenvolvidas em Python, com uso das bibliotecas listadas no arquivo `requirements.txt`.

---

## Como citar

Caso utilize este repositório, cite conforme indicado no arquivo `CITATION.cff`.

Referência sugerida:

> SILVA, Monica Gomes de Souza; LIMA, Fábio. *Determinantes do desempenho escolar: integração de regressão e agrupamentos em escolas públicas paulistas*. Trabalho de Conclusão de Curso apresentado ao MBA em Data Science e Analytics, USP/ESALQ, 2026.

---
