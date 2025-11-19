
# Projeto: Agente de IA para Interpretação de Modelos de Crédito

Este projeto demonstra um **pipeline completo de Machine Learning**, desde a **geração sintética dos dados** até a **interpretação automática dos modelos** por meio de um **Agente de IA Explicativo**.  

O foco é integrar **modelos preditivos tradicionais** (como Regressão Logística e Random Forest) com **IA explicativa**, capaz de analisar, comparar e interpretar modelos de forma autônoma — *um passo em direção à integração entre Machine 
Learning e IA generativa*.

---
## Estrutura do Projeto

```
agente-ia/
├── data/
│   └── clientes_sinteticos.csv               # Base de dados sintética gerada
│
├── model/
│   ├── modelo_logistico.pkl                  # Modelo treinado (Regressão Logística)
│   └── scaler.pkl                            # Objeto de padronização
│
├── notebooks/
│   ├── 01_gera_dados.ipynb                   # Geração dos dados sintéticos
│   ├── analise_dados.ipynb                   # Análise exploratória dos dados
│   ├── 02_modelagem.ipynb                    # Treinamento, tuning e avaliação dos modelos
│   └── 03_agente_ia.ipynb                    # Agente de IA para interpretação automática
│
├── scripts/
│   ├── gera_dados.py                         # Versão script para gerar dados
│   ├── train_model.py                        # Treina e salva o modelo
│   └── predict.py                            # Faz previsões em novos dados
│
├── resultados/
│   ├── resultados_modelos.csv                # Métricas comparativas dos modelos
│   └── explicacao_modelos.txt                # Saída textual do agente de IA
│
├── requirements.txt
│
└── README.md
```
---

## Etapas do Projeto

### 1. Geração de Dados Sintéticos (`01_gera_dados.ipynb`/ `scripts/gera_dados.py`)

Simula um conjunto de **5.000 clientes**, com variáveis como idade, renda, tempo de emprego, score de crédito, valor do empréstimo e risco de inadimplência.

**Principais pontos:**
- Uso de distribuições estatísticas (normal, exponencial, lognormal);
- Cálculo de probabilidade de inadimplência com base em múltiplos fatores;
- Salvamento automático em `data/clientes_sinteticos.csv`.
- A estrutura foi adaptada para detectar automaticamente se o código está sendo executado no **Google Colab** ou localmente no **VS Code**.

---

### 2. Análise Exploratória (`analise_dados.ipynb`)

Responsável por explorar o dataset gerado e compreender padrões, correlações e tendências.

Inclui:

- **Estatísticas descritivas e informações gerais**
- **Visualização da distribuição de renda, score e inadimplência**
- **Correlação entre variáveis financeiras e o risco de inadimplência**

---

### 3. Modelagem e Avaliação (`02_modelagem.ipynb` / `scripts/train_model.py`)

Treina e compara três modelos, avaliando desempenho com métricas de classificação:

- **Regressão Logística (balanceada)**
- **Árvore de Decisão**
- **Random Forest**

**Etapas executadas:**
1. Divisão treino/teste e padronização das variáveis;
2. Treinamento dos modelos;
3. Ajuste de hiperparâmetros com **GridSearchCV**;
4. Comparação das métricas de **Acurácia** e **F1-score**;
5. Salvamento dos melhores modelos e métricas em resultados/.

**Resultado obtido (exemplo):**
| Modelo | Acurácia | F1-score |
|--------|-----------|----------|
| Regressão Logística (Tuned) | 0.5900 | 0.4137 |
| Random Forest (Tuned) | 0.6227 | 0.3953 |

> A Regressão Logística apresentou melhor equilíbrio entre precisão e recall, 
sendo mais interpretável para stakeholders.

---

## 4. Predição Automatizada (`scripts/predict.py`)

Utiliza o modelo salvo (modelo_logistico.pkl) e o scaler.pkl para realizar previsões em novos dados.

Funções:

- **Leitura do modelo e do scaler (model/);**
- **Criação de novos exemplos de clientes;**
- **Predição da classe (Adimplente/Inadimplente);**
- **Salvamento dos resultados em data/previsoes_clientes.csv.**

Essa etapa é essencial para integrar o modelo a sistemas reais, APIs ou interfaces web.

---

### 5. Agente de IA Explicativo (`03_agente_ia.ipynb`)

Cria um **agente de IA local** capaz de interpretar automaticamente os resultados dos modelos e gerar explicações em **linguagem natural**.

**Funções principais:**
- Identificar o modelo com melhor desempenho (usando `F1-score`);
- Gerar análises textuais automáticas sobre cada modelo;
- Comparar modelos de forma interpretável.

**Exemplo de saída:**
> “O modelo com melhor desempenho geral foi **Regressão Logística (Tuned)**, com F1-score de 0.414 e acurácia de 0.590.  
> Embora o Random Forest apresente acurácia ligeiramente superior, a Regressão Logística manteve melhor equilíbrio entre precisão e recall, o que a torna mais consistente para bases desbalanceadas.”

**Extensão opcional (IA Generativa):**
> As explicações e resultados são automaticamente salvos em:
   - resultados/resultados_modelos.csv
   - resultados/explicacao_modelos.txt

---

### Extensão: Integração com IA Generativa

O notebook 03_agente_ia.ipynb inclui um bloco opcional (comentado) que mostra 
como integrar o agente a modelos generativos reais (como OpenAI GPT, Claude ou LLaMA)
para criar explicações ainda mais detalhadas.

---

## Como Executar o Projeto Localmente

### 1. Clonar o repositório

```
bash
git clone https://github.com/nycolleAgnes/agente-ia.git
cd agente-ia
```
---

### 2. Criar o ambiente virtual
```
bash

python -m venv venv
venv\Scripts\activate  # Windows

### ou
source venv/bin/activate  # Linux/Mac
```

---

### 3. Instalar dependências
```
bash
pip install -r requirements.txt
```
---

### 4. Executar o pipeline completo

### 4.1 Gerar os dados sintéticos
python scripts/gera_dados.py
### 4.2 Treinar e salvar o modelo
python scripts/train_model.py
### 4.3 Realizar previsões
python scripts/predict.py

Ou, se preferir executar de forma interativa:

"01_gera_dados.ipynb"
"analise_dados.ipynb"
"02_modelagem.ipynb"
"03_agente_ia.ipynb"

---

## Como Executar o Projeto utilizando o Google Colab + link do GitHub

### Link GitHub: https://github.com/nycolleAgnes/agente-ia.git

---

### 1. Abrir o Colab com o repositórior 

- Acesse https://colab.research.google.com**
- Clique em " + Novo notebook"
---

### 2. Preparar o ambiente (cole e execute)

- Cole esta sequência no início do seu notebook Colab
e execute — ela faz o clone, instala dependências e mostra arquivos:

---

python

### 2.1) Remover pasta antiga (se existir no Colab)
!rm -rf agente-ia

### 2.2) clone o repositório
!git clone https://github.com/nycolleAgnes/agente-ia.git

### 2.3) Montar o Google Drive
from google.colab import drive
drive.mount('/content/drive')

### 2.3) entre na pasta do projeto
%cd /content/agente-ia

### 2.3) inspecione a estrutura
!ls -la
!ls -la data
!ls -la model
!sed -n '1,120p' requirements.txt || true

### 2.4) instale dependências do requirements (se houver)
!pip install -r requirements.txt

**Observações :**
- ! -> executa comandos shell;
- %cd -> altera diretório do notebook

---

### 2.5) copiar o projeto para o Drive (opcional, para salvar modificações)
!cp -r /content/agente-ia /content/drive/MyDrive/agente-ia_copy

**Dica: deixe saídas (ex.: resultados/, model/) dentro de /content/drive/MyDrive/ para não perder.**

---

### 3. rodar scripts Python (manualmente)

- Se você preferir executar os scripts em scripts/ (ex.: gerar dados, treinar, prever), use:

```
bash
### gerar dados com o script
!python3 scripts/gera_dados.py

### treinar modelo
!python3 scripts/train_model.py

### previsões
!python3 scripts/predict.py --input data/novos_dados.csv --output resultados/preds.csv
```

---

### 4. salvar resultados no Drive
```
python

!cp -r resultados /content/drive/MyDrive/agente-ia_resultados/
!cp -r model /content/drive/MyDrive/agente-ia_modelos/
```
---

## Principais Tecnologias

Linguagem:	Python 3.x
Manipulação de Dados:	Pandas, NumPy
Modelagem:	Scikit-learn
Visualização:	Matplotlib, Seaborn
IA Explicativa:	IA local (função Python) / OpenAI API (opcional)
Ambiente:	Jupyter Notebook, Google Colab, VS Code

---

### Autora

Nycolle Franco

GitHub
https://github.com/nycolleAgnes

LinkedIn
https://www.linkedin.com/in/nycolle-franco-67b757261/