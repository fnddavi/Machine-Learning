# Machine Learning - Projeto de Estudos

Este repositório contém notebooks e exercícios práticos de Machine Learning, incluindo análise de dados, algoritmos de classificação, redes neurais e técnicas de ensemble.

## Estrutura do Projeto

- **AnaliseDados/**: Exercícios de análise e visualização de dados
- **AprendizadoEnsemble/**: Implementações de técnicas de ensemble (Random Forest, Voting, AdaBoost, Bagging)
- **ArvoresDecisao/**: Algoritmos de árvores de decisão
- **Aula**/**: Notebooks das aulas organizados por data
- **copy/**: Notebooks de apoio e exemplos adicionais

## Configuração do Ambiente

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Instalação

1. Clone o repositório:

```bash
git clone <url-do-repositorio>
cd Machine-Learning
```

2. Crie e ative o ambiente virtual:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Inicie o Jupyter Notebook:

```bash
jupyter notebook
```

## Principais Dependências

- **Jupyter**: Ambiente de desenvolvimento interativo
- **NumPy**: Computação numérica
- **Pandas**: Manipulação e análise de dados
- **Scikit-learn**: Algoritmos de machine learning
- **TensorFlow/Keras**: Deep learning
- **Matplotlib/Seaborn/Plotly**: Visualização de dados

## Como Usar

1. Ative o ambiente virtual antes de trabalhar nos notebooks
2. Execute os notebooks na ordem sugerida pelos nomes das pastas
3. Cada notebook contém explicações detalhadas e código comentado

## Principais Tópicos Abordados

- Análise exploratória de dados
- Pré-processamento de dados
- Algoritmos de classificação e regressão
- Técnicas de ensemble
- Redes neurais e deep learning
- Validação cruzada e métricas de avaliação

## Estrutura dos Dados

Os datasets utilizados incluem:

- MNIST (dígitos manuscritos)
- Iris (classificação de flores)
- Dados hospitalares (análise médica)
- Datasets personalizados para exercícios específicos
