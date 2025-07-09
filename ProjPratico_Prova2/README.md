# Projeto de Classificação de Uso do Solo - Sentinel-2

## Descrição

Este projeto implementa classificação automática de uso do solo para a região do Vale do Paraíba usando imagens do satélite Sentinel-2 e três métodos de aprendizagem de máquina.

## Objetivos

- Classificar automaticamente o uso do solo em: **vegetação**, **água** e **não vegetação**
- Comparar três métodos de classificação: Random Forest, SVM e Rede Neural (MLP)
- Otimizar hiperparâmetros para melhor performance
- Gerar relatórios e análises comparativas

## Dados

### Entrada
- **Fonte**: Imagens Sentinel-2 da região do Vale do Paraíba
- **Bandas espectrais**: B02, B03, B04, B05, B06, B07, B8A, B11, B12 (resolução 20m)
- **Rótulos**: Scene Classification Layer (SCL)

### Classes
- **Vegetação** (classe 4): Áreas com cobertura vegetal
- **Água** (classe 2): Corpos d'água
- **Não vegetação** (classe 1): Áreas urbanas, solo exposto, etc.

## Estrutura do Projeto

```
ProjPratico_Prova2/
├── main.py                           # Script principal
├── data_loader.py                    # Carregamento e pré-processamento
├── ml_models.py                      # Modelos de machine learning
├── hyperparameter_optimization.py   # Otimização de hiperparâmetros
├── exploratory_analysis.py          # Análise exploratória
├── requirements.txt                  # Dependências
├── README.md                         # Este arquivo
├── ImagensSentinel-2/               # Imagens Sentinel-2
│   ├── T23KMQ_*_B02_20m.jp2        # Banda azul
│   ├── T23KMQ_*_B03_20m.jp2        # Banda verde
│   ├── T23KMQ_*_B04_20m.jp2        # Banda vermelha
│   ├── T23KMQ_*_B05_20m.jp2        # Red Edge 1
│   ├── T23KMQ_*_B06_20m.jp2        # Red Edge 2
│   ├── T23KMQ_*_B07_20m.jp2        # Red Edge 3
│   ├── T23KMQ_*_B8A_20m.jp2        # NIR (narrow)
│   ├── T23KMQ_*_B11_20m.jp2        # SWIR 1
│   ├── T23KMQ_*_B12_20m.jp2        # SWIR 2
│   └── T23KMQ_*_SCL_20m.jp2        # Scene Classification Layer
├── scripts/                         # Scripts de apoio
│   ├── sentinel_B4.py               # Visualização banda B4
│   ├── sentinel_B432.py             # Composição RGB
│   └── sentinel_scl.py              # Visualização SCL
└── results/                         # Resultados gerados
    ├── processed_data.pkl           # Dados processados
    ├── optimization_results.pkl     # Resultados da otimização
    └── model_*.pkl                  # Modelos treinados
```

## Instalação

### 1. Criar Ambiente Virtual

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Verificar Estrutura de Dados

Certifique-se de que as imagens Sentinel-2 estão no diretório `ImagensSentinel-2/`.

## Uso

### Execução Principal

```bash
python main.py
```

### Execução por Módulos

#### 1. Análise Exploratória
```bash
python exploratory_analysis.py
```

#### 2. Teste de Carregamento de Dados
```bash
python data_loader.py
```

#### 3. Teste de Modelos
```bash
python ml_models.py
```

#### 4. Otimização de Hiperparâmetros
```bash
python hyperparameter_optimization.py
```

## Metodologia

### 1. Pré-processamento
- Carregamento das bandas espectrais
- Normalização dos valores de pixel (0-1)
- Mapeamento das classes SCL
- Amostragem estratificada dos dados

### 2. Modelos Utilizados

#### Random Forest
- **Características**: Ensemble de árvores de decisão
- **Vantagens**: Robusto, interpreta features
- **Parâmetros**: n_estimators, max_depth, min_samples_split

#### SVM (Support Vector Machine)
- **Características**: Classificador baseado em margens
- **Vantagens**: Eficaz em alta dimensionalidade
- **Parâmetros**: C, gamma, kernel

#### Rede Neural (MLP)
- **Características**: Perceptron multicamadas
- **Vantagens**: Captura relações não-lineares
- **Parâmetros**: hidden_layer_sizes, activation, solver

### 3. Otimização
- **Método**: RandomizedSearchCV
- **Validação**: 3-fold cross-validation
- **Métrica**: Acurácia

### 4. Avaliação
- **Métricas**: Acurácia, Precisão, Recall, F1-score
- **Visualização**: Matriz de confusão
- **Análise**: Importância das features

## Configurações

### Parâmetros Principais (main.py)

```python
N_SAMPLES = 50000          # Amostras para treinamento
OPTIMIZE_HYPERPARAMS = True  # Otimizar hiperparâmetros
SAVE_RESULTS = True         # Salvar resultados
```

### Divisão dos Dados
- **Treino**: 60%
- **Validação**: 10%
- **Teste**: 30%

## Resultados Esperados

### Métricas de Performance
- **Acurácia**: > 85%
- **Precisão por classe**: Vegetação > 90%, Água > 80%, Não vegetação > 80%

### Arquivos Gerados
- `processed_data.pkl`: Dados processados
- `optimization_results.pkl`: Melhores hiperparâmetros
- `model_*.pkl`: Modelos treinados
- `confusion_matrix_*.png`: Matrizes de confusão

## Bandas Espectrais Utilizadas

| Banda | Nome | Resolução | Uso |
|-------|------|-----------|-----|
| B02 | Blue | 20m | Análise atmosférica |
| B03 | Green | 20m | Vegetação saudável |
| B04 | Red | 20m | Clorofila |
| B05 | Red Edge 1 | 20m | Status da vegetação |
| B06 | Red Edge 2 | 20m | Status da vegetação |
| B07 | Red Edge 3 | 20m | Status da vegetação |
| B8A | NIR narrow | 20m | Biomassa |
| B11 | SWIR 1 | 20m | Umidade do solo |
| B12 | SWIR 2 | 20m | Umidade do solo |

## Troubleshooting

### Problemas Comuns

1. **Erro de memória**: Reduzir `N_SAMPLES` no main.py
2. **Demora na execução**: Desabilitar `OPTIMIZE_HYPERPARAMS`
3. **Imagens não encontradas**: Verificar diretório `ImagensSentinel-2/`

### Requisitos de Sistema
- **RAM**: Mínimo 8GB (recomendado 16GB)
- **Espaço**: ~5GB para dados e resultados
- **Python**: 3.8+

## Contribuição

### Melhorias Possíveis
- Adicionar mais modelos (XGBoost, LightGBM)
- Implementar deep learning (CNN)
- Incluir mais classes de uso do solo
- Otimizar performance com GPU

### Estrutura de Código
- Código modular e bem documentado
- Tratamento de erros
- Logging detalhado
- Testes unitários (futuro)

## Licença

Este projeto é parte de um trabalho acadêmico para a disciplina de Machine Learning.

## Autor

Fernando
Julho 2025

---

**Nota**: Este projeto foi desenvolvido como parte da avaliação prática P2 da disciplina de Machine Learning, focando na aplicação de técnicas de classificação em dados de sensoriamento remoto.
