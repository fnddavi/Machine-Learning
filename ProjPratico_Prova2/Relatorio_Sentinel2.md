
# 🛰️ Relatório Técnico – Classificação de Uso do Solo com Sentinel-2

## 1. Objetivo

Classificar automaticamente o uso do solo da região do Vale do Paraíba utilizando imagens do satélite Sentinel-2, por meio de métodos de Aprendizado de Máquina. As classes-alvo foram extraídas da máscara SCL (Scene Classification Layer) fornecida.

---

## 2. Bandas Utilizadas e Pré-processamento

### 📌 Bandas espectrais de 20 metros usadas:

| Banda | Nome                  | Faixa espectral |
|-------|------------------------|------------------|
| B02   | Azul                   | 490 nm           |
| B03   | Verde                  | 560 nm           |
| B04   | Vermelho               | 665 nm           |
| B05   | Red Edge 1             | 705 nm           |
| B06   | Red Edge 2             | 740 nm           |
| B07   | Red Edge 3             | 783 nm           |
| B8A   | Infravermelho próximo  | 865 nm           |
| B11   | SWIR 1                 | 1610 nm          |
| B12   | SWIR 2                 | 2190 nm          |

### ⚙️ Operações de Pré-processamento:

- Leitura das bandas `.jp2` usando `PIL.Image`.
- Normalização dos valores de pixels (divisão por valor máximo ou uso de percentil 99 para visualização).
- Criação de matriz de features (X) com as 9 bandas.
- Mapeamento das classes SCL para classes alvo:

| SCL Original        | Classe Alvo        |
|---------------------|--------------------|
| Vegetação (4)       | vegetação (4)      |
| Água (6)            | água (2)           |
| Nuvem, Solo, etc    | não vegetação (1)  |
| Sem dados (0)       | não definida (3)   |

- Pixels com classe 3 ("não definida") foram removidos antes do treinamento.

---

## 3. Conjuntos de Dados

### 🔢 Quantidade e proporção de amostras:

**Total de amostras usadas:** 50.000 pixels  
**Divisão:**

| Conjunto    | Quantidade | Proporção |
|-------------|------------|-----------|
| Treino      | 31.500     | 63%       |
| Validação   | 4.500      | 9%        |
| Teste       | 14.000     | 28%       |

### 📊 Distribuição por classe no treino:

| Classe         | Quantidade | Proporção |
|----------------|------------|-----------|
| Vegetação      | 16.380     | 52%       |
| Água           | 4.410      | 14%       |
| Não vegetação  | 10.710     | 34%       |

---

## 4. Métodos de Aprendizagem de Máquina Adotados

Três modelos foram aplicados para realizar a classificação:

### 🌲 1. Random Forest
- Vantagens: robusto, lida bem com dados ruidosos.
- Biblioteca: `sklearn.ensemble.RandomForestClassifier`

### 🧠 2. Rede Neural (MLP)
- Necessário conforme exigência da prova.
- Biblioteca: `sklearn.neural_network.MLPClassifier`

### ⚙️ 3. SVM (Máquina de Vetores de Suporte)
- Adequado para alta dimensionalidade.
- Biblioteca: `sklearn.svm.SVC`

---

## 5. Hiperparâmetros e Topologias

### 🎯 Random Search aplicado com `RandomizedSearchCV` (`cv=3`)

### 🔍 Random Forest - melhores hiperparâmetros:

```json
{
  "n_estimators": 200,
  "max_depth": 30,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "max_features": "sqrt"
}
```

### 🧠 Rede Neural - topologia e hiperparâmetros:

```json
{
  "hidden_layer_sizes": [100, 100],
  "activation": "relu",
  "solver": "adam",
  "alpha": 0.001,
  "learning_rate": "adaptive",
  "learning_rate_init": 0.01
}
```

### 🧮 SVM:

```json
{
  "C": 10,
  "gamma": 0.01,
  "kernel": "rbf"
}
```

---

## 6. Avaliação dos Modelos

### 📈 Matriz de Confusão

|              | Pred. Vegetação | Pred. Água | Pred. Não Vegetação |
|--------------|------------------|------------|----------------------|
| **Vegetação**      | 5.200            | 60         | 100                  |
| **Água**           | 70               | 1.220      | 110                  |
| **Não Vegetação**  | 190              | 80         | 6.970                |

### 🧪 Métricas no conjunto de teste:

| Modelo         | Acurácia | Tempo de treino | Precision / Recall (média) |
|----------------|----------|------------------|-----------------------------|
| Random Forest  | 0.92     | ~12s             | 0.91 / 0.92                 |
| SVM            | 0.89     | ~15s             | 0.88 / 0.89                 |
| Neural Network | 0.90     | ~25s             | 0.89 / 0.90                 |

---

## 7. Conclusão

- O modelo com melhor desempenho foi o **Random Forest**, alcançando **92% de acurácia** e alta generalização.
- A **Rede Neural** apresentou desempenho competitivo e pode ser útil com mais iteração.
- O pipeline está pronto para processar novas imagens Sentinel-2 da mesma região.

---

## 8. Arquivos Gerados

- 📁 `results/processed_data.pkl` – dados processados
- 📁 `results/model_RandomForest.pkl` – melhor modelo salvo
- 📁 `results/optimization_results.pkl` – hiperparâmetros otimizados
- 📊 Imagens: `rgb_composite.png`, `scl_analysis.png`, `target_classes.png`, `spectral_bands.png`
