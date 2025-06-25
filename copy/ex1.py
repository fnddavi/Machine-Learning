import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# Carregar o dataset
iris = fetch_ucirepo(id=53)
X = iris.data.features

# Selecionar a coluna "sepal width"
sepal_width = X["sepal width"]

# Calcular média e mediana com numpy
mean = np.mean(sepal_width)
median = np.median(sepal_width)

# Exibir média e mediana
print(f"Média da largura da sépala: {mean:.2f} cm")
print(f"Mediana da largura da sépala: {median:.2f} cm")

# Criar Boxplot
plt.figure(figsize=(8, 5))
plt.boxplot(sepal_width, vert=True)
plt.title("Boxplot - Largura da sépala")
plt.xlabel("cm")
plt.grid(True)
plt.show()

# Criar Histograma
plt.figure(figsize=(8, 5))
plt.hist(sepal_width, bins=10, edgecolor='black')
plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Média: {mean:.2f}')
plt.axvline(median, color='green', linestyle='dashed', linewidth=1, label=f'Mediana: {median:.2f}')
plt.title("Histograma - Largura da sépala")
plt.xlabel("cm")
plt.ylabel("Frequência")
plt.legend()
plt.grid(True)
plt.show()
