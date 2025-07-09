"""
Análise Exploratória dos Dados Sentinel-2

Este script realiza análise exploratória das imagens Sentinel-2 para entender
as características dos dados antes da classificação.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd
from data_loader import SentinelDataLoader
import warnings
warnings.filterwarnings('ignore')

def analyze_image_properties():
    """Analisa propriedades básicas das imagens"""
    print("=== ANÁLISE DAS PROPRIEDADES DAS IMAGENS ===")
    
    # Carregar uma imagem exemplo
    sample_path = "ImagensSentinel-2/T23KMQ_20250604T131239_B04_20m.jp2"
    if os.path.exists(sample_path):
        img = Image.open(sample_path)
        img_array = np.array(img)
        
        print(f"Dimensões da imagem: {img_array.shape}")
        print(f"Tipo de dados: {img_array.dtype}")
        print(f"Modo da imagem: {img.mode}")
        print(f"Formato: {img.format}")
        print(f"Tamanho total: {img_array.size:,} pixels")
        
        # Estatísticas básicas
        print(f"\nEstatísticas da banda B04:")
        print(f"  Mínimo: {img_array.min()}")
        print(f"  Máximo: {img_array.max()}")
        print(f"  Média: {img_array.mean():.2f}")
        print(f"  Desvio padrão: {img_array.std():.2f}")
        
        # Histograma
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(img_array.flatten(), bins=50, alpha=0.7, color='blue')
        plt.title('Histograma da Banda B04')
        plt.xlabel('Valor do Pixel')
        plt.ylabel('Frequência')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_array, cmap='gray')
        plt.title('Banda B04 (Vermelho)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"Arquivo não encontrado: {sample_path}")

def analyze_all_bands():
    """Analisa todas as bandas espectrais"""
    print("\n=== ANÁLISE DE TODAS AS BANDAS ===")
    
    loader = SentinelDataLoader()
    loader.load_all_bands()
    
    if not loader.bands:
        print("Nenhuma banda foi carregada")
        return
    
    # Criar DataFrame com estatísticas
    stats_data = []
    for band_name, band_data in loader.bands.items():
        stats_data.append({
            'Banda': band_name,
            'Mínimo': band_data.min(),
            'Máximo': band_data.max(),
            'Média': band_data.mean(),
            'Desvio_Padrão': band_data.std(),
            'Mediana': np.median(band_data)
        })
    
    stats_df = pd.DataFrame(stats_data)
    print("\nEstatísticas por banda:")
    print(stats_df.round(2))
    
    # Plotar histogramas de todas as bandas
    n_bands = len(loader.bands)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (band_name, band_data) in enumerate(loader.bands.items()):
        if i < len(axes):
            axes[i].hist(band_data.flatten(), bins=50, alpha=0.7)
            axes[i].set_title(f'Banda {band_name}')
            axes[i].set_xlabel('Valor do Pixel')
            axes[i].set_ylabel('Frequência')
    
    # Remover subplots vazios
    for i in range(len(loader.bands), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.suptitle('Histogramas das Bandas Espectrais', y=1.02)
    plt.show()

def analyze_scl_classes():
    """Analisa as classes do SCL"""
    print("\n=== ANÁLISE DAS CLASSES SCL ===")
    
    loader = SentinelDataLoader()
    loader.load_scl()
    
    if loader.scl_image is None:
        print("Imagem SCL não foi carregada")
        return
    
    # Contar classes
    unique_classes, counts = np.unique(loader.scl_image, return_counts=True)
    total_pixels = loader.scl_image.size
    
    print("Distribuição das classes SCL:")
    scl_class_names = {
        0: "NO_DATA",
        1: "SATURATED_DEFECTIVE", 
        2: "DARK_AREA_PIXELS",
        3: "CLOUD_SHADOWS",
        4: "VEGETATION",
        5: "NOT_VEGETATED",
        6: "WATER",
        7: "UNCLASSIFIED",
        8: "CLOUD_MEDIUM_PROBABILITY",
        9: "CLOUD_HIGH_PROBABILITY",
        10: "THIN_CIRRUS",
        11: "SNOW"
    }
    
    for cls, count in zip(unique_classes, counts):
        percentage = (count / total_pixels) * 100
        class_name = scl_class_names.get(cls, f"UNKNOWN_{cls}")
        print(f"  Classe {cls} ({class_name}): {count:,} pixels ({percentage:.2f}%)")

def analyze_target_classes():
    """Analisa as classes target mapeadas"""
    print("\n=== ANÁLISE DAS CLASSES TARGET ===")
    
    loader = SentinelDataLoader()
    loader.load_scl()
    
    # Criar vetor de classes target
    targets = loader.create_target_vector()
    
    # Contar classes target
    unique_targets, counts = np.unique(targets, return_counts=True)
    total_pixels = len(targets)
    
    print("Distribuição das classes target:")
    for cls, count in zip(unique_targets, counts):
        percentage = (count / total_pixels) * 100
        class_name = loader.class_names.get(cls, f"Desconhecida_{cls}")
        print(f"  {class_name}: {count:,} pixels ({percentage:.2f}%)")

def analyze_sample_data():
    """Analisa dados amostrados para ML"""
    print("\n=== ANÁLISE DOS DADOS AMOSTRADOS ===")
    
    loader = SentinelDataLoader()
    loader.load_all_bands()
    loader.load_scl()
    
    # Amostrar dados
    features, targets = loader.sample_data(n_samples=10000)
    
    print(f"Dados amostrados: {features.shape[0]} amostras, {features.shape[1]} features")
    
    # Estatísticas das features
    print("\nEstatísticas das features:")
    feature_names = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]
    
    for i, name in enumerate(feature_names):
        if i < features.shape[1]:
            print(f"  {name}: min={features[:, i].min():.3f}, "
                  f"max={features[:, i].max():.3f}, "
                  f"mean={features[:, i].mean():.3f}, "
                  f"std={features[:, i].std():.3f}")
    
    # Distribuição por classe
    print("\nDistribuição das classes na amostra:")
    class_dist = loader.get_class_distribution(targets)
    total_samples = len(targets)
    
    for classe, count in class_dist.items():
        percentage = (count / total_samples) * 100
        print(f"  {classe}: {count:,} amostras ({percentage:.1f}%)")

def main():
    """Função principal da análise exploratória"""
    print("=" * 60)
    print("ANÁLISE EXPLORATÓRIA - DADOS SENTINEL-2")
    print("=" * 60)
    
    try:
        # 1. Propriedades das imagens
        analyze_image_properties()
        
        # 2. Análise de todas as bandas
        analyze_all_bands()
        
        # 3. Análise das classes SCL
        analyze_scl_classes()
        
        # 4. Análise das classes target
        analyze_target_classes()
        
        # 5. Análise dos dados amostrados
        analyze_sample_data()
        
        print("\n" + "=" * 60)
        print("ANÁLISE EXPLORATÓRIA CONCLUÍDA!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Erro durante a análise: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
