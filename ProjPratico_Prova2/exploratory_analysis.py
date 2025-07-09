"""
Análise exploratória dos dados Sentinel-2

Este script realiza uma análise exploratória completa dos dados,
incluindo visualizações e estatísticas descritivas.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

# Configurar diretórios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, 'ImagensSentinel-2')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Criar diretório de output se não existir
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_scl_layer():
    """
    Analisa a camada SCL (Scene Classification Layer)
    """
    print("Analisando camada SCL...")
    
    # Carregar SCL
    scl_path = os.path.join(IMAGES_DIR, 'T23KMQ_20250604T131239_SCL_20m.jp2')
    scl_img = Image.open(scl_path)
    scl_array = np.array(scl_img)
    
    # Definir paleta de cores para SCL
    scl_palette = [
        '#000000', '#ff0000', '#2f2f2f', '#643200', '#00a000', '#ffe65a',
        '#0000ff', '#808080', '#c0c0c0', '#ffffff', '#64c8ff', '#ff96ff'
    ]
    
    # Mapeamento de classes SCL
    scl_classes = {
        0: 'Sem dados',
        1: 'Pixels saturados/defeituosos',
        2: 'Sombra escura',
        3: 'Sombra de nuvem',
        4: 'Vegetação',
        5: 'Solo nu',
        6: 'Água',
        7: 'Nuvem (baixa probabilidade)',
        8: 'Nuvem (média probabilidade)',
        9: 'Nuvem (alta probabilidade)',
        10: 'Cirrus',
        11: 'Neve/gelo'
    }
    
    # Estatísticas das classes
    unique_classes, counts = np.unique(scl_array, return_counts=True)
    total_pixels = scl_array.size
    
    print(f"Dimensões da imagem: {scl_array.shape}")
    print(f"Total de pixels: {total_pixels}")
    print("\nDistribuição das classes:")
    
    for cls, count in zip(unique_classes, counts):
        percentage = (count / total_pixels) * 100
        class_name = scl_classes.get(cls, f'Classe {cls}')
        print(f"  {cls}: {class_name} - {count} pixels ({percentage:.2f}%)")
    
    # Visualizar SCL
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Imagem SCL
    cmap = ListedColormap(scl_palette)
    bounds = list(range(len(scl_palette) + 1))
    norm = BoundaryNorm(bounds, cmap.N)
    
    im1 = ax1.imshow(scl_array, cmap=cmap, norm=norm)
    ax1.set_title('Scene Classification Layer (SCL)')
    ax1.set_xlabel('Pixel X')
    ax1.set_ylabel('Pixel Y')
    
    # Histograma das classes
    ax2.bar(unique_classes, counts, color=[scl_palette[i] if i < len(scl_palette) else '#000000' for i in unique_classes])
    ax2.set_title('Distribuição das Classes SCL')
    ax2.set_xlabel('Classe')
    ax2.set_ylabel('Número de Pixels')
    ax2.set_xticks(unique_classes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'scl_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return scl_array, unique_classes, counts

def analyze_spectral_bands():
    """
    Analisa as bandas espectrais
    """
    print("\nAnalisando bandas espectrais...")
    
    bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
    band_info = {
        'B02': 'Azul (490 nm)',
        'B03': 'Verde (560 nm)',
        'B04': 'Vermelho (665 nm)',
        'B05': 'Red Edge 1 (705 nm)',
        'B06': 'Red Edge 2 (740 nm)',
        'B07': 'Red Edge 3 (783 nm)',
        'B8A': 'NIR (865 nm)',
        'B11': 'SWIR 1 (1610 nm)',
        'B12': 'SWIR 2 (2190 nm)'
    }
    
    bands_data = {}
    bands_stats = {}
    
    for band in bands:
        band_path = os.path.join(IMAGES_DIR, f'T23KMQ_20250604T131239_{band}_20m.jp2')
        if os.path.exists(band_path):
            band_img = Image.open(band_path)
            band_array = np.array(band_img)
            bands_data[band] = band_array
            
            # Calcular estatísticas
            bands_stats[band] = {
                'min': np.min(band_array),
                'max': np.max(band_array),
                'mean': np.mean(band_array),
                'std': np.std(band_array),
                'median': np.median(band_array)
            }
            
            print(f"Banda {band} ({band_info[band]}):")
            print(f"  Min: {bands_stats[band]['min']}")
            print(f"  Max: {bands_stats[band]['max']}")
            print(f"  Média: {bands_stats[band]['mean']:.2f}")
            print(f"  Desvio padrão: {bands_stats[band]['std']:.2f}")
    
    # Visualizar algumas bandas
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    selected_bands = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
    
    for i, band in enumerate(selected_bands):
        if band in bands_data:
            # Normalizar para visualização
            band_norm = np.clip(bands_data[band] / np.percentile(bands_data[band], 99), 0, 1)
            
            im = axes[i].imshow(band_norm, cmap='gray')
            axes[i].set_title(f'{band} - {band_info[band]}')
            axes[i].set_xlabel('Pixel X')
            axes[i].set_ylabel('Pixel Y')
            plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'spectral_bands.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return bands_data, bands_stats

def create_rgb_composite():
    """
    Cria composição RGB (bandas 4, 3, 2)
    """
    print("\nCriando composição RGB...")
    
    # Carregar bandas RGB
    b4_path = os.path.join(IMAGES_DIR, 'T23KMQ_20250604T131239_B04_20m.jp2')
    b3_path = os.path.join(IMAGES_DIR, 'T23KMQ_20250604T131239_B03_20m.jp2')
    b2_path = os.path.join(IMAGES_DIR, 'T23KMQ_20250604T131239_B02_20m.jp2')
    
    b4_img = np.array(Image.open(b4_path))
    b3_img = np.array(Image.open(b3_path))
    b2_img = np.array(Image.open(b2_path))
    
    # Criar composição RGB
    rgb_composite = np.zeros((b4_img.shape[0], b4_img.shape[1], 3), dtype=np.float32)
    
    # Normalizar bandas (usando percentil 99 para evitar outliers)
    rgb_composite[:,:,0] = np.clip(b4_img / np.percentile(b4_img, 99), 0, 1)  # Red
    rgb_composite[:,:,1] = np.clip(b3_img / np.percentile(b3_img, 99), 0, 1)  # Green
    rgb_composite[:,:,2] = np.clip(b2_img / np.percentile(b2_img, 99), 0, 1)  # Blue
    
    # Aplicar correção de gamma para melhor visualização
    rgb_composite = np.power(rgb_composite, 0.5)
    
    # Visualizar
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb_composite)
    plt.title('Composição RGB (B4-B3-B2) - Cores Verdadeiras')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rgb_composite.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return rgb_composite

def analyze_target_classes():
    """
    Analisa as classes de interesse para o projeto
    """
    print("\nAnalisando classes de interesse...")
    
    # Carregar SCL
    scl_path = os.path.join(IMAGES_DIR, 'T23KMQ_20250604T131239_SCL_20m.jp2')
    scl_array = np.array(Image.open(scl_path))
    
    # Classes de interesse
    target_classes = {
        2: 'Não vegetação (sombra escura)',
        4: 'Vegetação',
        6: 'Água',
        8: 'Não definida (nuvem média prob.)'
    }
    
    # Criar máscara para classes de interesse
    target_mask = np.isin(scl_array, list(target_classes.keys()))
    
    # Estatísticas das classes de interesse
    print("Classes de interesse para o projeto:")
    total_target_pixels = 0
    
    for cls, name in target_classes.items():
        count = np.sum(scl_array == cls)
        total_target_pixels += count
        print(f"  Classe {cls} ({name}): {count} pixels")
    
    print(f"\nTotal de pixels úteis: {total_target_pixels}")
    print(f"Percentual de pixels úteis: {(total_target_pixels / scl_array.size) * 100:.2f}%")
    
    # Visualizar classes de interesse
    target_image = np.zeros_like(scl_array)
    colors = {2: 1, 4: 2, 6: 3, 8: 4}
    
    for cls in target_classes.keys():
        target_image[scl_array == cls] = colors[cls]
    
    plt.figure(figsize=(12, 8))
    cmap = ListedColormap(['black', 'brown', 'green', 'blue', 'gray'])
    plt.imshow(target_image, cmap=cmap)
    plt.title('Classes de Interesse para Classificação')
    
    # Criar legenda
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=cmap(colors[cls]), 
                                    label=f'{cls}: {name}') 
                      for cls, name in target_classes.items()]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'target_classes.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Executa análise exploratória completa
    """
    print("ANÁLISE EXPLORATÓRIA DOS DADOS SENTINEL-2")
    print("=" * 50)
    
    # Analisar camada SCL
    scl_array, unique_classes, counts = analyze_scl_layer()
    
    # Analisar bandas espectrais
    bands_data, bands_stats = analyze_spectral_bands()
    
    # Criar composição RGB
    rgb_composite = create_rgb_composite()
    
    # Analisar classes de interesse
    analyze_target_classes()
    
    print(f"\nAnálise concluída! Resultados salvos em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
