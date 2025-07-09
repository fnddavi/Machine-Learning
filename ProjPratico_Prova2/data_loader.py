"""
Módulo para carregar e pré-processar dados das imagens Sentinel-2
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

class SentinelDataLoader:
    """Classe para carregar e processar dados Sentinel-2"""
    
    def __init__(self, data_dir="ImagensSentinel-2"):
        self.data_dir = data_dir
        self.bands = {}
        self.scl_image = None
        self.image_size = None
        
        # Mapeamento das classes SCL para nossas classes target
        self.class_mapping = {
            0: 3,   # NO_DATA -> não definida
            1: 1,   # SATURATED_DEFECTIVE -> não vegetação
            2: 1,   # DARK_AREA_PIXELS -> não vegetação
            3: 1,   # CLOUD_SHADOWS -> não vegetação
            4: 4,   # VEGETATION -> vegetação
            5: 1,   # NOT_VEGETATED -> não vegetação
            6: 2,   # WATER -> água
            7: 1,   # UNCLASSIFIED -> não vegetação
            8: 1,   # CLOUD_MEDIUM_PROBABILITY -> não vegetação
            9: 1,   # CLOUD_HIGH_PROBABILITY -> não vegetação
            10: 1,  # THIN_CIRRUS -> não vegetação
            11: 1   # SNOW -> não vegetação
        }
        
        # Nomes das classes
        self.class_names = {
            1: "não vegetação",
            2: "água", 
            3: "não definida",
            4: "vegetação"
        }
    
    def load_band(self, band_name):
        """Carrega uma banda específica"""
        filename = f"T23KMQ_20250604T131239_{band_name}_20m.jp2"
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        
        img_pil = Image.open(filepath)
        img_array = np.array(img_pil)
        
        if self.image_size is None:
            self.image_size = img_array.shape
        
        return img_array
    
    def load_all_bands(self):
        """Carrega todas as bandas espectrais de 20m"""
        band_names = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]
        
        for band in band_names:
            try:
                self.bands[band] = self.load_band(band)
                print(f"Banda {band} carregada com sucesso")
            except FileNotFoundError as e:
                print(f"Erro ao carregar banda {band}: {e}")
    
    def load_scl(self):
        """Carrega a imagem de classificação SCL"""
        self.scl_image = self.load_band("SCL")
        print("Imagem SCL carregada com sucesso")
    
    def normalize_band(self, band_array, max_value=8000):
        """Normaliza uma banda espectral"""
        return np.where(band_array > max_value, 1.0, band_array / max_value)
    
    def create_feature_matrix(self, band_list=None):
        """Cria matriz de features a partir das bandas selecionadas"""
        if band_list is None:
            band_list = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]
        
        if not self.bands:
            self.load_all_bands()
        
        features = []
        for band in band_list:
            if band in self.bands:
                normalized_band = self.normalize_band(self.bands[band])
                features.append(normalized_band.flatten())
        
        return np.array(features).T
    
    def create_target_vector(self):
        """Cria vetor de classes target a partir da imagem SCL"""
        if self.scl_image is None:
            self.load_scl()
        
        # Mapear classes SCL para nossas classes target
        target = np.zeros_like(self.scl_image)
        for scl_class, target_class in self.class_mapping.items():
            target[self.scl_image == scl_class] = target_class
        
        return target.flatten()
    
    def sample_data(self, n_samples=50000, stratify=True):
        """Amostra dados para treinamento"""
        # Criar features e targets
        features = self.create_feature_matrix()
        targets = self.create_target_vector()
        
        # Remover pixels com classe "não definida" (classe 3)
        valid_mask = targets != 3
        features_clean = features[valid_mask]
        targets_clean = targets[valid_mask]
        
        # Amostragem
        if len(features_clean) > n_samples:
            if stratify:
                _, features_sampled, _, targets_sampled = train_test_split(
                    features_clean, targets_clean, 
                    test_size=n_samples, 
                    stratify=targets_clean, 
                    random_state=42
                )
            else:
                indices = np.random.choice(len(features_clean), n_samples, replace=False)
                features_sampled = features_clean[indices]
                targets_sampled = targets_clean[indices]
        else:
            features_sampled = features_clean
            targets_sampled = targets_clean
        
        return features_sampled, targets_sampled
    
    def split_data(self, features, targets, test_size=0.3, val_size=0.1):
        """Divide dados em treino, validação e teste"""
        # Primeiro split: treino + validação vs teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, targets, test_size=test_size, stratify=targets, random_state=42
        )
        
        # Segundo split: treino vs validação
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42
            )
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            return X_temp, X_test, y_temp, y_test
    
    def get_class_distribution(self, targets):
        """Retorna distribuição das classes"""
        unique, counts = np.unique(targets, return_counts=True)
        distribution = {}
        for cls, count in zip(unique, counts):
            if cls in self.class_names:
                distribution[self.class_names[cls]] = count
        return distribution
    
    def visualize_scl(self):
        """Visualiza a imagem SCL com cores apropriadas"""
        if self.scl_image is None:
            self.load_scl()
        
        scl_palette = [
            '#000000', '#ff0000', '#2f2f2f', '#643200', '#00a000', '#ffe65a',
            '#0000ff', '#808080', '#c0c0c0', '#ffffff', '#64c8ff', '#ff96ff'
        ]
        
        cmap = ListedColormap(scl_palette)
        bounds = list(range(len(scl_palette) + 1))
        norm = BoundaryNorm(bounds, cmap.N)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(self.scl_image, cmap=cmap, norm=norm)
        plt.title("Scene Classification Layer (SCL)")
        plt.colorbar(label="Classes SCL")
        plt.show()
    
    def visualize_rgb(self, bands=["B04", "B03", "B02"]):
        """Visualiza composição RGB"""
        if not all(band in self.bands for band in bands):
            print("Carregando bandas necessárias...")
            self.load_all_bands()
        
        rgb_image = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.float32)
        
        for i, band in enumerate(bands):
            rgb_image[:, :, i] = self.normalize_band(self.bands[band])
        
        plt.figure(figsize=(12, 10))
        plt.imshow(rgb_image)
        plt.title(f"Composição RGB ({'/'.join(bands)})")
        plt.axis('off')
        plt.show()
    
    def save_processed_data(self, features, targets, filepath):
        """Salva dados processados"""
        data = {
            'features': features,
            'targets': targets,
            'class_names': self.class_names,
            'feature_bands': list(self.bands.keys())
        }
        joblib.dump(data, filepath)
        print(f"Dados salvos em: {filepath}")
    
    def load_processed_data(self, filepath):
        """Carrega dados processados"""
        data = joblib.load(filepath)
        return data['features'], data['targets'], data['class_names']

if __name__ == "__main__":
    # Exemplo de uso
    loader = SentinelDataLoader()
    
    # Carregar dados
    loader.load_all_bands()
    loader.load_scl()
    
    # Visualizar dados
    print("Visualizando SCL...")
    loader.visualize_scl()
    
    print("Visualizando RGB...")
    loader.visualize_rgb()
    
    # Preparar dados para ML
    print("Preparando dados para ML...")
    features, targets = loader.sample_data(n_samples=50000)
    
    # Dividir dados
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(features, targets)
    
    print(f"\nTamanhos dos conjuntos:")
    print(f"Treino: {X_train.shape[0]} amostras")
    print(f"Validação: {X_val.shape[0]} amostras")
    print(f"Teste: {X_test.shape[0]} amostras")
    
    print(f"\nDistribuição das classes no conjunto de treino:")
    train_dist = loader.get_class_distribution(y_train)
    for classe, count in train_dist.items():
        print(f"{classe}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # Salvar dados processados
    loader.save_processed_data(features, targets, "processed_data.pkl")
