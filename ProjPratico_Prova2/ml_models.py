"""
Módulo para implementar diferentes modelos de Machine Learning
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

class MLModels:
    """Classe para gerenciar diferentes modelos de ML"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.class_names = {
            1: "não vegetação",
            2: "água", 
            4: "vegetação"
        }
    
    def add_random_forest(self, **params):
        """Adiciona modelo Random Forest"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(params)
        
        self.models['RandomForest'] = RandomForestClassifier(**default_params)
        print(f"Random Forest adicionado com parâmetros: {default_params}")
    
    def add_svm(self, **params):
        """Adiciona modelo SVM"""
        default_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'random_state': 42
        }
        default_params.update(params)
        
        self.models['SVM'] = SVC(**default_params)
        print(f"SVM adicionado com parâmetros: {default_params}")
    
    def add_neural_network(self, **params):
        """Adiciona Rede Neural (MLP)"""
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'max_iter': 500,
            'random_state': 42
        }
        default_params.update(params)
        
        self.models['NeuralNetwork'] = MLPClassifier(**default_params)
        print(f"Rede Neural adicionada com parâmetros: {default_params}")
    
    def train_model(self, model_name, X_train, y_train, X_val=None, y_val=None, scale_features=True):
        """Treina um modelo específico"""
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} não encontrado")
        
        print(f"\n--- Treinando {model_name} ---")
        start_time = time.time()
        
        # Escalar features se necessário
        if scale_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scalers[model_name] = scaler
            
            if X_val is not None:
                X_val_scaled = scaler.transform(X_val)
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # Treinar modelo
        model = self.models[model_name]
        model.fit(X_train_scaled, y_train)
        
        training_time = time.time() - start_time
        
        # Avaliar no conjunto de treino
        train_pred = model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        # Avaliar no conjunto de validação se disponível
        val_accuracy = None
        if X_val is not None and y_val is not None:
            val_pred = model.predict(X_val_scaled)
            val_accuracy = accuracy_score(y_val, val_pred)
        
        # Salvar resultados
        self.results[model_name] = {
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'trained': True
        }
        
        print(f"Tempo de treinamento: {training_time:.2f}s")
        print(f"Acurácia no treino: {train_accuracy:.4f}")
        if val_accuracy is not None:
            print(f"Acurácia na validação: {val_accuracy:.4f}")
        
        return model
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None):
        """Treina todos os modelos adicionados"""
        for model_name in self.models.keys():
            try:
                self.train_model(model_name, X_train, y_train, X_val, y_val)
            except Exception as e:
                print(f"Erro ao treinar {model_name}: {e}")
    
    def predict(self, model_name, X_test):
        """Faz predições com um modelo específico"""
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} não encontrado")
        
        model = self.models[model_name]
        
        # Escalar features se necessário
        if model_name in self.scalers:
            X_test_scaled = self.scalers[model_name].transform(X_test)
        else:
            X_test_scaled = X_test
        
        return model.predict(X_test_scaled)
    
    def evaluate_model(self, model_name, X_test, y_test, show_plots=True):
        """Avalia um modelo específico"""
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} não encontrado")
        
        print(f"\n--- Avaliação do {model_name} ---")
        
        # Fazer predições
        y_pred = self.predict(model_name, X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Acurácia no teste: {accuracy:.4f}")
        
        # Relatório de classificação
        print("\nRelatório de Classificação:")
        target_names = [self.class_names[i] for i in sorted(self.class_names.keys())]
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        
        if show_plots:
            self.plot_confusion_matrix(cm, model_name, target_names)
        
        # Atualizar resultados
        if model_name in self.results:
            self.results[model_name]['test_accuracy'] = accuracy
            self.results[model_name]['confusion_matrix'] = cm
        
        return accuracy, cm
    
    def plot_confusion_matrix(self, cm, model_name, target_names):
        """Plota matriz de confusão"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Matriz de Confusão - {model_name}')
        plt.ylabel('Classe Real')
        plt.xlabel('Classe Predita')
        plt.tight_layout()
        plt.show()
    
    def compare_models(self):
        """Compara performance de todos os modelos"""
        if not self.results:
            print("Nenhum modelo foi treinado ainda")
            return
        
        print("\n--- Comparação de Modelos ---")
        print(f"{'Modelo':<15} {'Treino':<10} {'Validação':<12} {'Teste':<10} {'Tempo (s)':<10}")
        print("-" * 65)
        
        for model_name, result in self.results.items():
            train_acc = f"{result['train_accuracy']:.4f}" if result['train_accuracy'] else "N/A"
            val_acc = f"{result['val_accuracy']:.4f}" if result['val_accuracy'] else "N/A"
            test_acc = f"{result['test_accuracy']:.4f}" if 'test_accuracy' in result else "N/A"
            time_str = f"{result['training_time']:.2f}" if result['training_time'] else "N/A"
            
            print(f"{model_name:<15} {train_acc:<10} {val_acc:<12} {test_acc:<10} {time_str:<10}")
    
    def get_best_model(self, metric='test_accuracy'):
        """Retorna o nome do melhor modelo baseado em uma métrica"""
        if not self.results:
            return None
        
        best_model = None
        best_score = -1
        
        for model_name, result in self.results.items():
            if metric in result and result[metric] is not None:
                if result[metric] > best_score:
                    best_score = result[metric]
                    best_model = model_name
        
        return best_model
    
    def save_model(self, model_name, filepath):
        """Salva um modelo treinado"""
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} não encontrado")
        
        model_data = {
            'model': self.models[model_name],
            'scaler': self.scalers.get(model_name, None),
            'results': self.results.get(model_name, {}),
            'class_names': self.class_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"Modelo {model_name} salvo em: {filepath}")
    
    def load_model(self, model_name, filepath):
        """Carrega um modelo salvo"""
        model_data = joblib.load(filepath)
        
        self.models[model_name] = model_data['model']
        if model_data['scaler'] is not None:
            self.scalers[model_name] = model_data['scaler']
        self.results[model_name] = model_data['results']
        
        print(f"Modelo {model_name} carregado de: {filepath}")
    
    def get_feature_importance(self, model_name):
        """Retorna importância das features (quando disponível)"""
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} não encontrado")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        else:
            print(f"Modelo {model_name} não suporta importância de features")
            return None
    
    def plot_feature_importance(self, model_name, feature_names=None):
        """Plota importância das features"""
        importance = self.get_feature_importance(model_name)
        
        if importance is None:
            return
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importance))]
        
        # Ordenar por importância
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Importância das Features - {model_name}')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Exemplo de uso
    print("Exemplo de uso da classe MLModels")
    
    # Criar dados de exemplo
    np.random.seed(42)
    X_train = np.random.rand(1000, 9)
    y_train = np.random.choice([1, 2, 4], 1000)
    X_val = np.random.rand(200, 9)
    y_val = np.random.choice([1, 2, 4], 200)
    X_test = np.random.rand(300, 9)
    y_test = np.random.choice([1, 2, 4], 300)
    
    # Inicializar modelos
    ml_models = MLModels()
    
    # Adicionar modelos
    ml_models.add_random_forest(n_estimators=50)
    ml_models.add_svm(C=0.1)
    ml_models.add_neural_network(hidden_layer_sizes=(50,))
    
    # Treinar todos os modelos
    ml_models.train_all_models(X_train, y_train, X_val, y_val)
    
    # Avaliar modelos
    for model_name in ml_models.models.keys():
        ml_models.evaluate_model(model_name, X_test, y_test, show_plots=False)
    
    # Comparar modelos
    ml_models.compare_models()
    
    # Melhor modelo
    best_model = ml_models.get_best_model()
    print(f"\nMelhor modelo: {best_model}")
