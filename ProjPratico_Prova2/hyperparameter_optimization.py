"""
Módulo para otimização de hiperparâmetros dos modelos
"""
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import time
import joblib

class HyperparameterOptimizer:
    """Classe para otimização de hiperparâmetros"""
    
    def __init__(self, cv=5, n_jobs=-1, scoring='accuracy'):
        self.cv = cv
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.best_params = {}
        self.best_scores = {}
        self.scalers = {}
    
    def optimize_random_forest(self, X_train, y_train, search_type='grid'):
        """Otimiza hiperparâmetros do Random Forest"""
        print("Otimizando Random Forest...")
        
        # Definir espaço de busca
        if search_type == 'grid':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        else:  # random search
            param_grid = {
                'n_estimators': [50, 100, 150, 200, 250],
                'max_depth': [10, 20, 30, 40, None],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None]
            }
        
        # Modelo base
        rf = RandomForestClassifier(random_state=42, n_jobs=self.n_jobs)
        
        # Busca de hiperparâmetros
        if search_type == 'grid':
            search = GridSearchCV(rf, param_grid, cv=self.cv, 
                                scoring=self.scoring, n_jobs=self.n_jobs)
        else:
            search = RandomizedSearchCV(rf, param_grid, cv=self.cv, 
                                      scoring=self.scoring, n_jobs=self.n_jobs,
                                      n_iter=50, random_state=42)
        
        start_time = time.time()
        search.fit(X_train, y_train)
        optimization_time = time.time() - start_time
        
        self.best_params['RandomForest'] = search.best_params_
        self.best_scores['RandomForest'] = search.best_score_
        
        print(f"Melhores parâmetros: {search.best_params_}")
        print(f"Melhor score CV: {search.best_score_:.4f}")
        print(f"Tempo de otimização: {optimization_time:.2f}s")
        
        return search.best_estimator_
    
    def optimize_svm(self, X_train, y_train, search_type='grid'):
        """Otimiza hiperparâmetros do SVM"""
        print("Otimizando SVM...")
        
        # Escalar dados para SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['SVM'] = scaler
        
        # Definir espaço de busca
        if search_type == 'grid':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
        else:  # random search
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100, 1000],
                'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
        
        # Modelo base
        svm = SVC(random_state=42)
        
        # Busca de hiperparâmetros
        if search_type == 'grid':
            search = GridSearchCV(svm, param_grid, cv=self.cv, 
                                scoring=self.scoring, n_jobs=self.n_jobs)
        else:
            search = RandomizedSearchCV(svm, param_grid, cv=self.cv, 
                                      scoring=self.scoring, n_jobs=self.n_jobs,
                                      n_iter=30, random_state=42)
        
        start_time = time.time()
        search.fit(X_train_scaled, y_train)
        optimization_time = time.time() - start_time
        
        self.best_params['SVM'] = search.best_params_
        self.best_scores['SVM'] = search.best_score_
        
        print(f"Melhores parâmetros: {search.best_params_}")
        print(f"Melhor score CV: {search.best_score_:.4f}")
        print(f"Tempo de otimização: {optimization_time:.2f}s")
        
        return search.best_estimator_
    
    def optimize_neural_network(self, X_train, y_train, search_type='grid'):
        """Otimiza hiperparâmetros da Rede Neural"""
        print("Otimizando Rede Neural...")
        
        # Escalar dados para RNA
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['NeuralNetwork'] = scaler
        
        # Definir espaço de busca
        if search_type == 'grid':
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'lbfgs'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        else:  # random search
            param_grid = {
                'hidden_layer_sizes': [(30,), (50,), (100,), (150,), 
                                     (50, 30), (100, 50), (150, 100),
                                     (50, 50, 30), (100, 100, 50)],
                'activation': ['relu', 'tanh', 'logistic'],
                'solver': ['adam', 'lbfgs'],
                'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'adaptive', 'invscaling'],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
        
        # Modelo base
        mlp = MLPClassifier(max_iter=1000, random_state=42)
        
        # Busca de hiperparâmetros
        if search_type == 'grid':
            search = GridSearchCV(mlp, param_grid, cv=self.cv, 
                                scoring=self.scoring, n_jobs=self.n_jobs)
        else:
            search = RandomizedSearchCV(mlp, param_grid, cv=self.cv, 
                                      scoring=self.scoring, n_jobs=self.n_jobs,
                                      n_iter=40, random_state=42)
        
        start_time = time.time()
        search.fit(X_train_scaled, y_train)
        optimization_time = time.time() - start_time
        
        self.best_params['NeuralNetwork'] = search.best_params_
        self.best_scores['NeuralNetwork'] = search.best_score_
        
        print(f"Melhores parâmetros: {search.best_params_}")
        print(f"Melhor score CV: {search.best_score_:.4f}")
        print(f"Tempo de otimização: {optimization_time:.2f}s")
        
        return search.best_estimator_
    
    def optimize_all_models(self, X_train, y_train, search_type='random'):
        """Otimiza todos os modelos"""
        optimized_models = {}
        
        # Random Forest
        try:
            optimized_models['RandomForest'] = self.optimize_random_forest(
                X_train, y_train, search_type)
        except Exception as e:
            print(f"Erro na otimização do Random Forest: {e}")
        
        # SVM
        try:
            optimized_models['SVM'] = self.optimize_svm(
                X_train, y_train, search_type)
        except Exception as e:
            print(f"Erro na otimização do SVM: {e}")
        
        # Neural Network
        try:
            optimized_models['NeuralNetwork'] = self.optimize_neural_network(
                X_train, y_train, search_type)
        except Exception as e:
            print(f"Erro na otimização da Rede Neural: {e}")
        
        return optimized_models
    
    def evaluate_optimized_model(self, model, model_name, X_test, y_test):
        """Avalia um modelo otimizado"""
        print(f"\n--- Avaliação do {model_name} Otimizado ---")
        
        # Escalar dados de teste se necessário
        if model_name in self.scalers:
            X_test_scaled = self.scalers[model_name].transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Fazer predições
        y_pred = model.predict(X_test_scaled)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Acurácia no teste: {accuracy:.4f}")
        print(f"Melhores parâmetros: {self.best_params.get(model_name, 'N/A')}")
        print(f"Score CV: {self.best_scores.get(model_name, 'N/A'):.4f}")
        
        # Relatório de classificação
        class_names = ["não vegetação", "água", "vegetação"]
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        return accuracy, y_pred
    
    def compare_optimization_results(self):
        """Compara resultados da otimização"""
        print("\n--- Resultados da Otimização ---")
        print(f"{'Modelo':<15} {'Score CV':<10} {'Parâmetros'}")
        print("-" * 80)
        
        for model_name in self.best_scores:
            score = f"{self.best_scores[model_name]:.4f}"
            params = str(self.best_params[model_name])[:50] + "..."
            print(f"{model_name:<15} {score:<10} {params}")
    
    def save_optimization_results(self, filepath):
        """Salva resultados da otimização"""
        results = {
            'best_params': self.best_params,
            'best_scores': self.best_scores,
            'scalers': self.scalers
        }
        joblib.dump(results, filepath)
        print(f"Resultados salvos em: {filepath}")
    
    def load_optimization_results(self, filepath):
        """Carrega resultados da otimização"""
        results = joblib.load(filepath)
        self.best_params = results['best_params']
        self.best_scores = results['best_scores']
        self.scalers = results['scalers']
        print(f"Resultados carregados de: {filepath}")

if __name__ == "__main__":
    # Exemplo de uso
    print("Exemplo de otimização de hiperparâmetros")
    
    # Criar dados de exemplo
    np.random.seed(42)
    X_train = np.random.rand(1000, 9)
    y_train = np.random.choice([1, 2, 4], 1000)
    X_test = np.random.rand(200, 9)
    y_test = np.random.choice([1, 2, 4], 200)
    
    # Inicializar otimizador
    optimizer = HyperparameterOptimizer(cv=3)
    
    # Otimizar modelos
    optimized_models = optimizer.optimize_all_models(X_train, y_train, 'random')
    
    # Comparar resultados
    optimizer.compare_optimization_results()
    
    # Avaliar modelos otimizados
    for model_name, model in optimized_models.items():
        optimizer.evaluate_optimized_model(model, model_name, X_test, y_test)
