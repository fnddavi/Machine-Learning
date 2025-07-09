"""
Projeto Principal - Classificação de Uso do Solo com Imagens Sentinel-2

Este é o script principal que executa todo o pipeline de classificação:
1. Carregamento e pré-processamento dos dados
2. Análise exploratória
3. Treinamento dos modelos
4. Otimização de hiperparâmetros
5. Avaliação e comparação dos resultados
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import SentinelDataLoader
from ml_models import MLModels
from hyperparameter_optimization import HyperparameterOptimizer
import time
import warnings
warnings.filterwarnings('ignore')

def main():
    """Função principal do projeto"""
    print("=" * 60)
    print("PROJETO DE CLASSIFICAÇÃO DE USO DO SOLO - SENTINEL-2")
    print("=" * 60)
    
    # Configurações
    N_SAMPLES = 50000  # Número de amostras para treinamento
    OPTIMIZE_HYPERPARAMS = True  # Se deve otimizar hiperparâmetros
    SAVE_RESULTS = True  # Se deve salvar resultados
    
    # Criar diretório de resultados
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # ============================================
    # 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
    # ============================================
    print("\n1. CARREGANDO E PRÉ-PROCESSANDO DADOS...")
    
    # Inicializar o carregador de dados
    loader = SentinelDataLoader(data_dir="ImagensSentinel-2")
    
    # Carregar todas as bandas
    print("Carregando bandas espectrais...")
    loader.load_all_bands()
    
    # Carregar SCL (Scene Classification Layer)
    print("Carregando SCL (Scene Classification Layer)...")
    loader.load_scl()
    
    # Preparar dados para ML
    print(f"Preparando dados para ML ({N_SAMPLES} amostras)...")
    features, targets = loader.sample_data(n_samples=N_SAMPLES)
    
    # Dividir dados em treino, validação e teste
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
        features, targets, test_size=0.3, val_size=0.1
    )
    
    print(f"\nTamanhos dos conjuntos:")
    print(f"  Treino: {X_train.shape[0]} amostras")
    print(f"  Validação: {X_val.shape[0]} amostras")
    print(f"  Teste: {X_test.shape[0]} amostras")
    print(f"  Features: {X_train.shape[1]} bandas espectrais")
    
    # Mostrar distribuição das classes
    print(f"\nDistribuição das classes no conjunto de treino:")
    train_dist = loader.get_class_distribution(y_train)
    total_samples = len(y_train)
    for classe, count in train_dist.items():
        percentage = (count / total_samples) * 100
        print(f"  {classe}: {count:,} amostras ({percentage:.1f}%)")
    
    # ============================================
    # 2. ANÁLISE EXPLORATÓRIA
    # ============================================
    print("\n2. ANÁLISE EXPLORATÓRIA DOS DADOS...")
    
    # Visualizar imagens (opcional - pode ser demorado)
    visualize_images = input("Deseja visualizar as imagens? (s/n): ").lower().strip() == 's'
    
    if visualize_images:
        print("Visualizando SCL...")
        loader.visualize_scl()
        
        print("Visualizando composição RGB...")
        loader.visualize_rgb()
    
    # ============================================
    # 3. TREINAMENTO DOS MODELOS BASE
    # ============================================
    print("\n3. TREINANDO MODELOS BASE...")
    
    # Inicializar classe de modelos
    ml_models = MLModels()
    
    # Adicionar modelos com parâmetros padrão
    print("Configurando modelos...")
    ml_models.add_random_forest(n_estimators=100, max_depth=20)
    ml_models.add_svm(C=1.0, kernel='rbf')
    ml_models.add_neural_network(hidden_layer_sizes=(100, 50), max_iter=500)
    
    # Treinar todos os modelos
    print("Treinando modelos...")
    ml_models.train_all_models(X_train, y_train, X_val, y_val)
    
    # ============================================
    # 4. OTIMIZAÇÃO DE HIPERPARÂMETROS
    # ============================================
    if OPTIMIZE_HYPERPARAMS:
        print("\n4. OTIMIZANDO HIPERPARÂMETROS...")
        
        # Usar amostra menor para otimização (mais rápido)
        sample_size = min(10000, len(X_train))
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_train_sample = X_train[indices]
        y_train_sample = y_train[indices]
        
        # Inicializar otimizador
        optimizer = HyperparameterOptimizer(cv=3, n_jobs=-1)
        
        # Otimizar todos os modelos
        optimized_models = optimizer.optimize_all_models(
            X_train_sample, y_train_sample, search_type='random'
        )
        
        # Comparar resultados da otimização
        optimizer.compare_optimization_results()
        
        # Treinar modelos otimizados no conjunto completo
        print("\nTreinando modelos otimizados no conjunto completo...")
        ml_models_optimized = MLModels()
        
        for model_name, model in optimized_models.items():
            # Obter parâmetros otimizados
            best_params = optimizer.best_params[model_name]
            
            # Adicionar modelo com parâmetros otimizados
            if model_name == 'RandomForest':
                ml_models_optimized.add_random_forest(**best_params)
            elif model_name == 'SVM':
                ml_models_optimized.add_svm(**best_params)
            elif model_name == 'NeuralNetwork':
                ml_models_optimized.add_neural_network(**best_params)
        
        # Treinar modelos otimizados
        ml_models_optimized.train_all_models(X_train, y_train, X_val, y_val)
        
        # Usar modelos otimizados para avaliação final
        final_models = ml_models_optimized
        
        if SAVE_RESULTS:
            optimizer.save_optimization_results(
                os.path.join(results_dir, "optimization_results.pkl")
            )
    else:
        final_models = ml_models
    
    # ============================================
    # 5. AVALIAÇÃO FINAL DOS MODELOS
    # ============================================
    print("\n5. AVALIAÇÃO FINAL DOS MODELOS...")
    
    # Avaliar todos os modelos no conjunto de teste
    test_results = {}
    for model_name in final_models.models.keys():
        print(f"\nAvaliando {model_name}...")
        accuracy, cm = final_models.evaluate_model(
            model_name, X_test, y_test, show_plots=True
        )
        test_results[model_name] = {
            'accuracy': accuracy,
            'confusion_matrix': cm
        }
    
    # ============================================
    # 6. COMPARAÇÃO E RESULTADOS FINAIS
    # ============================================
    print("\n6. COMPARAÇÃO E RESULTADOS FINAIS...")
    
    # Comparar modelos
    final_models.compare_models()
    
    # Encontrar melhor modelo
    best_model_name = final_models.get_best_model('test_accuracy')
    if best_model_name:
        print(f"\nMelhor modelo: {best_model_name}")
        print(f"Acurácia no teste: {test_results[best_model_name]['accuracy']:.4f}")
        
        # Mostrar importância das features (se disponível)
        feature_names = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]
        if hasattr(final_models.models[best_model_name], 'feature_importances_'):
            print("\nImportância das features:")
            final_models.plot_feature_importance(best_model_name, feature_names)
    
    # ============================================
    # 7. SALVAR RESULTADOS
    # ============================================
    if SAVE_RESULTS:
        print("\n7. SALVANDO RESULTADOS...")
        
        # Salvar dados processados
        loader.save_processed_data(
            features, targets, 
            os.path.join(results_dir, "processed_data.pkl")
        )
        
        # Salvar melhor modelo
        if best_model_name:
            final_models.save_model(
                best_model_name, 
                os.path.join(results_dir, f"best_model_{best_model_name}.pkl")
            )
        
        # Salvar todos os modelos
        for model_name in final_models.models.keys():
            final_models.save_model(
                model_name,
                os.path.join(results_dir, f"model_{model_name}.pkl")
            )
        
        print("Resultados salvos no diretório 'results/'")
    
    # ============================================
    # 8. RESUMO FINAL
    # ============================================
    print("\n" + "=" * 60)
    print("RESUMO FINAL DO PROJETO")
    print("=" * 60)
    
    print(f"Dados utilizados:")
    print(f"  • {N_SAMPLES:,} amostras de pixels")
    print(f"  • {X_train.shape[1]} bandas espectrais")
    print(f"  • 3 classes: vegetação, água, não vegetação")
    
    print(f"\nModelos treinados:")
    for model_name in final_models.models.keys():
        accuracy = test_results[model_name]['accuracy']
        print(f"  • {model_name}: {accuracy:.4f} de acurácia")
    
    if best_model_name:
        print(f"\nMelhor modelo: {best_model_name}")
        print(f"Acurácia final: {test_results[best_model_name]['accuracy']:.4f}")
    
    print(f"\nHiperparâmetros otimizados: {'Sim' if OPTIMIZE_HYPERPARAMS else 'Não'}")
    print(f"Resultados salvos: {'Sim' if SAVE_RESULTS else 'Não'}")
    
    print("\n" + "=" * 60)
    print("PROJETO CONCLUÍDO COM SUCESSO!")
    print("=" * 60)

if __name__ == "__main__":
    # Executar projeto
    start_time = time.time()
    main()
    end_time = time.time()
    
    print(f"\nTempo total de execução: {end_time - start_time:.2f} segundos")
