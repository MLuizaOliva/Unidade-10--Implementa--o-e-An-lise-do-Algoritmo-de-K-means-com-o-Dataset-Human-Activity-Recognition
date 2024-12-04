import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path

# O diretório do dataset deve estar na pasta raiz do programa, com o nome "UCI HAR Dataset"
# A constante é padronizada em Path para maximizar compatibilidade.
PASTA_BASE = Path("UCI HAR Dataset")
ARQUIVO_DADOS = PASTA_BASE / "train" / "X_train.txt"
ARQUIVO_FEATURES = PASTA_BASE / "features.txt"

def carregar_dados():
    features = pd.read_csv(ARQUIVO_FEATURES, sep=' ', header=None, names=['id', 'nome'])
    dados = pd.read_csv(ARQUIVO_DADOS, sep='\s+', header=None)
    dados.columns = features['nome']
    return dados

def analise_exploratoria(dados):
    # Estatísticas descritivas são armazenadas aqui.
    estatisticas = dados.describe()
    
    # Matriz de correlação (amostragem das 20 primeiras features)
    plt.figure(figsize=(12, 8))
    sns.heatmap(dados.iloc[:, :20].corr(), cmap='coolwarm', center=0)
    plt.title('Matriz de Correlação (20 primeiras features)')
    plt.tight_layout()
    plt.show()
    
    return estatisticas

# Redução de dimensionalidade utilizando técnica PCA
def aplicar_pca(dados_normalizados, n_componentes=3):
    pca = PCA(n_components=n_componentes)
    dados_pca = pca.fit_transform(dados_normalizados)
    variancia_explicada = pca.explained_variance_ratio_
    
    return dados_pca, variancia_explicada, pca

# Método do cotovelo para encontrar K ideal
def metodo_cotovelo(dados_normalizados, range_k=range(1, 11)):
    inercias = []
    silhouette_scores = []
    
    for k in range_k:
        if k > 1:  # Silhouette score precisa de pelo menos 2 clusters devido ao cálculo mínima
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
            kmeans.fit(dados_normalizados)
            inercias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(dados_normalizados, kmeans.labels_))
    
    # Plotagem do método do cotovelo
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(list(range_k)[1:], inercias, 'bo-')
    ax1.set_xlabel('Número de Clusters (K)')
    ax1.set_ylabel('Inércia')
    ax1.set_title('Método do Cotovelo')
    
    ax2.plot(list(range_k)[1:], silhouette_scores, 'ro-')
    ax2.set_xlabel('Número de Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Análise Silhouette')
    
    plt.tight_layout()
    plt.show()
    
    return inercias, silhouette_scores

def realizar_clustering(dados_pca, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(dados_pca)
    
    # Visualização 2D dos clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(dados_pca[:, 0], 
                          dados_pca[:, 1], 
                          c=clusters, 
                          cmap='viridis')
    plt.colorbar(scatter)
    plt.xlabel('Componente 1 (X)')
    plt.ylabel('Componente 2 (Y)')
    plt.title(f'Clusters K-means em 2D (K={n_clusters})')
    plt.show()
    
    # Visualização 3D dos clusters
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(dados_pca[:, 0], 
                         dados_pca[:, 1], 
                         dados_pca[:, 2],
                         c=clusters,
                         cmap='viridis')
    
    plt.colorbar(scatter)
    ax.set_xlabel('Componente 1 (X)')
    ax.set_ylabel('Componente 2 (Y)')
    ax.set_zlabel('Componente 3 (Z)')
    plt.title(f'Clusters K-means 3D (K={n_clusters})')
    plt.show()
    
    return clusters, kmeans

def main():
    dados = carregar_dados()
    
    estatisticas = analise_exploratoria(dados)
    
    # Normalização dos dados
    scaler = StandardScaler()
    dados_normalizados = scaler.fit_transform(dados)
    
    # Aplicação do PCA
    dados_pca, variancia_explicada, pca = aplicar_pca(dados_normalizados)
    print(f"Variância explicada pelas componentes: {variancia_explicada.cumsum()}")
    
    # Encontrar K ideal
    inercias, silhouette_scores = metodo_cotovelo(dados_normalizados)
    
    # Realizar clustering com K=5 (exemplo)
    clusters, modelo_kmeans = realizar_clustering(dados_pca, n_clusters=5)
    
    return dados, dados_pca, clusters, modelo_kmeans

if __name__ == "__main__":
    dados, dados_pca, clusters, modelo_kmeans = main()