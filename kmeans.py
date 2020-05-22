#Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

def kmeans():
    # Cargamos los datos con panda
    dataset = pd.read_csv("winequality-red.csv")

    X = dataset.iloc[:, [3, 6]].values

    # Metodo del codo para averiguar el numero de clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init="k-means++", max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss) 
    plt.title("Metodo de codo")
    plt.xlabel("Numero de clusteres")
    plt.ylabel("WCSS(k)")
    plt.show()

    # Aplicar metodo KMEANS para segmentar el dataset
    kmeans = KMeans(n_clusters = 3, init="k-means++", max_iter = 300, random_state = 0)
    y_kmeans = kmeans.fit_predict(X) 

    # Vizualizacion de los clusters 
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = "red", label = "cluster 1")
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c= "blue", label = "cluster 2")
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = "green", label = "cluster 3")
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = "yellow", label = "Baricentro")
    plt.title("Cluster de vinos")
    plt.xlabel("Total de azucar (g)")
    plt.ylabel("Dioxido de sulfuro")
    plt.legend()
    plt.show()