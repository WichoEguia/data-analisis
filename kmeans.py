import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def kmeans():
  data = pd.read_csv('metacritic_games.csv')
  dataframe = pd.DataFrame(data)
  x = dataframe['metascore'].values
  y = dataframe['user_score'].values

  print("METASCORE ", dataframe['metascore'].mean())
  print("USER SCORE ", dataframe['user_score'].mean())

  X = np.array(list(zip(x, y)))
  print(X)

  kmeans = KMeans(n_clusters=4)
  kmeans = kmeans.fit(X)
  labels = kmeans.predict(X)
  centroids = kmeans.cluster_centers_

  colors = ["m.", "r.", "c.", "y.", "b."]

  for i in range(len(X)):
    print("Coord: ", X[i], "Label: ", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
  
  plt.scatter(centroids[:,0], centroids[:,1], marker="x", s=159, linewidths=5, zorder=10)
  plt.show()