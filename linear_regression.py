import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

'''
metodo: Regresión lineal
autor: José Luis Eguía Téllez
matricula: 1791916
'''
def linear_regression():
  # Cargando y combinando datasets
  dataset_wine_red = pd.read_csv('winequality-red.csv')
  dataset_wine_white = pd.read_csv('winequality-white.csv')
  dataset = pd.concat([dataset_wine_red, dataset_wine_white])

  # Limpiando dataset
  dataset.isnull().any()

  X = dataset.iloc[:, :-1]
  Y = dataset.iloc[:, -1].values # Obteniendo propiedad quality

  # Dividiendo conjunto de prueba y entrenamiento
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

  # Usando los conjuntos de entrenamiento para entrenar el modelo
  regressor = linear_model.LinearRegression()
  regressor.fit(X_train,Y_train)

  # Prediciendo usando el conjunto de prueba
  Y_pred = regressor.predict(X_test)

  # Obteniendo relacion de cada propiedad con quality
  coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
  print(coeff_df)

  # Comparando resultado actual con el predicho
  df = pd.DataFrame({ 'Actual': Y_test, 'Predicho': Y_pred })
  df = df.head(25)
  df.plot(kind='bar',figsize=(10,8))
  plt.grid(which='major', linewidth='0.5', color='green')
  plt.grid(which='minor', linewidth='0.5', color='black')
  plt.title('Actual VS Predicho')
  plt.show()

  print('\nError absoluto:', metrics.mean_absolute_error(Y_test, Y_pred))  
  print('Error cuadrado:', metrics.mean_squared_error(Y_test, Y_pred))  
  print('Error cuadratico:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

  # Presición del modelo
  accuracy = regressor.score(X_test, Y_test)
  print("\nPresición: {}%".format(int(round(accuracy * 100))))

if __name__ == '__main__':
  linear_regression()