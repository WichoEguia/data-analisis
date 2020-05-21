import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split

def linear_regression():
  dataset_wine_red = pd.read_csv('winequality-red.csv')
  dataset_wine_white = pd.read_csv('winequality-white.csv')
  dataset = pd.concat([dataset_wine_red, dataset_wine_white])

  dataset.isnull().any() # Limpiando dataset

  X = dataset.iloc[:, :-1]
  Y = dataset.iloc[:, -1].values # Obteniendo propiedad quality

  # Entrenando X y Y
  # Dividiendo conjunto de prueba y entrenamiento
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

  # Usando los conjuntos de entrenamiento para entrenar el modelo
  reg = linear_model.LinearRegression()
  reg.fit(X_train,Y_train)

  # Entrenando conjunto de prueba
  y_test_pred = reg.predict(X_test)

  df = pd.DataFrame({
    'Actual': Y_test,
    'Predicted': y_test_pred
  })

  df = df.head(40)
  print('\nDiferencia entre datos de entrenamiento y los predichos (error)')
  print(df)

  print('\nError absoluto:', metrics.mean_absolute_error(Y_test, y_test_pred))  
  print('Error cuadrado:', metrics.mean_squared_error(Y_test, y_test_pred))  
  print('Error cuadratico:', np.sqrt(metrics.mean_squared_error(Y_test, y_test_pred)))

  accuracy = reg.score(X_test, Y_test)
  print("\nPresición: {}%".format(int(round(accuracy * 100))))