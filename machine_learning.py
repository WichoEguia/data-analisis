import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def machine_learning():
  dataset_wine_red = pd.read_csv('winequality-red.csv')
  dataset_wine_white = pd.read_csv('winequality-white.csv')
  dataset = pd.concat([dataset_wine_red, dataset_wine_white])

  dataset.isnull().any() # Limpiando dataset

  X = dataset.iloc[:, :-1]
  Y = dataset.iloc[:, -1].values # Obteniendo propiedad quality

  # Entrenando X y Y
  # Dividiendo conjunto de prueba y entrenamiento
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

  # Optimizando el resultado con escalado estandar
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.fit_transform(X_test)

  # Prediciendo usado una Maquina de Soporte Vectorial
  svc = SVC()
  svc.fit(X_train, Y_train)
  pred_svc = svc.predict(X_test)

  df = pd.DataFrame({
    'Actual': Y_test,
    'Predicted': pred_svc
  })

  df = df.head(40)
  print('\nDiferencia entre datos de entrenamiento y los predichos (error)')
  print(df)

  print('\nError absoluto:', metrics.mean_absolute_error(Y_test, pred_svc))  
  print('Error cuadrado:', metrics.mean_squared_error(Y_test, pred_svc))  
  print('Error cuadratico:', np.sqrt(metrics.mean_squared_error(Y_test, pred_svc)))

  accuracy = svc.score(X_test, Y_test)
  print("\nPresición: {}%".format(int(round(accuracy * 100))))