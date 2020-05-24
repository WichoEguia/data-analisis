import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def svm():
  # Cargando y combinando datasets
  dataset_wine_red = pd.read_csv('winequality-red.csv')
  dataset_wine_white = pd.read_csv('winequality-white.csv')
  dataset = pd.concat([dataset_wine_red, dataset_wine_white])

  # Limpiando dataset
  dataset.isnull().any()

  bins = (2, 4, 7, 9)
  group_names = ['malo', 'medio', 'bueno']
  dataset['quality'] = pd.cut(dataset['quality'], bins = bins, labels = group_names)

  label_quality = LabelEncoder()
  dataset['quality'] = label_quality.fit_transform(dataset['quality'])

  X = dataset.iloc[:, :-1]
  Y = dataset.iloc[:, -1].values # Obteniendo propiedad quality

  # Dividiendo conjunto de prueba y entrenamientos
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

  # Optimizando el resultado con escalado estandar
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.fit_transform(X_test)

  # Entrenando modelo con una Máquina de Soporte Vectorial
  svc = SVC(C=1.2, gamma=0.9, kernel='rbf')
  svc.fit(X_train, Y_train)

  # Prediciendo usado una Maquina de Soporte Vectorial
  Y_pred = svc.predict(X_test)
  print(metrics.classification_report(Y_test, Y_pred))

  # Comparando resultado actual con el predicho
  df = pd.DataFrame({ 'Actual': Y_test, 'Predicho': Y_pred })
  df = df.head(25)
  df.plot(kind='bar',figsize=(10,8))
  plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
  plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
  plt.title('Actual VS Predicho')
  plt.show()

  print('\nError absoluto:', metrics.mean_absolute_error(Y_test, Y_pred))  
  print('Error cuadrado:', metrics.mean_squared_error(Y_test, Y_pred))  
  print('Error cuadratico:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

  # Presición del modelo
  accuracy = svc.score(X_test, Y_test)
  print("\nPresición: {}%".format(int(round(accuracy * 100))))
