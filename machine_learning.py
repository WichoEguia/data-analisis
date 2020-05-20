import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def machine_learning():
  dataset_wine_red = pd.read_csv('winequality-red.csv')
  dataset_wine_white = pd.read_csv('winequality-white.csv')
  dataset = pd.concat([dataset_wine_red, dataset_wine_white])

  dataset.isnull().any() # Limpiando dataset

  print('\nValores del campo quality:')
  print(dataset['quality'].value_counts()) # Los valores quality tienen un rango de 2 a 8

  # Clasificando la calidad del vino [quality] en malo y bueno según el rango
  # reviews = []
  # for i in dataset['quality']:
  #     if i >= 1 and i <= 3:
  #         reviews.append('1')
  #     elif i >= 4 and i <= 7:
  #         reviews.append('2')
  #     elif i >= 8 and i <= 10:
  #         reviews.append('3')
  # dataset['quality'] = reviews

  # Conviertiendo categorias "buena" y "mala" en valores numericos
  # -> 1 = malo
  # -> 2 = medio
  # -> 3 = bueno
  # label_quality = LabelEncoder()
  # dataset['quality'] = label_quality.fit_transform(dataset['quality'])

  X = dataset.iloc[:, :-1]
  Y = dataset.iloc[:, -1].values # Obteniendo propiedad quality

  # Entrenando X y Y
  # Dividiendo conjunto de prueba y entrenamiento
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

  # Optimizando el resultado con escalado estandar
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.fit_transform(X_test)

  # Prediciendo usado una Maquina de Soporte Vectorial
  svc = SVC()
  svc.fit(X_train, Y_train)
  pred_svc = svc.predict(X_test)

  print()
  print(metrics.classification_report(Y_test, pred_svc))