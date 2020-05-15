import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def linear_regression():
  dataset = pd.read_csv('winequality-red.csv')
  dataset.isnull().any()

  X = dataset.iloc[:, :-1]
  Y = dataset.iloc[:, -1].values

  print('\nEntradas del dataset')
  print('X = ', X.values)
  print('Y = ', Y)

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

  regressor = LinearRegression()
  regressor.fit(X_train,Y_train)

  y_pred = regressor.predict(X_test)

  df = pd.DataFrame({
    'Actual': Y_test,
    'Predicted': y_pred
  })

  df1 = df.head(25)
  print('\nDiferencia entre datos de entrenamiento y los predichos (error)')
  print(df1)

  print()
  print('Error absoluto:', metrics.mean_absolute_error(Y_test, y_pred))  
  print('Error cuadrado:', metrics.mean_squared_error(Y_test, y_pred))  
  print('Error cuadratico:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
