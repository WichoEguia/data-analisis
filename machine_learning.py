import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def machine_learning():
  dataset = pd.read_csv('winequality-red.csv')

  bins = (2, 6.5, 8)
  group_names = ['bad', 'good']
  dataset['quality'] = pd.cut(dataset['quality'], bins = bins, labels = group_names)

  label_quality = LabelEncoder()
  dataset['quality'] = label_quality.fit_transform(dataset['quality'])
  dataset['quality'].value_counts()

  X = dataset.drop('quality', axis = 1)
  y = dataset['quality']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

  sc = StandardScaler()

  X_train = sc.fit_transform(X_train)
  X_test = sc.fit_transform(X_test)

  svc = SVC()
  svc.fit(X_train, y_train)
  pred_svc = svc.predict(X_test)

  print(classification_report(y_test, pred_svc))