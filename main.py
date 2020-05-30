import argparse
import pprint

# Metodos de solucion
from linear_regression import linear_regression
from svm import svm
from kmeans import kmeans

parser = argparse.ArgumentParser(description="Analisis de datos")
parser.add_argument('--method', help='1) linear regression\n2) Máquina de Soporte Vectorial\n3) K Means')
args = parser.parse_args()

def main():
  method = args.method
  switcher = {
    'linear-regression': linear_regression,
    'svm': svm,
    'kmeans': kmeans
  }

  if not method in switcher:
    if method != None:
      print('Método %s no disponible' % method)
      return
    else:
      print('Falta parametro --method con alguno de los siguientes valores:')
      print('1) linear regression\n2) Máquina de Soporte Vectorial\n3) K Means\n')
      print('ej:\npython3 main.py --method svm')
      return

  switcher[method]()

if __name__ == '__main__':
  main();