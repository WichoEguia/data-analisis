import argparse
import pprint

# k-means
from kmeans import *

# linear regression
from linear_regression import *

parser = argparse.ArgumentParser(description="Analisis de datos")
parser.add_argument('--method', help='1) k means\n2) linear regression\n3) algo más')

args = parser.parse_args()

def main():
  method = args.method
  switcher = {
    'k-means': kmeans_solver,
    'linear-regression': linear_regression_solver
  }

  if not method in switcher:
    print('Método %s no disponible' % method)
    return

  switcher[method]()

def kmeans_solver():
  print('Analisando por k-means')
  kmeans()

def linear_regression_solver():
  print('Analisando por regresión lineal')
  linear_regression()

if __name__ == '__main__':
  main();