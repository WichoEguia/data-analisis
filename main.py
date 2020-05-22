import argparse
import pprint

# Metodos de solucion
from linear_regression import linear_regression
from machine_learning import machine_learning
from kmeans import kmeans

parser = argparse.ArgumentParser(description="Analisis de datos")
parser.add_argument('--method', help='1) k means\n2) linear regression\n3) algo más')
args = parser.parse_args()

def main():
  method = args.method
  switcher = {
    'linear-regression': linear_regression,
    'machine-learning': machine_learning,
    'kmeans': kmeans
  }

  if not method in switcher:
    print('Método %s no disponible' % method)
    return

  switcher[method]()

if __name__ == '__main__':
  main();