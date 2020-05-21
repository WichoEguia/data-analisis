import argparse
import pprint

from linear_regression import *
from machine_learning import *

parser = argparse.ArgumentParser(description="Analisis de datos")
parser.add_argument('--method', help='1) k means\n2) linear regression\n3) algo más')
args = parser.parse_args()

def main():
  method = args.method
  switcher = {
    'machine-learning': machine_learning,
    'linear-regression': linear_regression
  }

  if not method in switcher:
    print('Método %s no disponible' % method)
    return

  switcher[method]()

if __name__ == '__main__':
  main();