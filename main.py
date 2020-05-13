import argparse
import pprint

# k-means
from kmeans import kmeans

parser = argparse.ArgumentParser(description="Analisis de datos")
parser.add_argument('--method', help='sum the integers (default: find the max)')

args = parser.parse_args()

def main():
  method = args.method
  switcher = {
    'k-means': kmeans_solver
  }

  if not method in switcher:
    print('Método %s no disponible' % method)
    return

  switcher[method]()

def kmeans_solver():
  print('Analisando por k-means')
  kmeans()

if __name__ == '__main__':
  main();