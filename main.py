import argparse
import csv
import pprint

parser = argparse.ArgumentParser(description="Analisis de datos")
parser.add_argument('--method', help='sum the integers (default: find the max)')

args = parser.parse_args()

def main():
  csv_data = get_csv_data()

  method = args.method
  switcher = {
    'k-means': kmeans_solver
  }

  if not method in switcher:
    print('Método %s no disponible' % method)
    return

  switcher[method]()

def get_csv_data():
  reader = csv.DictReader(open('vgsales.csv', 'rt'))
  dict_list = []

  for line in reader:
    dict_list.append(line)

  return dict_list

def kmeans_solver():
  print('k-means solve')

if __name__ == '__main__':
  main();