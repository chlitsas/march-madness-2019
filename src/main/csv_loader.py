import csv


def load_csv_data(filename):
    with open(filename) as csv_file:
        reader = csv.DictReader(csv_file)
        return reader
    raise ValueError('load_csv_data(): error on loading the data18.')
