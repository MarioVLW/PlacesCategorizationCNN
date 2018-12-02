import csv
from pathlib import Path

class RecorderCSV:

    def __init__(self, file_name):
        self.file_name = file_name
        self.file = file_name

    def create_file(self, column_names):
        index_file = 0
        self.file = self.file_name
        file = self.file + str(index_file) + ".txt"
        while Path(file).is_file():
            index_file = index_file + 1
            file = self.file + str(index_file) + ".txt"

        with open(file, mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(column_names)

        self.file = file

    def append_into_file(self, data):
        with open(self.file, mode='a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(data)