import os
import csv
import json
import re
import pickle

def read_json_from_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data
 
def read_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row[0].split(' '))
    return data

def read_txt(file_path):
    data = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            data.append(line.strip())
    return data

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return

def save_dict(path, dict_file, name, add=False):
    pickle_path = os.path.join(path, name + '.pkl')
    if os.path.exists(pickle_path) and add:
        remain_file = load_pkl(pickle_path)
        os.remove(pickle_path)
    else:
        remain_file = {}
    with open(os.path.join(path, name + '.pkl'), 'wb') as file:
        remain_file.update(dict_file)
        pickle.dump(remain_file, file)
    return

def load_pkl(path):
    file = open(path, 'rb')
    data = pickle.load(file)
    return data
