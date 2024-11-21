import csv
import json
import random



BANK_TRAIN_PATH = "datasets/bank/train.csv"
BANK_TEST_PATH = "datasets/bank/test.csv"
BANK_LABEL_PATH = "datasets/bank/categories.json"


PADDING = '<PAD>'
UNKNOWN =  '<UNK>'


def load_bank_data():
    with open(BANK_LABEL_PATH, encoding='utf-8') as f:
        labels = json.load(f)
    labels_dict = {}
    for idx,val in enumerate(labels):
        labels_dict[val] = idx
    with open(BANK_TRAIN_PATH, newline='', encoding='utf-8') as f:
        train_list = []
        reader = csv.reader(f)
        heads = next(reader)
        for row in reader:
            train_list.append([row[0], labels_dict[row[1]]])
        random.shuffle(train_list)
        split_point = int(len(train_list) * 0.9)
        valid_list = train_list[split_point:]
        train_list = train_list[:split_point]
    with open(BANK_TEST_PATH, newline='', encoding='utf-8') as f:
        test_list = []
        reader = csv.reader(f)
        heads = next(reader)
        for row in reader:
            test_list.append([row[0], labels_dict[row[1]]])

    return train_list, valid_list, test_list, labels_dict
