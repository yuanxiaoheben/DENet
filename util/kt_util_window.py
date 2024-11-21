import nltk
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import DataLoader,Dataset
import numpy as np
MASKED_TOKEN = "[MASK]"
PADDING = '<PAD>'
UNKNOWN =  '<UNK>'

def mask_sentence_generate_window(text, left_wing = 0, right_wing = 0):
    token_arr = word_tokenize(text)
    part_masked_list = []
    for idx,word in enumerate(token_arr):
        curr_masked_sentence = token_arr.copy()
        rest_masked_sentence = [MASKED_TOKEN for i in range(len(curr_masked_sentence))]
        start = max(idx - left_wing, 0)
        end =  min(idx + right_wing + 1, len(curr_masked_sentence))
        for i in range(start, end):
            curr_masked_sentence[i] = MASKED_TOKEN
            rest_masked_sentence[i] = token_arr[i]
        curr_masked_sentence = ' '.join(curr_masked_sentence)
        rest_masked_sentence = ' '.join(rest_masked_sentence)
        part_masked_list.append({"text": text, "masked_sentence": curr_masked_sentence, \
            "rest_masked_sentence": rest_masked_sentence, \
            "masked_word": word, "masked_index": idx, "masked_start": start, "masked_end": end})
    return part_masked_list

class MaskedWindowDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer, left, right):
        super(Dataset, self).__init__()
        processed_data = []
        label_list = []
        for row in raw_dataset:
            masked_list = mask_sentence_generate_window(row[0], left, right)
            processed_data += masked_list
            label_list += [row[1]] * len(masked_list)
        self.processed_data = processed_data
        self.label_list = label_list
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        record = self.processed_data[index]
        return record, record['masked_sentence'], record["rest_masked_sentence"], record['text'], self.label_list[index]

    def __len__(self):
        return len(self.processed_data)

class TestDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer, left, right):
        super(Dataset, self).__init__()
        processed_data = []
        label_list = []
        for row in raw_dataset:
            masked_list = mask_sentence_generate_window(row[0], left, right)
            processed_data.append(masked_list[-1]) # only last element
            label_list.append(row[1])
        self.processed_data = processed_data
        self.label_list = label_list
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        record = self.processed_data[index]
        return record, record['masked_sentence'], record["rest_masked_sentence"], record['text'], self.label_list[index]

    def __len__(self):
        return len(self.processed_data)


