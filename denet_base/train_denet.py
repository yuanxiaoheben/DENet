import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW, get_scheduler
from model.model_denet import DENet,AverageMeter
import util.kt_util_window as kt_util_window
from util.data_util import load_bank_data
from torch.utils.data import DataLoader
import os
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import json
import argparse
""" Hyperparameter"""
parser = argparse.ArgumentParser()
# data parameters
parser.add_argument('--num_epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=50, help='batch size')
parser.add_argument('--learning_rate', type=float, default=3e-5, help='learning_rate')
parser.add_argument('--dropout_prob', type=float, default=0.5, help='dropout prob')
parser.add_argument('--save_path', type=str, default='./SAVED', help='save folder')
parser.add_argument('--out_dim', type=int, default=77, help='number of class')
parser.add_argument('--lambda_epochs', type=int, default=50, help='lambda epochs')
parser.add_argument('--pretrain_path', type=str, default="/bert-base-uncase/", help='path for encoding')
parser.add_argument('--left', type=int, default=1, help='left wing for masking')
parser.add_argument('--right', type=int, default=1, help='right wing for masking')
configs = parser.parse_args()
num_epochs = configs.num_epochs
learning_rate=configs.learning_rate
batch_size=configs.batch_size
bert_out_dim = 768
dropout_prob = configs.dropout_prob
out_dim = configs.out_dim
lambda_epochs  = configs.lambda_epochs
view_number = 3
LEFT_WING = configs.left
RIGHT_WING = configs.right



LOCAL_BERT_PATH = configs.pretrain_path
SAVED_PATH = configs.save_path

# load data
train_data, valid_data, test_data,labels_dict = load_bank_data()

tokenizer = BertTokenizer.from_pretrained(LOCAL_BERT_PATH)
bert_model = BertModel.from_pretrained(LOCAL_BERT_PATH)


def fix_data_collator(data):
    records, masked, masked_rest, raw, label = zip(*data)
    raw_inputs = tokenizer(raw, padding="longest", return_tensors="pt")
    rest_inputs = tokenizer(masked_rest, padding="longest", return_tensors="pt")
    masked_inputs = tokenizer(masked, padding="longest", return_tensors="pt")
    labels = torch.tensor(label, dtype=torch.int64)
    return records, raw_inputs, masked_inputs, rest_inputs, labels
def loader_generate(raw_data, test=False):
    if not test:
        data = kt_util_window.MaskedWindowDataset(raw_data, tokenizer, LEFT_WING, RIGHT_WING)
    else:
        data = kt_util_window.TestDataset(raw_data, tokenizer, LEFT_WING, RIGHT_WING)
    curr_dataloader = DataLoader(
        data, shuffle=True, batch_size=batch_size, collate_fn=fix_data_collator
    )
    return curr_dataloader

train_dataloader = loader_generate(train_data)
valid_dataloader = loader_generate(valid_data)
test_dataloader = loader_generate(test_data, True)
print("Number of Train: %i, Valid: %i, Test: %i"\
     % (len(train_dataloader), len(valid_dataloader), len(test_dataloader)))



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
bert_model.to(device)
d_model = DENet(out_dim, view_number, [[bert_out_dim], [bert_out_dim], [bert_out_dim]], bert_model, lambda_epochs)

d_model.to(device)

num_training_steps = num_epochs * len(train_dataloader)
progress_bar = tqdm(range(num_training_steps))

loss = torch.nn.CrossEntropyLoss()

optimizer = AdamW(d_model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
def views_construct(batch):
    _, raw_inputs, masked_inputs, rest_inputs, labels = batch
    raw_inputs = {k: v.to(device) for k, v in raw_inputs.items()}
    rest_inputs = {k: v.to(device) for k, v in rest_inputs.items()}
    masked_inputs = {k: v.to(device) for k, v in masked_inputs.items()}
    labels = labels.to(device)
    views = {
        0:masked_inputs,
        1:raw_inputs,
        2:rest_inputs,
    }
    return views, labels

def model_eval(eval_dataloader, model):
    loss_meter = AverageMeter()
    model.eval()
    true_list = []
    pred_list = []
    for batch in eval_dataloader:
        views, labels = views_construct(batch)
        with torch.no_grad():
            evidences, evidence_a, loss  = model(views, labels, num_epochs)
            loss_meter.update(loss.item())
        predictions = torch.argmax(evidence_a, dim=-1)
        pred_list += predictions.cpu().tolist()
        true_list += labels.cpu().tolist()
    return accuracy_score(true_list, pred_list)
   
def model_train(train_dataloader, model): 
    max_acc = -1.0    
    best_model = None
    for epoch in range(num_epochs):
        model.train()
        loss_meter = AverageMeter()
        for batch in train_dataloader:
            views, labels = views_construct(batch)
            evidences, evidence_a, loss = model(views, labels, epoch)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            loss_meter.update(loss.item())
        valid_acc = model_eval(valid_dataloader, model)
        train_acc = model_eval(train_dataloader, model)
        print("Epoch %i, Train Accuracy: %f, Validation Accuracy: %f" % ((epoch+1), train_acc, valid_acc))
        if valid_acc > max_acc:
            torch.save(model, os.path.join(SAVED_PATH, "model.pkl"))
            max_acc = valid_acc
    return max_acc
best_valid = model_train(train_dataloader, d_model)
print("Best Valid Accuracy: %f" % (best_valid))
best_model = torch.load(os.path.join(SAVED_PATH, "model.pkl"))
print("Test Accuracy: %f" % (model_eval(test_dataloader, best_model)))
