import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,confusion_matrix
import torch
import gc
import pickle
from transformers import TrainingArguments, Trainer,RobertaForSequenceClassification,RobertaTokenizer
from transformers import EarlyStoppingCallback
from sklearn.utils import class_weight
import ray
import argparse
from tqdm import tqdm
import os
from os.path import join, exists
import re
import torch.nn as nn
from collections import Counter
import random
import pickle
import json
import hashlib

from collections import defaultdict

ray.init(_plasma_directory="/tmp")
torch.cuda.empty_cache()
gc.collect()

def getMD5(s):
    hl = hashlib.md5()
    hl.update(s.encode("utf-8"))
    return hl.hexdigest()

def removeComments(text):
    """ remove c-style comments.
        text: blob of text with comments (can include newlines)
        returns: text with comments removed
    """
    pattern = r"""
                            ##  --------- COMMENT ---------
           //.*?$           ##  Start of // .... comment
         |                  ##
           /\*              ##  Start of /* ... */ comment
           [^*]*\*+         ##  Non-* followed by 1-or-more *'s
           (                ##
             [^/*][^*]*\*+  ##
           )*               ##  0-or-more things which don't start with /
                            ##    but do end with '*'
           /                ##  End of /* ... */ comment
         |                  ##  -OR-  various things which aren't comments:
           (                ##
                            ##  ------ " ... " STRING ------
             "              ##  Start of " ... " string
             (              ##
               \\.          ##  Escaped char
             |              ##  -OR-
               [^"\\]       ##  Non "\ characters
             )*             ##
             "              ##  End of " ... " string
           |                ##  -OR-
                            ##
                            ##  ------ ' ... ' STRING ------
             '              ##  Start of ' ... ' string
             (              ##
               \\.          ##  Escaped char
             |              ##  -OR-
               [^'\\]       ##  Non '\ characters
             )*             ##
             '              ##  End of ' ... ' string
           |                ##  -OR-
                            ##
                            ##  ------ ANYTHING ELSE -------
             .              ##  Anything other char
             [^/"'\\]*      ##  Chars which doesn't start a comment, string
           )                ##    or escape
    """
    regex = re.compile(pattern, re.VERBOSE|re.MULTILINE|re.DOTALL)
    noncomments = [m.group(2) for m in regex.finditer(text) if m.group(2)]
    noncomments="".join(noncomments)
    return noncomments

    
class Dataset(torch.utils.data.Dataset):    
    def __init__(self, encodings, labels=None):          
        self.encodings = encodings        
        self.labels = labels
     
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item     
    def __len__(self):
        return len(self.encodings["input_ids"])
    
def create_token_chunks_vulnerable_samples(code_statements,all_special_ids,vulnerable_line_numbers):
    i=0
    samples,labels=[],[]
    while i<len(code_statements):
        tokens=[]
        k=i
        while i<len(code_statements):
            modified_input_ids=[]
            for j in range(len(code_statements[i])):
                if code_statements[i][j] not in all_special_ids:
                    modified_input_ids.append(code_statements[i][j])

            if len(tokens)+len(modified_input_ids)<=510:
                tokens.extend(modified_input_ids)
            else:
                break
            i+=1
        flag=False
        samples.append(removeComments(tokenizer.decode(tokens)))
        for line_number in vulnerable_line_numbers:
            if int(line_number) in range(k,i):
                flag=True
                break
        if flag:
            labels.append(1)
        else:
            labels.append(0)
    return samples,labels

@ray.remote 
def read_file_label(sample,tokenizer):
    label=1 if sample["vulnerable_line_numbers"] else 0
    all_special_ids=tokenizer.all_special_ids
    source_code=sample["processed_func"].split("\n")
    inputs,labels=[],[]
    if label==1:
        samples,mixed_labels=create_token_chunks_vulnerable_samples(tokenizer(source_code)["input_ids"],all_special_ids,sample["vulnerable_line_numbers"].split(","))
        inputs.extend(samples)
        labels.extend(mixed_labels)
    else:
        input_id=tokenizer(removeComments("".join(source_code)))["input_ids"]
        modified_input_ids=[]
        for i in range(len(input_id)):
            if input_id[i] not in all_special_ids:
                modified_input_ids.append(input_id[i])
        for i in range(0,len(modified_input_ids),510):
            inputs.append(tokenizer.decode(modified_input_ids[i:i+510]))
            labels.append(label)
    
    return inputs,labels
    
    
    
def prepare_dataset(samples,tokenizer):
        process_examples=[]
        for sample_i,sample in tqdm(samples.iterrows(),total=len(samples)):
            process_examples.append(read_file_label.remote(sample,tokenizer))
    
        result=ray.get(process_examples)
        source_codes,labels=[],[]
        for source_code,label in result:
            source_codes.extend(source_code)
            labels.extend(label)
        return source_codes,labels



def train_filter(source_codes,labels):
    final_samples=defaultdict(dict)
    modified_source_codes,modified_labels=[],[]
    for i,_ in tqdm(enumerate(labels),total=len(labels)):
            hash1=getMD5("".join(source_codes[i].split()))
            if hash1 not in final_samples:
                final_samples[hash1]["source_code"]=source_codes[i]
                final_samples[hash1]["label"]=labels[i]
            else:
                old_label=final_samples[hash1]["label"]
                if (old_label!=-1 and old_label!=labels[i]) or (old_label==-1):
                    final_samples[hash1]["label"]=-1

    
    for i in final_samples:
        if final_samples[i]["label"]!=-1:
            modified_source_codes.append(final_samples[i]["source_code"])
            modified_labels.append(final_samples[i]["label"])
            
    return [modified_source_codes,modified_labels]

def test_filter(source_codes,labels):
    final_samples=defaultdict(dict)
    modified_source_codes,modified_labels=[],[]
    for i,_ in tqdm(enumerate(labels),total=len(labels)):
            hash1=getMD5("".join(source_codes[i].split()))
            if hash1 not in final_samples:
                final_samples[hash1]["source_code"]=[source_codes[i]]
                final_samples[hash1]["label"]=[labels[i]]
            else:
                old_label=final_samples[hash1]["label"]
                if (old_label!=-1 and old_label!=labels[i]) or (old_label==-1):
                    final_samples[hash1]["label"]=-1
                    print(hash1)
                    collisions+=1
                else:
                    final_samples[hash1]["source_code"].append(source_codes[i])
                    final_samples[hash1]["label"].append(labels[i])
                    duplicates+=1
    
    for i in final_samples:
        if final_samples[i]["label"]!=-1:
            modified_source_codes.extend(final_samples[i]["source_code"])
            modified_labels.extend(final_samples[i]["label"])
    return [modified_source_codes,modified_labels]



def compute_metrics(p):    
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    confusion_matrix1 = confusion_matrix(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,"confusion_matrix":confusion_matrix1.tolist()
           }


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_csv_path", type=str, required=True,
                    help="The input training data file (a csv file).")
parser.add_argument("--tokenizer_name",  type=str, required=True,
                    help="tokenizer_name")
parser.add_argument("--model_name", type=str, required=True,
                    help="model path")
parser.add_argument("--output_dir", type=str, required=True,
                    help="output_dir")
parser.add_argument("--per_device_train_batch_size", type=int, required=True,
                    help="per_device_train_batch_size")
parser.add_argument("--per_device_eval_batch_size", type=int, required=True,
                    help="per_device_eval_batch_size")
parser.add_argument("--num_train_epochs", type=int, required=True,
                    help="num_train_epochs")
parser.add_argument("--dataset_path", type=str, required=True,
                    help="dataset_path")
parser.add_argument("--prepare_dataset", default=False,action='store_true',
                    help="prepare_dataset")
parser.add_argument("--train",default=False,action='store_true',
                    help="train")
parser.add_argument("--val_predict", default=False,action='store_true',
                    help="val_predict")
parser.add_argument("--test_predict", default=False,action='store_true',
                    help="test_predict")
parser.add_argument("--train_predict", default=False,action='store_true',
                    help="train_predict")

args = parser.parse_args()

print("Arguments", args)

tokenizer_name=args.tokenizer_name
model_name=args.model_name
dataset_csv_path=args.dataset_csv_path
output_dir=args.output_dir
dataset_path= args.dataset_path

if not exists(dataset_path):
        os.makedirs(dataset_path)

if not exists(output_dir):
        os.makedirs(output_dir)
        



project_df=pd.read_csv(dataset_csv_path)
project_df["vulnerable_line_numbers"]=project_df["vulnerable_line_numbers"].fillna("")
train_val=project_df[project_df["dataset_type"]=="train_val"]
test_data=project_df[project_df["dataset_type"]=="test"]



if args.prepare_dataset:
    print("Preparing Dataset...")
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    source_code,labels=prepare_dataset(train_val,tokenizer)

    filtered_source_code,filtered_labels=train_filter(source_code,labels)
    
    train_source_code, val_source_code,train_labels,val_labels = train_test_split(filtered_source_code,filtered_labels, test_size=0.1)

    X_chunked_train_tokenized = tokenizer(train_source_code,padding=True, truncation=True, max_length=512)
    X_chunked_val_tokenized = tokenizer(val_source_code,padding=True, truncation=True, max_length=512)

    train_dataset = Dataset(X_chunked_train_tokenized, train_labels)
    val_dataset = Dataset(X_chunked_val_tokenized, val_labels)
    
    test_source_code,test_labels=prepare_dataset(test_data,tokenizer)
    
    filtered_test_source_code,filtered_test_labels=test_filter(test_source_code,test_labels)


    X_chunked_test_tokenized = tokenizer(filtered_test_source_code,padding=True, truncation=True, max_length=512)
    test_dataset = Dataset(X_chunked_test_tokenized,filtered_test_labels) 

    with open(join(dataset_path,"train_dataset.pickle"), "wb") as output_file:
            pickle.dump(train_dataset, output_file)

    with open(join(dataset_path,"val_dataset.pickle"), "wb") as output_file:
            pickle.dump(val_dataset, output_file)
    
    with open(join(dataset_path,"test_dataset.pickle"), "wb") as output_file:
            pickle.dump(test_dataset, output_file)

else:
    print("Loading Dataset...")
    with open(join(dataset_path,"train_dataset.pickle"), "rb") as output_file_train:
            train_dataset = pickle.load(output_file_train)
    
    with open(join(dataset_path,"val_dataset.pickle"), "rb") as output_file_val:
           val_dataset= pickle.load(output_file_val)
    
    with open(join(dataset_path,"test_dataset.pickle"), "rb") as output_file_test:
           test_dataset= pickle.load(output_file_test)
    

if args.train:
    print("Training Dataset...")
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        seed=121,
        load_best_model_at_end=True)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    trainer.train()
    trainer.save_model(join(args.output_dir,"best_model"))

if args.val_predict or args.test_predict:
    best_model= RobertaForSequenceClassification.from_pretrained(join(args.output_dir,"best_model"), num_labels=2)
    train_args = TrainingArguments(output_dir=args.output_dir,per_device_eval_batch_size=args.per_device_eval_batch_size)
    trainer = Trainer(model=best_model,args=train_args)

if args.val_predict:
    print("Validation Results...")
    raw_pred_val, b, c = trainer.predict(val_dataset)
    y_pred_val = np.argmax(raw_pred_val, axis=1)
    val_metrics=compute_metrics([raw_pred_val,val_dataset.labels])
    print("Validation Metrics",val_metrics)
    
if args.test_predict:
    print("Test Results...")
    raw_pred_test, b, c = trainer.predict(test_dataset)
    y_pred_test = np.argmax(raw_pred_test, axis=1)
    test_preds=compute_metrics([raw_pred_test,test_dataset.labels])
    print("Test Metrics",test_preds)

    
if args.train_predict:
    print("Train Results...")
    raw_pred_train, b, c = trainer.predict(train_dataset)
    y_pred_train = np.argmax(raw_pred_train, axis=1)
    train_preds=compute_metrics([raw_pred_train,train_dataset.labels])
    print("Train Metrics",train_preds)
    log_file.write("Train Metrics:" +json.dumps(train_preds)+"\n")


