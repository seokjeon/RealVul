import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,confusion_matrix
import torch
import gc
import pickle
from transformers import TrainingArguments, Trainer, TrainerCallback,RobertaForSequenceClassification,RobertaTokenizer
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
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ray.init(_plasma_directory="/tmp")
torch.cuda.empty_cache()
gc.collect()

TRAIN_EPOCH_LOSS_PREFIX = "TRAIN_EPOCH_LOSS"
TRAINING_LOSS_LOG_NAME = "training_loss.log"
DEFAULT_BASELINE_MODEL_NAME = "/app/RealVul/Experiments/LineVul/best_model"
REQUIRED_MODEL_ARTIFACT_NAMES = ("config.json", "pytorch_model.bin")


class EpochTrainingLossLoggerCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.log_path = Path(output_dir) / TRAINING_LOSS_LOG_NAME
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" not in logs or "epoch" not in logs:
            return
        if "eval_loss" in logs or "train_loss" in logs:
            return

        payload = {
            "epoch": float(logs["epoch"]),
            "loss": float(logs["loss"]),
        }
        line = f"{TRAIN_EPOCH_LOSS_PREFIX} {json.dumps(payload, sort_keys=True)}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as output_file:
            output_file.write(line + "\n")


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


def parse_vulnerable_line_numbers(value):
    if pd.isna(value) or value == "":
        return []
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raw_values = [str(value)]

    vulnerable_line_numbers = []
    for raw_value in raw_values:
        vulnerable_line_numbers.append(int(float(raw_value)))
    return vulnerable_line_numbers
    
def create_token_chunks_vulnerable_samples(
    code_statements, all_special_ids, vulnerable_line_numbers, tokenizer
):
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

def read_file_label(sample,tokenizer):
    vulnerable_line_numbers = parse_vulnerable_line_numbers(sample["vulnerable_line_numbers"])
    label=1 if vulnerable_line_numbers else 0
    all_special_ids=tokenizer.all_special_ids
    source_code=sample["processed_func"].split("\n")
    inputs,labels=[],[]
    if label==1:
        samples,mixed_labels=create_token_chunks_vulnerable_samples(
            tokenizer(source_code)["input_ids"],
            all_special_ids,
            vulnerable_line_numbers,
            tokenizer,
        )
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

@ray.remote
def read_file_label_batch(samples,tokenizer):
    batch_inputs,batch_labels=[],[]
    for sample in samples:
        inputs,labels=read_file_label(sample,tokenizer)
        batch_inputs.extend(inputs)
        batch_labels.extend(labels)
    return batch_inputs,batch_labels



def prepare_dataset(samples,tokenizer):
        records=samples.to_dict("records")
        batch_size=3000
        chunk_size = 10  # 한 번에 10개 task만 처리
        source_codes, labels = [], []
        
        all_batches = list(range(0, len(records), batch_size))
        
        for chunk_start in tqdm(range(0, len(all_batches), chunk_size)):
            process_examples = []
            chunk_end = min(chunk_start + chunk_size, len(all_batches))
            
            for i in all_batches[chunk_start:chunk_end]:
                process_examples.append(
                    read_file_label_batch.remote(records[i:i+batch_size], tokenizer)
                )
            
            # 10개씩만 join
            result = ray.get(process_examples)
            for source_code, label in result:
                source_codes.extend(source_code)
                labels.extend(label)
        
        return source_codes, labels



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
    collisions=0
    duplicates=0
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

# Transformer 모델의 마지막 hidden state 벡터를 추출하여 저장하는 함수 (<s>/<cls> 벡터 사용)
def export_last_hidden_state_vectors(model, dataset, batch_size, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    feature_batches = []
    label_batches = []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Exporting hidden states to {Path(output_path).name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            cls_hidden_states = outputs.hidden_states[-1][:, 0, :] # CLS 토큰의 hidden state 벡터 추출
            feature_batches.append(cls_hidden_states.cpu().numpy())
            if "labels" in batch:
                label_batches.append(batch["labels"].cpu().numpy())

    hidden_size = model.config.hidden_size
    features = np.concatenate(feature_batches, axis=0) if feature_batches else np.empty((0, hidden_size), dtype=np.float32)
    labels = np.concatenate(label_batches, axis=0) if label_batches else np.empty((0,), dtype=np.int64)
    np.savez_compressed(output_path, features=features, labels=labels)
    print(f"Saved CLS last hidden states to {output_path}")
    from tsne import plot_embedding
    plot_embedding(features, labels, title=str(Path(output_path).with_suffix("")), new=True)

def build_hidden_state_output_path(output_dir, split_name, export_timestamp):
    return join(output_dir, f"{export_timestamp}_{split_name}_last_hidden_state_vectors.npz")


RAW_MODEL_EVAL_DIRNAME = "raw_model_eval"
COMBINED_TEST_TSNE_BASENAME = "combined_test_last_hidden_state_vectors"


def find_latest_hidden_state_output(output_dir, split_name="test"):
    candidates = sorted(
        Path(output_dir).glob(f"*_{split_name}_last_hidden_state_vectors.npz"),
        key=lambda path: path.name,
    )
    if not candidates:
        return None
    return candidates[-1]


def load_hidden_state_output(output_path):
    with np.load(output_path) as payload:
        return np.asarray(payload["features"]), np.asarray(payload["labels"])


# cls토큰의 hidden state 벡터가 정상인지 검증(생략해도 무방)
def validate_paired_hidden_state_outputs(
    fine_features,
    fine_labels,
    raw_features,
    raw_labels,
    *,
    fine_output_path,
    raw_output_path,
):
    # csl토큰의 hidden state 벡터는 2D 배열이어야 한다.
    if fine_features.ndim != 2 or raw_features.ndim != 2:
        raise ValueError(
            "Expected 2D feature arrays for paired TSNE: "
            f"{fine_output_path} -> {fine_features.shape}, "
            f"{raw_output_path} -> {raw_features.shape}"
        )

    # fine-tuned과 raw 모델의 feature 벡터 차원이 일치해야 한다.
    if fine_features.shape[1] != raw_features.shape[1]:
        raise ValueError(
            "Fine-tuned and raw test feature dimensions do not match: "
            f"{fine_output_path} -> {fine_features.shape}, "
            f"{raw_output_path} -> {raw_features.shape}"
        )

    # fine-tuned과 raw 모델의 테스트 샘플 수가 일치해야 한다.
    if fine_features.shape[0] != raw_features.shape[0]:
        raise ValueError(
            "Fine-tuned and raw test sample counts do not match: "
            f"{fine_output_path} -> {fine_features.shape[0]}, "
            f"{raw_output_path} -> {raw_features.shape[0]}"
        )


def maybe_export_paired_test_tsne(raw_output_path):
    raw_output_path = Path(raw_output_path)
    raw_output_dir = raw_output_path.parent
    if raw_output_dir.name != RAW_MODEL_EVAL_DIRNAME:
        return

    fine_output_dir = raw_output_dir.parent
    fine_output_path = find_latest_hidden_state_output(fine_output_dir, split_name="test")
    if fine_output_path is None:
        print(
            "Skipping paired TSNE: fine-tuned test hidden states not found under "
            f"{fine_output_dir}"
        )
        return

    # features은 마지막 hidden layer의 cls토큰의 벡터, labels는 취약/비취약
    fine_features, fine_labels = load_hidden_state_output(fine_output_path)
    raw_features, raw_labels = load_hidden_state_output(raw_output_path)

    # cls토큰의 hidden state 벡터가 정상인지 검증
    validate_paired_hidden_state_outputs(
        fine_features,
        fine_labels,
        raw_features,
        raw_labels,
        fine_output_path=fine_output_path,
        raw_output_path=raw_output_path,
    )

    from tsne import plot_paired_embedding

    combined_output_base = fine_output_dir / COMBINED_TEST_TSNE_BASENAME # vuln_patch/combined_test_last_hidden_state_vectors.npz 경로
    plot_paired_embedding(
        fine_features,
        fine_labels,
        raw_features,
        raw_labels,
        title=str(combined_output_base),
        new=True,   # tsne 를 위한 연산은 건너뛰고, 캐시를 활용해 jpeg 만 생성하려면 new=False 로 수정하세요.
    )
    print(f"Saved paired test TSNE to {combined_output_base}.jpeg")


def build_arg_parser():
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
    parser.add_argument("--eval_model_name", type=str, default=None,
                        help="optional model path used for eval/test instead of output_dir/best_model")
    parser.add_argument("--extended-realvul", dest="extended_realvul", default=False, action='store_true',
                        help="prepare the test split, evaluate both models, and emit paired TSNE artifacts")
    parser.add_argument("--baseline_model_name", type=str, default=DEFAULT_BASELINE_MODEL_NAME,
                        help="baseline model path used in --extended-realvul mode")
    return parser


def require_model_artifacts(model_dir, label):
    model_path = Path(model_dir)
    for artifact_name in REQUIRED_MODEL_ARTIFACT_NAMES:
        artifact_path = model_path / artifact_name
        if not artifact_path.exists():
            raise FileNotFoundError(f"Expected {label}/{artifact_name} not found: {artifact_path}")


def prepare_dataset_pickles(train_val, test_data, tokenizer, dataset_path, *, test_only=False):
    train_dataset = None
    val_dataset = None
    test_dataset = None

    if not test_only and len(train_val) > 0:
        source_code,labels=prepare_dataset(train_val,tokenizer)
        filtered_source_code,filtered_labels=train_filter(source_code,labels)
        train_source_code, val_source_code,train_labels,val_labels = train_test_split(
            filtered_source_code,
            filtered_labels,
            test_size=0.1,
        )
        X_chunked_train_tokenized = tokenizer(train_source_code,padding=True, truncation=True, max_length=512)
        X_chunked_val_tokenized = tokenizer(val_source_code,padding=True, truncation=True, max_length=512)
        train_dataset = Dataset(X_chunked_train_tokenized, train_labels)
        val_dataset = Dataset(X_chunked_val_tokenized, val_labels)
        with open(join(dataset_path,"train_dataset.pkl"), "wb") as output_file:
            pickle.dump(train_dataset, output_file)

        with open(join(dataset_path,"val_dataset.pkl"), "wb") as output_file:
            pickle.dump(val_dataset, output_file)

    if len(test_data) > 0:
        test_source_code,test_labels=prepare_dataset(test_data,tokenizer)
        filtered_test_source_code,filtered_test_labels=test_filter(test_source_code,test_labels)
        X_chunked_test_tokenized = tokenizer(filtered_test_source_code,padding=True, truncation=True, max_length=512)
        test_dataset = Dataset(X_chunked_test_tokenized,filtered_test_labels)

        with open(join(dataset_path,"test_dataset.pkl"), "wb") as output_file:
            pickle.dump(test_dataset, output_file)

    return train_dataset, val_dataset, test_dataset


def load_dataset_pickles(dataset_path):
    train_dataset = None
    val_dataset = None
    test_dataset = None

    print("Loading Dataset...")
    train_dataset_path = Path(join(dataset_path,"train_dataset.pkl"))
    if train_dataset_path.exists():
        with open(train_dataset_path, "rb") as output_file_train:
            train_dataset = pickle.load(output_file_train)

    val_dataset_path = Path(join(dataset_path,"val_dataset.pkl"))
    if val_dataset_path.exists():
        with open(val_dataset_path, "rb") as output_file_val:
            val_dataset= pickle.load(output_file_val)

    test_dataset_path = Path(join(dataset_path,"test_dataset.pkl"))
    if test_dataset_path.exists():
        with open(test_dataset_path, "rb") as output_file_test:
            test_dataset= pickle.load(output_file_test)

    return train_dataset, val_dataset, test_dataset


def build_eval_model_and_trainer(eval_model_name, output_dir, eval_batch_size, use_fp16):
    best_model= RobertaForSequenceClassification.from_pretrained(eval_model_name, num_labels=2)
    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=eval_batch_size,
        fp16=use_fp16,
    )
    trainer = Trainer(model=best_model,args=train_args)
    return best_model, trainer


def run_test_prediction(best_model, trainer, test_dataset, output_dir, eval_batch_size):
    print("Test Results...")
    export_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    test_hidden_state_output_path = None
    if test_dataset is not None:
        test_hidden_state_output_path = build_hidden_state_output_path(
            output_dir,
            "test",
            export_timestamp,
        )
        export_last_hidden_state_vectors(
            best_model,
            test_dataset,
            eval_batch_size,
            test_hidden_state_output_path,
        )

    raw_pred_test, b, c = trainer.predict(test_dataset)
    y_pred_test = np.argmax(raw_pred_test, axis=1)
    test_preds=compute_metrics([raw_pred_test,test_dataset.labels])
    print("Test Metrics",test_preds)
    df = pd.DataFrame({
        "label": test_dataset.labels,
        "pred": y_pred_test,
    })
    df.to_csv(join(output_dir, "test_pred_with_code.csv"), index=False)

    if test_hidden_state_output_path is not None:
        maybe_export_paired_test_tsne(test_hidden_state_output_path)

    return test_hidden_state_output_path


def run_extended_realvul(args, project_df, use_fp16):
    active_modes = [
        args.prepare_dataset,
        args.train,
        args.val_predict,
        args.test_predict,
        args.train_predict,
    ]
    if any(active_modes):
        raise ValueError("--extended-realvul does not accept other phase flags")

    require_model_artifacts(args.model_name, "fine-tuned model")
    require_model_artifacts(args.baseline_model_name, "baseline model")

    test_data=project_df[project_df["dataset_type"]=="test"]
    if len(test_data) == 0:
        raise ValueError("Expected dataset_type=test rows for --extended-realvul")

    print("Preparing Extended RealVul test dataset...")
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    prepare_dataset_pickles(
        train_val=project_df.iloc[0:0],
        test_data=test_data,
        tokenizer=tokenizer,
        dataset_path=args.dataset_path,
        test_only=True,
    )
    _, _, test_dataset = load_dataset_pickles(args.dataset_path)
    if test_dataset is None:
        raise RuntimeError(f"Expected test_dataset.pkl under {args.dataset_path}")

    print(f"Evaluating fine-tuned model from {args.model_name}...")
    best_model, trainer = build_eval_model_and_trainer(
        args.model_name,
        args.output_dir,
        args.per_device_eval_batch_size,
        use_fp16,
    )
    run_test_prediction(
        best_model,
        trainer,
        test_dataset,
        args.output_dir,
        args.per_device_eval_batch_size,
    )

    raw_output_dir = join(args.output_dir, RAW_MODEL_EVAL_DIRNAME)
    Path(raw_output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Evaluating baseline model from {args.baseline_model_name}...")
    raw_model, raw_trainer = build_eval_model_and_trainer(
        args.baseline_model_name,
        raw_output_dir,
        args.per_device_eval_batch_size,
        use_fp16,
    )
    run_test_prediction(
        raw_model,
        raw_trainer,
        test_dataset,
        raw_output_dir,
        args.per_device_eval_batch_size,
    )
    print(f"Extended RealVul evaluation completed under {args.output_dir}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    print("Arguments", args)

    tokenizer_name=args.tokenizer_name
    model_name=args.model_name
    dataset_csv_path=args.dataset_csv_path
    output_dir=args.output_dir
    dataset_path= args.dataset_path
    use_fp16 = torch.cuda.is_available()
    print(f"Using fp16: {use_fp16}")

    if not exists(dataset_path):
            os.makedirs(dataset_path)

    if not exists(output_dir):
            os.makedirs(output_dir)

    project_df=pd.read_csv(dataset_csv_path)
    project_df["vulnerable_line_numbers"]=project_df["vulnerable_line_numbers"].fillna("")
    train_val=project_df[project_df["dataset_type"]=="train_val"]
    test_data=project_df[project_df["dataset_type"]=="test"]

    if args.extended_realvul:
        run_extended_realvul(args, project_df, use_fp16)
        return

    train_dataset = None
    val_dataset = None
    test_dataset = None

    if args.prepare_dataset:
        print("Preparing Dataset...")
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        train_dataset, val_dataset, test_dataset = prepare_dataset_pickles(
            train_val,
            test_data,
            tokenizer,
            dataset_path,
        )
    else:
        train_dataset, val_dataset, test_dataset = load_dataset_pickles(dataset_path)

    if args.train:
        print("Training Dataset...")
        model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
        loss_logger_callback = EpochTrainingLossLoggerCallback(args.output_dir)
        train_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            seed=121,
            load_best_model_at_end=True,
            fp16=use_fp16)

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[loss_logger_callback, EarlyStoppingCallback(early_stopping_patience=3)])
        trainer.train()
        trainer.save_model(join(args.output_dir,"best_model"))
        export_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        if train_dataset is not None:
            export_last_hidden_state_vectors(
                trainer.model,
                train_dataset,
                args.per_device_eval_batch_size,
                build_hidden_state_output_path(args.output_dir, "train", export_timestamp),
            )

    if args.val_predict or args.test_predict:
        eval_model_name = args.eval_model_name or join(args.output_dir,"best_model")
        best_model, trainer = build_eval_model_and_trainer(
            eval_model_name,
            args.output_dir,
            args.per_device_eval_batch_size,
            use_fp16,
        )

    if args.val_predict:
        print("Validation Results...")
        export_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        if val_dataset is not None:
            export_last_hidden_state_vectors(
                best_model,
                val_dataset,
                args.per_device_eval_batch_size,
                build_hidden_state_output_path(args.output_dir, "val", export_timestamp),
            )

        raw_pred_val, b, c = trainer.predict(val_dataset)
        y_pred_val = np.argmax(raw_pred_val, axis=1)
        val_metrics=compute_metrics([raw_pred_val,val_dataset.labels])
        print("Validation Metrics",val_metrics)

    if args.test_predict:
        run_test_prediction(
            best_model,
            trainer,
            test_dataset,
            args.output_dir,
            args.per_device_eval_batch_size,
        )

    if args.train_predict:
        print("Train Results...")
        best_model= RobertaForSequenceClassification.from_pretrained(join(args.output_dir,"best_model"), num_labels=2)
        train_args = TrainingArguments(output_dir=args.output_dir,per_device_eval_batch_size=args.per_device_eval_batch_size,fp16=use_fp16)
        trainer = Trainer(model=best_model,args=train_args)

        raw_pred_train, b, c = trainer.predict(train_dataset)
        y_pred_train = np.argmax(raw_pred_train, axis=1)
        train_preds=compute_metrics([raw_pred_train,train_dataset.labels])
        print("Train Metrics",train_preds)


if __name__ == "__main__":
    main()
