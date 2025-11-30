import hashlib
from warnings import filterwarnings
import subprocess

from sklearn.model_selection import train_test_split
from typing import List, Union, Dict, Tuple
import numpy
import torch,ray
import json
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
os.environ['RAY_TMPDIR'] = '/data/tmp'
os.environ['TMPDIR'] = '/data/tmp'
os.environ['RAY_USE_MULTIPROCESSING_CPU_COUNT'] = '32'
import networkx as nx
from os.path import exists
from tqdm import tqdm
from os.path import join
from collections import defaultdict
import random
import pandas as pd
PAD = "<PAD>"
UNK = "<UNK>"
MASK = "<MASK>"
BOS = "<BOS>"
EOS = "<EOS>"


def getMD5(s):
    '''
    得到字符串s的md5加密后的值

    :param s:
    :return:
    '''
    hl = hashlib.md5()
    hl.update(s.encode("utf-8"))
    return hl.hexdigest()


def filter_warnings():
    # "The dataloader does not have many workers which may be a bottleneck."
    filterwarnings("ignore",
                   category=UserWarning,
                   module="pytorch_lightning.trainer.data_loading",
                   lineno=102)
    filterwarnings("ignore",
                   category=UserWarning,
                   module="pytorch_lightning.utilities.data",
                   lineno=41)
    # "Please also save or load the state of the optimizer when saving or loading the scheduler."
    filterwarnings("ignore",
                   category=UserWarning,
                   module="torch.optim.lr_scheduler",
                   lineno=216)  # save
    filterwarnings("ignore",
                   category=UserWarning,
                   module="torch.optim.lr_scheduler",
                   lineno=234)  # load
    filterwarnings("ignore",
                   category=DeprecationWarning,
                   module="pytorch_lightning.metrics.__init__",
                   lineno=43)
    filterwarnings("ignore",
                   category=UserWarning,
                   module="torch._tensor",
                   lineno=575)
    filterwarnings("ignore",
                   category=UserWarning,
                   module="src.models.modules.common_layers",
                   lineno=0)


def count_lines_in_file(file_path: str) -> int:
    command_result = subprocess.run(["wc", "-l", file_path],
                                    capture_output=True,
                                    encoding="utf-8")
    if command_result.returncode != 0:
        raise RuntimeError(
            f"Counting lines in {file_path} failed with error\n{command_result.stderr}"
        )
    return int(command_result.stdout.split()[0])

    
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
            

def unique_xfg_raw(XFG_root_path,xfg_path_list):
    """f
    unique xfg from xfg list
    Args:
        xfg_path_list:

    Returns:
        md5_dict: {xfg md5:{"xfg": xfg_path, "label": 0/1/-1}}, -1 stands for conflict
    """
    md5_dict = dict()
    mul_ct = 0
    conflict_ct = 0

    for xfg_path in tqdm(xfg_path_list,total=len(xfg_path_list)):
        xfg = nx.read_gpickle(join(XFG_root_path,xfg_path))
        label = xfg.graph["label"]
        file_path = xfg.graph["file_paths"][0]
        assert exists(file_path), f"{file_path} not exists!"
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            file_contents = f.readlines()
        for ln in xfg:
            ln_md5 = getMD5(file_contents[ln - 1])
            xfg.nodes[ln]["md5"] = ln_md5
        edges_md5 = list()
        for edge in xfg.edges:
            edges_md5.append(xfg.nodes[edge[0]]["md5"] + "_" + xfg.nodes[edge[1]]["md5"])
        xfg_md5 = getMD5(str(sorted(edges_md5)))
        if xfg_md5 not in md5_dict:
            md5_dict[xfg_md5] = dict()
            md5_dict[xfg_md5]["label"] = label
            md5_dict[xfg_md5]["xfg"] = join(XFG_root_path,xfg_path)
        else:
            md5_label = md5_dict[xfg_md5]["label"]
            if md5_label != -1 and md5_label != label:
                conflict_ct += 1
                md5_dict[xfg_md5]["label"] = -1
            else:
                mul_ct += 1
    
    print(f"total conflit: {conflict_ct}")
    print(f"total multiple: {mul_ct}")
    return md5_dict

@ray.remote(num_cpus=4)
def read_xfg(xfg_path):
        xfg_path_m="/".join(xfg_path.split("/")[-3:])
        return [nx.read_gpickle(xfg_path),xfg_path_m]

def read_normal_xfg(xfg_path):
        xfg_path_m="/".join(xfg_path.split("/")[-3:])
        return [nx.read_gpickle(xfg_path),xfg_path_m]
def unique_xfg_sym(xfg_path_list,config):
    """f
    unique xfg from xfg list
    Args:
        xfg_path_list:

    Returns:
        md5_dict: {xfg md5:{"xfg": xfg_path, "label": 0/1/-1}}, -1 stands for conflict
    """
    md5_dict = dict()
    mul_ct = 0
    conflict_ct = 0
    ray_xfgs=[]
    # for i in xfg_path_list:
    #     ray_xfgs.append(read_xfg.remote(i))
    # xfgs = ray.get(ray_xfgs) 
    # Ray parallel processing
    import ray
    from multiprocessing import cpu_count
    
    @ray.remote
    def process_xfg_batch(batch_paths, config):
        """XFG 배치를 병렬로 처리"""
        batch_results = []
        for path_item in batch_paths:
            try:
                xfg, xfg_path = read_normal_xfg(path_item)
                label = xfg.graph["label"]
                file_path = join(config.local_dir_source_code_path, xfg.graph["file_paths"][0])
                
                if not exists(file_path):
                    continue
                    
                # Calculate node MD5 (skip nodes without code_sym_token)
                nodes_to_remove = []
                for ln in xfg:
                    if "code_sym_token" not in xfg.nodes[ln]:
                        nodes_to_remove.append(ln)
                        continue
                    if len(xfg.nodes[ln]["code_sym_token"]) == 0:
                        nodes_to_remove.append(ln)
                        continue
                    ln_md5 = getMD5(str(xfg.nodes[ln]["code_sym_token"]))
                    xfg.nodes[ln]["md5"] = ln_md5
                
                # Remove nodes with empty tokens
                if nodes_to_remove:
                    xfg.remove_nodes_from(nodes_to_remove)
                
                # Skip if all nodes are removed
                if len(xfg.nodes) == 0:
                    continue
                
                # Calculate edge MD5
                edges_md5 = []
                for edge in xfg.edges:
                    edges_md5.append(xfg.nodes[edge[0]]["md5"] + "_" + xfg.nodes[edge[1]]["md5"])
                
                xfg_md5 = getMD5(str(sorted(edges_md5)))
                batch_results.append({
                    'xfg_md5': xfg_md5,
                    'label': label,
                    'xfg_path': xfg_path,
                    'file_path': file_path
                })
            except Exception as e:
                print(f"Error processing {path_item}: {e}")
                continue
                
        return batch_results
    
    # Split into batches for parallel processing
    batch_size = max(100, len(xfg_path_list) // (cpu_count() * 4))
    batches = [xfg_path_list[i:i+batch_size] for i in range(0, len(xfg_path_list), batch_size)]
    
    print(f"Processing {len(xfg_path_list)} XFGs in {len(batches)} batches using Ray...")
    
    # Process all batches in parallel
    batch_futures = [process_xfg_batch.remote(batch, config) for batch in batches]
    
    # Collect results
    with tqdm(total=len(xfg_path_list), desc="Processing XFGs") as pbar:
        for future in batch_futures:
            batch_results = ray.get(future)
            pbar.update(len(batch_results))
            
            # Merge results into main dictionary
            for result in batch_results:
                xfg_md5 = result['xfg_md5']
                label = result['label']
                xfg_path = result['xfg_path']
                file_path = result['file_path']
                
                if xfg_md5 not in md5_dict:
                    md5_dict[xfg_md5] = {
                        'label': label,
                        'xfg': xfg_path,
                        'file_path': file_path
                    }
                else:
                    md5_label = md5_dict[xfg_md5]["label"]
                    if md5_label == -1:
                        conflict_ct += 1
                    elif md5_label != -1 and md5_label != label:
                        conflict_ct += 1
                        md5_dict[xfg_md5]["label"] = -1
                        md5_dict[xfg_md5]["xfg"] = xfg_path
                        md5_dict[xfg_md5]["file_path"] = file_path
                    else:
                        mul_ct += 1
    print(f"total conflit: {conflict_ct}")
    print(f"total multiple: {mul_ct}")
    return md5_dict

def remove_conflicts_test(testcaseids,config):
    xfg_paths=[]
    for testcase in testcaseids:
        testcase_root_path = join(os.environ["SLURM_TMPDIR"],config.local_dir_xfg_path, testcase)
        for k in ["arith", "array", "call", "ptr"]:
            k_root_path = join(testcase_root_path, k)
            xfg_ps = list(listdir_nohidden(k_root_path))
            for xfg_p in xfg_ps:
                xfg_path = join(k_root_path, xfg_p)
                xfg_paths.append(xfg_path)
    md5_dict = dict()
    conflict_ct = 0
    ray_xfgs=[]
    for i in xfg_paths:
        ray_xfgs.append(read_xfg.remote(i))
    xfgs = ray.get(ray_xfgs) 
    for xfg,xfg_path in tqdm(xfgs, total=len(xfgs), desc="xfgs: "):
        label = xfg.graph["label"]
        file_path=join(os.environ["SLURM_TMPDIR"],config.local_dir_source_code_path,xfg.graph["file_paths"][0])
   
        assert exists(file_path), f"{file_path} not exists!"
        for ln in xfg:
            ln_md5 = getMD5(str(xfg.nodes[ln]["code_sym_token"]))
            xfg.nodes[ln]["md5"] = ln_md5
        edges_md5 = list()
        for edge in xfg.edges:
            edges_md5.append(xfg.nodes[edge[0]]["md5"] + "_" + xfg.nodes[edge[1]]["md5"])
        xfg_md5 = getMD5(str(sorted(edges_md5)))
        if xfg_md5 not in md5_dict:
            md5_dict[xfg_md5] = dict()
            md5_dict[xfg_md5]["label"] = label
            md5_dict[xfg_md5]["xfg"] = [xfg_path]
            md5_dict[xfg_md5]["file_path"]=[file_path]
            
        else:
            md5_label = md5_dict[xfg_md5]["label"]
            if md5_label==-1:
                conflict_ct+=1
            if md5_label!=-1 and md5_label != label:
                conflict_ct += 1
                md5_dict[xfg_md5]["label"] = -1
            md5_dict[xfg_md5]["xfg"].append(xfg_path)
            md5_dict[xfg_md5]["file_path"].append(file_path)
    xfg_paths=[]
    for md5 in md5_dict:
        if md5_dict[md5]["label"]!=-1:
            xfg_paths.extend(md5_dict[md5]["xfg"])
            
    print(f"total test conflit: {conflict_ct}")
    return xfg_paths

def split_list(files,unique_xfg_count, out_root_path: str,config,XFG_path):
    """

    Args:
        files:
        out_root_path:

    Returns:

    """
    print("splitting")
    
    X_train, X_test = train_test_split(files, test_size=0.2, random_state=13)
    X_train, X_val = train_test_split(files, test_size=0.125, random_state=13)
    # df= pd.read_csv(config.csv_data_path)
    # testcaseids=df[df["dataset_type"]=="test"]["file_name"]
    # X_test= remove_conflicts_test(testcaseids,config)

      
    with open(f"{config.root_folder_path}/{config.split_folder_name}/train.json", "w") as f:
        json.dump(files, f)
    with open(f"{config.root_folder_path}/{config.split_folder_name}/test.json", "w") as f:
        json.dump(X_test, f)
    with open(f"{config.root_folder_path}/{config.split_folder_name}/val.json", "w") as f:
        json.dump(X_val, f)
