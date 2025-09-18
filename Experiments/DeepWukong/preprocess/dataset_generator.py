from typing import List, cast
from os.path import join, exists
from argparse import ArgumentParser
import os
import sys
module_path = os.path.abspath('/code/models/DeepWukong/')

if module_path not in sys.path:
    sys.path.append(module_path) 
# print(sys.path)
from utils import unique_xfg_sym, split_list
import networkx as nx
# from preprocess.symbolizer import clean_gadget
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from multiprocessing import cpu_count, Manager, Pool, Queue
import functools
import dataclasses
# from preprocess.symbolizer import tokenize_code_line, clean_gadget
from collections import defaultdict

import pickle
import ray
import pandas as pd

# ray.init(_plasma_directory="/tmp")
# USE_CPU = cpu_count()

    
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c",
                            "--config",
                            help="Config File",
                            default="config.yaml",
                            type=str)
    return arg_parser


def unique_data(XFG_root_path,config):
    """
    unique raw data without symbolization
    Args:
        cweid:
        root:

    Returns:

    """
    testcaseids=[]
    df= pd.read_csv(config.csv_data_path)
    # df["vulnerable_line_numbers"] = df["flaw_line_index"]
    df['unique_file_name'] = df['unique_id'].astype(str) + ".c"
    testcaseids=list(df["unique_file_name"])
   
    xfg_paths = list()
    for testcase in tqdm(testcaseids):
        testcase_root_path = join(XFG_root_path, testcase)
        for k in ["arith", "array", "call", "ptr"]:
            k_root_path = join(testcase_root_path, k)
            xfg_ps = list(listdir_nohidden(k_root_path))
            for xfg_p in xfg_ps:
                xfg_path = join(k_root_path, xfg_p)
                xfg_paths.append(xfg_path)

    xfg_dict = unique_xfg_sym(xfg_paths,config)
    xfg_unique_paths = list()
    
    for md5 in xfg_dict:
        if xfg_dict[md5]["label"]!=-1:
            xfg_unique_paths.append(xfg_dict[md5]["xfg"])
            
    return xfg_unique_paths


def process_parallel(testcaseid: str,XFG_root_path: str, split_token: bool,config):
    """

    Args:
        testcaseid:
        queue:
        XFG_root_path

    Returns:

    """
    import sys
    os.environ["SLURM_TMPDIR"] = "data/realvul"
    module_path = os.path.abspath('/code/models/DeepWukong/')
    if module_path not in sys.path:
        sys.path.append(module_path) 
    from preprocess.symbolizer import tokenize_code_line, clean_gadget

    testcase_root_path = join(XFG_root_path, testcaseid)
    for k in ["arith", "array", "call", "ptr"]:
        k_root_path = join(testcase_root_path, k)
        xfg_ps = list(listdir_nohidden(k_root_path))
        for xfg_p in xfg_ps:
            xfg_path = join(k_root_path, xfg_p)
            xfg: nx.DiGraph = nx.read_gpickle(xfg_path)
            for idx, n in enumerate(xfg):
                if "code_sym_token" in xfg.nodes[n]:
                    return testcaseid
            file_path=join(config.local_dir_source_code_path,xfg.graph["file_paths"][0])
       
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                file_contents = f.readlines()
            code_lines = list()
            for n in xfg:
                code_lines.append(file_contents[n - 1])
            sym_code_lines = clean_gadget(code_lines)

            to_remove = list()
            for idx, n in enumerate(xfg):
                xfg.nodes[n]["code_sym_token"] = tokenize_code_line(sym_code_lines[idx], split_token)
                if len(xfg.nodes[n]["code_sym_token"]) == 0:
                    to_remove.append(n)
            xfg.remove_nodes_from(to_remove)
            if len(xfg.nodes) != 0:
                nx.write_gpickle(xfg, xfg_path)
            else:
                os.system(f"rm {xfg_path}")
    return testcaseid


def add_symlines(XFG_root_path: str, split_token: bool,config):
    """

    Args:
        cweid:
        root:

    Returns:

    """

    testcaseids = list(listdir_nohidden(XFG_root_path))
    testcase_len = len(testcaseids)

    result_ids = []
    for i in tqdm(testcaseids, desc="Processing testcases"):
        result_ids.append(process_parallel(i,XFG_root_path=XFG_root_path,
                                         split_token=split_token,config=config))
    print("Done")


if __name__ == '__main__':
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    config = cast(DictConfig, OmegaConf.load(__args.config))
    os.environ["SLURM_TMPDIR"] = "data/realvul"
    XFG_path=join(os.environ["SLURM_TMPDIR"],config.local_dir_xfg_path)

    
    add_symlines(XFG_path, config.split_token,config)
    
    xfg_unique_paths = unique_data(XFG_path,config)
    print("unique path done")
    
    
    split_folder_path=join(config.root_folder_path,config.split_folder_name)
    if not exists(split_folder_path):
        os.makedirs(split_folder_path)
    split_list(xfg_unique_paths,[],split_folder_path,config,XFG_path)
