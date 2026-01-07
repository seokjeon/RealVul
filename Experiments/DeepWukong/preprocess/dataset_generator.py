from typing import List, cast
from os.path import join, exists
from argparse import ArgumentParser
import os
import sys
module_path = os.path.abspath('.')

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

ray.init(_plasma_directory="/data/tmp")
USE_CPU = cpu_count()

    
def listdir_nohidden(path):
    if not exists(path):
        return
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
    try:
        df['unique_file_name'] = df['unique_id'].astype(str) + ".c"
    except KeyError:
        try:
            df['unique_file_name'] = df['file_name'].astype(str) + ".c"
        except KeyError:
            exit(1)
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


@ray.remote
def process_parallel(testcaseid: str, XFG_root_path: str, split_token: bool, config):
    """

    Args:
        testcaseid:
        queue:
        XFG_root_path

    Returns:

    """
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

    # Ray를 사용한 병렬 처리
    total_tasks = len(testcaseids)
    initial_batch_size = min(USE_CPU * 2, total_tasks)
    pending, results, failures = [], [], []
    
    # 진행률 표시 설정
    pbar = tqdm(total=total_tasks, desc="Processing testcases", unit="testcase", 
               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    # 초기 배치 제출
    for i in range(initial_batch_size):
        pending.append(process_parallel.remote(testcaseids[i], XFG_root_path, split_token, config))
    
    # 나머지 작업을 위한 인덱스
    next_task_idx = initial_batch_size
    
    # 메인 처리 루프
    while pending:
        # 완료된 작업들 수집
        done, pending = ray.wait(pending, 
                               num_returns=min(USE_CPU, len(pending)), 
                               timeout=None)
        
        # 완료된 작업들 처리
        for obj in done:
            try:
                result = ray.get(obj)
                results.append(result)
            except Exception as e:
                print(f"Task failed: {e}")
                failures.append((obj, str(e)))
                pbar.set_postfix_str(f"Failed: {len(failures)}")
        
        # 진행률 업데이트
        pbar.update(len(done))
        
        # 새로운 작업 제출 (완료된 만큼)
        for _ in range(len(done)):
            if next_task_idx < total_tasks:
                pending.append(process_parallel.remote(testcaseids[next_task_idx], XFG_root_path, split_token, config))
                next_task_idx += 1
    
    pbar.close()
    
    # 결과 요약
    print(f"\n✅ Processing completed!")
    print(f"   Success: {len(results)} testcases")
    if failures:
        print(f"   Failed: {len(failures)} testcases")
        print(f"   Success rate: {len(results)/(len(results)+len(failures))*100:.1f}%")


if __name__ == '__main__':
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    config = cast(DictConfig, OmegaConf.load(__args.config))
    os.environ["SLURM_TMPDIR"] = "."
    XFG_path=join(os.environ["SLURM_TMPDIR"],config.local_dir_xfg_path)

    
    add_symlines(XFG_path, config.split_token,config)
    
    xfg_unique_paths = unique_data(XFG_path,config)
    print("unique path done")
    
    
    split_folder_path=join(config.root_folder_path,config.split_folder_name)
    if not exists(split_folder_path):
        os.makedirs(split_folder_path)
    split_list(xfg_unique_paths,[],split_folder_path,config,XFG_path)
