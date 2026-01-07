import xml.etree.ElementTree as ET
import networkx as nx
from typing import List, Set, Tuple, Dict
from os.path import join, exists
from argparse import ArgumentParser
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
os.environ['RAY_TMPDIR'] = '/data/tmp'
os.environ['TMPDIR'] = '/data/tmp'
from tqdm import tqdm
from typing import cast
import dataclasses
from omegaconf import OmegaConf, DictConfig
from multiprocessing import cpu_count, Manager, Pool, Queue
import functools
import traceback
import pandas as pd
import ray
import logging
from ray.exceptions import RayTaskError, WorkerCrashedError

ray.init(
    _temp_dir="/data/tmp",
    dashboard_host="0.0.0.0",  # 모든 IP에서 접근 가능
    dashboard_port=8265,
    include_dashboard=True
)
USE_CPU = cpu_count()
xfg_ct=0

def getCodeIDtoPathDict(testcases: Dict,
                        sourceDir: str) -> Dict[str, Dict[str, Set[int]]]:
    
    codeIDtoPath: Dict[str, Set[int]] = {}
    for testcase in testcases:
        if len(testcases[testcase])>0:
            VulLine = set(map(lambda x: int(x)+1,str(testcases[testcase]).split(",")))
        else:
            VulLine={}
        codeIDtoPath[testcase] = VulLine
    return codeIDtoPath

def extract_line_number(idx: int, nodes: List) -> int:
    """
    return the line number of node index

    Args:
        idx (int): node index
        nodes (List)
    Returns: line number of node idx
    """
    try:
        while idx >= 0:
            c_node = nodes[idx]
            if 'location' in c_node.keys():
                location = c_node['location']
                if location.strip() != '':
                    try:
                        ln = int(location.split(':')[0])
                        return ln
                    except Exception as e:
                        print(e)
                        pass
            idx -= 1
        return -1
    except Exception as e:
                 logging.exception(e)



def read_csv(csv_file_path: str) -> List:
    """
    read csv file
    """
    try:
        assert exists(csv_file_path), f"no {csv_file_path}"
        data = []
        with open(csv_file_path) as fp:
            header = fp.readline()
            header = header.strip()
            h_parts = [hp.strip() for hp in header.split('\t')]
            for line in fp:
                line = line.strip()
                instance = {}
                lparts = line.split('\t')
                for i, hp in enumerate(h_parts):
                    if i < len(lparts):
                        content = lparts[i].strip()
                    else:
                        content = ''
                    instance[hp] = content
                data.append(instance)
            return data
    except Exception as e:
                 logging.exception(e)


def extract_nodes_with_location_info(nodes):
    """
    Will return an array identifying the indices of those nodes in nodes array
    another array identifying the node_id of those nodes
    another array indicating the line numbers
    all 3 return arrays should have same length indicating 1-to-1 matching.
    
    """

    node_indices = []
    node_ids = []
    line_numbers = []
    node_id_to_line_number = {}
    for node_index, node in enumerate(nodes):
        assert isinstance(node, dict)
        if 'location' in node.keys():
            location = node['location']
            if location == '':
                continue
            line_num = int(location.split(':')[0])
            node_id = node['key'].strip()
            node_indices.append(node_index)
            node_ids.append(node_id)
            line_numbers.append(line_num)
            node_id_to_line_number[node_id] = line_num
    return node_indices, node_ids, line_numbers, node_id_to_line_number


def build_PDG(code_path: str, sensi_api_path: str,
              source_path: str) -> Tuple[nx.DiGraph, Dict[str, Set[int]]]:
    """
    build program dependence graph from code

    Args:
        code_path (str): source code root path
        sensi_api_path (str): path to sensitive apis
        source_path (str): source file path

    Returns: (PDG, key line map)
    """
    try:
        nodes_path = join(code_path, "nodes.csv")
        edges_path = join(code_path, "edges.csv")
        # print(nodes_path)
        assert exists(sensi_api_path), f"{sensi_api_path} not exists!"
        with open(sensi_api_path, "r", encoding="utf-8") as f:
            sensi_api_set = set([api.strip() for api in f.read().split(",")])
        if not exists(nodes_path) or not exists(edges_path):
            return None, None
        nodes = read_csv(nodes_path)
        edges = read_csv(edges_path)
        call_lines = set()
        array_lines = set()
        ptr_lines = set()
        arithmatic_lines = set()
        if len(nodes) == 0:
            return None, None
        for node_idx, node in enumerate(nodes):
            ntype = node['type'].strip()
            if ntype == 'CallExpression':
                function_name = nodes[node_idx + 1]['code']
                if function_name is None or function_name.strip() == '':
                    continue
                if function_name.strip() in sensi_api_set:
                    line_no = extract_line_number(node_idx, nodes)
                    if line_no > 0:
                        call_lines.add(line_no)
            elif ntype == 'ArrayIndexing':
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    array_lines.add(line_no)
            elif ntype == 'PtrMemberAccess':
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    ptr_lines.add(line_no)
            elif node['operator'].strip() in ['+', '-', '*', '/']:
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    arithmatic_lines.add(line_no)

        PDG = nx.DiGraph(file_paths=[source_path])
        control_edges, data_edges = list(), list()
        node_indices, node_ids, line_numbers, node_id_to_ln = extract_nodes_with_location_info(
            nodes)
        for edge in edges:
            edge_type = edge['type'].strip()
            if True:  # edge_type in ['IS_AST_PARENT', 'FLOWS_TO']:
                start_node_id = edge['start'].strip()
                end_node_id = edge['end'].strip()
                if start_node_id not in node_id_to_ln.keys(
                ) or end_node_id not in node_id_to_ln.keys():
                    continue
                start_ln = node_id_to_ln[start_node_id]
                end_ln = node_id_to_ln[end_node_id]
                if edge_type == 'CONTROLS':  # Control
                    control_edges.append((start_ln, end_ln, {"c/d": "c"}))
                if edge_type == 'REACHES':  # Data
                    data_edges.append((start_ln, end_ln, {"c/d": "d"}))
        PDG.add_edges_from(control_edges)
        PDG.add_edges_from(data_edges)
        return PDG, {
            "call": call_lines,
            "array": array_lines,
            "ptr": ptr_lines,
            "arith": arithmatic_lines
        }
    except Exception as e:
                 logging.exception(e)



def build_XFG(PDG: nx.DiGraph, key_line_map: Dict[str, Set[int]],
              vul_lines: int) -> Dict[str, List[nx.DiGraph]]:
    """
    build XFGs
    Args:
        PDG (nx.DiGraph): program dependence graph
        key_line_map (Dict[str, Set[int]]): key lines
    Returns: XFG map
    """
    try:
        if PDG is None or key_line_map is None:
            return None
        res = {"call": [], "array": [], "ptr": [], "arith": []}
        for key in ["call", "array", "ptr", "arith"]:
            for ln in key_line_map[key]:
                sliced_lines = set()
                # backward traversal
                bqueue = list()
                visited = set()
                bqueue.append(ln)
                visited.add(ln)
                while bqueue:
                    fro = bqueue.pop(0)
                    sliced_lines.add(fro)
                    if fro in PDG._pred:
                        for pred in PDG._pred[fro]:
                            if pred not in visited:
                                visited.add(pred)
                                bqueue.append(pred)

                # forward traversal
                fqueue = list()
                visited = set()
                fqueue.append(ln)
                visited.add(ln)
                while fqueue:
                    fro = fqueue.pop(0)
                    sliced_lines.add(fro)
                    if fro in PDG._succ:
                        for succ in PDG._succ[fro]:
                            if succ not in visited:
                                visited.add(succ)
                                fqueue.append(succ)
                if len(sliced_lines) != 0:
                    XFG = PDG.subgraph(list(sliced_lines)).copy()
                    XFG.graph["key_line"] = ln
                    if len(sliced_lines.intersection(vul_lines)) != 0:
                        XFG.graph["label"] = 1
                        # ct1 += 1
                    else:
                        XFG.graph["label"] = 0
                        # ct0 += 1
                res[key].append(XFG)
            # print("ct1:", ct1)
            # print("ct0:", ct0)

        return res
    except Exception as e:
                 logging.exception(e)


def dump_XFG(res: Dict[str, List[nx.DiGraph]], out_root_path: str,
             testcase: str):
    # global xfg_ct
    """
    dump XFG to file

    Args:
        res: XFGs
        out_root_path: output root path
        testcaseid: testcase id
    Returns:
    """
    import logging
    try:
        #print(out_root_path)
        testcase_out_root_path = join(out_root_path, testcase)
        if not exists(testcase_out_root_path):
            os.makedirs(testcase_out_root_path)
        for k in res:
            k_root_path = join(testcase_out_root_path, k)
            if not exists(k_root_path):
                os.makedirs(k_root_path)
            for XFG in res[k]:
                out_path = join(k_root_path, f"{XFG.graph['key_line']}.xfg.pkl")
                nx.write_gpickle(XFG, out_path)
    except Exception as e:
        print(e)
        logging.exception(e)


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c",
                            "--config",
                            help="Config File",
                            default="config.yaml",
                            type=str)
    return arg_parser
    return arg_parser




@ray.remote
def process_parallel(testcase:str, doneIDs: Set, codeIDtoPath: Dict, root: str,
                     source_root_path: str,
                     out_root_path: str,config):
    """

    Args:
        testcase:
        doneIDs:
        codeIDtoPath:
        cwe_root:
        source_root_path:
        out_root_path:

    Returns:

    """
    import logging
    try:
        if testcase in doneIDs:
            return testcase
        vul_lines = codeIDtoPath[testcase]
        
        csv_path = join(os.environ["SLURM_TMPDIR"],config.local_dir_csv_path,testcase)
        source_path = testcase

        sensiAPI_path= config.sensiAPI_path
        PDG, key_line_map = build_PDG(csv_path, sensiAPI_path ,
                                      source_path)
        # print(PDG, key_line_map)
        res = build_XFG(PDG, key_line_map, vul_lines)
        # print("res", res)
        if res:
            dump_XFG(res, out_root_path,testcase)
        return testcase

    except Exception as e:
             logging.exception(e)


    
if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    config_path=__args.config
    
    config = cast(DictConfig, OmegaConf.load(config_path))
    root = config.root_folder_path
    # os.environ["SLURM_TMPDIR"] = "/data/RealVul/Experiments/DeepWukong/data/all"
    # print("SLURM_TMPDIR:",os.environ["SLURM_TMPDIR"])
    
    csv_path=config.csv_data_path
    
    source_root_path = join(os.environ["SLURM_TMPDIR"],config.local_dir_source_code_path)
    out_root_path=join(os.environ["SLURM_TMPDIR"],config.local_dir_xfg_path)
    vul_data_csv=pd.read_csv(csv_path)
    # vul_data_csv["vulnerable_line_numbers"] = vul_data_csv["flaw_line_index"]
    try:
        vul_data_csv['unique_file_name'] = vul_data_csv['unique_id'].astype(str) + ".c"
    except KeyError:
        try:
            vul_data_csv['unique_file_name'] = vul_data_csv['file_name'].astype(str) + ".c"
        except KeyError:
            exit(1)
    vul_data=pd.Series(vul_data_csv.vulnerable_line_numbers.values,index=vul_data_csv.unique_file_name).fillna('').to_dict()
    codeIDtoPath = getCodeIDtoPathDict(vul_data, source_root_path)
    # print(codeIDtoPath)

    if not exists(out_root_path):
        os.makedirs(out_root_path)


    # 작업 목록 준비
    task_list = [i for i in codeIDtoPath if isinstance(i, str)]
    total_tasks = len(task_list)
    
    if total_tasks == 0:
        print("No tasks to process.")
        results = []
    else:
        print(f"Processing {total_tasks} tasks (CPU cores: {USE_CPU})")
        
        # Initial task submission (2x CPU cores)
        initial_batch_size = min(USE_CPU * 2, total_tasks)
        pending, results, failures = [], [], []
        
        # Progress bar setup
        pbar = tqdm(total=total_tasks, desc="Processing", unit="files", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        # Initial batch submission
        for i in range(initial_batch_size):
            pending.append(process_parallel.remote(task_list[i], doneIDs=[], 
                                                 codeIDtoPath=codeIDtoPath,
                                                 root=root, source_root_path=source_root_path,
                                                 out_root_path=out_root_path, config=config))
        
        # Index for remaining tasks
        next_task_idx = initial_batch_size
        
        # Main processing loop
        while pending:
            # Collect completed tasks (multiple at once)
            done, pending = ray.wait(pending, 
                                   num_returns=min(USE_CPU, len(pending)), 
                                   timeout=None)
            
            # Process completed tasks
            for obj in done:
                try:
                    result = ray.get(obj)
                    results.append(result)
                except (RayTaskError, WorkerCrashedError) as e:
                    logging.error(f"Task failed: {e}")
                    failures.append((obj, str(e)))
                    pbar.set_postfix_str(f"Failed: {len(failures)}")
            
            # Update progress
            pbar.update(len(done))
            
            # Submit new tasks (as many as completed)
            for _ in range(len(done)):
                if next_task_idx < total_tasks:
                    pending.append(process_parallel.remote(task_list[next_task_idx], doneIDs=[], 
                                                         codeIDtoPath=codeIDtoPath,
                                                         root=root, source_root_path=source_root_path,
                                                         out_root_path=out_root_path, config=config))
                    next_task_idx += 1
        
        pbar.close()
        
        # Results summary
        print(f"\n✅ Processing completed!")
        print(f"   Success: {len(results)} files")
        if failures:
            print(f"   Failed: {len(failures)} files")
            print(f"   Success rate: {len(results)/(len(results)+len(failures))*100:.1f}%")

    # 원본
    # result_ids = []
    # for i in codeIDtoPath:
    #     if isinstance(i, str):
    #         result_ids.append(process_parallel.remote(i,doneIDs=[],
    #                                          codeIDtoPath=codeIDtoPath,
    #                                          root=root, source_root_path=source_root_path,out_root_path=out_root_path, config=config))
    # results = ray.get(result_ids)
