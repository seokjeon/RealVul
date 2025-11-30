import sys
from os import cpu_count
from gensim.models import Word2Vec, KeyedVectors
import os
import networkx as nx
import json
from omegaconf import OmegaConf, DictConfig
from typing import cast, List
from argparse import ArgumentParser
module_path = os.path.abspath(
    '.')

if module_path not in sys.path:
    sys.path.append(module_path)
#print(sys.path)
from utils import PAD, MASK, UNK
import functools
from multiprocessing import cpu_count, Manager, Pool
from tqdm import tqdm
from os.path import join


SPECIAL_TOKENS = [PAD, UNK, MASK]
USE_CPU = cpu_count()


def process_parallel(path: str, split_token: bool, config):
    """

    Args:
        path:

    Returns:

    """
    xfg = nx.read_gpickle(
        join(os.environ["SLURM_TMPDIR"], config.local_dir_xfg_path, path))

    tokens_list = list()
    for ln in xfg:
        code_tokens = xfg.nodes[ln]["code_sym_token"]

        if len(code_tokens) != 0:
            tokens_list.append(code_tokens)

    return tokens_list


def train_word_embedding(config_path: str):
    """
    train word embedding using word2vec

    Args:
        config_path:

    Returns:

    """
    config = cast(DictConfig, OmegaConf.load(config_path))
    root = config.root_folder_path
    split_folder = config.split_folder_name
    train_json = f"{root}/{split_folder}/train.json"
    with open(train_json, "r") as f:
        paths = json.load(f)
    tokens_list = list()
    with Manager():
        pool = Pool(USE_CPU)

        process_func = functools.partial(process_parallel,
                                         split_token=config.split_token, config=config)
        tokens: List = [
            res
            for res in tqdm(
                pool.imap_unordered(process_func, paths),
                desc=f"xfg paths: ",
                total=len(paths),
            )
        ]
        pool.close()
        pool.join()
    for token_l in tokens:
        tokens_list.extend(token_l)

    print("training w2v...")
    num_workers = cpu_count(
    ) if config.num_workers == -1 else config.num_workers
    model = Word2Vec(sentences=tokens_list, min_count=3, size=config.gnn.embed_size,
                     max_vocab_size=config.dataset.token.vocabulary_size, workers=num_workers, sg=1)
    model.wv.save(f"{root}/{split_folder}/w2v.wv")
    model.save(f"{root}/{split_folder}/w2v.model")
    print("Embedding Saved")


def load_wv(config_path: str):
    """

    Args:
        config_path:

    Returns:

    """
    config = cast(DictConfig, OmegaConf.load(config_path))
    cweid = config.dataset.name

    model = KeyedVectors.load(f"{root}/{split_folder}/w2v.wv", mmap="r")

    print()


if __name__ == '__main__':
    os.environ["SLURM_TMPDIR"] = "."
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("-c",
                              "--config",
                              help="Config File",
                              default="config.yaml",
                              type=str)
    __args = __arg_parser.parse_args()
    train_word_embedding(__args.config)
    # load_wv(__args.config)
