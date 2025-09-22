from argparse import ArgumentParser
import os,sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
from typing import cast

from commode_utils.common import print_config
from omegaconf import OmegaConf, DictConfig
from src.datas.datamodules import XFGDataModule
from src.models.vd import DeepWuKong
from src.train import train
from src.utils import filter_warnings, PAD
from src.vocabulary import Vocabulary
import os


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c",
                            "--config",
                            help="Config File",
                            default="config.yaml",
                            type=str)
    return arg_parser


def vul_detect(config_path: str):
    filter_warnings()
    config = cast(DictConfig, OmegaConf.load(config_path))
    print_config(config, ["gnn", "classifier", "hyper_parameters"])

    vocab = Vocabulary.build_from_w2v(config.gnn.w2v_path)
    vocab_size = vocab.get_vocab_size()
    pad_idx = vocab.get_pad_id()

    # Init datamodule
    data_module = XFGDataModule(config, vocab)

    # Init model
    model = DeepWuKong(config, vocab, vocab_size, pad_idx)

    train(model, data_module, config)


if __name__ == "__main__":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TORCH_USE_CUDA_DETERMINISTIC"] = "0"
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    vul_detect(__args.config)
