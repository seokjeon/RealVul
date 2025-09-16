from argparse import ArgumentParser
import os,sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from pytorch_lightning import seed_everything, Trainer
import torch
from src.models.vd import DeepWuKong
from src.datas.datamodules import XFGDataModule
from src.utils import filter_warnings
import os


def test(checkpoint_path: str, root_folder_path: str = None,split_folder_name: str = None, batch_size: int = None,sample_percentage=1,sub_folder=""):
    """

    test the trained model using specified files

    Args:
        checkpoint_path:
        data_folder:
        batch_size:

    Returns:

    """
    filter_warnings()
    model = DeepWuKong.load_from_checkpoint(checkpoint_path)
    config = model.hparams["config"]
    vocabulary = model.hparams["vocab"]
    if batch_size is not None:
        config.hyper_parameters.test_batch_size = batch_size
    if root_folder_path is not None:
        config.root_folder_path = root_folder_path
    if split_folder_name is not None:
        config.split_folder_name = split_folder_name
    

   
    data_module = XFGDataModule(config, vocabulary)
    gpu = 1 if torch.cuda.is_available() else None
    trainer = Trainer(gpus=gpu)
    trainer.test(model, datamodule=data_module)


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("checkpoint", type=str)
    arg_parser.add_argument("--root_folder_path", type=str, default=None)
    arg_parser.add_argument("--split_folder_name", type=str, default=None)
    arg_parser.add_argument("--batch-size", type=int, default=None)
    return arg_parser


if __name__ == '__main__':
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    test(__args.checkpoint, __args.root_folder_path,__args.split_folder_name, __args.batch_size)#, __args.sample_percentage,__args.sub_folder)
