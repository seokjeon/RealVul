from torch.utils.data import Dataset
from omegaconf import DictConfig
from src.datas.graphs import XFG
from src.datas.samples import XFGSample
from os.path import exists,join
import json
from src.vocabulary import Vocabulary
from tqdm import  tqdm
import ray
import os
ray.init(_plasma_directory="/tmp")
from random import sample
@ray.remote
def read_xfg(xfg_path):
        return XFG(xfg_path)

class XFGDataset(Dataset):
    def __init__(self, XFG_paths_json: str, config: DictConfig, vocab: Vocabulary) -> None:
        """
        Args:
            XFG_root_path: json file of list of XFG paths
        """
        super().__init__()
        self.__config = config
        assert exists(XFG_paths_json), f"{XFG_paths_json} not exists!"
        with open(XFG_paths_json, "r") as f:
            self.__XFG_paths_all = list(json.load(f))
        self.__vocab = vocab
        self.__XFGs = list()

        self.root_XFG_path = config.local_dir_xfg_path # os.environ["SLURM_TMPDIR"]
       
        # ray_xfgs=[]
        # for xfg_path in __XFG_paths_all:
        #     ray_xfgs.append(read_xfg.remote(join(root_XFG_path,xfg_path)))
        # self.__XFGs = ray.get(ray_xfgs)
        self.__n_samples = len(self.__XFG_paths_all)

    def __len__(self) -> int:
        return self.__n_samples

    def __getitem__(self, index) -> XFGSample:
        # xfg: XFG = self.__XFGs[index]
        xfg_path = join(self.root_XFG_path, self.__XFG_paths_all[index])
        xfg = XFG(xfg_path)  # 디스크에서 읽기
        return XFGSample(graph=xfg.to_torch(self.__vocab,
                                            self.__config.dataset.token.max_parts),
                         label=xfg.label,file_path=xfg.file_path)

    def get_n_samples(self):
        return self.__n_samples
