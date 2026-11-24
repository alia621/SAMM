# Created by ttwu at 2022/8/14
from dataclasses import dataclass
from typing import AnyStr, List


@dataclass
#定义了一个基于数据类的配置基类基础类 MmreConfigBase
class MmreConfigBase:
    def __init__(self):
        self.dataset_dir: str  #dataset_dir：数据集目录（未设默认值，需用户指定）。
        self.num_labels: int = 23  #num_labels：标签数量，默认为23。

        self.do_prefix: bool = True    #是否使用前缀调优，默认为True。
        self.prefix_type: str = "input"  # or "full"#前缀类型，可以是"input"或"full"，默认为"input"。
        self.prefix_len: int = 5 #前缀长度，默认为5。
        # or "pointer",classification_feat：分类特征类型，可选值为 "avg"（平均特征）或 "pointer"
        # （使用前缀后的后的第一个 token 作为预测特征），默认值为 "avg"
        self.classification_feat: str = "avg"
        # then use the first token after prefix for prediction

        self.check_conflict()

    def check_conflict(self):
        # check for number of classes
        assert self.num_labels > 0

        # check for prefix length
        if self.do_prefix:
            assert self.prefix_len > 0
            assert self.prefix_type in ("input", "full")




