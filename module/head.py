import torch
import torch.nn as nn

from transformers import PretrainedConfig
from configs.config_base import MmreConfigBase

from module.modeling_vilt import ViltPooler


class ClassificationHead(nn.Module):#定义分类头
    def __init__(self, backbone_config: PretrainedConfig, training_config: MmreConfigBase):#初始化分类头，接收两个配置参数
        super(ClassificationHead, self).__init__()
        self.config = backbone_config
        self.training_config = training_config
        print("config.num_labels", self.config.num_labels)
        self.hidden_size = 100
        self.net = nn.Sequential(#构建一个神经网络序列，包含以下层，包括线性层、层归一化、GELU激活函数和输出层
            nn.Linear(self.config.hidden_size*2, self.config.hidden_size),#将输入的隐藏状态维度从hidden_size*2映射到hidden_size
            nn.LayerNorm(self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.training_config.num_labels)
        )

    def forward(self, output):
        if self.training_config.classification_feat == "avg":
            processed_hs = torch.mean(output.last_hidden_state, dim=1)
        elif self.training_config.classification_feat == "pointer":
            # otherwise using the first embedding as the presentation
            processed_hs = output.pooler_output
        else:
            # default method is average pooling of the last hidden state
            processed_hs = torch.mean(output.last_hidden_state, dim=1)
        print("pooled hidden states", processed_hs.shape)
        output = self.net(processed_hs)
        print(type(output))
        return output
