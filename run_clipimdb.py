import os
import argparse
import logging
import sys
sys.path.append("..")
import logging
import pdb
import os
import wandb

# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader

from processor.dataset import imdbclipdataset
# from processor.testdata import gmudataset
from configs.exp_1 import Exp1
from modules.train import clipweightTrainer
import warnings
from module.modeling_vilt import ViltModel
import torch
from PIL import Image
import open_clip
from model.clipclass import clipweightClassifier
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# MODEL_CLASSES = {
#     'MRE': ConditionedVilt,

# }
#参数为clipweight时使用imdbclipdataset数据集类
DATASET = {

    'clipweight':imdbclipdataset,
}


        
# }
#
TRAINER_CLASSES = {
    'clipweight':clipweightTrainer,

}


def set_seed(seed=2021):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='MRE', type=str, help="The name of dataset.")
    parser.add_argument('--model_name', default='dandelin/vilt-b32-mlm', type=str, help="Pretrained language model path")
    parser.add_argument('--num_epochs', default=30, type=int, help="num training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=1, type=int, help="epoch to start evluate")
    parser.add_argument('--seed', default=1, type=int, help="random seed, default is 1")
    parser.add_argument('--prompt_len', default=10, type=int, help="prompt length")
    parser.add_argument('--prompt_dim', default=800, type=int, help="mid dimension of prompt project layer")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default=None, type=str, help="save model at save_path")
    parser.add_argument('--write_path', default=None, type=str, help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--use_prompt', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--max_seq', default=128, type=int)
    parser.add_argument('--vis_dim', default=4096, type=int)
    parser.add_argument('--text_dim', default=300, type=int)
    parser.add_argument('--kg_dim', default=400, type=int)
    parser.add_argument('--n_classes', default=23, type=int)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--ignore_idx', default=-100, type=int)
    parser.add_argument('--modelclass', default='kg', type=str)
    parser.add_argument('--sample_ratio', default=1.0, type=float, help="only for low resource.")
    parser.add_argument('--name', default='default-run', type=str, help="The name of this experiment for WandB")
    # 改进1消融开关：协作加权KG特征
    parser.add_argument('--use_collab_weight', action='store_true', default=True, help="启用改进1：协作加权KG特征")
    parser.add_argument('--no_collab_weight', dest='use_collab_weight', action='store_false', help="消融实验：关闭改进1，使用原版等权求和")
    # 改进3消融开关：多视角自适应对比学习MAGCL
    parser.add_argument('--use_magcl', action='store_true', default=True, help="启用改进3：MAGCLLoss双视角对比学习")
    parser.add_argument('--no_magcl', dest='use_magcl', action='store_false', help="消融实验：关闭改进3，使用原版SupConLoss")
    # 改进2消融开关：CLIP相似度伪标签注意力损失
    parser.add_argument('--use_clip_att', action='store_true', default=True, help="启用改进2：CLIP伪标签注意力损失L_C")
    parser.add_argument('--no_clip_att', dest='use_clip_att', action='store_false', help="消融实验：关闭改进2，只用L_K")
    # 损失权重参数
    parser.add_argument('--lambda_atten', default=0.1, type=float, help="注意力损失L_atten的权重")
    parser.add_argument('--lambda_cl', default=0.1, type=float, help="对比学习损失L_cl的权重")
    parser.add_argument('--magcl_temperature', default=0.1, type=float, help="MAGCL/SupConLoss InfoNCE 温度系数（IDKG 原版=0.1）")
    # 改进4：融合层MoE参数
    parser.add_argument('--use_moe', action='store_true', default=False, help="启用改进4：融合层MoE")
    parser.add_argument('--num_experts', default=4, type=int, help="MoE专家数量")
    parser.add_argument('--top_k', default=2, type=int, help="MoE每次激活的专家数")
    parser.add_argument('--load_balance_weight', default=0.01, type=float, help="融合层MoE负载均衡损失权重（默认0.01，权重救援实验扫描）")
    # 改进5（重构 方案B）：类型簇级专家分工分类头 + router 层次分类
    parser.add_argument('--use_cluster_moe', action='store_true', default=False, help="启用改进5（重构 方案B）：cluster_moe 分类头 + router")
    parser.add_argument('--lambda_cluster_aux', default=0.1, type=float, help="cluster_moe 辅助损失权重（router 的 cluster-membership BCE）")
    # 改进5-Learnable：可学习类型簇分配（Gumbel-Softmax hard 采样）
    parser.add_argument('--use_learnable_cluster', action='store_true', default=False, help="启用改进5-Learnable：可学习类型簇分配 + 类型簇级专家")
    parser.add_argument('--num_clusters', default=5, type=int, help="learnable_cluster 簇数 K（默认5，与人工先验一致；扫 K 时改这里）")
    parser.add_argument('--cluster_tau', default=1.0, type=float, help="Gumbel-Softmax 温度（默认1.0，固定不退火）")
    parser.add_argument('--cluster_init', default='manual', choices=['manual', 'cooccur', 'random'], help="learnable_cluster 初始化策略（K=5 时 manual；K≠5 自动 fallback 到 cooccur）")
    parser.add_argument('--lambda_collapse', default=0.01, type=float, help="cluster size entropy collapse 正则权重（防止所有 genre 塌到一个 cluster）")
    # 改进6：模态专属MoE参数（消融开关，不加 --use_modality_moe 则不启用）
    parser.add_argument('--use_modality_moe', action='store_true', default=False, help="启用改进6：模态专属MoE（图像/文本/KG各一组专家，残差连接）")
    parser.add_argument('--mod_num_experts', default=4, type=int, help="模态MoE每个模态的专家数量")
    parser.add_argument('--mod_top_k', default=2, type=int, help="模态MoE每次激活的专家数")
    parser.add_argument('--mod_load_balance_weight', default=0.01, type=float, help="模态MoE负载均衡损失权重（默认0.01，诊断实验建议尝试0.05/0.1）")
    args = parser.parse_args()
    wandb.init(
        project="IDKG-Movie-Genre-New", #
        name=args.name,
        config=args
    )

    # data_path, img_path = DATA_PATH[args.dataset_name], IMG_PATH[args.dataset_name]
    Trainer = TRAINER_CLASSES[args.dataset_name]
    # data_process, dataset_class = DATA_PROCESS[args.dataset_name]
    dataset=DATASET[args.dataset_name]
    #clipmodel的作用是提取图像/文本特征，preprocess是clip的预处理函数，用于处理输入图像
    clipmodel, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')

    set_seed(args.seed) # set seed, default is 1
    if args.save_path is not None:  # make save_path dir
        # args.save_path = os.path.join(args.save_path, args.dataset_name+"_"+str(args.batch_size)+"_"+str(args.lr)+"_"+args.notes)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
    print(args)
    # pdb.set_trace()
    train_dataset = dataset(mode='train',processor=preprocess,use_collab_weight=args.use_collab_weight)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # for i,sample in enumerate(train_dataloader):
    #     print(sample)
    #     pdb.set_trace()
    dev_dataset = dataset(mode='dev',processor=preprocess,use_collab_weight=args.use_collab_weight)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_dataset = dataset(mode='test',processor=preprocess,use_collab_weight=args.use_collab_weight)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # pdb.set_trace()

    # 改进5-Learnable：根据 cluster_init 准备初始化 assignment 矩阵
    cluster_init_assignment = None
    if args.use_learnable_cluster:
        init_mode = args.cluster_init
        if init_mode == 'manual' and args.num_clusters != 5:
            logger.warning(
                f"--cluster_init=manual 要求 num_clusters=5，但收到 num_clusters={args.num_clusters}，自动 fallback 到 cooccur")
            init_mode = 'cooccur'
        if init_mode == 'manual':
            from model.clipclass import build_manual_cluster_assignment
            cluster_init_assignment = build_manual_cluster_assignment(
                train_dataset.target_names, num_clusters=args.num_clusters)
        elif init_mode == 'cooccur':
            from model.clipclass import build_cooccur_cluster_assignment
            gengt = train_dataset.gengt
            if isinstance(gengt, list):
                Y_train = np.stack([np.asarray(y) for y in gengt], axis=0)
            else:
                Y_train = np.asarray(gengt)
            cluster_init_assignment = build_cooccur_cluster_assignment(
                Y_train, args.num_clusters)
        # random: 保持 None，让模型内部 randn 初始化
        shape = tuple(cluster_init_assignment.shape) if cluster_init_assignment is not None else None
        logger.info(f"learnable_cluster init: mode={init_mode}, K={args.num_clusters}, shape={shape}")

    if args.modelclass=='clipweight':
        model = clipweightClassifier(args.vis_dim, args.text_dim,args.kg_dim,args.n_classes,args.hidden_size,
                                     use_magcl=args.use_magcl, use_moe=args.use_moe,
                                     num_experts=args.num_experts, top_k=args.top_k,
                                     args=args, target_names=train_dataset.target_names,
                                     cluster_init_assignment=cluster_init_assignment)
        # for name, param in model.named_parameters():
        #     print(name)
    # pdb.set_trace()
    trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model, clipmodel=clipmodel,args=args, logger=logger, target_names=train_dataset.target_names)
    if args.do_train:
        # train
        trainer.train()
        # test best model
        args.load_path = os.path.join(args.save_path, 'best_model.pth')
        trainer.test()

    if args.only_test:
        # only do test
        trainer.test()

    torch.cuda.empty_cache()
    

if __name__ == "__main__":
    main()
