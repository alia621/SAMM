



# Created by ttwu at 2022/8/14
import copy
import torch.nn as nn
import torch
from collections import namedtuple
from transformers import ViltConfig, PreTrainedTokenizer, PreTrainedModel
from transformers.modeling_utils import *

from module.modeling_vilt import ViltModel
from module.head import ClassificationHead
from module.encoders import TextEncoder, ImageEncoder
from utils import signature
from configs.config_base import MmreConfigBase
from torch.nn import init
import torch.nn.functional as F
from model.losscon import MAGCLLoss,SupConLoss
from torch.nn.parameter import Parameter
import math
import pdb
import numpy as np
class GatedBimodal(nn.Module):
    def __init__(self, dim, activation=None,gate_activation=None):
        super(GatedBimodal, self).__init__()
        self.dim=dim
        if not activation:
            activation = nn.Tanh()
        if not gate_activation:
            gate_activation = nn.Sigmoid()
        self.activation = activation
        self.gate_activation = gate_activation
        self.W1 = nn.Parameter(torch.Tensor(self.dim, self.dim),requires_grad=True)
        self.W2 = nn.Parameter(torch.Tensor(self.dim, self.dim),requires_grad=True)
        self.W3 = nn.Parameter(torch.Tensor(self.dim, self.dim),requires_grad=True)

        nn.init.kaiming_normal(self.W1, mode='fan_out')
        nn.init.kaiming_normal(self.W2, mode='fan_out')
        nn.init.kaiming_normal(self.W3, mode='fan_out')
        # sampling
    def forward(self, x1,x2,x3):
        # x=torch.cat([x1,x2,x3],dim=1)
        h1=self.activation(x1)
        h2=self.activation(x2)
        h3=self.activation(x3)
        z1=self.gate_activation(torch.matmul(x1,self.W1))
        z2=self.gate_activation(torch.matmul(x2,self.W2))
        z3=self.gate_activation(torch.matmul(x3,self.W3))

        # z=self.gate_activation(torch.matmul(x,self.W))
        # TODO: split text and image embedding from embedding_output

        return z1*h1+z2*h2 +z3*h3 , z1+z2+z3

# 多模态注意力融合模块（AT Module）
#dim:hidden_dim，统一后的维度
class classweight(nn.Module):
    def __init__(self, dim, activation=None,gate_activation=None):
        super(classweight, self).__init__()
        self.dim=dim
        #默认使用tanh处理特征，使用sigmoid生成门控权重
        if not activation:
            activation = nn.Tanh()
        if not gate_activation:
            gate_activation = nn.Sigmoid()
        self.activation = activation
        self.gate_activation = gate_activation
        self.W1 = nn.Parameter(torch.Tensor(self.dim, self.dim),requires_grad=True)
        self.W2 = nn.Parameter(torch.Tensor(self.dim, self.dim),requires_grad=True)
        self.W3 = nn.Parameter(torch.Tensor(self.dim, self.dim),requires_grad=True)
        #初始化每个模态的权重矩阵
        nn.init.kaiming_normal(self.W1, mode='fan_out')
        nn.init.kaiming_normal(self.W2, mode='fan_out')
        nn.init.kaiming_normal(self.W3, mode='fan_out')
        # sampling
        #input：三个模态特征（B，hidden_size）
    def forward(self, x1,x2,x3):
        h1=self.activation(x1)#Tanh,特征归一化（-1，1）
        h2=self.activation(x2)
        h3=self.activation(x3)
        # x=torch.cat([x1,x2,x3],dim=1)
        z1=self.gate_activation(torch.matmul(x1,self.W1))#Sigmoid，门控权重（0，1）
        z2=self.gate_activation(torch.matmul(x2,self.W2))
        z3=self.gate_activation(torch.matmul(x3,self.W3))

        # z=self.gate_activation(torch.matmul(x,self.W))
        # TODO: split text and image embedding from embedding_output
        #融合特征（B,H），三个模态的注意力权重z_i（B,H）->对dim1求均值，得到一个直观的模态整体权重（B）,标量
        return z1*h1+z2*h2 +z3*h3 , torch.mean(z1,dim=1), torch.mean(z2,dim=1), torch.mean(z3,dim=1)


class FusionExpert(nn.Module):
    """单个融合专家：独立的三模态门控融合网络"""
    def __init__(self, dim):
        super(FusionExpert, self).__init__()
        self.dim = dim
        self.activation = nn.Tanh()
        self.gate_activation = nn.Sigmoid()
        self.W1 = nn.Parameter(torch.Tensor(dim, dim))
        self.W2 = nn.Parameter(torch.Tensor(dim, dim))
        self.W3 = nn.Parameter(torch.Tensor(dim, dim))
        nn.init.kaiming_normal_(self.W1, mode='fan_out')
        nn.init.kaiming_normal_(self.W2, mode='fan_out')
        nn.init.kaiming_normal_(self.W3, mode='fan_out')

    def forward(self, x1, x2, x3):
        h1 = self.activation(x1)
        h2 = self.activation(x2)
        h3 = self.activation(x3)
        z1 = self.gate_activation(torch.matmul(x1, self.W1))
        z2 = self.gate_activation(torch.matmul(x2, self.W2))
        z3 = self.gate_activation(torch.matmul(x3, self.W3))
        fused = z1 * h1 + z2 * h2 + z3 * h3
        return fused, torch.mean(z1, dim=1), torch.mean(z2, dim=1), torch.mean(z3, dim=1)


class MoEFusion(nn.Module):
    """融合层MoE：多个融合专家 + Router，替代原classweight"""
    def __init__(self, dim, num_experts=4, top_k=2, load_balance_weight=0.01):
        super(MoEFusion, self).__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight

        # 专家池：同时创建多个FusionExpert实例，每个专家独立参数

  
        self.experts = nn.ModuleList([FusionExpert(dim) for _ in range(num_experts)])

        # Router：三模态concat -> softmax门控权重
        #全连接层，输入拼接三维特征，输出各专家的评分
        self.router = nn.Sequential(
            nn.Linear(dim * 3, num_experts),
        )

        # 记录负载均衡损失，供训练时使用
        self.aux_loss = 0.0

    def forward(self, x1, x2, x3):
        batch_size = x1.shape[0]

        # Router输入：三模态特征拼接
        #计算一个batch中每个样本的专家得分
        router_input = torch.cat([x1, x2, x3], dim=1)  # [B, 3*dim]
        router_logits = self.router(router_input)        # [B, num_experts]

        # Top-k选择
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=1)  # [B, top_k]
        top_k_gates = F.softmax(top_k_logits, dim=1)  # [B, top_k] 归一化权重

        # 计算负载均衡损失（训练时）
        if self.training:
            self.aux_loss = self._load_balance_loss(router_logits)

        # 所有专家前向计算
        expert_outputs = []  # 每个专家计算出的一个batch的 (fused, w1, w2, w3)
        for expert in self.experts:
            expert_outputs.append(expert(x1, x2, x3))

        # 
        fused = torch.zeros(batch_size, self.dim, device=x1.device)
        w1 = torch.zeros(batch_size, device=x1.device)
        w2 = torch.zeros(batch_size, device=x1.device)
        w3 = torch.zeros(batch_size, device=x1.device)

        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]     # [B]
            gate_val = top_k_gates[:, k]         # [B]

            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)     # [B] bool
                if mask.any():
                    e_fused, e_w1, e_w2, e_w3 = expert_outputs[e_idx]
                    fused[mask] += gate_val[mask].unsqueeze(1) * e_fused[mask]
                    w1[mask] += gate_val[mask] * e_w1[mask]
                    w2[mask] += gate_val[mask] * e_w2[mask]
                    w3[mask] += gate_val[mask] * e_w3[mask]

        return fused, w1, w2, w3

    def _load_balance_loss(self, router_logits):
        """负载均衡损失：鼓励各专家被均匀使用"""
        # router_probs: [B, num_experts]
        router_probs = F.softmax(router_logits, dim=1)

        # 每个专家的平均使用概率
        avg_probs = router_probs.mean(dim=0)  # [num_experts]

        # 每个专家被选为top-1的频率
        top1_indices = router_logits.argmax(dim=1)  # [B]
        freq = torch.zeros(self.num_experts, device=router_logits.device)
        for i in range(self.num_experts):
            freq[i] = (top1_indices == i).float().mean()

        # Switch Transformer式负载均衡损失
        aux_loss = self.num_experts * (freq * avg_probs).sum()
        return self.load_balance_weight * aux_loss


class ModalityMoE(nn.Module):
    """改进6：模态专属MoE。对单模态投影特征做Top-k稀疏专家变换 + 残差连接。
    每个专家为两层MLP（Linear-GELU-Linear），Router只看本模态特征。"""
    def __init__(self, dim, num_experts=4, top_k=2, load_balance_weight=0.01):
        super(ModalityMoE, self).__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        #expert pool单个modality_expert结构两层全连接
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            for _ in range(num_experts)
        ])
        #router只用本模态特征
        self.router = nn.Linear(dim, num_experts)

        self.aux_loss = 0.0

    def forward(self, x):
        # x: [B, dim]
        # Router 计算专家选择概率
        router_logits = self.router(x)                          # [B, E]分数
        router_probs = F.softmax(router_logits, dim=-1)         #softmax归一化，每一行的专家logist之和为1
        # Top-k 稀疏激活，选择top-k个概率大的expert[B,top-k]，内部有排序，按照概率从大到小
        topk_vals, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)#激活的expert的权重和expert索引
        topk_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)
        #用0补全router_probs [B,top-k]->[B,num_expert]
        route_weight = torch.zeros_like(router_probs)
        route_weight.scatter_(1, topk_indices, topk_weights)     # [B, E]
        #所有expert分别处理输入
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # [B, E, dim]
        delta = (expert_outputs * route_weight.unsqueeze(-1)).sum(dim=1)   # expert的dim进行求和，[B, dim]
        #训练时计算负载的均衡损失
        if self.training:
            # ===== 旧版（只统计 top-1，忽略 top-k 中其余位置，top_k>1 时 freq 与实际负载不一致）=====
            # top1_indices = topk_indices[:, 0]#[B],取每个sample概率最高的expert
            # #freq是每个expert真正被选为top1的频率（真实）
            # freq = torch.zeros(self.num_experts, device=x.device)#[num_expert]
            # for i in range(self.num_experts):
            #     freq[i] = (top1_indices == i).float().mean()
            #
            #     #表示router给每个expert的平均概率（期望）
            # avg_probs = router_probs.mean(dim=0) #对batch维度求平概率[B,E]->[E]
            # #期望 点积 真实 计算两个向量的相似度
            # self.aux_loss = self.num_experts * (freq * avg_probs).sum() * self.load_balance_weight

            # ===== 新版：top-k 全部计入 freq，归一到 Σf_i=1，与 Switch 公式量级一致 =====
            # 1. 展平所有 top-k 选择：[B, top_k] -> [B * top_k]
            flat_indices = topk_indices.flatten()
            # 2. bincount 高效统计每个 expert 被选中次数，minlength 保证未被选中的 expert 计数为 0
            expert_counts = torch.bincount(flat_indices, minlength=self.num_experts).float()
            # 3. 归一化得 f_i，总和 = 1（分母为激活路径总数 B * top_k）
            f_i = expert_counts / flat_indices.numel()
            # 4. 平均路由概率 p_i：对 batch 维度求均值
            p_i = router_probs.mean(dim=0)
            # 5. Switch 公式：E * Σ f_i * p_i，均衡时最小值 ≈ 1
            self.aux_loss = self.num_experts * torch.sum(f_i * p_i) * self.load_balance_weight
        else:
            self.aux_loss = 0.0

        return x + delta  # 残差


# 改进5（重构 方案B）：人工 genre 聚类先验。每个 cluster 由一个专家独立负责。
# 5 簇覆盖 MM-IMDb 2.0 top-23 genre，互斥且并集等于全部 23 类。
GENRE_CLUSTERS = {
    "drama_realism": ["Drama", "Biography", "History", "War"],
    "comedy_romance_family": ["Comedy", "Romance", "Family"],
    "action_crime_thriller": ["Action", "Adventure", "Crime", "Thriller", "Western"],
    "fantasy_horror_mystery": ["Horror", "Mystery", "Sci-Fi", "Fantasy"],
    "form_music_tail": ["Animation", "Documentary", "Music", "Musical", "Sport", "Short", "Film-Noir"],
}


class ClusterMoEClassifier(nn.Module):
    """改进5（重构 方案B）：类型簇级专家分工 + 层次分类 router。

    - 静态分工：每个 expert 输出本 cluster 内 genre 的 logits（维度 = cluster size）
    - Router：sigmoid 输出 K 维 cluster gate，独立判断每个 cluster 是否激活
    - 层次分解：P(g | x) = sigmoid(router_k(h)) · sigmoid(expert_k(h)_g)
    - 辅助损失：BCE(router_logits, cluster_label)，cluster_label[k] = max(label[g], g ∈ cluster_k)
      由 self.aux_loss 暴露给外层 forward 累加
    """
    def __init__(self, input_dim, hidden_size, target_names, clusters=GENRE_CLUSTERS,
                 lambda_aux=0.1):
        super().__init__()
        flat = [g for c in clusters.values() for g in c]
        assert len(flat) == len(set(flat)), f"GENRE_CLUSTERS 内有重复 genre: {flat}"
        missing = [g for g in target_names if g not in flat]
        assert not missing, f"target_names 中以下 genre 未出现在任何 cluster: {missing}"

        self.lambda_aux = lambda_aux
        target_pos = {g: i for i, g in enumerate(target_names)}
        cluster_target_indices = []
        self.cluster_names = []
        self.cluster_sizes = []
        self.experts = nn.ModuleList()
        for cname, glist in clusters.items():
            idxs = sorted(target_pos[g] for g in glist if g in target_pos)
            if not idxs:
                continue
            cluster_target_indices.append(idxs)
            self.cluster_names.append(cname)
            self.cluster_sizes.append(len(idxs))
            self.experts.append(nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, len(idxs)),
            ))

        num_clusters = len(self.experts)
        output_dim = len(target_names)
        flat_target_idx = [i for idxs in cluster_target_indices for i in idxs]
        assert len(flat_target_idx) == output_dim, (
            f"cluster 总输出维度 {len(flat_target_idx)} 与 target_names {output_dim} 不一致")
        perm = [0] * output_dim
        for j, target_i in enumerate(flat_target_idx):
            perm[target_i] = j
        self.register_buffer('perm', torch.tensor(perm, dtype=torch.long))

        cluster_mask = torch.zeros(num_clusters, output_dim)
        for k, idxs in enumerate(cluster_target_indices):
            for g in idxs:
                cluster_mask[k, g] = 1.0
        self.register_buffer('cluster_mask', cluster_mask)

        self.router = nn.Linear(input_dim, num_clusters)
        self.aux_loss = torch.zeros(())

    def forward(self, h, label=None):
        outs = [e(h) for e in self.experts]
        concat = torch.cat(outs, dim=-1)
        expert_logits = concat[:, self.perm]
        expert_probs = torch.sigmoid(expert_logits)

        router_logits = self.router(h)
        gate = torch.sigmoid(router_logits)
        gate_per_genre = gate @ self.cluster_mask

        y_hat = expert_probs * gate_per_genre

        if self.training and label is not None:
            cluster_label = (label.float() @ self.cluster_mask.t() > 0).float()
            self.aux_loss = self.lambda_aux * F.binary_cross_entropy_with_logits(
                router_logits, cluster_label)
        else:
            self.aux_loss = torch.zeros((), device=h.device)
        return y_hat


def build_manual_cluster_assignment(target_names, clusters=GENRE_CLUSTERS,
                                    num_clusters=None, scale=5.0):
    """根据人工 GENRE_CLUSTERS 构造 one-hot 初始化 logits（× scale 给强信号）。

    Returns
    -------
    Tensor [G, K]，K = len(clusters)
    """
    K = len(clusters)
    if num_clusters is not None and num_clusters != K:
        raise ValueError(
            f"manual init 要求 num_clusters={K}（GENRE_CLUSTERS 簇数），但收到 {num_clusters}")
    target_pos = {g: i for i, g in enumerate(target_names)}
    G = len(target_names)
    A = torch.zeros(G, K)
    for k, (_cname, glist) in enumerate(clusters.items()):
        for g in glist:
            if g in target_pos:
                A[target_pos[g], k] = 1.0
    empty_rows = (A.sum(dim=-1) == 0).nonzero(as_tuple=True)[0].tolist()
    if empty_rows:
        missing = [target_names[i] for i in empty_rows]
        raise ValueError(f"以下 genre 未出现在 GENRE_CLUSTERS 中: {missing}")
    return A * scale


def build_cooccur_cluster_assignment(label_matrix, num_clusters, scale=5.0,
                                     random_state=42):
    """基于训练集 label 共现矩阵的 K-means 聚类初始化。
    它只是一次性离线初始化 self.A，完全不在 forward / 训练循环内。
    这点务必在论文里说清楚——不要把 K-means 写成参与前向或被损失监督的组件。

    Parameters
    ----------
    label_matrix : np.ndarray [N, G]，训练集 multi-hot 标签,[训练集samples数，23]，训练集的gengt
    num_clusters : int 分到几个cluster里
    scale：输出初始化矩阵的放大系数
    Returns
    -------
    Tensor [G, K]，one-hot × scale
    """
    from sklearn.cluster import KMeans
    Y = np.asarray(label_matrix).astype(np.float32)
    G = Y.shape[1]  #取genre数23
    C = Y.T @ Y  # [G, G] 共现矩阵
    #不对电影样本聚类，而是对genre本身进行聚类
    # 共现矩阵归一化
    # 后面 KMeans 用的是欧氏距离。
    # 如果直接拿原始计数 C 聚类，高频 genre（Drama 这类）的计数绝对值远大于长尾 genre
    # 距离会被频次量级主导——结果基本只是"按出现频次分簇"，而不是按语义共现模式。
    row_sum = C.sum(axis=1, keepdims=True)#对每一行进行求和，求每个genre共现的次数[G,1]，
    feat = C / np.clip(row_sum, 1e-6, None)  #[G,G]共现矩阵每一行除以行和，每行变成一个和为1的概率分布 row-normalize 作为 genre 特征
    #构造一个kmeans聚类器，n_init=10：跑 10 组不同初始质心，保留 inertia（簇内平方和，这里 0.1394）最小的那次
    km = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    #(1) 每个点归到最近的质心;(2) 质心更新为所属点的均值。
    cluster_id = km.fit_predict(feat)  # [G]，表示第i个genre分配到第几cluster
    #转为one-hot矩阵[G,cluster_num],每一行代表一个genre
    A = torch.zeros(G, num_clusters)#
    for g, k in enumerate(cluster_id):
        A[g, int(k)] = 1.0
        #如果某一列全为0，即K-means产生空簇
    empty_cols = (A.sum(dim=0) == 0).nonzero(as_tuple=True)[0].tolist()
    for k in empty_cols:
        A[:, k] = torch.randn(G) * 0.01#用一个很小的随机值填充这一列
    return A * scale


class LearnableClusterMoEClassifier(nn.Module):
    """改进5-Learnable：可学习类型簇分配 + 类型簇级专家分类头。

    - 可学习 assignment A ∈ R^{G×K}，每 genre 在 K 个 cluster 上的分配 logits
    - 训练：Gumbel-Softmax hard=True 采样 → 每行 one-hot 的 M ∈ {0,1}^{G×K}（ST estimator）
    - 推理：argmax(A) → one-hot，确定性
    - K 个 expert 均输出完整 G 维 logits；用 M 选择"genre g 取哪个 expert 的输出"
      （hard one-hot 下等于每个 expert 参数只在自己负责的 genre 上有梯度）
    - Router: Linear(d, K) sigmoid → cluster gate
    - y_hat = sigmoid(expert_logits) · (gate @ M.t())
    - 辅助损失：
      * BCE(router_logits, cluster_label)，cluster_label[b, k]=max_g{label[b,g]·M[g,k]}
      * Cluster size entropy collapse 正则：log(K) − H(cluster size 分布)
    """
    def __init__(self, input_dim, hidden_size, target_names, num_clusters=5,
                 tau=1.0, lambda_aux=0.1, lambda_collapse=0.01,
                 init_assignment=None):
        super().__init__()
        self.num_clusters = num_clusters
        self.tau = tau
        self.lambda_aux = lambda_aux
        self.lambda_collapse = lambda_collapse
        output_dim = len(target_names)
        self.output_dim = output_dim

        if init_assignment is not None:
            assert init_assignment.shape == (output_dim, num_clusters), (
                f"init_assignment 形状 {tuple(init_assignment.shape)} ≠ "
                f"({output_dim}, {num_clusters})")
            self.A = nn.Parameter(init_assignment.clone().float())
        else:
            self.A = nn.Parameter(torch.randn(output_dim, num_clusters) * 0.1)
    #每个expert输入[B,Hidden_dim]->[B,G]
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim),
            )
            for _ in range(num_clusters)
        ])

        self.router = nn.Linear(input_dim, num_clusters)
        self.aux_loss = torch.zeros(())
        # 拆分日志：BCE / collapse 两个子项的原始值（未乘 lambda）
        self.bce_loss = torch.zeros(())
        self.collapse_loss = torch.zeros(())
#训练和推理两套方法
    def _sample_mask(self):
        if self.training:
            return F.gumbel_softmax(self.A, tau=self.tau, hard=True, dim=-1)
        idx = self.A.argmax(dim=-1)
        return F.one_hot(idx, num_classes=self.num_clusters).float()#推理直接argmax（A）

    def forward(self, h, label=None):
        M = self._sample_mask()  # [G, K]
        #融合特征的输入途径一:classifier分类
        expert_stack = torch.stack([e(h) for e in self.experts], dim=0)  # [K, B, G]
        # M.t(): [K, G]，broadcast 到 [K, 1, G]；与 [K, B, G] 相乘后 sum 得 [B, G]
        expert_logits = (M.t().unsqueeze(1) * expert_stack).sum(dim=0)#[B,G]
        expert_probs = torch.sigmoid(expert_logits)
        #途径二：各classifier的权重
        router_logits = self.router(h)
        gate = torch.sigmoid(router_logits)
        gate_per_genre = gate @ M.t()

        y_hat = expert_probs * gate_per_genre

        if self.training and label is not None:
            cluster_label = (label.float() @ M > 0).float()  # [B, K]
            bce_loss = F.binary_cross_entropy_with_logits(router_logits, cluster_label)

            cluster_size = M.sum(dim=0)  # [K]
            dist = cluster_size / cluster_size.sum().clamp(min=1e-6)
            entropy = -(dist * (dist + 1e-9).log()).sum()
            collapse_loss = math.log(self.num_clusters) - entropy

            self.bce_loss = bce_loss.detach()
            self.collapse_loss = collapse_loss.detach()
            self.aux_loss = (self.lambda_aux * bce_loss
                             + self.lambda_collapse * collapse_loss)
        else:
            self.aux_loss = torch.zeros((), device=h.device)
            self.bce_loss = torch.zeros((), device=h.device)
            self.collapse_loss = torch.zeros((), device=h.device)
        return y_hat


class clipweightClassifier(nn.Module):
    def __init__(self, visual_dim, text_dim,kg_dim,output_dim,hidden_size,use_magcl=True,
                 use_moe=False, num_experts=4, top_k=2,
                 args=None, target_names=None, cluster_init_assignment=None):
        super(clipweightClassifier, self).__init__()
        self.use_magcl=use_magcl  # 改进3消融开关
        self.use_moe=use_moe      # 改进4：MoE融合开关
        # 改进5（重构 方案B）消融开关：类型簇级专家分工分类头 + router
        self.use_cluster_moe = bool(args is not None and getattr(args, 'use_cluster_moe', False))
        # 改进5-Learnable 消融开关：可学习类型簇分配（Gumbel-Softmax hard）
        self.use_learnable_cluster = bool(args is not None and getattr(args, 'use_learnable_cluster', False))
        assert not (self.use_cluster_moe and self.use_learnable_cluster), (
            "use_cluster_moe 和 use_learnable_cluster 互斥，请只开一个")
        # 改进6消融开关：模态专属MoE
        self.use_modality_moe = bool(args is not None and getattr(args, 'use_modality_moe', False))
        # 模态投影层：简单的BN+Linear
        self.vis_proj = nn.Sequential(nn.BatchNorm1d(visual_dim), nn.Linear(visual_dim, hidden_size))
        self.text_proj = nn.Sequential(nn.BatchNorm1d(text_dim), nn.Linear(text_dim, hidden_size))
        self.kg_proj = nn.Sequential(nn.BatchNorm1d(kg_dim), nn.Linear(kg_dim, hidden_size))
        # 改进6：每个模态独立一组ModalityMoE，只在开关打开时构建
        if self.use_modality_moe:
            mod_num_experts = getattr(args, 'mod_num_experts', 4)
            mod_top_k = getattr(args, 'mod_top_k', 2)
            mod_lb_weight = getattr(args, 'mod_load_balance_weight', 0.01)
            self.vis_moe = ModalityMoE(hidden_size, num_experts=mod_num_experts, top_k=mod_top_k, load_balance_weight=mod_lb_weight)
            self.text_moe = ModalityMoE(hidden_size, num_experts=mod_num_experts, top_k=mod_top_k, load_balance_weight=mod_lb_weight)
            self.kg_moe = ModalityMoE(hidden_size, num_experts=mod_num_experts, top_k=mod_top_k, load_balance_weight=mod_lb_weight)
        self.labelmlp=nn.Sequential(
        nn.Linear(kg_dim,hidden_size))
        self.gbu=GatedBimodal(hidden_size)
        # 改进4：根据开关选择融合模块
        if self.use_moe:
            lb_weight = getattr(args, 'load_balance_weight', 0.01) if args is not None else 0.01
            self.clswei=MoEFusion(hidden_size, num_experts=num_experts, top_k=top_k, load_balance_weight=lb_weight)
        else:
            self.clswei=classweight(hidden_size)
        # 改进5：根据开关选择分类头（learnable_cluster 优先）
        if self.use_learnable_cluster:
            assert target_names is not None, "use_learnable_cluster=True 时必须传入 target_names"
            num_clusters = getattr(args, 'num_clusters', 5)
            cluster_tau = getattr(args, 'cluster_tau', 1.0)
            lambda_cluster_aux = getattr(args, 'lambda_cluster_aux', 0.1)
            lambda_collapse = getattr(args, 'lambda_collapse', 0.01)
            self.classifier = LearnableClusterMoEClassifier(
                hidden_size, hidden_size, target_names,
                num_clusters=num_clusters, tau=cluster_tau,
                lambda_aux=lambda_cluster_aux, lambda_collapse=lambda_collapse,
                init_assignment=cluster_init_assignment,
            )
        elif self.use_cluster_moe:
            assert target_names is not None, "use_cluster_moe=True 时必须传入 target_names"
            lambda_cluster_aux = getattr(args, 'lambda_cluster_aux', 0.1) if args is not None else 0.1
            self.classifier = ClusterMoEClassifier(
                hidden_size, hidden_size, target_names,
                lambda_aux=lambda_cluster_aux,
            )
        else:
            self.classifier = ClipGenreClassifier(hidden_size, output_dim, hidden_size)
        # 改进3：根据开关选择对比损失函数
        magcl_temp = getattr(args, 'magcl_temperature', 0.1) if args is not None else 0.1
        if self.use_magcl:
            self.contloss=MAGCLLoss(temperature=magcl_temp)   # 改进3：双视角对比学习
        else:
            self.contloss=SupConLoss(temperature=magcl_temp)  # 原版：监督对比学习
        self.t=nn.Parameter(torch.Tensor(1),requires_grad=True)
        # sampling

    # def contrasloss_features(self,xfuse,labelem,label):
        
    #     anchorlist=[]
    #     labellist=[]
    #     for i in range(len(xfuse)):
    #         emb=labelem[i]
    #         lab=label[i]
    #         mask=lab==1
    #         compute_emb=emb[mask]
    #         anchor=torch.mean(compute_emb,dim=0).tolist()
    #         anchorlist.append(anchor)
    #         labellist.append(i)

                    

    #     indtor=torch.LongTensor(labellist).cuda()
    #     pos=torch.tensor(anchorlist).cuda()
    #     indlabel=torch.cat([indtor,indtor]).cuda()
    #     feas=torch.cat([xfuse,pos]).cuda()
        
    #     # pdb.set_trace()
    #     contraloss=self.contloss(feas,indlabel)
    #     return contraloss

    def contrasloss_features(self, xfuse, labelem, label):
        """
        输入维度假设:
        xfuse:   [Batch, Hidden]
        labelem: [Batch, Num_Genres, Hidden]  <-- 3D 张量
        label:   [Batch, Num_Genres]
        """
        
        # 1. 调整 label 维度以适配 Batch 矩阵乘法
        # [Batch, Num_Genres] -> [Batch, 1, Num_Genres]
        label_unsqueezed = label.float().unsqueeze(1)
        
        # 2. 计算每个样本所属流派嵌入的和
        # bmm: [B, 1, N] @ [B, N, H] -> [B, 1, H]
        # 这一步利用了显卡并行计算，比 for 循环快得多
        genre_sum = torch.matmul(label_unsqueezed, labelem)
        
        # 3. 去掉中间的维度 -> [B, H]
        genre_sum = genre_sum.squeeze(1)
        
        # 4. 计算每个样本包含的流派数量 (分母)
        # [Batch, Num_Genres] -> [Batch, 1]
        num_genres = torch.sum(label, dim=1, keepdim=True)
        
        # 5. 求平均值得到质心 (Anchor)
        # 加上 1e-6 避免除以 0 (如果某个样本没有标签)
        anchors = genre_sum / (num_genres + 1e-6)

        # 改进3：根据开关选择不同的对比损失计算方式
        if self.use_magcl:
            # 改进3：MAGCLLoss，需要全局流派嵌入用于双视角负采样
            # labelem 是 3D [B,N,H]，取第一个样本的流派嵌入即可: [N, H]
            if labelem.dim() == 3:
                all_genre_embeddings = labelem[0]
            else:
                all_genre_embeddings = labelem
            contraloss = self.contloss(xfuse, anchors, all_genre_embeddings, label)
        else:
            # 原版：SupConLoss，将样本特征与其质心拼接，用样本索引作为标签
            batch_size = xfuse.shape[0]
            indlabel = torch.arange(batch_size).to(xfuse.device)
            indlabel = torch.cat([indlabel, indlabel])  # 样本和其质心共享同一标签
            feas = torch.cat([xfuse, anchors])           # [2B, H]
            contraloss = self.contloss(feas, indlabel)
        return contraloss
    
    def forward(self, x1,x2,x3,x4,label):
        # 简单模态投影
        vh = self.vis_proj(x1)
        th = self.text_proj(x2)
        kh = self.kg_proj(x3)

        # 改进6：模态专属MoE（残差），加入modmoe
        if self.use_modality_moe:
            vh = self.vis_moe(vh)
            th = self.text_moe(th)
            kh = self.kg_moe(kh)

        x4=self.labelmlp(x4)
        # 门控融合
        h,w1,w2,w3=self.clswei(vh,th,kh)

        contraloss=self.contrasloss_features(h,x4,label)

        # 改进5：cluster_moe / learnable_cluster 路径需要 label 监督 router
        if self.use_cluster_moe or self.use_learnable_cluster:
            y_hat = self.classifier(h, label)
        else:
            y_hat = self.classifier(h)

        # 改进4/改进5/改进6 的辅助损失独立打包，不再混入 contraloss
        moe_aux_loss = torch.zeros((), device=h.device)
        if self.training:
            if self.use_moe:
                moe_aux_loss = moe_aux_loss + self.clswei.aux_loss
            if self.use_modality_moe:
                moe_aux_loss = moe_aux_loss + self.vis_moe.aux_loss + self.text_moe.aux_loss + self.kg_moe.aux_loss
            if self.use_cluster_moe or self.use_learnable_cluster:
                moe_aux_loss = moe_aux_loss + self.classifier.aux_loss

        return y_hat,w1,w2,w3,contraloss,moe_aux_loss

class ClipGenreClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super(ClipGenreClassifier, self).__init__()
        # self.linear1=Maxout(input_dim,hidden_size,2)
        # self.linear2=Maxout(hidden_size,hidden_size,2)
        self.linear1=nn.Linear(input_dim,hidden_size)
        self.linear2=nn.Linear(hidden_size,hidden_size)

        self.linear3=nn.Linear(hidden_size,output_dim)
        # sampling
    def forward(self, x):


        return torch.sigmoid(self.linear3(self.linear2(self.linear1(x))))








