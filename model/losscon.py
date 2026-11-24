import torch
import torch.nn as nn
import torch.nn.functional as F
class SupConLoss(nn.Module):

    def __init__(self, temperature=0.1, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        """
        输入:
            features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            labels: 每个样本的ground truth标签，尺寸是[batch_size].
            mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 如果样本i和j属于同一个label，那么mask_{i,j}=1 
        输出:
            loss值
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # 关于labels参数
        if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
            raise ValueError('Cannot define both `labels` and `mask`') 
        elif labels is None and mask is None: # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None: # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        '''
        示例: 
        labels: 
            tensor([[1.],
                    [2.],
                    [1.],
                    [1.]])
        mask:  # 两个样本i,j的label相等时，mask_{i,j}=1
            tensor([[1., 0., 1., 1.],
                    [0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.]]) 
        '''
        # compute logits
        # import pdb
        # pdb.set_trace()
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        '''
        logits是anchor_dot_contrast减去每一行的最大值得到的最终相似度
        示例: logits: torch.size([4,4])
        logits:
            tensor([[ 0.0000, -0.0471, -0.3352, -0.2156],
                    [-1.2576,  0.0000, -0.3367, -0.0725],
                    [-1.3500, -0.1409, -0.1420,  0.0000],
                    [-1.4312, -0.0776, -0.2009,  0.0000]])       
        '''
        # 构建mask 
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)     
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        '''
        但是对于计算Loss而言，(i,i)位置表示样本本身的相似度，对Loss是没用的，所以要mask掉
        # 第ind行第ind位置填充为0
        得到logits_mask:
            tensor([[0., 1., 1., 1.],
                    [1., 0., 1., 1.],
                    [1., 1., 0., 1.],
                    [1., 1., 1., 0.]])
        positives_mask:
        tensor([[0., 0., 1., 1.],
                [0., 0., 0., 0.],
                [1., 0., 0., 1.],
                [1., 0., 1., 0.]])
        negatives_mask:
        tensor([[0., 1., 0., 0.],
                [1., 0., 1., 1.],
                [0., 1., 0., 0.],
                [0., 1., 0., 0.]])
        '''        
        num_positives_per_row  = torch.sum(positives_mask , axis=1) # 除了自己之外，正样本的个数  [2 0 2 2]       
        denominator = torch.sum(
        exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)  
        
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        

        log_probs = torch.sum(
            log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        '''
        计算正样本平均的log-likelihood
        考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        所以这里只计算正样本个数>0的    
        '''
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss
    

class MAGCLLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(MAGCLLoss, self).__init__()
        self.temperature = temperature

    def forward(self, fused_features, anchors, all_genre_embeddings, labels):
        """
        输入:
            fused_features: [Batch, Dim]
            anchors: [Batch, Dim] (样本的正向锚点/质心)
            all_genre_embeddings: [Total_Genres, Dim] (所有流派的投影后嵌入，用于负采样)
            labels: [Batch, Total_Genres] (用于计算Batch内负样本掩码)
        """
        batch_size = fused_features.shape[0]
        labels = labels.float()

        # 1. 归一化 (对比学习的标准操作)
        fused_features = F.normalize(fused_features, dim=1)
        anchors = F.normalize(anchors, dim=1)
        all_genre_embeddings = F.normalize(all_genre_embeddings, dim=1)

        # 2. 计算正样本相似度 (Sample vs Centroid)
        # exp(F_i * C_i / tau)
        # 这里的 anchors 必须带有梯度！不能是 .tolist() 出来的
        pos_sim = torch.sum(fused_features * anchors, dim=1, keepdim=True) / self.temperature#计算每个样本的融合特征与其锚点特征的相似度，正样本是anchor
        exp_pos = torch.exp(pos_sim)

        # 3. View 1: 流派嵌入负样本 (Sample vs All Genres)
        # q^GE
        # 计算与所有流派的相似度
        all_genre_sim = torch.matmul(fused_features, all_genre_embeddings.t()) / self.temperature#计算每个样本的融合特征与所有流派嵌入特征的相似度
        exp_all_genre = torch.exp(all_genre_sim)
        
        # 掩码：过滤掉样本自带的流派 (labels=1的位置设为0)
        neg_mask_ge = (1.0 - labels)#label形状[23]
        q_ge = torch.sum(exp_all_genre * neg_mask_ge, dim=1, keepdim=True)#类型嵌入负样本的相似度之和

        # 4. View 2: Batch内互斥样本负样本 (Sample vs Batch)
        # q^CB
        # 找出标签无交集的样本对
        intersection = torch.matmul(labels, labels.t())#(batch_size, batch_size)，第一行代表第一个样本与所有样本的标签是否有交集，只有为0才能证明没有交集，有几个交集，数值就为几
        neg_mask_cb = (intersection == 0).float()#无交集为0的设为1，有交集的设为0
        
        # 排除对角线 (虽然交集判断已排除，但双重保险)
        identity = torch.eye(batch_size).to(labels.device)
        neg_mask_cb = neg_mask_cb * (1 - identity)

        batch_sim = torch.matmul(fused_features, fused_features.t()) / self.temperature
        exp_batch = torch.exp(batch_sim)
        
        q_cb = torch.sum(exp_batch * neg_mask_cb, dim=1, keepdim=True)

        # 5. 最终 Loss
        # -log( exp_pos / (exp_pos + q_ge + q_cb) )
        denominator = exp_pos + q_ge + q_cb
        loss = -torch.log(exp_pos / (denominator + 1e-8))

        return loss.mean()

