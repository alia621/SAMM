import os
import torch
from torch import optim
from tqdm import tqdm
import torch.nn as nn
import random
from sklearn.metrics import classification_report as sk_classification_report
from seqeval.metrics import classification_report
from transformers.optimization import get_linear_schedule_with_warmup
import pdb
import wandb
import numpy as np
from .metrics import eval_result
from sklearn import metrics
from pytorch_optimizer import Ranger
import torchvision.models as models
class BaseTrainer(object):
    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()



class clipweightTrainer(BaseTrainer):
    #加载数据train_data,dev_data,test_data
    #model:IDKG的大体框架clipclass；clipmodel:clip模型提取特征
    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, clipmodel=None,processor=None, args=None, logger=None, target_names=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.target_names=target_names
        self.logger = logger
        
        self.refresh_step = 2#每两步刷新一次进度条
        self.best_dev_metric = 0 #最佳验证集指标
        self.best_test_metric = 0 #最佳测试集指标
        self.clipmodel=clipmodel.cuda()
        self.best_dev_epoch = None #最佳模型出现的epoch
        self.best_test_epoch = None #最佳测试集出现的epoch
        self.optimizer = None 
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs #训练总步数
        self.step = 0
        self.args = args
        self.lossfn=torch.nn.BCELoss() #二分类交叉熵损失，多标签分类损失
        self.before_gmu_train()
        self.lossself=Lossself()#自监督注意力损失AT模块损失
        self.mse_loss = torch.nn.MSELoss()
        # pdb.set_trace()
    def train(self):
        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data)*self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")
            #postfix进度条右侧显示信息，首个参数为0#
            # dynamic_ncols= 动态调整进度条的宽度
            # leave=进度条结束后是否保留在屏幕上
        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True, initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            for epoch in range(1, self.args.num_epochs+1):#左侧显示第几个epoch，
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                epoch_true_labels, epoch_pred_labels = [], []  # 累积本 epoch 训练样本的标签/预测，用于绘制 train 曲线
                for batch in self.train_data:#遍历训练集中的每一个batch
                    self.step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    loss, logits, labels ,lossweight,contra,moe_aux_loss,z,weightlabel= self._step(batch, mode="train")#loss指多标签损失，lossweight指AT模块损失，contra对比学习损失，moe_aux_loss 改进4/5/6 辅助损失合计
                    lambda_cl = getattr(self.args, 'lambda_cl', 0.1)  # 对比学习损失权重
                    lambda_atten = getattr(self.args, 'lambda_atten', 0.1)  # 注意力损失权重
                    contra = lambda_cl * contra
                    lossweight = lambda_atten * lossweight
                    total_loss = loss + lossweight + contra + moe_aux_loss  # L = L_class + L_atten + L_cl + L_moe_aux(改进5/6)
                    avg_loss += total_loss.detach().cpu().item()
                    total_loss.backward()

                    self.optimizer.step()#根据梯度更新模型参数
                    self.scheduler.step()#更新学习率，使用的是线性warmup调度器
                    self.optimizer.zero_grad()#清零所有参数梯度

                    epoch_true_labels.extend(labels.detach().cpu().tolist())
                    epoch_pred_labels.extend(logits.detach().cpu().tolist())

                    def _to_float(v):
                        return v.item() if isinstance(v, torch.Tensor) else float(v)
                    log_dict = {
                        "train/total_loss": total_loss.item(),
                        "train/loss_cls": loss.item(),
                        "train/loss_att": lossweight.item(),
                        "train/loss_contra": contra.item(),
                        "train/loss_moe_aux": _to_float(moe_aux_loss),
                        "train/lr": self.optimizer.param_groups[0]['lr'],
                    }
                    # 改进4：融合层MoE 负载均衡损失（已独立到 moe_aux_loss）
                    if hasattr(self.model, 'use_moe') and self.model.use_moe:
                        log_dict["train/loss_fusion_aux"] = _to_float(self.model.clswei.aux_loss)
                    # 改进5（重构 方案B）：cluster_moe router 辅助损失
                    if hasattr(self.model, 'use_cluster_moe') and self.model.use_cluster_moe:
                        log_dict["train/loss_cluster_aux"] = _to_float(self.model.classifier.aux_loss)
                    # 改进5-Learnable：可学习 cluster 辅助损失（合计 + BCE / collapse 拆分）
                    if hasattr(self.model, 'use_learnable_cluster') and self.model.use_learnable_cluster:
                        log_dict["train/loss_learnable_cluster_aux"] = _to_float(self.model.classifier.aux_loss)
                        log_dict["train/loss_learnable_cluster_bce"] = _to_float(self.model.classifier.bce_loss)
                        log_dict["train/loss_learnable_cluster_collapse"] = _to_float(self.model.classifier.collapse_loss)
                    # 改进6：模态专属MoE 负载均衡损失（按模态拆分 + 三模态合计）
                    if hasattr(self.model, 'use_modality_moe') and self.model.use_modality_moe:
                        vis_aux = _to_float(self.model.vis_moe.aux_loss)
                        text_aux = _to_float(self.model.text_moe.aux_loss)
                        kg_aux = _to_float(self.model.kg_moe.aux_loss)
                        log_dict["train/loss_modality_aux_vis"] = vis_aux
                        log_dict["train/loss_modality_aux_text"] = text_aux
                        log_dict["train/loss_modality_aux_kg"] = kg_aux
                        log_dict["train/loss_modality_aux_total"] = vis_aux + text_aux + kg_aux
                    wandb.log(log_dict, step=self.step)

                    if self.step % self.refresh_step == 0:#每refresh_step（2步）进行一次日志输出
                        avg_loss = float(avg_loss) / self.refresh_step#平均损失=累计损失/refreh_step个batch
                        print_output = "loss:{:<6.5f},losscls:{:<6.5f},losswei:{:<6.5f},contra:{:<6.5f},".format(avg_loss,loss,lossweight,contra)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        # pdb.set_trace()
                        avg_loss = 0  #

                # 每个 epoch 结束后，用累积的 train 预测计算 accuracy 曲线
                train_acc = (np.array(epoch_pred_labels) > 0.4) == np.array(epoch_true_labels)
                train_acc = train_acc.all(axis=1).mean()
                wandb.log({"train/accuracy": float(train_acc), "epoch": epoch}, step=self.step)

                if epoch >= self.args.eval_begin_epoch:#设置为1，每一轮都进行评估
                    self.evaluate(epoch)   # generator to dev.在测试集上评估模型

            pbar.close()
            self.pbar = None
            self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch, self.best_dev_metric))
            self.logger.info("Get best test performance at epoch {}, best test f1 score is {}".format(self.best_test_epoch, self.best_test_metric))

    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        step = 0 #记录处理到第几个batch
        true_labels, pred_labels = [], []
        with torch.no_grad():#无需反向传播更新参数
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0
                for batch in self.dev_data:
                    step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
                    loss, logits, labels ,loss2,contra,moe_aux_loss,_,_= self._step(batch, mode="dev")    # logits: batch, 3
                    loss=loss.detach().cpu().item()+loss2.detach().cpu().item()+contra.detach().cpu().item()+moe_aux_loss.detach().cpu().item()
                    total_loss += loss

                    true_labels.extend(labels.detach().cpu().tolist())
                    pred_labels.extend(logits.detach().cpu().tolist())
                    pbar.update()
                results=self.report_performance(true_labels,pred_labels,0.5)
                # evaluate done
                pbar.close()
                wandb.log({
                    "val/macro_f1": results['macro'][2],
                    "val/micro_f1": results['micro'][2],
                    "val/weighted_f1": results['weighted'][2],
                    "val/samples_f1": results['samples'][2],
                    "val/accuracy": results['micro'][4],
                    "epoch": epoch
                })
                if results['macro'][2] >= self.best_dev_metric:  # this epoch get best performance，macro f1作为评估指标
                    # pdb.set_trace()
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = results['macro'][2] # update best metric(f1 score)
                    if self.args.save_path is not None:
                        torch.save(self.model.state_dict(), self.args.save_path+"/best_model.pth")
                        self.logger.info("Save best model at {}".format(self.args.save_path))
                

        self.model.train()#每一个epoch评估一次，评估后重新变为训练模式
    def report_performance(self,y_true, y_prob, threshold, print_results=True, multilabel=True):
        # y_true=y_true.cpu().numpy()
        # y_prob=y_prob.cpu().numpy()
        y_true=np.array(y_true)#将列表转为numpy数组，满足metrics函数输入
        y_prob=np.array(y_prob)
        y_pred = y_prob > threshold#将概率转为二值预测，超过0.5为1

        # print("y_pred",y_pred)
        # y_true=y_true[0]
        # y_prob=y_prob[0]
        results = {}
        averages = ('micro', 'macro', 'weighted', 'samples')
        if multilabel:#多标签分类
            acc = metrics.accuracy_score(y_true, y_pred)#对于多标签分类，只有样本的所有标签全部对应，才算预测成功
        else:
            acc = metrics.accuracy_score(
                y_true.argmax(axis=1), y_prob.argmax(axis=1))#单标签分类，取每个样本概率最大的
        # print(acc)
        # pdb.set_trace()
       
       #macro：先计算每个类别的指标，然后取平均
       #micro：将所有样本的所有标签汇总计算
       #weighted按每个类别的样本数加权平均
       #samples：先为每个样本计算指标，再平均
        for average in averages:#计算各种平均方法下的指标
            results[average] = metrics.precision_recall_fscore_support(y_true, y_pred, average=average)[:3] + (
                # metrics.roc_auc_score(y_true, y_prob, average=average),
                metrics.hamming_loss(y_true, y_pred),#预测错误的标签比例，更加细致，可以判断所有标签有多少预测成功
                acc)

        if print_results:
            print('average\tprecisi\trecall\tf_score\thamming\taccuracy')
            for avg, vals in results.items():
                print('{0:.7}\t{1:0.3f}\t{2:0.3f}\t{3:0.3f}\t{4:0.3f}\t{5:0.3f}'.format(
                    avg, *vals))
            
            print(metrics.classification_report(
                y_true, y_pred, target_names=self.target_names, digits=3))
        return results

    def test(self):
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        
        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))#加载最优模型
            self.logger.info("Load model successful!")
        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device  
                    loss, logits, labels,loss2,contra,moe_aux_loss,_,_ = self._step(batch, mode="dev")    # logits: batch, 3
                    loss=loss.detach().cpu().item()+loss2.detach().cpu().item()+contra.detach().cpu().item()+moe_aux_loss.detach().cpu().item()
                    total_loss += loss
                    # pdb.set_trace()
                    true_labels.extend(labels.detach().cpu().tolist())
                    pred_labels.extend(logits.detach().cpu().tolist())
                    pbar.update()
                results=self.report_performance(true_labels,pred_labels,0.4)
                # evaluate done
                pbar.close()
                wandb.log({
                    "test/macro_f1": results['macro'][2],
                    "test/micro_f1": results['micro'][2],
                    "test/weighted_f1": results['weighted'][2],
                    "test/samples_f1": results['samples'][2],
                    "test/accuracy": results['micro'][4],
                })
                    
        self.model.train()
        
    def _step(self, batch, mode="train"):#处理单个batch
        if mode != "predict":

            image,text,label,kgem,labelem,weightlabel=batch#batch的数据结构
            text=text.to(self.args.device)#图像（还未被提取特征），形状: [batch_size, 3, H, W]
            image=image.to(self.args.device) # 文本，形状: [batch_size, max_seq_len]
            kgem=kgem.to(self.args.device).to(torch.float32)#知识图谱特征，形状: [batch_size, kg_dim]
            labelem=labelem.to(self.args.device).to(torch.float32)#标签嵌入向量 [batch_size, num_classes, emb_dim]
            weightlabel=weightlabel.to(self.args.device).to(torch.float32)#权重标签，监督模态权重
            label=label.to(self.args.device)# 真实标签，形状: [batch_size, num_classes]
            # raw_norm = raw_norm.to(self.args.device).to(torch.float32)

            with torch.no_grad(), torch.cuda.amp.autocast():
                #clip提取图像，文本特征
                image_features = self.clipmodel.encode_image(image)
                text_features = self.clipmodel.encode_text(text)

                #特征归一化
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)   


                # 改进2：计算CLIP相似度伪标签，论文公式(3)
                # CLIP特征已L2归一化，点积即余弦相似度
                similarity = (image_features * text_features).sum(dim=1)
                similarity = torch.clamp(similarity, 0.0, 1.0)  # 确保在[0,1]范围内
                clip_pseudo_label = 1.0 - similarity  # Y_tilde^C，相似度越大，伪标签越小



            # 将提取出的特征进行数据转换   ，转化为float32 
            image_features=image_features.float()
            text_features=text_features.float()
            # inputs=inputs.cuda()
            # pdb.set_trace()

# 完整模型（clip/clipweight）：
# 输入：图像特征、文本特征、知识图谱嵌入、标签嵌入、标签
# 输出：outputs（预测概率）, x, y, z（）, contra（对比损失）

# 基线模型（baseline）：
# 输入：图像特征、文本特征
# 输出：outputs（预测概率）, z（）

# KG模型（kg）：
# 输入：图像特征、文本特征、知识图谱嵌入
# 输出：outputs（预测概率）, z（）

            if self.args.modelclass=='clip' or self.args.modelclass=='clipweight':

                outputs,x,y,z,contra,moe_aux_loss=self.model(image_features,text_features,kgem,labelem,label)
            #如果使用gmu，输入文本、图像两个特征
            if self.args.modelclass=='baseline':
                outputs,z=self.model(image_features,text_features)     
            #如果使用kggmu和kglabel，则输入3个特征         
            if self.args.modelclass=='kg':
                outputs,z=self.model(image_features,text_features,kgem)  

            outputs=outputs.to(torch.float32)
            label=label.to(torch.float32)
            # rn = torch.log1p(raw_norm)#构建一个0-1的合作强度权重
            # kg_strength_label = (rn - rn.min()) / (rn.max() - rn.min() + 1e-8)
            # alpha = 0.5
            # target_z = (1 - alpha) * weightlabel + alpha * kg_strength_label
            # loss2 = self.mse_loss(target_z, z)
            loss1=self.lossfn(outputs, label)#多标签损失，论文公式(14)

            # 改进2：注意力自监督损失，论文公式(5)
            # L_K: KG注意力伪标签损失：L_K = -sum(log(1 - |A^K - Y_tilde^K|))
            # 原版IDKG使用MSE: loss_K = self.mse_loss(weightlabel, z)
            loss_K = -torch.log(1.0 - torch.abs(z - weightlabel) + 1e-6).mean()

            # L_C: CLIP相似度伪标签损失（改进2新增）
            # L_C = -sum(log(1 - ||A^T - A^I| - Y_tilde^C|))
            use_clip_att = getattr(self.args, 'use_clip_att', True)  # 消融实验开关，默认开启
            if use_clip_att and self.args.modelclass == 'clipweight':
                att_diff = torch.abs(x - y)  # |A^I - A^T|，图像和文本注意力分数之差
                loss_C = -torch.log(1.0 - torch.abs(att_diff - clip_pseudo_label) + 1e-6).mean()
                loss2 = loss_C + loss_K  # L_atten = L_C + L_K，论文公式(5)
            else:
                loss2 = loss_K  # 消融实验：只用 L_K
            
            
            
            # if self.args.use_prompt:
            #     input_ids, token_type_ids, attention_mask, labels, images, aux_imgs = batch
            # else:
            #     images, aux_imgs = None, None
            #     input_ids, token_type_ids, attention_mask, labels= batch
            # outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, images=images, aux_imgs=aux_imgs)
            return loss1,outputs, label,loss2,contra,moe_aux_loss,z,weightlabel

    def before_gmu_train(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)#Adamw优化器
        # self.optimizer = Ranger(optimizer_grouped_parameters, lr=self.args.lr)
        # self.optimizer = optim.Adam(optimizer_grouped_parameters, lr=self.args.lr,amsgrad=True)
        # self.optimizer = optim.SGD(optimizer_grouped_parameters, lr=self.args.lr,momentum =0.9)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, 
                                                            num_warmup_steps=self.args.warmup_ratio*self.train_num_steps, 
                                                                num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)


class Lossself(nn.Module):
    def __init__(self,alpha=0):
        super(Lossself, self).__init__()
        # self.linear1=Maxout(input_dim,hidden_size,2)
        # self.linear2=Maxout(hidden_size,hidden_size,2)
        self.alpha=alpha
        # sampling
    def forward(self, x,y):
        return -torch.log(torch.abs(x-y)+self.alpha).mean()

