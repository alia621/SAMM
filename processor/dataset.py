# from msilib import sequence

import random
import os
import torch
import json
import ast
from PIL import Image
import torchvision.models as models
# from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer
from transformers import ViltProcessor
from torchvision import transforms
import logging
logger = logging.getLogger(__name__)
import re
import numpy as np
import pdb
import torch
from PIL import Image
import open_clip
import random
random.seed(10)
from collections import OrderedDict, Counter
from sklearn.preprocessing import MultiLabelBinarizer
import itertools

Image.MAX_IMAGE_PIXELS = 2300000000

''''''
class imdbDataset(Dataset):
    def __init__(self, data_path='/data1/amax/mmimdb1.0', img_path=None,max_seq=40, sample_ratio=1.0, mode="train") -> None:#输入参数为数据路径，图片路径，最大序列长度，采样比例，模式
        self.data_path=data_path
        with open(os.path.join(self.data_path,'list.txt'), 'r') as f:#读取数据集的每一个文件
            files = f.read().splitlines()#读取每一行作为文件名
        self.vocab_counts=[]#定义词汇表
        self.movies=[]#统计有效电影数据
        n_classes=23

        for i, file in enumerate(files):#遍历数据集每个文件
            with open(os.path.join(self.data_path,file)) as f:
                data = json.load(f)#加载每个电影的JSON数据，字典
                data['imdb_id'] = file.split('/')[-1].split('.')[0]#从list的每一行获取电影ID，加到数据字典中
                im_file = os.path.join(self.data_path,file.replace('json', 'jpg'))
                if all([k in data for k in ('genres', 'plot')] + [os.path.isfile(im_file)]):#每个样本必须有genres、plot、poser
                    # plot_id = np.array([len(p) for p in data['plot']]).argmax()
                    # pdb.set_trace()
                    data['plot'] = self.normalizeText(data['plot'])#对电影的剧情文本进行处理，分词等
                    if len(data['plot']) > 0:
                        self.vocab_counts.extend(data['plot'])
                        self.movies.append(data)#将有效电影数据添加到列表中，有效指的是包含类型和剧情文本，且剧情文本非空的电影数据


        '''电影标签处理'''
        counts = OrderedDict(
            Counter([g for m in self.movies for g in m['genres']]).most_common())#统计电影类型的出现频率，按频率从高到低排序
        target_names = list(counts.keys())[:n_classes]#选择出现频率最高的n_classes个电影类型作为目标类别，n_classes=23
        # print(counts)
        # print(target_names)
        # 
        le = MultiLabelBinarizer()
        self.Y = le.fit_transform([m['genres'] for m in self.movies])#全部数据集的全部电影类型去重并且按照字母排序，再转为二进制矩阵（多热标签）(电影数量, 所有类型数量)
        self.le=le
        self.labels = np.nonzero(le.transform([[t] for t in target_names]))[1]## 获取这23个类型在完整标签列表中的索引[1]=所有类型数量
        self.gt=self.Y[:,self.labels]
        self.target_names=target_names

        '''数据集划分'''
        index = list(range(len(self.movies)))
        random.shuffle(index)
        train_len = int(len(self.movies) * 0.6)#训练集长度
        test_len = int(len(self.movies) * 0.3)#测试集长度
        train_data = [self.movies[index[i]] for i in range(train_len)]#训练集数据
        train_labels = [self.gt[index[i]] for i in range(train_len)]#训练集标签
        test_data = [self.movies[index[i]] for i in range(train_len,train_len+test_len)]#测试集数据
        test_labels = [self.gt[index[i]] for i in range(train_len,train_len+test_len)]#测试集标签
        dev_data = [self.movies[index[i]] for i in range(train_len+test_len,len(index))]#验证集数据
        dev_labels = [self.gt[index[i]] for i in range(train_len+test_len,len(index))]#验证集标签



        with open('imdbtrain.txt','w') as f:
                for i in train_data:
                    id=i['imdb_id']
                    f.write(str(id)+'\n')
        with open('imdbtest.txt','w') as f:
            for i in test_data:
                id=i['imdb_id']
                f.write(str(id)+'\n')
        with open('imdbdev.txt','w') as f:
            for i in dev_data:
                id=i['imdb_id']
                f.write(str(id)+'\n')
        if mode=='train':#选择数据集模式,根据不同模式加载不同数据
            self.samples=train_data
            self.gengt=train_labels
        elif mode=='test':
            self.samples=test_data
            self.gengt=test_labels
        else:
            self.samples=dev_data
            self.gengt=dev_labels

        # pdb.set_trace()
        

        # self.tokenizer = self.processor.tokenizer
    def normalizeText(self,text):#处理
        text = text.lower()
        text = re.sub(r'<br />', r' ', text).strip()
        text = re.sub(r'^https?:\/\/.*[\r\n]*', ' L ', text, flags=re.MULTILINE)
        text = re.sub(r'[\~\*\+\^`_#\[\]|]', r' ', text).strip()
        text = re.sub(r'[0-9]+', r' N ', text).strip()
        text = re.sub(r'([/\'\-\.?!\(\)",:;])', r' \1 ', text).strip()
        return text.split()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        return 0

    
class imdbclipdataset(imdbDataset):
    # def __init__(self,data_path='/data1/amax/mmimdb1.0', entity_path='/home/amax/liuyali/OpenKE/benchmarks/imdb1.1/entity2id.txt',kgparam_path='/data1/amax/mmimdb1.0/embeddings1.1/rotate_imdb1.1.json',model=None,processor=None,mode='dev',use_collab_weight=True) -> None:
    def __init__(self,data_path='/home/amax/liuyali/data/imdb', entity_path='/home/amax/liuyali/entity2id.txt',kgparam_path='/home/amax/liuyali/data/rotateE_paramsimdb.json',model=None,processor=None,mode='dev',use_collab_weight=True) -> None:
        #调用父类的初始化方法，加载基本数据，包括self.movies等
        super().__init__(data_path=data_path,mode=mode)
        self.use_collab_weight=use_collab_weight  # 改进1消融开关
        self.entity_path=entity_path
        self.kg_path=kgparam_path
        self.entityid={}#字典类型映射[entity_name] ：entity_id
        with open(self.entity_path) as f:#读取实体ID映射文件
            lines=f.readlines()[1:]#跳过第一行
            for line in lines:#遍历每一行
                # pdb.set_trace()
                self.entityid[line.strip().split('\t')[0]]=int(line.strip().split('\t')[-1])#构建实体到ID的映射字典
        #加载知识图谱参数，获取实体嵌入权重，将知识图谱嵌入向量存储在self.kgemb中
        with open(self.kg_path,'r',encoding='utf-8') as f:#读取知识图谱参数文件
            data=f.read()
            if not data.strip().endswith('}'):
                data=data.rstrip('\n\t\r,')+'}'#确保JSON格式正确
            kgparams=json.loads(data)#加载JSON数据
            self.kgemb=kgparams['ent_embeddings.weight']#获取实体嵌入权重

        ''''统计演员和导演的合作次数(默认导演只有一个)'''
        self.collab_map = {} # 用于存储 {(人A, 人B): 合作次数}
        # 必须确保随机种子与父类一致，才能保证划分出的训练集是一样的
        indices = list(range(len(self.movies)))
        random.seed(10) # 关键：重置种子以匹配 dataset.py 顶部的设置
        random.shuffle(indices)
        train_len = int(len(self.movies) * 0.6)
        train_indices = indices[:train_len] # 防止数据泄露，只取前60%作为训练集用来统计
        for idx in train_indices:
            movie = self.movies[idx] # 遍历训练集每一部电影的原始电影数据
            # 收集每一部电影中所有有效的参与者（演员 + 导演）
            participants = []#
            if 'cast' in movie:
                participants.extend([actor for actor in movie['cast'] if actor in self.entityid])#
            if 'director' in movie:
                director = movie['director']
                if director in self.entityid:#如果演员/导演不在 entity2id.txt 中，说明没有对应的知识图谱表示，跳过
                    participants.append(director)
            # 去重：防止某人既是导演又是演员被统计成自己和自己合作
            participants = list(set(participants))

            #两两配对，并统计两人的合作次数，如果一部电影有 A, B, C 三人，combinations会生成 (A,B), (A,C), (B,C)
            for personA, personB in itertools.combinations(participants, 2):
                pair = tuple(sorted((personA, personB))) # 确保顺序一致，movie1: (Tom_Hanks, Morgan_Freeman)，movie2: (Morgan_Freeman, Tom_Hanks)，按字母排序，变成同一个键
                if pair not in self.collab_map:#如果该组合不在合作字典中，初始化为0
                    self.collab_map[pair] = 0
                self.collab_map[pair] += 1#（A,B）的合作次数加1
        # print(len(self.collab_map))
        
        # pdb.set_trace()




        vocab_counts = OrderedDict(Counter(self.vocab_counts).most_common())#构建词汇表，用于文本处理
        vocab = ['_UNK_'] + [v for v in vocab_counts.keys()]

        self.ix_to_word = dict(zip(range(len(vocab)), vocab))
        self.word_to_ix = dict(zip(vocab, range(len(vocab))))
 
        self.vocab=vocab
        self.data_path=data_path
        self.preprocessor=processor
        # self.vgg = models.vgg19(pretrained=True).cuda()
        # 图像预处理的transform，但实际使用的是processor
        self.transform = transforms.Compose([
            transforms.Resize((160,256)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.samples)
    def normalizeText(self,text):
        text = text.lower()
        text = re.sub(r'<br />', r' ', text).strip()
        text = re.sub(r'^https?:\/\/.*[\r\n]*', ' L ', text, flags=re.MULTILINE)
        text = re.sub(r'[\~\*\+\^`_#\[\]|]', r' ', text).strip()
        text = re.sub(r'[0-9]+', r' N ', text).strip()
        text = re.sub(r'([/\'\-\.?!\(\)",:;])', r' \1 ', text).strip()
        return text#输出处理后的文本，以字符串形式返回，而非父类的分词列表
    
    def __getitem__(self, idx):#获取单个样本数据
        # wordinput = np.zeros((300), dtype='float32')
        # pdb.set_trace()
        words=self.samples[idx]['plot']#获取电影剧情文本
        words = open_clip.tokenize(words).squeeze(0)#使用OpenCLIP的tokenizer对文本进行tokenize，转换为模型输入格式

        #处理知识图谱里的电影类型信息（labelem）
        labelem=[self.kgemb[i] for i in range(23)]
        labelem=np.array(labelem)

        # kgembedding=np.array()
        # kgembedding
        # 改进1：协作加权KG特征（Collaboration-Weighted KG Features）
        #收集当前电影的所有实体
        current_participants = []
        kgweight = 0  # 知识图谱权重初始化
        # 获取演员
        if 'cast' in self.samples[idx]:
            for actor in self.samples[idx]['cast']:
                if actor in self.entityid:
                    current_participants.append(actor)
        # 获取导演
        if 'director' in self.samples[idx]:
            d = self.samples[idx]['director']
            if d in self.entityid:
                # 避免重复（导演兼演员）
                if d not in current_participants:
                    current_participants.append(d)
        #统计当前电影的实体数量
        num_entities = len(current_participants)
        kgweight = num_entities / (num_entities + 6)  

        # 计算KG特征向量
        if len(current_participants) > 0:
            kgem = np.zeros(400)
            if self.use_collab_weight:
                # 改进1：协作加权KG特征，论文公式(2)，嵌入乘以合作次数 a
                for p1 in current_participants:
                    p1_id = self.entityid[p1]#获取实体ID,p1是一个key
                    p1_emb = np.array(self.kgemb[p1_id])#获取实体嵌入向量

                    # 改进1：累计p1与当前电影其他合作者的合作次数

                    total_collab = 0.0
                    for p2 in current_participants:
                        if p1 == p2:#跳过自己
                            continue
                        pair = tuple(sorted((p1, p2)))#创建一个有序的组合键，确保 (A,B) 和 (B,A) 被视为同一组合
                        if pair in self.collab_map:
                            total_collab += self.collab_map[pair]#累计合作次数
                    weight = 1.0 + np.log1p(total_collab)
                    
                    kgem += weight * p1_emb
            else:
                # 原版IDKG：所有实体嵌入等权求和
                for p1 in current_participants:
                    p1_id = self.entityid[p1]
                    p1_emb = np.array(self.kgemb[p1_id])
                    kgem += p1_emb
        else:
            #没有找到任何演员或导演的嵌入向量，使用全零向量代替
            kgem=np.zeros(400)
                
        # 读取电影海报图像并进行预处理
        img_path=os.path.join(self.data_path,'dataset',self.samples[idx]['imdb_id']+'.jpg')#拼接出海报图像的完整路径
        # image = Image.open(img_path).convert('RGB')
        image = Image.open(img_path)#打开图像文件
        image = self.preprocessor(image)#用模型要求的processor对图像进行预处理，转换为模型输入格式马，便于后续模型处理


        return image,words,self.gengt[idx],kgem,labelem,kgweight
   
if __name__ == "__main__":
    # data=mmimdbDataset(
    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
    # data=clipdataset(processor=preprocess,mode='dev')
    # # print(data[0])
    # x=0
    # # print("123",data[0][1].shape)
    # train_dataloader = DataLoader(data, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    # for i,k in enumerate(train_dataloader):
    #     for s in k[2]:
    #         if s[-1]==1:
    #             x+=1
    
    data=imdbclipdataset()
    


    # pdb.set_trace()
    # pdb.set_trace()


    # data=mmimdbDataset()
