
import json
import os
import pdb
#读取训练集电影id
# with open('/home/amax/liuyali/imdbtrain.txt') as f:
with open('/home/amax/liuyali/imdbtrain.txt') as f:  # 用../表示上一级目录
    trainli=[]
    lines=f.readlines()#读取所有行
    for line in lines:#遍历每一行
        trainli.append(line.strip())#将去除换行符的电影id添加到列表中
        # print(trainli)
        # pdb.set_trace()


# with open('entity2id.txt','w') as f:
# gens={'Drama':0,'Comedy':1,'Romance':2,'Thriller':3,'Crime':4,'Action':5,'Adventure':6,'Horror':7,'Documentary':8,'Mystery':9,'Sci-Fi':10,'Fantasy':11,'Family':12,'Biography':13,'War':14,
# 'History':15,'Music':16,'Animation':17,'Musical':18,'Western':19,'Sport':20,'Short':21,'Film-Noir':22}
# k=23
# enti={'Drama':0,'Comedy':1,'Romance':2,'Thriller':3,'Crime':4,'Action':5,'Adventure':6,'Horror':7,'Documentary':8,'Mystery':9,'Sci-Fi':10,'Fantasy':11,'Family':12,'Biography':13,'War':14,
# 'History':15,'Music':16,'Animation':17,'Musical':18,'Western':19,'Sport':20,'Short':21,'Film-Noir':22}

#给出所有类型的初始id
gens={'Drama':0,'Comedy':1,'Action':2,'Adventure':3,'Romance':4,'Crime':5,'Horror':6,'Thriller':7,'Biography':8,'Animation':9,'Family':10,'Mystery':11,'Fantasy':12,'Music':13,'History':14,
'Western':15,'Sci-Fi':16,'Musical':17,'Sport':18,'Short':19,'War':20,'Documentary':21,'Film-Noir':22}
k=23
enti={'Drama':0,'Comedy':1,'Action':2,'Adventure':3,'Romance':4,'Crime':5,'Horror':6,'Thriller':7,'Biography':8,'Animation':9,'Family':10,'Mystery':11,'Fantasy':12,'Music':13,'History':14,
'Western':15,'Sci-Fi':16,'Musical':17,'Sport':18,'Short':19,'War':20,'Documentary':21,'Film-Noir':22}


k=23#实体id计数器，从23开始，因为前23（0-22）个id分配给了电影类型
for i in trainli:#遍历训练集电影id列表
    with open(os.path.join('/home/amax/liuyali/data/imdb/dataset',i+'.json')) as f:#打开对应电影的json文件，i是电影id
        data=json.load(f)
        #电影本身作为实体
        #title
        enti[i]=k#电影id作为实体，分配唯一数字id，值为当前计数器k的值
        k+=1
        #director
        # print(i)
        #如果导演不在实体字典中，则添加导演实体并分配唯一数字id
        if 'director' in data.keys():#如果导演信息存在
            dir=data['director']#获取导演名字
            if dir not in enti.keys():#如果导演不在实体字典中
                enti[dir]=k#添加导演实体并分配唯一数字id
                k+=1

            
        #actor
        # 
        if 'cast' in data.keys():
            for actor in data['cast']:
                
                if actor not in enti.keys():
                    enti[actor]=k
                    k+=1
# pdb.set_trace()
#entiti cons
ENTITY_OUT = '/home/amax/liuyali/entity2id.txt'
TRAIN_OUT = '/home/amax/liuyali/train2id.txt'

with open(ENTITY_OUT, 'w') as f:#写入实体到id的映射文件
    f.write(str(len(enti)) + '\n')  # OpenKE 风格：首行为实体总数
    for i in enti.keys():
        f.write(i+'\t'+str(enti[i])+'\n')

#构建三元组文件
dirac={}#导演拍过的演员
dirge={}#导演拍过的类型
acge={}#演员演过的类型
triples = []  # 先收集，最后再写（OpenKE 首行需要写总数）

for i in trainli:
    with open(os.path.join('/home/amax/liuyali/data/imdb/dataset', i+'.json')) as fa:#再次遍历每部训练集电影的json文件
        data=json.load(fa)
        title_id=enti[i]#获取电影实体id
        #电影-导演关系，关系id为0
        if 'director' in data.keys():#检查当前的json文件中是否有导演信息
            dir=data['director']#获取导演名字
            triples.append((title_id, enti[dir], 0))
            if enti[dir] not in dirge.keys():
                dirge[enti[dir]]=[enti[gen] for gen in data['genres'] if gen in gens.keys()]
            else:
                for gen in data['genres']:
                    if gen in gens.keys() and enti[gen] not in dirge[enti[dir]]:
                        dirge[enti[dir]].append(enti[gen])
        #电影-演员关系，关系id为1
        if 'cast' in data.keys():
            for act in data['cast']:
                triples.append((title_id, enti[act], 1))
                if enti[act] not in acge.keys():
                    acge[enti[act]]=[enti[gen] for gen in data['genres'] if gen in gens.keys()]
                else:
                    for gen in data['genres']:
                        if gen in gens.keys() and enti[gen] not in acge[enti[act]]:
                            acge[enti[act]].append(enti[gen])
        #电影-类型关系，关系id为2
        for gen in data['genres']:
            if gen in gens.keys():
                triples.append((title_id, enti[gen], 2))
        if 'cast' in data.keys() and 'director' in data.keys():
            dir=data['director']
            if enti[dir] not in dirac.keys():
                dirac[enti[dir]]=[enti[act] for act in data['cast']]
            else:
                for act in data['cast']:
                    if enti[act] not in dirac[enti[dir]]:
                        dirac[enti[dir]].append(enti[act])

#导演-演员关系，关系id为3
for dir in dirac.keys():
    for act in dirac[dir]:
        triples.append((dir, act, 3))
#导演-类型关系，关系id为4
for dir in dirge.keys():
    for gen in dirge[dir]:
        triples.append((dir, gen, 4))
#演员-类型关系，演员经常演的类型，关系id为5
for act in acge.keys():
    for gen in acge[act]:
        triples.append((act, gen, 5))

with open(TRAIN_OUT, 'w') as f:
    f.write(str(len(triples)) + '\n')  # OpenKE 风格：首行为三元组总数
    for h, t, r in triples:
        f.write(f'{h}\t{t}\t{r}\n')

print(f'[DONE] entities = {len(enti)}, triples = {len(triples)}')
print(f'[DONE] wrote {ENTITY_OUT}')
print(f'[DONE] wrote {TRAIN_OUT}')