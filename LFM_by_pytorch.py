import numpy as np
from torch.utils.data import DataLoader
from torch import nn
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score,recall_score,accuracy_score

def __getParis(n_users,n_items,dataset):
    ndataset=[]
    for data in dataset:
        u,i,r=data.strip().split()
        u,i,r=int(u),int(i),int(r)
        if u>n_users:n_users=u
        if i>n_items:n_items=i
        ndataset.append([u,i,r])
    return ndataset,n_users,n_items

def loadData(dataFile='ml-latest-small/rating_index.tsv',test_ratio=0.2):
    with open(dataFile, 'r', encoding='utf-8') as f:
        allData = f.read().split('\n')
    allData=list(set(filter(None,allData)))
    testData = np.random.choice(allData, int(len(allData)*test_ratio), replace=False)
    trainData = list(set(allData)-set(testData))
    n_users,n_items=0,0
    testData,n_users,n_items=__getParis(n_users,n_items, testData)
    trainData,n_users,n_items=__getParis(n_users,n_items, trainData)
    return torch.LongTensor(trainData),torch.LongTensor(testData),n_users+1,n_items+1

class LFM(nn.Module):
    def __init__(self,n_users,n_items,dim=50):
        super(LFM, self).__init__()
        self.users = nn.Embedding(n_users, dim, max_norm=1)
        self.items = nn.Embedding(n_items, dim, max_norm=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, u,v):
        u = self.users(u)
        v = self.items(v)
        uv = torch.sum(u*v,axis=1)
        logit = self.sigmoid(uv)
        return logit

def doEva(net,d):
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    with torch.no_grad():
        out = net(u,i)
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    y_true = r.detach().numpy()
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc =accuracy_score(y_true,y_pred)
    return p,r,acc

def train(epochs=10,batchSize=1024):
    trainData, testData,n_users,n_items = loadData()
    net=LFM(n_users,n_items,)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01)
    loss_fun=torch.nn.BCELoss()
    for e in range(epochs):
        all_lose=0
        for d in tqdm(DataLoader(trainData,batch_size=batchSize,shuffle=True)):
            optimizer.zero_grad()
            u,i,r = d[:,0],d[:,1],d[:,2]
            r=torch.FloatTensor(r.detach().numpy())
            result = net(u,i)
            loss = loss_fun(result,r)
            all_lose+=loss
            loss.backward()
            optimizer.step()
        print('epoch{},avg_loss={}'.format(e,all_lose/(len(trainData)//batchSize)))
        p, r, acc=doEva(net,testData)
        print('p:{},r:{},acc:{}'.format(round(p,3),round(r,3),round(acc,3)))


train()