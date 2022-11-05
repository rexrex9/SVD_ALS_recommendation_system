
import numpy as np

def __getParis(n_users,n_items,dataset):
    ndataset=[]
    for data in dataset:
        u,i,r,_=data.strip().split(',')
        u,i,r=int(u),int(i),float(r)
        if u>n_users:n_users=u
        if i>n_items:n_items=i
        ndataset.append([u,i,r])
    return ndataset,n_users,n_items

def loadData(dataFile='../ml-latest-small/ratings.csv',test_ratio=0.1):
    with open(dataFile, 'r', encoding='utf-8') as f:
        allData = f.read().split('\n')[1:]
    allData=list(set(filter(None,allData)))
    testData = np.random.choice(allData, int(len(allData)*test_ratio), replace=False)
    trainData = list(set(allData)-set(testData))
    n_users,n_items=0,0
    testData,n_users,n_items=__getParis(n_users,n_items, testData)
    trainData,n_users,n_items=__getParis(n_users,n_items, trainData)
    return trainData,testData,n_users+1,n_items+1

