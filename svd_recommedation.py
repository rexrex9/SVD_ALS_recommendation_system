__author__='雷克斯掷骰子'
'''
B站:https://space.bilibili.com/497998686
头条:https://www.toutiao.com/c/user/token/MS4wLjABAAAAAxu8A9lNX1qfkRKEyU9Uecqa2opPcZufDLWHbv7m-hVdMVPOe7r_i-k6nw4RY61i/
'''
'''
数据下载地址：https://grouplens.org/datasets/movielens/
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

def readDatas():
    path = 'ml-latest-small/ratings.csv'
    odatas = pd.read_csv(path,usecols=[0,1,2])
    return odatas

def splitTrainSetTestSet(odatas):
    testset = odatas.sample(frac=0.2, axis=0)
    trainset = odatas.drop(index=testset.index.values.tolist(), axis=0)
    return trainset,testset

def getMatrix(trainset):
    userSet,itemSet = set(),set()

    for d in trainset.values:
        userSet.add(int(d[0]))
        itemSet.add(int(d[1]))

    userList = list(userSet)
    itemList = list(itemSet)

    df = pd.DataFrame(0, index=userList, columns=itemList,dtype=float)
    for d in tqdm(trainset.values):
        df[d[1]][d[0]]=d[2]

    return df,userList,itemList

def svd(m,k):
    u, i, v =np.linalg.svd(m)
    return u[:,0:k],np.diag(i[0:k]),v[0:k,:]

def predict(u,i,v,user_index,item_index):
    return float(u[user_index].dot(i).dot(v.T[item_index].T))

def getPredicts(testSet,userList,itemList,u,i,v):
    y_true,y_hat = [],[]

    for d in tqdm(testSet.values):
        user = int(d[0])
        item = int(d[1])
        if user in userList and item in itemList:
            user_index = userList.index(user)
            item_index = itemList.index(item)
            y_true.append(d[2])
            y_hat.append(predict(u,i,v,user_index,item_index))
    return y_true,y_hat

def play():
    k=200 #奇异值数量

    trainset, testSet = splitTrainSetTestSet(readDatas())
    df, userList, itemList = getMatrix(trainset)
    u, i, v = svd(np.mat(df),k)

    train_y_true, train_y_hat = getPredicts(trainset, userList, itemList, u, i, v)
    test_y_true, test_y_hat = getPredicts(testSet, userList, itemList,u, i, v)

    print(RMSE(train_y_true,train_y_hat))
    print(RMSE(test_y_true,test_y_hat))

def RMSE(a,b):
    return(np.average((np.array(a) - np.array(b)) ** 2)) ** 0.5


if __name__ == '__main__':
    play()





