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
import pickle


def splitTrainSetTestSet(odatas,frac):
    testset = odatas.sample(frac=frac, axis=0)
    trainset = odatas.drop(index=testset.index.values.tolist(), axis=0)
    return trainset,testset

def readDatas():
    path = 'ml-latest-small/ratings.csv'
    odatas = pd.read_csv(path,usecols=[0,1,2])
    return odatas

class LFM():
    def __init__(self,dataset,factors, epochs, lr, lamda):
        self.dataset = dataset

        self.userList, self.itemList = self.__getListMap()
        self.factors=factors
        self.epochs=epochs
        self.lr=lr
        self.lamda=lamda

        self.p = pd.DataFrame(np.random.randn(len(self.userList), factors), index=self.userList)
        self.q = pd.DataFrame(np.random.randn(len(self.itemList), factors), index=self.itemList)
        self.bu = pd.DataFrame(np.random.randn(len(self.userList)), index=self.userList)
        self.bi = pd.DataFrame(np.random.randn(len(self.itemList)), index=self.itemList)

    def __prediction(self,pu, qi, bu, bi):
        return (np.dot(pu, qi.T) + bu + bi)[0]

    def __getError(self,r, pu, qi, bu, bi):
        return r - self.__prediction(pu, qi, bu, bi)

    def __getListMap(self):
        userSet, itemSet = set(), set()
        for d in self.dataset.values:
            userSet.add(int(d[0]))
            itemSet.add(int(d[1]))
        userList = list(userSet)
        itemList = list(itemSet)
        return userList, itemList

    def fit(self):
        for e in tqdm(range(self.epochs)):
            for d in self.dataset.values:
                u, i, r = d[0], d[1], d[2]
                error = self.__getError(r, self.p.ix[u], self.q.ix[i], self.bu.ix[u], self.bi.ix[i])
                self.p.ix[u] += self.lr * (error * self.q.ix[i] - self.lamda * self.p.ix[u])
                self.q.ix[i] += self.lr * (error * self.p.ix[u] - self.lamda * self.q.ix[i])
                self.bu.ix[u] += self.lr * (error - self.lamda * self.bu.ix[u])
                self.bi.ix[i] += self.lr * (error - self.lamda * self.bi.ix[i])

    def __RMSE(self,a, b):
        return(np.average((np.array(a) - np.array(b)) ** 2)) ** 0.5

    def testRMSE(self,testSet):
        y_true, y_hat = [], []
        for d in tqdm(testSet.values):
            user = int(d[0])
            item = int(d[1])
            if user in self.userList and item in self.itemList:
                hat=self.__prediction(self.p.ix[user], self.q.ix[item], self.bu.ix[user], self.bi.ix[item])
                y_hat.append(hat)
                y_true.append(d[2])
        rmse = self.__RMSE(y_true,y_hat)
        return rmse

    def save(self,path):
        with open(path,'wb+') as f:
            pickle.dump(self,f)

    @staticmethod
    def load(path):
        with open(path,'rb') as f:
            return pickle.load(f)

def play():
    factors=100 #隐因子数量
    epochs=10 #迭代次数
    lr=0.01 #学习率
    lamda=0.1 #正则项系数

    model_path='model/lfm.model'

    trainset, testSet = splitTrainSetTestSet(readDatas(),0.2)

    #lfm=LFM.load(model_path)

    lfm=LFM(trainset,factors, epochs, lr, lamda)
    lfm.fit()
    lfm.save(model_path)

    rmse_test = lfm.testRMSE(testSet)
    rmse_train = lfm.testRMSE(trainset)

    print(rmse_train)
    print(rmse_test)

if __name__ == '__main__':
    play()

