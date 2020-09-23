__author__='雷克斯掷骰子'
'''
B站:https://space.bilibili.com/497998686
头条:https://www.toutiao.com/c/user/token/MS4wLjABAAAAAxu8A9lNX1qfkRKEyU9Uecqa2opPcZufDLWHbv7m-hVdMVPOe7r_i-k6nw4RY61i/
'''

from surprise import Dataset,Reader,SVD,dump
from surprise import accuracy
from surprise.model_selection import train_test_split
import pandas as pd

'''
https://surprise.readthedocs.io
'''

def readData():
    path = 'ml-latest-small/ratings.csv'
    odatas = pd.read_csv(path, usecols=[0, 1, 2])
    return odatas

def splitTrainSetTestSet(odatas,frac):
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(odatas[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=frac)
    return trainset,testset

def train():
    trainset, testset = splitTrainSetTestSet(readData(), 0.2)
    algo=SVD(n_factors=100,n_epochs=10,lr_all=0.005,reg_all=0.02,biased=True,verbose=True)
    algo.fit(trainset)
    accuracy.rmse(algo.test(testset))
    dump.dump('model/superise_lfm.model',algo=algo)

def play():
    _,algo = dump.load('model/superise_lfm.model')
    print(algo.predict(1,1))


if __name__ == '__main__':

    train()
    #play()