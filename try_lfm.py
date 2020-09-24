__author__='雷克斯掷骰子'
'''
B站:https://space.bilibili.com/497998686
头条:https://www.toutiao.com/c/user/token/MS4wLjABAAAAAxu8A9lNX1qfkRKEyU9Uecqa2opPcZufDLWHbv7m-hVdMVPOe7r_i-k6nw4RY61i/
'''

import numpy as np

def prediction(pu,qi):
    return np.dot(pu,qi.T)

def getError(r,pu,qi):
    return r - prediction(pu, qi)

def tryTrain():
    real=np.mat([[1,2,3,0,3],[3,0,3,1,3],[3,2,0,3,1]])
    print(real)

    factors=3
    p = np.random.randn(3, factors)
    q = np.random.randn(5, factors)

    ul,il=real.shape

    lr=0.05
    lamda=0.1

    for e in range(30):
        for u in range(ul):
            for i in range(il):
                r=real[u,i]
                if r!=0:
                    error = getError(r,p[u],q[i])
                    p[u] -= lr * (-2 * error * q[i] + 2 * lamda * p[u])
                    q[i] -= lr * (-2 * error * p[u] + 2 * lamda * q[i])


    print(prediction(p,q))




if __name__ == '__main__':
    tryTrain()