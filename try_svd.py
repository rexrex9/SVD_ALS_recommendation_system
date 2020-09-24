__author__='雷克斯掷骰子'
'''
B站:https://space.bilibili.com/497998686
头条:https://www.toutiao.com/c/user/token/MS4wLjABAAAAAxu8A9lNX1qfkRKEyU9Uecqa2opPcZufDLWHbv7m-hVdMVPOe7r_i-k6nw4RY61i/
'''
import numpy as np

def svd(data,k):
    u,i,v = np.linalg.svd(data)
    u=u[:,0:k]
    i=np.diag(i[0:k])
    v=v[0:k,:]

    return u,i,v

def predictSingle(u_index,i_index,u,i,v):
    return u[u_index].dot(i).dot(v.T[i_index].T)

def play():
    import sys
    k=4
    data = np.mat([[1,2,3,1,1],[1,3,3,1,2],[3,1,1,2,1],[1,2,3,3,1]])
    u,i,v = svd(data,k)
    print(u.dot(i).dot(v))
    print(predictSingle(0, 0, u, i, v))

if __name__ == '__main__':
    play()

