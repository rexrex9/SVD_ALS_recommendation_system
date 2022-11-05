from torch.utils.data import DataLoader
from version2.evaluate import doEva
from version2.model import ALS
from version2.data_prepare import loadData
import torch


def train(epochs = 5, batchSize = 1024):
    trainData, testData,n_users,n_items = loadData()
    net=ALS(n_users,n_items,)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01)
    criterion=torch.nn.MSELoss()
    for e in range(epochs):
        all_lose=0
        for u,i,r in DataLoader(trainData,batch_size=batchSize,shuffle=True):
            optimizer.zero_grad()
            r=torch.FloatTensor(r.detach().numpy())
            result = net(u,i)
            loss = criterion(result,r)
            all_lose+=loss
            loss.backward()
            optimizer.step()

        print('epoch{}, avg_loss={}'.format(e,all_lose/(len(trainData)//batchSize)))
        rms = doEva( net,testData )
        print('rms:{}'.format(float(rms)))
    torch.save(net,'ALS.model')

if __name__ == '__main__':

    train()