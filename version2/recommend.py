import torch


class Recommend():

    def __init__(self):
        self.user_items, self.all_items = self.get_all_items()
        self.net = torch.load('ALS.model')

    def get_all_items(self,dataFile='../ml-latest-small/ratings.csv'):
        with open(dataFile, 'r', encoding='utf-8') as f:
            allData = f.read().split('\n')[1:]
        allData=list(set(filter(None,allData)))
        user_items={}
        all_items=set()
        for data in allData:
            data = data.split(',')
            if int(data[0]) in user_items:
                user_items[int(data[0])].add(int(data[1]))
            else:
                user_items[int(data[0])]={int(data[1])}
            all_items.add(int(data[1]))
        return user_items,all_items

    def recommend(self,user,topk):
        als = torch.load('ALS.model')
        items = self.user_items[user]
        items = self.all_items-items
        user_boardcast=torch.LongTensor([user for _ in items])
        order_items = list(items)
        items = torch.LongTensor(order_items)
        logits = als.forward(user_boardcast,items)
        logits = logits.detach().numpy().tolist()
        a = sorted(zip(logits,order_items),reverse=True)
        return a[:topk]

if __name__ == '__main__':
    k = 5
    r = Recommend()
    tops = r.recommend(1,k)
    print(tops)