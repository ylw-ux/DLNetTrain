import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn

class MyDadaset(Dataset):
    def __init__(self,filepath):
        super().__init__()
        # 去导入我们的数据集
        xy = np.loadtxt(fname=filepath,delimiter=',',dtype=np.float32)
        self.len = xy.shape([0])
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])
         
    def __getitem__(self, index):
        # return super().__getitem__(index)
        return self.x_data[index],self.y_data[index]
        # (x,y)
        # 通过索引去调取第几个数据 
    def __len__(self):
        # 获取数据值的大小
        return self.len


class modul(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = nn.Linear(8,6)
        self.l2 = nn.Linear(6,6)
        self.l3 = nn.Sigmoid(6,1)


    def forward(self,x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_modul = modul().to(device)
    dataset = MyDadaset("你的数据集的地址")
    train_data = DataLoader(dataset=dataset,batch_size=10,shuffle=True,num_workers=2)

    optimizer = torch.optim.Adam(my_modul.parameters(), lr=0.001)
    # 第一个告诉他我的数据集是哪个。第二个讲它的batch size是多少,第三个讲是不是打乱
    # num workers代表我读取这个数据的时候是否需要并行，如果比较是强大的硬件的话，可以并形，这样提升效率
    for epoch in range(100):
        for data,i in enumerate(train_data,0):
            x,y = data
            x=x.to(device)
            y = y.to(device)
            # 后面就是 清空梯度正向传播、反向传播，计算损失，优化器优化
            my_modul.zero_grad()
            x_pre = my_modul.forward(x)
            loss = torch.nn.functional.nll_loss(x_pre,y)
            loss.backward()
            optimizer.step()



if __name__ == "__main__":
    main()