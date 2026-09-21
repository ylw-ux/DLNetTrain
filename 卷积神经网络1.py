import  torch
import  torch.nn as nn
import  torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np

def data_set():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])
    data_set1 = datasets.MNIST(root='./data',transform=transform,download=True,train=True)
    data_set2 = datasets.MNIST(root='./data',transform=transform,download=True,train=False)
    train_set = DataLoader(data_set1,batch_size=10,shuffle=True)
    test_set = DataLoader(data_set2,batch_size=10,shuffle=False)
    print("data加载完成\n")
    print("len_train",len(train_set),"\n")  
    print("len_test",len(test_set),"\n")
    return train_set,test_set

class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1,10,kernel_size=5,stride=1)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5,stride=1)
        self.conv3 = nn.Conv2d(20,30,kernel_size=3,stride=1)
        self.relu = nn.ReLU()
        self.maxpol = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Lin1 = nn.Linear(30,40)
        self.Lin2 = nn.Linear(40,50)
        self.Lin3 = nn.Linear(50,10)
        
        
    def forward(self,x):
        x = self.maxpol(self.relu(self.conv1(x)))
        x = self.maxpol(self.relu(self.conv2(x)))
        x = self.maxpol(self.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1) 
        x = self.relu(self.Lin1(x))
        x = self.relu(self.Lin2(x))
        '''
        最后一层保持线性输出，保留原本数值
        '''
        x = self.Lin3(x)
        return x




def train(device,train_set,epoch):
    net = Net().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for i in range(epoch):
        for x,y in train_set:
            x,y = x.to(device),y.to(device)
            '''使用优化器来掌控梯度清除更加好,管理多个参数组'''
            optimizer.zero_grad()
            y_pre = net.forward(x)
            loss = F.cross_entropy(y_pre,y)
            loss.backward()
            optimizer.step()
    print("训练完毕\n")
    return net


def test(device,test_set,net):
    n_correct = 0   # 正确预测的样本数
    n_total = 0     # 总样本数
    # 禁用梯度计算（评估时不需要反向传播，节省内存和计算）
    n=0
    with torch.no_grad():
        for n,(x, y) in enumerate(test_set):
            # 将数据和标签移到同一个设备（GPU）
            x, y = x.to(device), y.to(device)
            # 将 28x28 图像展平为 784 维向量，前向传播得到输出
            outputs = net.forward(x)  # outputs 形状: (batch_size, 10)
            # 对每个样本，比较预测类别（argmax）与真实标签
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
            print(f"第{n}批次验证完毕\n")
    return n_correct / n_total
      

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set,test_set = data_set()
    new_net = train(device,train_set,5)
    rul = test(device,test_set,new_net)
    print(f"successful pre{rul}")

if __name__ == "__main__":
    main()

