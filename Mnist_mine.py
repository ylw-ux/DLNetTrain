import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


# 创建一个类,继承nn.Module可以使用他的所有方法
class Net(nn.Module):
    #  Python 类的初始化方法。
    # 每当你创建实例 net = Net() 的时候，这个方法会被自动调用。
    def __init__(self):
        # 激活父类，继承了之后要激活才能使用吧
        super().__init__()
        self.fcl1 = nn.Linear(28*28,64)
        self.fcl2 = nn.Linear(64,64)
        self.fcl3 = nn.Linear(64,64)
        self.fcl4 = nn.Linear(64,10)
    def forward(self,x):
        x = F.relu(self.fcl1(x))
        x = F.relu(self.fcl2(x))
        x = F.relu(self.fcl3(x))
        x = F.log_softmax(self.fcl4(x),dim = 1)
        return x

def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST(root="./data", train=is_train, transform=to_tensor, download=True)
    # 创建 DataLoader，批量大小 15，训练集打乱顺序（测试集通常不打乱)
    if(is_train == True):
        return DataLoader(data_set, batch_size=16, shuffle=True)
    if(is_train == False):
        return DataLoader(data_set,batch_size=16,shuffle=False)


def get_eval_data():
    # 灰度图化为张量
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set =MNIST(root="./data",train=False,transform=to_tensor,download=True)
    return DataLoader(dataset=data_set,batch_size=4,shuffle=False)


def train_loss_col(net,dataloader,device,cnt):
    net.eval()
    net.to(device)
    with  torch.no_grad():
        corr=0
        totle=0
        sumloss=0.0
        for (x,y) in dataloader:
            # x,y [batch,weight] 
            x= x.to(device)
            y = y.to(device)
            rul = net.forward(x.view(-1,28*28))
            pre_num = rul.argmax(dim=1)
            # rul = rul.to(device)

            # 每一批的平均loss
            loss = F.nll_loss(rul,y)
            # 得到总loss
            # 所有pytorch参与训练的返回结果都是Tensor格式，为了传播计算
            # 要用.item() 化为数值形式
            sumloss += loss.item()*y.size(0)
            # pre_num  y 为dataloader返回 计算的结果 都为多batch的结果 比较结果为batch数目量的Tensor 里面为【True，...False】这样的结果
            # 所以借sum来计算 最后不要忘了Tensor item 为数值
            corr += (pre_num == y).sum().item()
            # size（0）为batch数  每一批batch加起来就是所有的批次数 即所有y的个数
            totle += y.size(0)
            
                # :.4f 保留4位float结果
        print(f"当前epoch{cnt}轮网络的损失值{sumloss/totle:.4f},\n当前准确率{corr/totle:.4f}\n")



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")
    # 加载训练集和测试集

    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    # 创建网络实例嗯，把它转移到我们的GPU上
    net = Net().to(device)
    # 选择我们，这个网络的参数要如何优化,就是一个优化器
    # 
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    #  训练前把模型调用到训练模式
    
    for i in range(2):
        net.train()
        for (x,y) in train_data:
            x= x.view(-1,28*28)
            x = x.to(device)
            y = y.to(device)
            # 由于我们是分批次来输送数据进行训练
            # 所以每一次的训练前，都把上一次训练计算好的梯度进行清零
            # 这个梯度也是和X、W和B相关的，不可能每次都一样
            net.zero_grad()
            # 向前传播
            output = net.forward(x)
            # 到结果之后计算一下损失
            loss = F.nll_loss(output,y)
            # 然后反向传播，计算梯度
            # 之后就可以用这个梯度来对参数进行更新了
            loss.backward()
            # 用我们选择的优化方式进行更新参数
            optimizer.step()
        dataloader = get_eval_data()
        train_loss_col(net,dataloader,device,i)

    # ===== 可视化预测结果（取测试集前 4 个样本） =====
    # 因为 test_data 迭代返回的 x 在 CPU 上（DataLoader 默认 CPU），
    # 为了推理我们需要将其移到 GPU，但绘图时需要移回 CPU（matplotlib 不支持 CUDA）

    # 把模型调到测试模式
    net.eval()
    with torch.no_grad():#  推理测试的时候关掉梯度计算，不需要，养成区分训练和测试 的意识   
        for (n, (x, _)) in enumerate(test_data):
            if n > 3:   # 只取前 4 个样本
                break
            # 将当前图像（形状 [1, 28, 28]）移到 GPU 并展平进行预测
            x_gpu = x.to(device) # 移到 GPU
            output = net.forward(x_gpu.view(-1, 28 * 28))  
            predict = torch.argmax(output,dim=1)
            # 绘图：x[0] 是原始 CPU 张量（因为 x 在 CPU），可以直接显示
            plt.figure(n)
            # plt.imshow(x[0].view(28, 28), cmap='gray')  # 显示为灰度图
            # 预测结果需要移回 CPU 才能转为 Python 数字
            # plt.title("predict:" + str(int(predict.cpu())))
            plt.imshow(x[0].squeeze(), cmap='gray')
            plt.title(f"predict: {predict[0].item()}")
        plt.show()   # 显示所有图像

if __name__ == "__main__":
    main()




