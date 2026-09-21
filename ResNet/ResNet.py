import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split

# torchvision里面有大量的数据集
from torchvision import datasets
from torchvision import transforms
# from torch.utils.data.sampler import SubsetRandomSampler




def dataloader (batch=16,istest= True,shuffle = False,val_size = None,random_seed=42,data_dir = "./data"):
    # norminlze to data
    
    normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
        )
    # create a to tensor for data
    toTensor = transforms.Compose([
                    transforms.Resize((64, 64)),   # 先 Resize
                    transforms.ToTensor(),           # 再 ToTensor
                    normalize,                       # 最后 Normalize
                ])

    # dif ttain and test // if shuffle
    # get datasets
    if istest == True:
        dataset = datasets.CIFAR10(root=data_dir, 
                                           train=False,download=True, transform=toTensor,)
        # create dataloader 
        return DataLoader(dataset=dataset,shuffle=shuffle,batch_size=batch)

    

    if istest == False:
        # 这一步是Shuffle再定为true,保险
        shuffle = True
        dataset = datasets.CIFAR10(root=data_dir,
                                   train=True,download=True,transform=toTensor)

        # if val_size:valid_size=0.1：10%验证，90%训练
        if val_size is not None:
            # 划分：训练子集 + 验证集
            generator = torch.Generator().manual_seed(random_seed)
            val_len = int(len(dataset) * val_size)
            train_len = len(dataset) - val_len
            train_dataset, val_dataset = random_split(dataset=dataset, lengths=[train_len, val_len], 
                                                      generator=generator)
            # 训练集打乱
            train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch)
            # value 不打乱
            val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch)
            
            return train_loader, val_loader
        # 不要求验证集，那就只返回训练集
        else:
            return DataLoader(dataset= dataset,shuffle=shuffle,batch_size=batch)



class ResidualBlock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=3,stride=1,padding=1,downsample = None):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=kernel,stride=stride,padding=padding),
            nn.BatchNorm2d(out_channel)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel,out_channel,kernel_size=kernel,stride=1,padding=padding),
            nn.BatchNorm2d(out_channel)
        )
    #     downsample = nn.Sequential(
    #           nn.Conv2d(..., kernel_size=1, stride=stride, bias=False),
    #           nn.BatchNorm2d(...)
    #       )
        self.downsample = downsample

    def forward(self,x):
        residual = x # [b,c,w,h]
        # 先保存初始进来的X作为research，然后X通过两层向前传播，
        # 最后在结尾判断一下是否需要downsample，
        # 如果需要的话，就把X通过downsample，使它的维度对齐
        if self.downsample is not None:
                residual  = self.downsample(residual)

        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        
        # 我最终是要改变原始保留的residual，使它和变换后的能够对齐
        
        out = F.relu(out)
        return out


class DownsampleProj(nn.Module):
    def __init__(self, in_channel,out_channel,kernel=1,stride=1,padding=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel,stride,padding)
        self.norm = nn.BatchNorm2d(out_channel)
    def forward (self,x):
        x = self.norm(self.conv1(x))
        return x 


class Net(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=3,stride=1,padding=1,downsample = None):
        super().__init__()
        self.ResBlock1 =  ResidualBlock(in_channel,out_channel,kernel,stride,padding,downsample)
        self.ResBlock2 =  ResidualBlock(out_channel,out_channel,kernel,stride=1,padding=1,downsample=None)
        # 加一个全局平均池化层降低wh的尺寸，
        # 后面层的线性层 展平的话，不降低尺寸会有维度爆炸
        self.avepool = nn.AdaptiveAvgPool2d((1,1))
        self.lin1 = nn.Linear(out_channel,10)
        self.downsample1 = downsample
    def forward(self,x):
            x = self.ResBlock1(x) #[b,ic,w,h]
            x = self.ResBlock2(x) # [b,oc,w,h]
            x = self.avepool(x) # [b,oc,1,1]
            ### 第一维开始全部展平。变成二维tensor [B,C*W*H]  [b,oc]
            x = torch.flatten(x,start_dim=1)
            x = self.lin1(x) # [b,10]
            x = F.log_softmax(x,dim=1)
            return x

        


def epoch_val_arr(net,test_set,device):
    net.eval()
    net.to(device)
    # how many epoch to test
    epo_loss =0.0
    corr = 0
    totle= 0
    # 验证不需要计算梯度
    print("验证:")
    with torch.no_grad():
        for (n,(x,y)) in  enumerate(test_set):
            x,y = x.to(device),y.to(device)
            out = net(x)
            # 计算得到 max index 
            pre_out = out.argmax(dim=1)
            # 计算得到 
            one_batch_ave_loss = F.nll_loss(out,y)  # 当前batch的平均loss
            epo_loss += one_batch_ave_loss.item() * y.size(0) # ave * num -> all loss 
            # 比较结果 True的个数 sum Tensor--> float
            corr += (pre_out == y).sum().item()
            totle += y.size(0)
            
        #计算loss 和 acc
        loss = epo_loss / totle
        acc = corr / totle

        print(f"当前batch{n},loss={loss},arr={acc}\n")



def train(epoch,train_set,val_set,device):
    print(f"正在使用设备: {device}")
    # CIFAR输入通道=3，第一个残差块把3映射到32，stride=2下采样
    downsample = DownsampleProj(3,32,stride=2)
    net = Net(in_channel=3,out_channel=32,stride=2,downsample=downsample)
    
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001)
    for i in range(epoch):
        net.train()
        for (n,(x,y)) in enumerate(train_set):
            x,y = x.to(device),y.to(device)
            ### x = x.contiguous()
            x = x.contiguous()
            optimizer.zero_grad()
            # feed
            out = net.forward(x)
            # loss 
            loss  = F.nll_loss(out,y)
            # back
            loss.backward()
            # optim
            optimizer.step()
               
        # current epoch acc
        epoch_val_arr(net,val_set,device)

    return net

def test(net,test_set,device):
    # 测试只跑一遍，不需要epoch
    net.eval()
    net.to(device)
    # 测试不需要计算梯度
    loss=0.0
    corr=0
    totle =0
    with torch.no_grad():
        for (x,y) in test_set:
            x,y = x.to(device),y.to(device)
            out = net(x)
            pre_img = out.argmax(dim=1)

            # batch_ave_loss
            ave_loss = F.nll_loss(out,y)
            loss+= ave_loss.item() * y.size(0)

            # acc compare sum 
            corr += (pre_img==y).sum().item()
            totle += y.size(0)
        rul_loss = loss/totle
        rul_acc = corr/totle
        return rul_loss,rul_acc



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set , val_set = dataloader(istest=False,shuffle=True,val_size=0.1,random_seed=42)
    test_set = dataloader(istest=True,shuffle=False)

    trained_net =  train(10,train_set=train_set,val_set=val_set,device=device)
    rul_loss,rul_acc = test(net=trained_net,test_set=test_set,device=device)
    print(f"eval结果rul_loss={rul_loss},rul_acc={rul_acc}\n")



if __name__ == "__main__":
    main()