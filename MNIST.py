# ==================== 1. 导入必要的库 ====================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# ==================== 2. 定义神经网络模型 ====================
class Net(nn.Module):
    """
    一个简单的全连接神经网络，用于手写数字识别（MNIST）。
    输入：28x28 像素的图像（展平为 784 维向量）
    输出：10 个类别的对数概率（log-softmax）
    """
    def __init__(self):
        super(Net, self).__init__()
        # 定义四个全连接层（线性层）
        self.fc1 = nn.Linear(28 * 28, 64)   # 输入 784 -> 隐藏层 64
        self.fc2 = nn.Linear(64, 64)        # 64 -> 64
        self.fc3 = nn.Linear(64, 64)        # 64 -> 64
        self.fc4 = nn.Linear(64, 10)        # 64 -> 输出 10 个类别

    def forward(self, x):
        """
        前向传播：输入 x 是形状为 (batch_size, 784) 的张量
        经过三个 ReLU 激活的隐藏层，最后输出 log_softmax
        """
        x = F.relu(self.fc1(x))        # 第一层 + ReLU
        x = F.relu(self.fc2(x))        # 第二层 + ReLU
        x = F.relu(self.fc3(x))        # 第三层 + ReLU
        x = F.log_softmax(self.fc4(x), dim=1)  # 输出层 + log_softmax（用于分类）
        return x

# ==================== 3. 数据加载函数 ====================
def get_data_loader(is_train):
    """
    加载 MNIST 数据集并返回 DataLoader。
    参数：
        is_train (bool): True 表示训练集，False 表示测试集
    返回：
        DataLoader 对象，每次迭代返回 (图像, 标签)
    """
    # 数据预处理：将 PIL 图像转换为 PyTorch 张量，像素值归一化到 [0,1]
    to_tensor = transforms.Compose([transforms.ToTensor()])
    # 下载 MNIST 数据集（如果本地没有）
    data_set = MNIST(root="./data", train=is_train, transform=to_tensor, download=True)
    # 创建 DataLoader，批量大小 15，训练集打乱顺序（测试集通常不打乱，这里为简单统一 shuffle=True）
    return DataLoader(data_set, batch_size=15, shuffle=True)

# ==================== 4. 评估函数（支持 GPU） ====================
def evaluate(test_data, net, device):
    """
    在给定的测试数据集上评估模型的准确率。
    参数：
        test_data (DataLoader): 测试数据加载器
        net (Net): 训练好的神经网络模型（已在 device 上）
        device (torch.device): 模型所在的设备（'cuda' 或 'cpu'）
    返回：
        float: 准确率（正确预测数 / 总样本数）
    """
    n_correct = 0   # 正确预测的样本数
    n_total = 0     # 总样本数
    # 禁用梯度计算（评估时不需要反向传播，节省内存和计算）
    with torch.no_grad():
        for (x, y) in test_data:
            # 将数据和标签移到同一个设备（GPU）
            x, y = x.to(device), y.to(device)
            # 将 28x28 图像展平为 784 维向量，前向传播得到输出
            outputs = net.forward(x.view(-1, 28 * 28))  # outputs 形状: (batch_size, 10)
            # 对每个样本，比较预测类别（argmax）与真实标签
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total

# ==================== 5. 主函数 ====================
def main():
    # 检测是否有可用 GPU，定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 加载训练集和测试集
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)

    # 创建模型实例，并将其移动到指定设备（GPU 或 CPU）
    net = Net().to(device)

    # 评估初始模型（未训练）的准确率（通常接近 10%，随机猜测）
    print("初始准确率:", evaluate(test_data, net, device))

    # 定义优化器（Adam，学习率 0.001）
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # 训练 2 个 epoch（完整遍历训练集两次）
    for epoch in range(2):
        # 在每个 epoch 中，遍历训练集的所有批次
        for (x, y) in train_data:
            # 将数据和标签移到 GPU
            x, y = x.to(device), y.to(device)
            # 梯度清零（避免累积）
            net.zero_grad()
            # 前向传播：展平图像，得到输出
            output = net.forward(x.view(-1, 28 * 28))
            # 计算负对数似然损失（因为输出是 log_softmax）
            loss = F.nll_loss(output, y)
            # 反向传播计算梯度
            loss.backward()
            # 优化器更新参数
            optimizer.step()

        # 每个 epoch 结束后，在测试集上评估当前模型准确率
        acc = evaluate(test_data, net, device)
        print(f"Epoch {epoch} 准确率: {acc}")

    # ===== 可视化预测结果（取测试集前 4 个样本） =====
    # 因为 test_data 迭代返回的 x 在 CPU 上（DataLoader 默认 CPU），
    # 为了推理我们需要将其移到 GPU，但绘图时需要移回 CPU（matplotlib 不支持 CUDA）
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:   # 只取前 4 个样本
            break
        # 将当前图像（形状 [1, 28, 28]）移到 GPU 并展平进行预测
        x_gpu = x.to(device)                     # 移到 GPU
        predict = torch.argmax(net.forward(x_gpu.view(-1, 28 * 28)))
        # 绘图：x[0] 是原始 CPU 张量（因为 x 在 CPU），可以直接显示
        plt.figure(n)
        plt.imshow(x[0].view(28, 28), cmap='gray')  # 显示为灰度图
        # 预测结果需要移回 CPU 才能转为 Python 数字
        plt.title("预测: " + str(int(predict.cpu())))
    plt.show()   # 显示所有图像

# ==================== 6. 入口点 ====================
if __name__ == "__main__":
    main()