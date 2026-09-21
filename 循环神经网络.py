import torch
import torch.nn as nn

# ---------------------- 超参数定义 ----------------------
batch_size = 3       # 批次：3个样本，张量第0维，一行代表1个样本
input_size = 10      # 每个时间步输入特征数，张量第1维（列）
hidden_size = 20     # 隐藏状态维度，h的列数
seq_len = 5          # 序列一共有5个时间步 t0,t1,t2,t3,t4

# 1. 创建RNNCell单元（单时间步最小单元）
cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

# 2. 构造模拟输入序列
# 注意：完整序列 shape = [seq_len, batch_size, input_size]
# seq_len：时间步；batch：样本（行）；input_size：特征（列）
x_all = torch.randn(seq_len, batch_size, input_size)
print(f"完整输入序列x_all shape: {x_all.shape}  [seq_len, batch, input_size]")
print("="*70)

# 3. 【必须手动初始化h0】第一个时刻没有历史记忆，RNNCell不会帮你初始化
# h_prev 就是 h_{t-1}, shape=[batch_size, hidden_size]
# 行：batch样本，列：隐藏记忆维度
h_prev = torch.zeros(batch_size, hidden_size)
print(f"初始隐藏状态 h0 shape: {h_prev.shape}  [batch, hidden_size]")
print("="*70)

# 保存每一步输出的隐藏状态
h_list = []

# ！！！重点：RNNCell没有内置时间循环，序列遍历必须自己手写for循环
for t in range(seq_len):
    # 取出第t时刻所有样本的输入 x_t
    x_t = x_all[t]
    print(f"\n------时间步 t={t} ------")
    print(f"x_t shape = {x_t.shape}    [batch, input_size]")
    print(f"输入cell的h_prev(h_t-1) shape = {h_prev.shape} [batch, hidden_size]")

    # 执行RNNCell前向传播，内部自动做：x@Wih.T + hprev@Whh.T + bias + tanh
    h_t = cell(x_t, h_prev)

    print(f"cell输出 h_t shape = {h_t.shape} [batch, hidden_size]")
    h_list.append(h_t)

    # 更新：当前输出h_t，作为下一个时间步的历史状态 h_prev
    h_prev = h_t


print("\n"+"="*70)
# h_list存放t0~t4全部时刻的隐藏状态
# 如果要做预测分类，拿最后时刻 h_prev 外接Linear层
print(f"最后时刻隐藏状态 h_final shape: {h_prev.shape} [batch, hidden_size]")

# ========== 外接线性层做预测（RNNCell本身不会输出y，必须自己写）==========
num_classes = 8  # 假设任务输出8个类别
head = nn.Linear(hidden_size, num_classes)
y_pred = head(h_prev)
print(f"经过外接Linear得到预测输出y_pred shape:{y_pred.shape} [batch, num_classes]")