import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# 1. 基础配置与数据准备
# ============================================================

# 字符表：我们只使用 a b c d e 五个字符
chars = ['a', 'b', 'c', 'd', 'e']
# 构建字符到索引的映射（one-hot 的本质就是用索引表示）
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}

vocab_size = len(chars)    # 词汇表大小 = 5
embed_dim = 8             # embedding 输出维度
hidden_size = 16          # RNN 隐藏层维度
num_layers = 2            # RNN 层数，按你的要求设为 2
seq_len = 5               # 序列长度，abcde 共 5 个字符
num_epochs = 100         # 训练轮数
learning_rate = 0.01

# 手动定义输入与目标（倒序）
# 输入: abcde -> [0, 1, 2, 3, 4]
# 目标: edcba -> [4, 3, 2, 1, 0]
input_seq = torch.tensor([char2idx[c] for c in 'abcde'], dtype=torch.long)  # (seq_len,)
target_seq = torch.tensor([char2idx[c] for c in 'edcba'], dtype=torch.long) # (seq_len,)

# 工程化说明：RNN 输入形状一般是 (batch_size, seq_len) 或 (seq_len, batch_size)
# PyTorch 默认 batch_first=False，即 (seq_len, batch, feature)
# 这里我们手动增加 batch 维度，batch_size=1
input_seq = input_seq.unsqueeze(1)   # (seq_len, batch_size=1)
target_seq = target_seq.unsqueeze(1) # (seq_len, batch_size=1)


# ============================================================
# 2. 模型定义（类形式）
# ============================================================

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, output_size):
        super(SimpleRNN, self).__init__()
        
        # —— Embedding 层 ——
        # 作用：将 one-hot 形式的字符索引映射为稠密向量
        # 输入形状：(任意) 里面每个元素是 0~vocab_size-1 的整数索引
        # 输出形状：(..., embed_dim)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,  # 词汇表大小
            embedding_dim=embed_dim     # 每个词向量的维度
        )
        
        # —— 2 层 RNN ——
        # batch_first=False：输入形状为 (seq_len, batch, input_size)
        self.rnn = nn.RNN(
            input_size=embed_dim,    # 输入特征维度 = embedding 输出维度
            hidden_size=hidden_size, # 隐藏状态维度
            num_layers=num_layers,   # RNN 层数
            batch_first=False,       # 不把 batch 放在第一维
            nonlinearity='tanh'      # 激活函数，RNN 默认 tanh
        )
        
        # —— 输出全连接层 ——
        # 把每个时间步的隐藏状态映射到词汇表大小的 logits
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0):
        # x 形状: (seq_len, batch_size)，里面是字符索引
        # 1. 过 embedding，得到 (seq_len, batch_size, embed_dim)
        embed = self.embedding(x)
        
        # 2. 过 RNN
        # out: 所有时间步的隐藏状态，形状 (seq_len, batch, hidden_size)
        # hn:  最后一个时间步的隐藏状态，形状 (num_layers, batch, hidden_size)
        out, hn = self.rnn(embed, h0)
        
        # 3. 每个时间步都过 fc 做预测
        # 工程化技巧：把 (seq_len, batch, hidden) 展平成 (seq_len*batch, hidden)
        # 一次性过线性层，再 reshape 回去，效率更高
        seq_len, batch_size, _ = out.shape
        out = out.reshape(seq_len * batch_size, -1)
        out = self.fc(out)
        out = out.reshape(seq_len, batch_size, -1)
        
        return out, hn

    def init_hidden(self, batch_size):
        """初始化隐藏状态 h0，全零初始化"""
        # 形状：(num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size)
        return h0


# ============================================================
# 3. 训练函数
# ============================================================

def train(model, input_seq, target_seq, num_epochs, lr,device):
    # 损失函数：交叉熵，常用于分类/词汇预测
    # 注意：CrossEntropyLoss 内部自带 softmax，模型最后一层不要加 softmax
    criterion = nn.CrossEntropyLoss()
    # 优化器：Adam 是最常用的基础优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()  # 切换到训练模式（对 Dropout、BatchNorm 等生效，这里虽没用但是规范写法）
    model = model.to(device)
    input_seq,target_seq = input_seq.to(device),target_seq.to(device)

    for epoch in range(num_epochs):
        # 1. 初始化隐藏状态
        batch_size = input_seq.shape[1]
        h0 = model.init_hidden(batch_size)
        h0 = h0.to(device)
        # 2. 前向传播
        output, _ = model(input_seq, h0)

        # output 形状: (seq_len, batch, vocab_size)
        
        # 3. 计算损失
        # CrossEntropyLoss 要求输入形状 (N, C)，目标形状 (N,)
        # 所以把 seq_len 和 batch 维度合并
        output_flat = output.reshape(-1, vocab_size)   # (seq_len*batch, vocab_size)
        target_flat = target_seq.reshape(-1)           # (seq_len*batch,)

        loss = criterion(output_flat, target_flat)
        
        # 4. 反向传播 + 参数更新
        optimizer.zero_grad()  # 梯度清零，防止累积
        loss.backward()        # 反向传播计算梯度
        optimizer.step()       # 更新参数
        
        # 每 50 轮打印一次
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# ============================================================
# 4. 验证/推理函数
# ============================================================

def evaluate(model,input_seq,device):
    model.eval()  # 切换到评估模式

    model = model.to(device)
    input_seq = input_seq.to(device)
    with torch.no_grad():  # 推理阶段不计算梯度，节省显存、加速
        batch_size = input_seq.shape[1]

        h0 = model.init_hidden(batch_size).to(device)
        
        output, _ = model(input_seq, h0)
        
        # 取概率最大的索引作为预测结果
        # output 形状 (seq_len, batch, vocab_size)，在最后一维取 argmax
        pred_idx = torch.argmax(output, dim=-1)  # (seq_len, batch)
        
        # 把索引转回字符
        pred_chars = [idx2char[i.item()] for i in pred_idx.squeeze()]
        input_chars = [idx2char[i.item()] for i in input_seq.squeeze()]
        
        print(f"\n输入序列: {''.join(input_chars)}")
        print(f"预测序列: {''.join(pred_chars)}")
        print(f"目标序列: edcba")


# ============================================================
# 5. 主程序入口
# ============================================================

if __name__ == '__main__':
    # 实例化模型
    model = SimpleRNN(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=vocab_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 训练
    print("开始训练...")
    train(model, input_seq, target_seq, num_epochs, learning_rate,device)
    
    # 验证
    evaluate(model, input_seq,device)


# 总结一下:第一点就是在模型内的创建里，不要忘记了初始的H0
# 第二点就是对于运算设备的移动，嗯，模型和我们的初始的数据张量都需要进行移动
# 因为这个位置的移动，本质上是在哪个硬件上面进行计算.
# 你有一个东西不在同一位置上的话，不就相当于少东西了 肯定无法计算
# 然后就是这个embedding和我们的输入数据啊，最主要的是要做一个输入数据的序列化映射
# 然后再通过我们自定义的embedding转化为稠密的特征向量.然后这个向量在输入网络中
# 还有一个要点就是对于我们的outputs和hn的处理
# Outputs, 它是一个三维向量，而我们的这个结果输入线性层的话只能是二维向量。
# 我们可以对每个seqlen进行拆分，再分别输入，但是过于麻烦，
# 可以直接对结果reshape，保留权重为一个单独维度，
# 输入线性层，最后连接完再输出recepe回去