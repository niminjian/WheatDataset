import torch.nn as nn
import torch


# 定义一个残差块，用于构建深度共享网络
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x)
        out = self.relu(out + shortcut)
        return out


# 定义多头自注意力模块，加入LayerNorm稳定输出
class MultiHeadSelfAttentionLayer(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadSelfAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, dim = x.size()
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = self.softmax(scores)
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


# 定义优化后的多任务 DNN 模型
class EnhancedMultiTaskDNN(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedMultiTaskDNN, self).__init__()
        self.relu = nn.ReLU()
        # 初始映射层，将输入映射到256维
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        # 共享层：堆叠3个残差块
        self.res_block1 = ResidualBlock(256, 256, dropout_rate=0.3)
        self.res_block2 = ResidualBlock(256, 256, dropout_rate=0.3)
        self.res_block3 = ResidualBlock(256, 256, dropout_rate=0.3)

        # 多头自注意力层：对共享表示进行非线性特征提取
        # 输入维度为256，使用4个头
        self.attention = MultiHeadSelfAttentionLayer(256, num_heads=4)

        # 任务专用分支，每个任务使用独立的分支（使用较小的残差块结构）
        def task_branch():
            return nn.Sequential(
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            )

        self.task1_branch = task_branch()
        self.task2_branch = task_branch()
        self.task3_branch = task_branch()
        self.task4_branch = task_branch()

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.fc_in(x)  # (batch_size, 256)
        x = self.res_block1(x)  # 共享层残差块1
        x = self.res_block2(x)  # 共享层残差块2
        x = self.res_block3(x)  # 共享层残差块3

        # 将共享表示扩展维度以适应自注意力模块： (batch_size, seq_len=1, 256)
        x_unsq = x.unsqueeze(1)
        x_att = self.attention(x_unsq).squeeze(1)  # (batch_size, 256)
        # 融合共享表示与注意力输出，例如取平均
        x_combined = (x + x_att) / 2

        # 任务专用分支
        out1 = self.task1_branch(x_combined)
        out2 = self.task2_branch(x_combined)
        out3 = self.task3_branch(x_combined)
        out4 = self.task4_branch(x_combined)

        return out1, out2, out3, out4

class FeatureInteractionLayer(nn.Module):
    def __init__(self, input_dim):
        super(FeatureInteractionLayer, self).__init__()
        # 使用一个线性层来实现特征交互
        self.interaction = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        interaction_features = self.relu(self.interaction(x))
        return interaction_features


class ConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_sizes):
        super(ConvLayer, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=ks, padding=ks // 2)
            for ks in kernel_sizes
        ])
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv_outs = [conv(x) for conv in self.convs]
        out = sum(conv_outs)  # 多个卷积核的输出相加
        out = self.batch_norm(out)
        out = self.relu(out)
        return out


class MixedPoolingLayer(nn.Module):
    def __init__(self, pool_size):
        super(MixedPoolingLayer, self).__init__()
        self.max_pool = nn.MaxPool1d(pool_size)
        self.avg_pool = nn.AvgPool1d(pool_size)

    def forward(self, x):
        max_pooled = self.max_pool(x)
        avg_pooled = self.avg_pool(x)
        return max_pooled + avg_pooled  # 混合池化输出相加


class MultiHeadSelfAttentionLayer(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadSelfAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5))
        attention_output = torch.matmul(attention_scores, V).transpose(1, 2).contiguous().view(batch_size, seq_len, dim)

        return attention_output + x  # 残差连接


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, input_dim):
        super(AdaptiveFeatureFusion, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        fusion_weights = self.sigmoid(self.fc(x))
        return fusion_weights * x + (1 - fusion_weights) * y  # 自适应加权融合

class MoistureContentMLP(nn.Module):
    def __init__(self):
        super(MoistureContentMLP, self).__init__()

        # 卷积层，使用不同尺寸的卷积核
        self.conv = ConvLayer(input_dim=1, output_dim=16, kernel_sizes=[1, 3, 5])

        # 混合池化层
        self.mixed_pool = MixedPoolingLayer(pool_size=2)

        # 初始层
        self.fc1 = nn.Sequential(
            nn.Linear(16, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 多头自注意力机制
        self.multi_head_attention = MultiHeadSelfAttentionLayer(256, num_heads=4)

        # 特征交互层
        self.feature_interaction = FeatureInteractionLayer(256)

        # 自适应特征融合
        self.adaptive_fusion = AdaptiveFeatureFusion(256)

        # 后续层
        self.fc2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fc5 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.fc6 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # 输出层（多任务）
        self.output_layer_task1 = nn.Linear(32, 1)  # 任务 1 输出层
        self.output_layer_task2 = nn.Linear(32, 1)  # 任务 2 输出层
        self.output_layer_task3 = nn.Linear(32, 1)  # 任务 3 输出层
        self.output_layer_task4 = nn.Linear(32, 1)  # 任务 4 输出层

    def forward(self, x):
        if len(x.shape) != 3:
            x = x.unsqueeze(1)  # 增加一个通道维度以匹配卷积输入要求

        # 卷积层提取特征
        x = self.conv(x)

        # 混合池化层提取特征
        x = self.mixed_pool(x)
        x = x.view(x.size(0), -1)  # 展平卷积输出以输入到全连接层

        x = self.fc1(x)  # 第一层

        # 多头自注意力层
        x_attention = self.multi_head_attention(x.unsqueeze(1)).squeeze(1)  # 提取自注意力特征
        x = self.adaptive_fusion(x, x_attention)  # 自适应融合自注意力特征与原始特征

        # 特征交互
        x_interaction = self.feature_interaction(x)  # 提取交互特征
        x = x + x_interaction  # 与原始特征相加以融合

        x = self.fc2(x)  # 第二层
        x = self.fc3(x)  # 第三层
        x = self.fc4(x)  # 第四层
        x = self.fc5(x)  # 第五层
        x = self.fc6(x)  # 第六层

        # 多任务输出
        output_task1 = self.output_layer_task1(x)  # 任务 1 输出
        output_task2 = self.output_layer_task2(x)  # 任务 2 输出
        output_task3 = self.output_layer_task3(x)  # 任务 3 输出
        output_task4 = self.output_layer_task4(x)  # 任务 4 输出

        return output_task1, output_task2, output_task3, output_task4
