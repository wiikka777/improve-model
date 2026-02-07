import torch
import torch.nn as nn
from torchfm.layer import FeaturesEmbedding, CrossNetwork, MultiLayerPerceptron


class My_DeepCrossNetworkModel(torch.nn.Module):
    """
    A pytorch implementation of Deep & Cross Network.

    Reference:
        R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.
    """

    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        h_l2 = self.mlp(embed_x)
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        p = self.linear(x_stack)
        # return torch.sigmoid(p.squeeze(1))
        return p.squeeze(1)


class My_DeepCrossNetworkModel_withCommentsRanking(nn.Module):
    def __init__(self, field_dims, comments_dims, embed_dim, num_layers, mlp_dims, dropout, text_embeddings, 
                 attention_dim=64, nhead=5):
        super().__init__()

        # 独立嵌入层用于前 -6 列
        self.individual_embedding = FeaturesEmbedding(field_dims, embed_dim)

        # 共享嵌入层用于 -6:-1 列
        self.shared_embedding = FeaturesEmbedding([comments_dims], embed_dim)
        
        self.embed_dim = embed_dim
        
        self.text_embeddings = text_embeddings[0]
        self.text_embed_dim = self.text_embeddings.size(1)
        self.user_comment_embeddings = text_embeddings[1]

        # 添加降维线性层，将 text_embed_dim 降到 embed_dim
        self.text_dim_reducer = nn.Linear(self.text_embed_dim, embed_dim)
        self.comment_dim_reducer = nn.Linear(self.text_embed_dim, embed_dim)

        # 计算总的嵌入输出维度
        # query(用户+视频) + attn_output(注意力增强) + comment_embeds(原始评论)
        self.seq_len = (len(field_dims) + 1) * 2 + 6
        self.embed_output_dim = self.seq_len * embed_dim

        # 初始化 CrossNetwork, MLP 和 Linear 层
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)
        
        # MultiheadAttention 模块，用于额外的评论打分
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=nhead, dropout=dropout, batch_first=True)
        
        # 早期评论权重预测网络（基于用户+视频特征）
        self.early_comment_scorer = nn.Sequential(
            nn.Linear((len(field_dims) + 1) * embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 6),
            nn.Softmax(dim=1)
        )
        
        self.comment_score_linear = nn.Sequential(
            nn.Linear(self.embed_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
        )
        self.comment_score_linear_ = nn.Sequential(
            nn.Linear(self.embed_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
        )
        self.softmax = nn.Softmax(dim=1)  # 添加 softmax 层

    def forward(self, x, explicit_weights=None):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        individual_embed_x = self.individual_embedding(x[:, :-6])

        # Comment embeddings (B,6)
        comment_ids = x[:, -6:]
        # Ensure ids on same device as embeddings
        comment_ids = comment_ids.to(self.user_comment_embeddings.device)
        comment_embeds = self.user_comment_embeddings[comment_ids]  # (B,6,text_embed_dim)
        comment_embeds = self.comment_dim_reducer(comment_embeds)  # (B,6,embed_dim)

        # 不再使用手动传入的 explicit_weights，让模型通过后续网络自动学习评论权重

        # Text embeddings (single per example)
        text_embed_ids = x[:, -8]
        text_embed_ids = text_embed_ids.to(self.text_embeddings.device)
        text_embeds = self.text_embeddings[text_embed_ids]
        text_embeds = self.text_dim_reducer(text_embeds)

        # === 方案B：早期评论权重预测 + 反馈机制 ===
        # 1. 基于用户+视频特征，预测评论重要性权重
        query_flat = torch.cat([individual_embed_x, text_embeds.unsqueeze(1)], dim=1).view(individual_embed_x.size(0), -1)
        early_comment_weights = self.early_comment_scorer(query_flat)  # (B, 6)
        
        # 2. 用权重对评论特征进行加权（门控机制）
        weighted_comment_embeds = comment_embeds * early_comment_weights.unsqueeze(-1)  # (B, 6, embed_dim)

        # Cross-Attention: 用户/视频(query) 挑选 加权后的评论(key/value)
        # Query: 用户 + 视频特征 (决定我们要找什么样的评论)
        query = torch.cat([individual_embed_x, text_embeds.unsqueeze(1)], dim=1)  # (B, len(field_dims)+1, embed_dim)
        
        # Key/Value: 使用加权后的评论特征
        key_value = weighted_comment_embeds  # (B, 6, embed_dim)
        
        # 执行 Cross-Attention: 用户/视频去"查询"哪些评论最相关
        # attn_output: 加权后的评论表示, attn_weights: 自动分配的评论权重
        attn_output, attn_weights = self.multihead_attn(
            query=query,
            key=key_value,
            value=key_value
        )  # attn_output: (B, len(field_dims)+1, embed_dim), attn_weights: (B, len(field_dims)+1, 6)
        
        # 保存自动生成的评论权重 (平均所有 query 位置的注意力)
        self.auto_comment_weights = attn_weights.mean(dim=1)  # (B, 6) - 每个评论的平均权重
        self.early_comment_weights = early_comment_weights  # 保存早期权重用于分析
        
        # 重新组合所有特征: 原始用户/视频特征 + 注意力增强后的特征 + 加权后的评论特征
        embed_x = torch.cat([query, attn_output, weighted_comment_embeds], dim=1)
        embed_x = embed_x.contiguous().view(-1, self.embed_output_dim)

        # Cross Network 和 MLP 处理
        x_l1 = self.cn(embed_x)
        h_l2 = self.mlp(embed_x)
        x_stack = torch.cat([x_l1, h_l2], dim=1)
        p = self.linear(x_stack).squeeze(1)  # 原始的预测输出

        comment_scores = self.comment_score_linear(embed_x)  # 输出6个评论的打分
        comment_scores_ = self.comment_score_linear_(embed_x)  # 输出6个评论的打分

        # 通过 softmax 归一化为概率分布
        comment_probs = self.softmax(comment_scores)
        comment_probs_ = self.softmax(comment_scores_)
        self.comment_probs = comment_probs
        self.comment_probs_ = comment_probs_

        return p

    def get_comment_probs(self):
        return self.comment_probs
    
    def get_comment_probs_(self):
        return self.comment_probs_