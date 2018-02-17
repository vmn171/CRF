import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self, vocab_size, embed_dim, h_dim, tag_size):
        """GRU网络，通过输入句子，生成句子特征表达
        Arguments:
            vocab_size：字典长度
            embed_dim：嵌入长度
            h_dim：隐藏层维度
            tag_size：标签种类
        """
        super(Net, self).__init__()
        self.h_dim = h_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, h_dim, num_layers=1,
                          bidirectional=True, batch_first=True)
        self.h2tag = nn.Linear(h_dim*2, tag_size)  # 因为是双向的，输出*2
        self.crfs = CRFs(tag_size)

    def forward(self, x, tags):
        n = x.size(0)
        x = self.embed(x)  # Input: LongTensor (N, W) Output: (N, W, embed_dim)
        h0 = Variable(x.data.new(2, n, self.h_dim).fill_(0))
        feats, _ = self.gru(x, h0)  # input (seq_len, batch, input_size)
        feats = self.h2tag(feats)
        out = self.crfs(feats, tags)
        return out

    def viterbi(self, x):
        """
        维特比算法
        """
        n = x.size(0)
        x = self.embed(x)  # Input: LongTensor (N, W) Output: (N, W, embed_dim)
        h0 = Variable(x.data.new(2, n, self.h_dim).fill_(0))
        feats, _ = self.gru(x, h0)  # input (seq_len, batch, input_size)
        feats = self.h2tag(feats)
        scores, path = self.crfs.decode(feats)
        return scores,path


class CRFs(nn.Module):
    def __init__(self, tag_size):
        """CRFs层
        Arguments:
            tag_size：标签种类
        """
        super(CRFs, self).__init__()
        self.tag_size = tag_size
        # 转移矩阵，因为增加了<start>标签，所以加1
        self.trans = nn.Parameter(torch.randn(tag_size + 1, tag_size))

    def forward(self, feats, tags):
        """
        forward是金标准引导训练模式
        feats: batch, seq, dim # 传入RNN的特征
        tags: batch, seq # 正确的标签
        """
        seq_len = tags.size(1)
        n = tags.size(0)
        start_idx = self.tag_size + 1  # <start>的标签是最大标签加1

        t = self.trans  # t是转移矩阵
        t = t.unsqueeze(0)
        t = t.expand(n * seq_len, -1, -1)

        # tags需要处理，因为转移矩阵的设计，第一位是<start>，最后一位不要
        pad = Variable(tags.data.new(n, 1).fill_(start_idx))
        tags = torch.cat((pad, tags), 1)
        tags = tags[:, :-1]
        tags = tags.view(-1)

        t = t[np.arange(feats.size(0)), tags]
        t = t.view(n, seq_len, -1)
        return feats + t

    def decode(self, feats, **kwargs):  #
        """ 
        维特比算法进行推断,暂时只允许一个批次进入
        Arguments:
            feats:RNN输出的特征
        Returns:
            [type] -- [description]
        """

        seq_len = feats.size(1)
        scores = []  # 记录总的分数
        point_list = []  # 记录指征
        start_trans_score = self.trans[-1]  # start开始的转移分数
        s0 = start_trans_score + feats[0, 0, :]  # 第一个位置的总分
        scores.append(s0.view(1, -1))
        for i in range(1, seq_len):
            s = scores[i-1].view(1, -1)  # 取出上一步的分数
            h = feats[0, i, :].view(-1, 1)  # 取得这一步feats的分数
            max_s, max_ix = (s + h).max(1)  # 相加后取最大的分数，并获得路径的指针
            scores.append(max_s.view(1, -1))  # 添加分数到列表
            point_list.append(max_ix)  # 添加指征
        scores = torch.cat(scores)

        # 计算路径
        path = []
        _, point = scores[-1].view(1, -1).max(1)  # 从最后一个分数中获得最大索引
        path.append(point.view(-1, 1))  # 加入路径
        for i in reversed(range(seq_len-1)):
            point = point_list[i][point]  # 用索引从poin_list中查到前一个索引
            path.insert(0, point.view(-1, 1))
        path = torch.cat(path, dim=1)
        return scores, path
