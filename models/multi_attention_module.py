import torch
import torch.nn.functional as F
from torch import nn


class GroupModule(nn.Module):
    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        hidden_channels = in_channels if hidden_channels is None else in_channels
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_channels, in_channels, bias=False),
            nn.Sigmoid(),
        )
        self.pool2 = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        b, c, h, w = x.size()

        # (b, c, h, w) -> (b, c, 1, 1) -> (b, c)
        d = self.pool1(x).contiguous().view(b, c)
        # (b, c) -> (b, c, 1, 1)
        d = self.linear(d).unsqueeze(-1).unsqueeze(-1)

        # (b, c, h, w) * (b, c, 1, 1) -> (b, h, w)
        M = torch.sum(x * d, dim=1)
        # (b, h, w) -> (b, h*w) -> (b, h, w)
        M = F.normalize(M.contiguous().view(b, -1), dim=-1, p=2).view(b, h, w)

        # (b, c, h, w) * (b, 1, h, w) -> (b, c, h, w) -> (b, h, w)
        P = torch.sum(x * M.unsqueeze(1), dim=1)

        # (b, h, w), (b, h, w)
        return P, M
    

class MultiAttentionModule(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, n_groups=4):
        super().__init__()
        self.group_modules = nn.ModuleList([GroupModule(in_channels, hidden_channels) for _ in range(n_groups)])
    
    def forward(self, x):
        """
        : param x tensor<b, c, h, w>
        """
        P, M = [], []
        for group_module in self.group_modules:
            P_, M_ = group_module(x)
            P.append(P_)
            M.append(M_)
        
        # [(b, h, w), ...] -> (b, c, h, w)
        P = torch.stack(P, dim=1)
        M = torch.stack(M, dim=1)
        
        return P, M


class MultiAttentionLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, M):
        """
        : param x tensor<b, c, h, w>
        """
        dis_loss = self.cal_dis_loss(M, reduction=self.reduction)
        div_loss = self.cal_div_loss(M, reduction=self.reduction)
        return dis_loss, div_loss
    
    def cal_dis_loss(self, M, reduction='mean'):
        """
        : param M tensor<b, c, h, w>
        """
        def get_max_index(M):
            b, h, w = M.size()
            M_flatten = M.view(b, h*w)
            _, ind = M_flatten.max(1)
            indy = torch.div(ind, w, rounding_mode='floor')
            indx = ind % w
            # (2, b)
            return torch.stack([indy, indx])
        
        b, c, h, w = M.size()
        # (b*c, h, w)
        M = M.view(b*c, h, w)
        # (b*c,), (b*c,)
        y_max, x_max = get_max_index(M)
        # (h, w), (h, w)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        y = y.to(M.device)
        x = x.to(M.device)
        
        # (b*c, 1, 1) * (1, h, w) -> (b*c, h, w)
        dist_square_y = torch.pow(y_max.unsqueeze(-1).unsqueeze(-1) - y.unsqueeze(0), 2)
        dist_square_x = torch.pow(x_max.unsqueeze(-1).unsqueeze(-1) - x.unsqueeze(0), 2)
        dist_square = dist_square_x + dist_square_y
        
        loss = dist_square * M
        
        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
    
    def cal_div_loss(self, M, reduction='mean'):
        """
        : param M tensor<b, c, h, w>
        """
        loss = []
        for c in range(M.size(1)):
            # (b, h, w)
            M_this = M[:, c, :, :]
            M_other = M.clone()
            # 排除当前通道
            M_other[:, c, :, :] = -999999
            # (b, c, h, w) -> (b, h, w)
            M_max = torch.max(M_other, dim=1).values
            loss.append(M_max * M_this)
        
        # [(b, h, w), ...] -> (b, c, h, w)
        loss = torch.stack(loss, dim=1)
        
        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
        

if __name__ == '__main__':
    embed = MultiAttentionModule(64)
    ma_loss = MultiAttentionLoss()
    feat = torch.rand(1, 64, 16, 16)
    P, M = embed(feat)
    print('P, M:')
    print(P.size())
    print(M.size())
    
    print(ma_loss(M))
    