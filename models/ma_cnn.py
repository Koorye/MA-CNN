import torch
from torch import nn
from torchvision.models import resnet18

from models.multi_attention_module import MultiAttentionModule, MultiAttentionLoss


class MACNN(nn.Module):
    def __init__(self, n_clses, n_groups=16):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet18(pretrained=True).children())[:-5])
        self.multi_attention_module = MultiAttentionModule(64, n_groups=n_groups)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(64 + n_groups, n_clses)

        self.loss_cls = nn.CrossEntropyLoss()
        self.ma_loss = MultiAttentionLoss()

        print(self)
    
    def forward(self, x, gt_score):
        # (b, 3, h0, w0) -> (b, 512, h, w)
        feat = self.backbone(x)
        # (b, 512, h, w) -> (b, group, h, w)
        group_feat, M = self.multi_attention_module(feat)
        # (b, 512, h, w) | (b, group, h, w) -> (b, 512 + group, h, w)
        feat = torch.cat([feat, group_feat], dim=1)
        # (b, 512 + group, h, w) -> (b, 512 + group, 1, 1) -> (b, 512 + group)
        feat = self.pool(feat).squeeze(-1).squeeze(-1)
        # (b, 512 + group) -> (b, n_clses)
        logit = self.linear(feat)

        if self.training:
            loss_cls = self.loss_cls(logit, gt_score)
            loss_dis, loss_div = self.ma_loss(M)
            return dict(
                loss_cls=loss_cls,
                loss_dis=loss_dis,
                loss_div=loss_div,
            )
    
        return logit.argmax(1)


if __name__ == '__main__':
    ma_cnn = MACNN(10)
    x = torch.rand(16, 3, 64, 64)
    gt_score = torch.ones(16).long()
    loss_dict = ma_cnn(x, gt_score)
    print(loss_dict)