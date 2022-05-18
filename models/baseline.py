import torch
from torch import nn
from torchvision.models import resnet18


class MACNN(nn.Module):
    def __init__(self, n_clses):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet18(pretrained=True).children())[:-5])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(64, n_clses)

        self.loss_cls = nn.CrossEntropyLoss()
        print(self)
    
    def forward(self, x, gt_score):
        # (b, 3, h0, w0) -> (b, 512, h, w)
        feat = self.backbone(x)
        # (b, 512, h, w) -> (b, 512, 1, 1) -> (b, 512)
        feat = self.pool(feat).squeeze(-1).squeeze(-1)
        # (b, 512) -> (b, n_clses)
        logit = self.linear(feat)

        if self.training:
            loss_cls = self.loss_cls(logit, gt_score)
            return dict(loss_cls=loss_cls)
    
        return logit.argmax(1)
