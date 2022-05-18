import random
import torch
from collections import defaultdict
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from tqdm import tqdm

from models.ma_cnn import MACNN

seed = 999
n_classes = 10
n_epoches = 20
batch_size = 16
historical_weights = None

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

loss_weight = dict(
    loss_cls=1.,
    loss_dis=.5,
    loss_div=.5,
)

random.seed(seed)
torch.manual_seed(seed)

device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f'Use {device}')

trans = T.Compose([
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([.485, .456, .406], [.229, .224, .225]),
])
train_set = CIFAR10(root='/root/Datasets', train=True, download=True, transform=trans)
val_set = CIFAR10(root='/root/Datasets', train=False, download=True, transform=trans)

train_loader = DataLoader(train_set, batch_size=batch_size,
                          shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=batch_size,
                        shuffle=False, num_workers=4)

model = MACNN(len(classes)).to(device)

if historical_weights is not None:
    model.load_state_dict(torch.load(f'weights/{historical_weights}'))
    historical_epoch = int(historical_weights.split('_')[4][5:]) + 1
else:
    historical_epoch = 0

optim = Adam(model.parameters(), lr=1e-3, betas=(.5, .999))

for epoch in range(historical_epoch, n_epoches):
    model.train()
    loss_dict = defaultdict(float)

    for step, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        loss = model(data, target)
        loss_total = torch.stack([loss[key] * loss_weight[key] for key in loss]).sum()

        optim.zero_grad()
        loss_total.backward()
        optim.step()

        for key in loss:
            loss_dict[key] += loss[key].item()
        
        if step != 0 and step % 50 == 0:
            print(f'epoch: {epoch}, step: {step}/{len(train_loader)}, ', end='', flush=True)
            for idx, key in enumerate(loss_dict):
                print(f'{key}: {loss_dict[key] / 50}', end='', flush=True)
                loss_dict[key] = 0.
                if idx != len(loss_dict) - 1:
                    print(', ', end='', flush=True)
            print('', flush=True)
                        
    model.eval()
    n_correct = 0

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc=f'Epoch{epoch} Val'):
            data = data.to(device)
            target = target.to(device)

            scores = model(data, target)
            n_correct += torch.eq(scores, target).sum().item()

        acc = n_correct / len(val_set)
        print(f'epoch: {epoch}, acc: {acc}', flush=True)

    torch.save(model.state_dict(
    ), f'weights/epoch{epoch}_acc{acc:.2%}.pth')
