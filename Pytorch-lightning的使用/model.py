import pytorch_lightning as pl
import os
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

"""
https://zhuanlan.zhihu.com/p/370185203
Pytorch-lightning(以下简称pl)可以非常简洁得构建深度学习代码。但是其实大部分人用不到很多复杂得功能。
而pl有时候包装得过于深了，用的时候稍微有一些不灵活。通常来说，在你的模型搭建好之后，
大部分的功能都会被封装在一个叫trainer的类里面。一些比较麻烦但是需要的功能通常如下：

    1保存checkpoints
    2输出log信息
    3resume training 即重载训练，我们希望可以接着上一次的epoch继续训练
    4记录模型训练的过程(通常使用tensorboard)
    5设置seed，即保证训练过程可以复制
    
    好在这些功能在pl中都已经实现。
"""





"""
模型 - 继承pytorch_lightning.LightningModule
 class MyModule(pytorch_lightning.LightningModule): 
    def __init__(self, ): 
        super().__init__() 
        self.model = Model() 
        #... 
    def forward(self, inputs): 
        return self.model(inputs) 
    def configure_optimizers(self): # -> optimizer 
        pass 
    def training_step(self, batch, batch_idx): # -> train loss 
        pass 
完成这两部分之后就可以使用Trainer自动地完成最基本的训练

model = MyModule() 
data = MyData() 
trainer = Trainer() 
trainer.fit(model, data)  # 
调用fit函数时，训练集、验证集、测试集的选择，梯度的开关闭，设备的选择全部会自动进行


trainer的底层逻辑如下， 与平常使用Pytorch时的一致

trainer.fit()就会执行下面的底下实现底层逻辑
model.train()
torch.set_grad_enabled(True)
losses = []
for batch in train_dataloader:
    loss = training_step(batch) # 自定义模型类里面的继承父类的training_step()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss)

"""















class CoolSystem(pl.LightningModule):

    def __init__(self, hparams):
        super(CoolSystem, self).__init__()

        self.params = hparams

        self.data_dir = self.params.data_dir
        self.num_classes = self.params.num_classes

        ########## define the model ##########
        arch = torchvision.models.resnet18(pretrained=True)
        num_ftrs = arch.fc.in_features

        modules = list(arch.children())[:-1]  # ResNet18 has 10 children
        self.backbone = torch.nn.Sequential(*modules)  # [bs, 512, 1, 1]
        self.final = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, self.num_classes),
            torch.nn.Softmax(dim=1))

    def forward(self, x):
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1)
        x = self.final(x)

        return x

    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.SGD([
            {'params': self.backbone.parameters()},
            {'params': self.final.parameters(), 'lr': 1e-2}
        ], lr=1e-3, momentum=0.9)

        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        return [optimizer], [exp_lr_scheduler]

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)

        _, preds = torch.max(y_hat, dim=1)
        acc = torch.sum(preds == y.data) / (y.shape[0] * 1.0)

        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return {'loss': loss, 'train_acc': acc}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        _, preds = torch.max(y_hat, 1)
        acc = torch.sum(preds == y.data) / (y.shape[0] * 1.0)

        self.log('val_loss', loss)
        self.log('val_acc', acc)

        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        _, preds = torch.max(y_hat, 1)
        acc = torch.sum(preds == y.data) / (y.shape[0] * 1.0)

        return {'test_loss': loss, 'test_acc': acc}

    # 训练数据加载器， 必须的
    def train_dataloader(self):
        # REQUIRED

        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_set = torchvision.datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

        return train_loader
    # 验证数据加载器
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_set = torchvision.datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True, num_workers=4)

        return val_loader
    # 测试数据记载器
    def test_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_set = torchvision.datasets.ImageFolder(os.path.join(self.data_dir, 'val'), transform)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=8, shuffle=True, num_workers=4)

        return val_loader


def main(hparams):
    model = CoolSystem(hparams)

    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        gpus=1,
        accelerator='dp'
    )

    trainer.fit(model) # 调用fit函数时，训练集、验证集、测试集的选择，梯度的开关闭，设备的选择全部会自动进行