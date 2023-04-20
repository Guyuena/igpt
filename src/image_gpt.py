import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
import numpy as np
import math
from argparse import ArgumentParser

from gpt2 import GPT2
from utils import quantize


def _to_sequence(x):
    """shape batch of images for input into GPT2 model"""
    x = x.view(x.shape[0], -1)  # flatten images into sequences
    x = x.transpose(0, 1).contiguous()  # to shape [seq len, batch]
    return x


class ImageGPT(pl.LightningModule):
    def __init__(
        self,
        centroids,
        embed_dim=16,
        num_heads=2,
        num_layers=8,
        num_pixels=28,
        num_vocab=16,
        num_classes=10,
        classify=False,
        learning_rate=3e-3,
        steps=10_000,
        warmup_steps=500,
        **kwargs,
    ):
        super(ImageGPT, self).__init__()
        self.save_hyperparameters()
        # IGPT就是在GPT2的基础上来的
        self.gpt = GPT2(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_positions=num_pixels * num_pixels,
            num_vocab=num_vocab,
            num_classes=num_classes,
        )
        # centroids：质心
        # 创建这个参数，要从聚类处理后中得到
        # 先对图像数据进行聚类处理，然后再获得这个centroids参数  computer_centroids.py聚类处理脚本
        # torch.nn.Parameter()：将一个不可训练的tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面
        # 聚类中心
        self.centroids = nn.Parameter(torch.from_numpy(np.load(centroids)), requires_grad=False)
        # criterion： 判据，标准
        self.criterion = nn.CrossEntropyLoss()
        self.classify = classify
        self.learning_rate = learning_rate
        self.steps = steps
        self.warmup_steps = warmup_steps
        print(self.classify)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--embed_dim", type=int, default=16)
        parser.add_argument("--num_heads", type=int, default=2)
        parser.add_argument("--num_layers", type=int, default=8)
        parser.add_argument("--num_pixels", type=int, default=28)
        parser.add_argument("--num_vocab", type=int, default=16)
        parser.add_argument("--num_classes", type=int, default=10)
        parser.add_argument("--classify", action="store_true", default=False)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--learning_rate", type=float, default=3e-3)
        parser.add_argument("--steps", type=int, default=25_000)
        return parser

    # 重写configure_optimizers()函数即可
    # 设置优化器
    # 只要在training_step()函数中返回了loss，就会自动反向传播，
    # 并自动调用loss.backward()和optimizer.step()和stepLR.step()了
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.gpt.parameters(), lr=self.learning_rate)

        # no learning rate schedule for fine-tuning
        # 意思就是说如果加入了分类任务时，预训练的时候的学习率就是定的，不是变的
        if self.classify:
            return optimizer

        # 这里设置warmup_steps后学习率为动态变化的
        scheduler = {
            "scheduler": LambdaLR(
                optimizer, learning_rate_schedule(self.warmup_steps, self.steps)
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]




    def forward(self, x):
        return self.gpt(x)

    # 训练主要是重写def training_setp(self, batch, batch_idx)函数，并返回要反向传播的loss即可，其中batch
    # 即为从 train_dataloader 采样的一个batch的数据，batch_idx即为目前batch的索引
    def training_step(self, batch, batch_idx):
        x, y = batch
        print("x.shape, y.shape ",x.shape, y.shap)
        # 图像先量化
        # 计算X和聚类中心的距离
        # 比如输入 x=[3,5,7]  聚类中心 centroids=[1,2,3]
        # 每个 x 都要和聚类中心计算量化距离，
        x = quantize(x, self.centroids)


        # 再序列化
        x = _to_sequence(x)

        if self.classify:
            clf_logits, logits = self.gpt(x, classify=True)
            "交叉熵损失"
            clf_loss = self.criterion(clf_logits, y) # 计算交叉熵损失
            gen_loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1)) # 计算交叉熵损失
            # joint loss for classification
            loss = clf_loss + gen_loss  # 总损失等于生成损失加分类损失
        else:
            logits = self.gpt(x)
            loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1))

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    # 重写def validation_step(self, batch, batch_idx) 函数，
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # 量化处理
        # 模型压缩之聚类量化  https://zhuanlan.zhihu.com/p/466620131
        # iGPT论文中讲了，大尺寸的图片数据量会很大，所以当时openAI对ImageNet数据集做聚类，压缩到512内
        x = quantize(x, self.centroids)
        # 若自己的数据集没必要进行聚类，那就没有聚类中心，就不需要上面的量化，直接输入就行
        x = _to_sequence(x)

        if self.classify or self.hparams.classify:
            clf_logits, logits = self.gpt(x, classify=True)
            clf_loss = self.criterion(clf_logits, y)
            gen_loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1))
            # joint loss for classification
            loss = clf_loss + gen_loss
            _, preds = torch.max(clf_logits, 1)
            correct = preds == y
            return {"val_loss": loss, "correct": correct}
        else:
            logits = self.gpt(x)
            loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1))
            return {"val_loss": loss}



    # validation_epoch_end - - 即每一个 validation 的epoch完成之后会自动调用
    def validation_epoch_end(self, outs):
        avg_loss = torch.stack([x["val_loss"] for x in outs]).mean()
        logs = {"val_loss": avg_loss}
        if self.classify or self.hparams.classify:
            correct = torch.cat([x["correct"] for x in outs])
            logs["val_acc"] = correct.sum().float() / correct.shape[0]
        return {"val_loss": avg_loss, "log": logs}

    # 在pytoch_lightning框架中，test 在训练过程中是不调用的，也就是说是不相关，在训练过程中只进行training和validation，
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    # test_epoch_end - - 即每一个 test 的epoch完成之后会自动调用
    def test_epoch_end(self, outs):
        result = self.validation_epoch_end(outs)

        # replace valid stats with test stats becuase we are reusing function
        result["log"]["test_loss"] = result["log"].pop("val_loss")
        result["test_loss"] = result.pop("val_loss")
        if self.hparams.classify:
            result["log"]["test_acc"] = result["log"].pop("val_acc")
        return result


def learning_rate_schedule(warmup_steps, total_steps):
    """Linear warmup for warmup_steps, with cosine annealing to 0 at total_steps"""

    def learning_rate_fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return learning_rate_fn
