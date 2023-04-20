import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np
import argparse
import yaml


from image_gpt import ImageGPT
from data import dataloaders


def train(args):

    with open(args.config, "rb") as f:
        config = yaml.safe_load(f)

    # experiment name
    name = f"{config['name']}_{args.dataset}"

    # 如果有预训练权重模型
    if args.pretrained is not None:
        print("useing pre-train model")
        model = ImageGPT.load_from_checkpoint(args.pretrained)
        # potentially modify model for finetuning
        model.learning_rate = config["learning_rate"]
        model.classify = config["classify"]
    else: # 没有预训练权重就重新搭建模型开始从头训练
        print("building new model")
        model = ImageGPT(centroids=args.centroids, **config)
    # 训练-验证-测试数据生成
    train_dl, valid_dl, test_dl = dataloaders(args.dataset, config["batch_size"])
    # 训练日志工具
    logger = pl_loggers.TensorBoardLogger("logs", name=name) # 使用Tensorboard来进行训练日志记录
    # 如果进行分类任务
    if config["classify"]:
        # classification
        # stop early for best validation accuracy for finetuning
        # 尽早停止以获得最佳验证精度以进行微调
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_acc", patience=3, mode="max"
        )

        checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_acc") # 模型检测点保存
        # monitor - - 需要监控的量，string类型。例如'val_loss'

        trainer = pl.Trainer(
            max_steps=config["steps"],
            gpus=config["gpus"],
            accumulate_grad_batches=config.get("accumulate_grad_batches", 1), # # 梯度累加的含义为：每累计k个step的梯度之后，进行一次参数的更新
            precision=config.get("precision", 32), # 训练精度
            early_stop_callback=early_stopping, # 监控validation_step()中某一个量，如果其不能再变得更优，则提前停止训练
            checkpoint_callback=checkpoint,
            logger=logger,
        )

    else:
        # pretraining
        checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss")
        trainer = pl.Trainer(
            max_steps=config["steps"], # 最大网络权重更新次数
            gpus=config["gpus"],
            precision=config["precision"],
            accumulate_grad_batches=config["accumulate_grad_batches"], # 梯度累加的含义为：每累计k个step的梯度之后，进行一次参数的更新
            checkpoint_callback=checkpoint,
            logger=logger,
        )
    # 训练
    trainer.fit(model, train_dl, valid_dl)



def test(args):
    with open(args.config, "rb") as f:
        config = yaml.safe_load(f)
    trainer = pl.Trainer(gpus=config["gpus"])
    _, _, test_dl = dataloaders(args.dataset, config["batch_size"])
    model = ImageGPT.load_from_checkpoint(args.checkpoint)
    if config["classify"]:
        model.hparams.classify = True
    trainer.test(model, test_dataloaders=test_dl)
    # 如果没有预训练模型，就会使用自己搭建的模型，就会进入image_gpt.py


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="mnist")

    subparsers = parser.add_subparsers()
    # 在使用Trainer时，还是推荐将其放在main() 函数中然后进行调用，这样会避免调用多线程时出现问题
    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--pretrained", type=str, default=None)
    parser_train.add_argument("config", type=str)
    parser_train.set_defaults(func=train)

    parser_test = subparsers.add_parser("test")
    parser_test.add_argument("checkpoint", type=str)
    parser_test.add_argument("config", type=str)
    parser_test.set_defaults(func=test)

    args = parser.parse_args()
    # 图像聚类分析的质心结果
    args.centroids = f"data/{args.dataset}_centroids.npy"


    args.func(args)


# Trainer可以接受的参数可以直接使用Trainer.add_argparse_args来添加，免去手动去写一条条的argparse