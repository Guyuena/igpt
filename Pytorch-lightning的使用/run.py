# from argparse import Namespace
# from model import main
# args = {
#     'num_classes': 2,
#     'epochs': 5,
#     'data_dir': "/content/hymenoptera_data",
# }
#
# hyperparams = Namespace(**args)


if __name__ == '__main__':
    # main(hyperparams)

    default= None
    if default is  None:
        print("yes")



# 如果希望重载训练的话，可以按如下方式：

# resume training

# RESUME = True
#
# if RESUME:
#     resume_checkpoint_dir = './lightning_logs/version_0/checkpoints/'
#     checkpoint_path = os.listdir(resume_checkpoint_dir)[0]
#     resume_checkpoint_path = resume_checkpoint_dir + checkpoint_path
#
#     args = {
#         'num_classes': 2,
#         'data_dir': "/content/hymenoptera_data"}
#
#     hparams = Namespace(**args)
#
#     model = CoolSystem(hparams)
#
#     trainer = pl.Trainer(gpus=1,
#                          max_epochs=10,
#                          accelerator='dp',
#                          resume_from_checkpoint=resume_checkpoint_path)
#
#     trainer.fit(model)