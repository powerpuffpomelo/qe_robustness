# 画图，展示训练过程
import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib
import os

matplotlib.use('Agg')



# 读取loss记录
#loss_dict_path = "/home/data_ti4_c/yanym/save/enzh/discriminator_pretrained_256_log_v2/loss_dict.json"
loss_dict_prefix = "/home/yanym/save/enzh/discriminator_finetuned_256_log_ckpt_end_end/"
plot_prefix = "/home/yanym/save/enzh/discriminator_finetuned_256_log_ckpt_end_end/"
loss_dict_path = loss_dict_prefix + "loss_dict.json"
with open(loss_dict_path, 'r', encoding='utf-8') as f:
    loss_dict = json.load(f)


# 保存路径
plot_dir = plot_prefix + "loss_plot/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir, exist_ok=True)


# train_loss画图
def plot_train_loss(plot_path, key):
    train_loss_word = loss_dict[ key + '_train_loss']['loss_word']
    train_loss_hter = loss_dict[ key + '_train_loss']['loss_hter']

    x_iters = np.arange(len(train_loss_word))  # 横坐标

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    axes = ax.flatten()

    axes[0].plot(x_iters, train_loss_word, label='train_loss_word')
    axes[0].set_title(key + "_train_loss_word", fontsize=16)
    axes[0].set_xlabel("iters", fontsize=14)
    axes[0].set_ylabel("loss", fontsize=14)
    axes[0].legend()

    axes[1].plot(x_iters, train_loss_hter, label='train_loss_hter')
    axes[1].set_title(key + "_train_loss_hter", fontsize=16)
    axes[1].set_xlabel("iters", fontsize=14)
    axes[1].set_ylabel("loss", fontsize=14)
    axes[1].legend()

    plt.savefig(plot_path)



# eval 画图
def plot_eval(plot_path, key):
    items = ['pretrain_train_eval', 'pretrain_val_eval', 'finetune_train_eval', 'finetune_val_eval']
    labels = ['pseudo_train_set', 'pseudo_valid_set', 'qe_train_set', 'qe_valid_set']
    records = ['loss_word', 'loss_hter', 'f1_ok', 'f1_bad', 'f1_multi', 'pearsonr']

    x_iters = np.arange(len(loss_dict['pretrain_train_eval']['loss_word']))

    fig, ax = plt.subplots(3, 2, figsize=(12,18))
    axes = ax.flatten()

    for i, record in enumerate(records):
        for item,label in zip(items,labels):
            axes[i].plot(x_iters, loss_dict[item][record], label=label)
        axes[i].set_title(record, fontsize=16)
        axes[i].set_xlabel("val_times", fontsize=14)
        axes[i].set_ylabel(record, fontsize=14)
        axes[i].legend()


    plt.savefig(plot_path)
    


if 'finetune' in loss_dict_path:
    key = 'finetune'
elif 'pretrain' in loss_dict_path:
    key = 'pretrain'
train_loss_plot_path = plot_dir + key + "_loss.png"
eval_plot_path = plot_dir + key + "_eval.png"

plot_train_loss(train_loss_plot_path, key)
plot_eval(eval_plot_path, key)





# python scripts/analysis/plot/plot_loss.py