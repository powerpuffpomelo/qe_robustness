# 画图，展示训练过程
import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib
import os

matplotlib.use('Agg')


# 读取loss记录
loss_dict_prefix_1 = "/home/data_ti4_c/yanym/save/enzh/discriminator_pretrained_256_log_v3/"
loss_dict_prefix_2 = "/home/data_ti4_c/yanym/save/enzh/discriminator_finetuned_256_log_v3_3360_sent/"
loss_dict_prefix_3 = "/home/yanym/save/enzh/discriminator_finetuned_256_log_v3_3360_sent_dense_1200/"

plot_prefix = "/home/yanym/save/enzh/"
loss_dict_path_1 = loss_dict_prefix_1 + "loss_dict.json"
loss_dict_path_2 = loss_dict_prefix_2 + "loss_dict.json"
loss_dict_path_3 = loss_dict_prefix_3 + "loss_dict.json"
with open(loss_dict_path_1, 'r', encoding='utf-8') as f1, open(loss_dict_path_2, 'r', encoding='utf-8') as f2, \
    open(loss_dict_path_3, 'r', encoding='utf-8') as f3:
    loss_dict_1 = json.load(f1)
    loss_dict_2 = json.load(f2)
    loss_dict_3 = json.load(f3)


# 保存路径
plot_dir = plot_prefix + "loss_plot/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir, exist_ok=True)


# eval 画图
def plot_eval_2(plot_path, key):
    items = ['pretrain_train_eval', 'pretrain_val_eval', 'finetune_train_eval', 'finetune_val_eval']
    labels = ['pseudo_train_set', 'pseudo_valid_set', 'qe_train_set', 'qe_valid_set']
    records = ['loss_word', 'loss_hter', 'f1_ok', 'f1_bad', 'f1_multi', 'pearsonr']

    x_iters = np.arange(len(loss_dict_1['pretrain_train_eval']['loss_word'][:42]) + len(loss_dict_2['pretrain_train_eval']['loss_word']))

    fig, ax = plt.subplots(3, 2, figsize=(12,18))
    axes = ax.flatten()

    for i, record in enumerate(records):
        for item,label in zip(items,labels):
            axes[i].plot(x_iters, loss_dict_1[item][record][:42] + loss_dict_2[item][record], label=label)
        axes[i].set_title(record, fontsize=16)
        axes[i].axvline(x = len(loss_dict_1[items[0]][record][:41]), color = 'grey', linestyle=':')
        axes[i].axvline(x = len(loss_dict_1[items[0]][record][:42]), color = 'grey', linestyle=':')
        axes[i].set_xlabel("val_times", fontsize=14)
        axes[i].set_ylabel(record, fontsize=14)
        axes[i].legend()

    plt.savefig(plot_path)
    

def plot_eval_3(plot_path, key):
    items = ['pretrain_train_eval', 'pretrain_val_eval', 'finetune_train_eval', 'finetune_val_eval']
    labels = ['pseudo_train_set', 'pseudo_valid_set', 'qe_train_set', 'qe_valid_set']
    records = ['loss_word', 'loss_hter', 'f1_ok', 'f1_bad', 'f1_multi', 'pearsonr']

    x_iters = np.arange(len(loss_dict_1['pretrain_train_eval']['loss_word'][:42]) + len(loss_dict_2['pretrain_train_eval']['loss_word'])
        + len(loss_dict_3['pretrain_train_eval']['loss_word']))

    fig, ax = plt.subplots(3, 2, figsize=(20,12))
    axes = ax.flatten()

    for i, record in enumerate(records):
        for item,label in zip(items,labels):
            axes[i].plot(x_iters, loss_dict_1[item][record][:42] + loss_dict_3[item][record] + loss_dict_2[item][record], label=label)
        axes[i].set_title(record, fontsize=16)
        axes[i].axvline(x = len(loss_dict_1[items[0]][record][:41]), color = 'grey', linestyle=':')
        #axes[i].axvline(x = len(loss_dict_1[items[0]][record][:42]), color = 'grey', linestyle=':')
        axes[i].axvline(x = len(loss_dict_1[items[0]][record][:41] + loss_dict_3[item][record]), color = 'grey', linestyle=':')
        #axes[i].set_xlabel("val_times", fontsize=14)
        axes[i].set_ylabel(record, fontsize=14)
        axes[i].legend()

    plt.savefig(plot_path)


key = 'cat'
eval_plot_path = plot_dir + key + "_eval.png"

# plot_eval_2(eval_plot_path, key)

plot_eval_3(eval_plot_path, key)



# python scripts/train_directqe_enzh/analysis/plot_loss_cat.py