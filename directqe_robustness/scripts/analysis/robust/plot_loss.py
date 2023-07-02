# 画图，展示训练过程
import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib
import os

matplotlib.use('Agg')


# 读取loss记录
loss_dict_prefix = "/data1/yanym/save/directqe_enzh_bpe_robust/imp_2finetune/discriminator_finetuned_256_imp_v1_c_mask1_word2vec/"
plot_prefix = "/data1/yanym/save/directqe_enzh_bpe_robust/imp_2finetune/discriminator_finetuned_256_imp_v1_c_mask1_word2vec/"
loss_dict_path = loss_dict_prefix + "loss_dict.json"
with open(loss_dict_path, 'r', encoding='utf-8') as f:
    loss_dict = json.load(f)


# 保存路径
plot_dir = plot_prefix + "loss_plot/"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

# eval 画图
def plot_eval(plot_path, key):
    items = ['normal_train_eval', 'val_eval']
    labels = ['normal_train_set', 'valid_set']
    records = ['loss_word', 'loss_hter', 'f1_ok', 'f1_bad', 'f1_multi', 'pearsonr']

    x_iters = np.arange(len(loss_dict['val_eval']['loss_word']))

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
    

eval_plot_path = plot_dir + "finetune_eval.png"

plot_eval(eval_plot_path, 'finetune')





# python scripts/analysis/robust/plot_loss.py