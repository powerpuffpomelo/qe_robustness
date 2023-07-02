import itertools
import os
import random
import time
import json
import copy
import faiss

import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn import metrics, manifold
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
from scipy.stats import pearsonr
from collections import Counter
#from transformers import BertModel, BertTokenizer

import src.distributed as dist
from src.data.data_iterator import DataIterator
from src.data.dataset import TextLineDataset, TextLineStrDataset, ZipDataset, QEHterDataset, QETagsDataset, QEAlignDataset
from src.data.vocabulary import Vocabulary
from src.decoding import beam_search
from src.models import build_model, load_predefined_configs
from src.models import LinearProbe
from src.modules.criterions import NMTCriterion, NMTCriterionFocalLoss, NMTCriterionMarginLoss
from src.optim import Optimizer
from src.optim.lr_scheduler import build_scheduler
from src.utils.common_utils import *
from src.utils.configs import pretty_configs, add_user_configs, default_base_configs
from src.utils.logging import *
from src.utils.moving_average import MovingAverage
from src.utils import faiss_utils
from src.utils.knn_utils import *


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True


def load_model_parameters(path, map_location="cpu"):
    state_dict = torch.load(path, map_location=map_location)

    if "model" in state_dict:
        return state_dict["model"]
    return state_dict


def prepare_configs(config_path: str, predefined_config: str = "") -> dict:
    """Prepare configuration file"""
    # 1. Default configurations
    default_configs = default_base_configs()

    # 2. [Optional] Load pre-defined configurations
    default_configs = load_predefined_configs(default_configs, predefined_config)

    # 3. Load user configs
    config_path = os.path.abspath(config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    configs = add_user_configs(default_configs, configs)

    return configs


def prepare_data(seqs_x, seqs_y, lm_ids=None, backward=False, cuda=False, batch_first=True):

    def _np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):

        batch_size = len(samples)

        sizes = [len(s) for s in samples]
        max_size = max(sizes)

        x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')

        for ii in range(batch_size):
            x_np[ii, :sizes[ii]] = samples[ii]

        if batch_first is False:
            x_np = np.transpose(x_np, [1, 0])

        x = torch.tensor(x_np)

        if cuda is True:
            x = x.cuda()
        return x

    seqs_x = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], seqs_x))
    x = _np_pad_batch_2D(samples=seqs_x, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    if seqs_y is None:
        return x

    # 若反向，那么需要把数据完全倒过来，不影响其他的部分。
    if backward is False:
        seqs_y = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], seqs_y))
    else:
        seqs_y = list(map(lambda s: [Constants.BOS] + s[::-1] + [Constants.EOS], seqs_y))
    y = _np_pad_batch_2D(seqs_y, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)
    if lm_ids is None:
        return x, y

    lm_ids = list(map(lambda s: [0] + s + [0], lm_ids))
    lm_ids = _np_pad_batch_2D(lm_ids, pad=Constants.PAD,
                              cuda=cuda, batch_first=batch_first)

    return x, y, lm_ids


def prepare_data_all(seqs_x, seqs_y, cuda=False, batch_first=True, wwm=False, seqs_y_str=None):

    def _np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):

        batch_size = len(samples)

        sizes = [len(s) for s in samples]
        max_size = max(sizes)

        x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')

        for ii in range(batch_size):
            x_np[ii, :sizes[ii]] = samples[ii]

        if batch_first is False:
            x_np = np.transpose(x_np, [1, 0])

        x = torch.tensor(x_np)

        if cuda is True:
            x = x.cuda()
        return x

    seqs_x = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], seqs_x))
    x = _np_pad_batch_2D(samples=seqs_x, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    if seqs_y is None:
        return x

    seqs_y_forward = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], seqs_y))
    y = _np_pad_batch_2D(seqs_y_forward, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    if not wwm:
        mask_seqs_y, output_labels = mask_target_data(seqs_y)  
    else:
        mask_seqs_y, mask_seqs_y_str, output_labels = mask_target_data_wwm(seqs_y, seqs_y_str, mask_prob = 0.7)
    
    mask_seqs_y = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], mask_seqs_y))
    y_mask = _np_pad_batch_2D(mask_seqs_y, pad=Constants.PAD,
                                cuda=cuda, batch_first=batch_first)

    output_labels = list(map(lambda s: [0] + s + [0], output_labels))
    lm_ids = _np_pad_batch_2D(output_labels, pad=0, cuda=cuda, batch_first=batch_first)

    seqs_y_backward = list(map(lambda s: [Constants.BOS] + s[::-1] + [Constants.EOS], seqs_y))
    y_reverse = _np_pad_batch_2D(seqs_y_backward, pad=Constants.PAD,
                                 cuda=cuda, batch_first=batch_first)

    return x, y, y_mask, y_reverse, lm_ids


def prepare_data_all_v2(seqs_x, seqs_y, cuda=False, batch_first=True, wwm=False, seqs_y_str=None):

    def _np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):

        batch_size = len(samples)

        sizes = [len(s) for s in samples]
        max_size = max(sizes)

        x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')

        for ii in range(batch_size):
            x_np[ii, :sizes[ii]] = samples[ii]

        if batch_first is False:
            x_np = np.transpose(x_np, [1, 0])

        x = torch.tensor(x_np)

        if cuda is True:
            x = x.cuda()
        return x

    seqs_x_v1 = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], seqs_x))
    x = _np_pad_batch_2D(samples=seqs_x_v1, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    seqs_x_v2 = list(map(lambda s: [Constants.BOS, Constants.HTER] + s + [Constants.EOS], seqs_x))
    x_v2 = _np_pad_batch_2D(samples=seqs_x_v2, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    if seqs_y is None:
        return x

    seqs_y_forward = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], seqs_y))
    y = _np_pad_batch_2D(seqs_y_forward, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    if not wwm:
        mask_seqs_y, output_labels = mask_target_data(seqs_y)  
    else:
        mask_seqs_y, mask_seqs_y_str, output_labels = mask_target_data_wwm(seqs_y, seqs_y_str, mask_prob = 0.7)
    
    mask_seqs_y = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], mask_seqs_y))
    y_mask = _np_pad_batch_2D(mask_seqs_y, pad=Constants.PAD,
                                cuda=cuda, batch_first=batch_first)

    output_labels = list(map(lambda s: [0] + s + [0], output_labels))
    lm_ids = _np_pad_batch_2D(output_labels, pad=0, cuda=cuda, batch_first=batch_first)

    seqs_y_backward = list(map(lambda s: [Constants.BOS] + s[::-1] + [Constants.EOS], seqs_y))
    y_reverse = _np_pad_batch_2D(seqs_y_backward, pad=Constants.PAD,
                                 cuda=cuda, batch_first=batch_first)

    return x, y, y_mask, y_reverse, lm_ids, x_v2


def prepare_data_all_robust(seqs_x, seqs_y, align_position, cuda=False, batch_first=True):

    def _np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):

        batch_size = len(samples)

        sizes = [len(s) for s in samples]
        max_size = max(sizes)

        x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')

        for ii in range(batch_size):
            x_np[ii, :sizes[ii]] = samples[ii]

        if batch_first is False:
            x_np = np.transpose(x_np, [1, 0])

        x = torch.tensor(x_np)

        if cuda is True:
            x = x.cuda()
        return x

    seqs_x_copy = [seqs_x[0][:]]

    x = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], seqs_x_copy))
    x = _np_pad_batch_2D(samples=x, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    y = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], seqs_y))
    y = _np_pad_batch_2D(samples=y, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    mask_seqs_x, output_labels_x = mask_target_data_idx(seqs_x_copy, align_position)
    x_mask = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], mask_seqs_x))
    x_mask = _np_pad_batch_2D(x_mask, pad=Constants.PAD,
                                cuda=cuda, batch_first=batch_first)

    output_labels_x = list(map(lambda s: [0] + s + [0], output_labels_x))
    lm_ids_x = _np_pad_batch_2D(output_labels_x, pad=0, cuda=cuda, batch_first=batch_first)

    return x, y, x_mask, lm_ids_x


def prepare_data_all_contrastive(seqs_x, seqs_y, cuda=False, batch_first=True):

    def _np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):

        batch_size = len(samples)

        sizes = [len(s) for s in samples]
        max_size = max(sizes)

        x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')

        for ii in range(batch_size):
            x_np[ii, :sizes[ii]] = samples[ii]

        if batch_first is False:
            x_np = np.transpose(x_np, [1, 0])

        x = torch.tensor(x_np)

        if cuda is True:
            x = x.cuda()
        return x

    seqs_x = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], seqs_x))
    x = _np_pad_batch_2D(samples=seqs_x, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    seqs_y_forward = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], seqs_y))
    y = _np_pad_batch_2D(seqs_y_forward, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    mask_seqs_x, output_labels_x = mask_target_data_contrastive(seqs_x)
    mask_seqs_x = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], mask_seqs_x))
    x_mask = _np_pad_batch_2D(mask_seqs_x, pad=Constants.PAD,
                              cuda=cuda, batch_first=batch_first)

    output_labels_x = list(map(lambda s: [0] + s + [0], output_labels_x))
    lm_ids_x = _np_pad_batch_2D(output_labels_x, pad=0, cuda=cuda, batch_first=batch_first)

    return x, y, x_mask, lm_ids_x


def prepare_data_fake(seqs_y, ori_seqs_y, backward=False, cuda=False, batch_first=True):

    def _np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):

        batch_size = len(samples)

        sizes = [len(s) for s in samples]
        max_size = max(sizes)

        x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')

        for ii in range(batch_size):
            x_np[ii, :sizes[ii]] = samples[ii]

        if batch_first is False:
            x_np = np.transpose(x_np, [1, 0])

        x = torch.tensor(x_np)

        if cuda is True:
            x = x.cuda()
        return x
    # seqs_y 和 origin_seqs_y不一样长

    # 先反向一下
    if backward is True:
        new_ori_seqs_y = []
        for each in ori_seqs_y:
            new_ori_seqs_y.append(each[::-1])
        ori_seqs_y = new_ori_seqs_y

    lm_ids = []
    for sentence, ori_sentence in zip(seqs_y, ori_seqs_y):
        lm_id = []
        for i in range(len(sentence)):
            if i < len(ori_sentence):
                if ori_sentence[i] == sentence[i]:
                    lm_id.append(1)
                else:
                    lm_id.append(2)
            else:
                lm_id.append(2)
        lm_ids.append(lm_id)

    # 若反向，那么需要把数据完全倒过来，不影响其他的部分。
    if backward is False:
        seqs_y = list(map(lambda s: [Constants.BOS, Constants.HTER] + s + [Constants.EOS], seqs_y))
        lm_ids = list(map(lambda s: [0, 0] + s + [0], lm_ids))
    else:
        seqs_y = list(map(lambda s: [Constants.BOS, Constants.HTER] + s[::-1] + [Constants.EOS], seqs_y))
        lm_ids = list(map(lambda s: [0, 0] + s[::-1] + [0], lm_ids))

    hter_label = []
    for lm_id in lm_ids:
        all_count = len(lm_id) - 3
        bad_count = lm_id.count(2)
        hter_label.append(bad_count / all_count)

    y = _np_pad_batch_2D(seqs_y, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)
    lm_ids = _np_pad_batch_2D(lm_ids, pad=Constants.PAD,
                              cuda=cuda, batch_first=batch_first)
    hter_label = torch.tensor(hter_label).view((-1, 1)).cuda()

    return y, lm_ids, hter_label


def prepare_data_qe(seqs_x, seqs_y, lm_ids=None, cuda=False, batch_first=True):

    def _np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):

        batch_size = len(samples)

        sizes = [len(s) for s in samples]
        max_size = max(sizes)

        x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')

        for ii in range(batch_size):
            x_np[ii, :sizes[ii]] = samples[ii]

        if batch_first is False:
            x_np = np.transpose(x_np, [1, 0])

        x = torch.tensor(x_np)

        if cuda is True:
            x = x.cuda()
        return x

    seqs_x = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], seqs_x))
    x = _np_pad_batch_2D(samples=seqs_x, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    seqs_y = list(map(lambda s: [Constants.BOS, Constants.HTER] + s + [Constants.EOS], seqs_y))

    y = _np_pad_batch_2D(seqs_y, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)
    if lm_ids is None:
        return x, y

    lm_ids = list(map(lambda s: [0, 0] + s + [0], lm_ids))
    lm_ids = _np_pad_batch_2D(lm_ids, pad=Constants.PAD,
                              cuda=cuda, batch_first=batch_first)

    return x, y, lm_ids


def np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):

    batch_size = len(samples)

    sizes = [len(s) for s in samples]
    max_size = max(sizes)

    x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')

    for ii in range(batch_size):
        x_np[ii, :sizes[ii]] = samples[ii]

    if batch_first is False:
        x_np = np.transpose(x_np, [1, 0])

    x = torch.tensor(x_np)

    if cuda is True:
        x = x.cuda()
    return x


def prepare_data_qe_with_pe(seqs_x, seqs_y, seqs_pe, lm_ids=None, cuda=False, batch_first=True):

    def _np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):

        batch_size = len(samples)

        sizes = [len(s) for s in samples]
        max_size = max(sizes)

        x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')

        for ii in range(batch_size):
            x_np[ii, :sizes[ii]] = samples[ii]

        if batch_first is False:
            x_np = np.transpose(x_np, [1, 0])

        x = torch.tensor(x_np)

        if cuda is True:
            x = x.cuda()
        return x

    seqs_x = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], seqs_x))
    x = _np_pad_batch_2D(samples=seqs_x, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    seqs_y = list(map(lambda s: [Constants.BOS, Constants.HTER] + s + [Constants.EOS], seqs_y))

    y = _np_pad_batch_2D(seqs_y, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)
    
    seqs_pe = list(map(lambda s: [Constants.BOS, Constants.HTER] + s + [Constants.EOS], seqs_pe))

    pe = _np_pad_batch_2D(seqs_pe, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    if lm_ids is None:
        return x, y, pe

    lm_ids = list(map(lambda s: [0, 0] + s + [0], lm_ids))
    lm_ids = _np_pad_batch_2D(lm_ids, pad=Constants.PAD,
                              cuda=cuda, batch_first=batch_first)

    return x, y, pe, lm_ids


def prepare_data_qe_with_src_real(seqs_x, seqs_y, seqs_x_real, lm_ids=None, cuda=False, batch_first=True):

    def _np_pad_batch_2D(samples, pad, batch_first=True, cuda=True):

        batch_size = len(samples)

        sizes = [len(s) for s in samples]
        max_size = max(sizes)

        x_np = np.full((batch_size, max_size), fill_value=pad, dtype='int64')

        for ii in range(batch_size):
            x_np[ii, :sizes[ii]] = samples[ii]

        if batch_first is False:
            x_np = np.transpose(x_np, [1, 0])

        x = torch.tensor(x_np)

        if cuda is True:
            x = x.cuda()
        return x

    seqs_x = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], seqs_x))
    x = _np_pad_batch_2D(samples=seqs_x, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    seqs_y = list(map(lambda s: [Constants.BOS, Constants.HTER] + s + [Constants.EOS], seqs_y))
    y = _np_pad_batch_2D(seqs_y, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)
    
    seqs_x_real = list(map(lambda s: [Constants.BOS] + s + [Constants.EOS], seqs_x_real))
    x_real = _np_pad_batch_2D(samples=seqs_x_real, pad=Constants.PAD,
                         cuda=cuda, batch_first=batch_first)

    if lm_ids is None:
        return x, y, x_real

    lm_ids = list(map(lambda s: [0, 0] + s + [0], lm_ids))
    lm_ids = _np_pad_batch_2D(lm_ids, pad=Constants.PAD,
                              cuda=cuda, batch_first=batch_first)

    return x, y, x_real, lm_ids


def precision_validation(logits, seqs_y):
    # logits:[batch_size , seq_len , vocab_size]
    # seqs_y:[batch_size , seq_len]
    # 都是有padding的
    correct = 0
    words = 0
    logits = logits.argmax(2)  # [batch_size , seq_len]
    logits = logits.view(-1)  # [batch_size*seq_len]
    seqs_y = seqs_y.view(-1)  # [batch_size*seq_len]

    for each_logit, each_y in zip(logits, seqs_y):
        if each_logit == each_y and each_y != Constants.PAD:
            correct += 1
            words += 1
        elif each_y != Constants.PAD:
            words += 1

    return correct, words


def compute_forward(model,
                    critic,
                    seqs_x,
                    seqs_y,
                    lm_ids=None,
                    lm=None,
                    eval=False,
                    normalization=1.0,
                    norm_by_words=False
                    ):

    assert lm in ['MLM', 'LM_forward', 'LM_backward']

    y_inp = seqs_y[:, :].contiguous()

    if lm != 'MLM':
        y_label = seqs_y[:, 1:].contiguous()
    else:
        if lm_ids is None:
            lm_ids = seqs_y
        y_label = lm_ids[:, :].contiguous()

    words_norm = y_label.ne(Constants.PAD).float().sum(1)

    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            log_probs = model(seqs_x, y_inp, lm)
            loss = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization)

            if norm_by_words:
                loss = loss.div(words_norm).sum()
            else:
                loss = loss.sum()
        torch.autograd.backward(loss)
        return loss.item()
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs = model(seqs_x, y_inp, lm)
            loss = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=True)
        return loss.item(), log_probs


def compute_forward_discriminator(model,
                                  critic,
                                  critic_mse,
                                  seqs_x,
                                  seqs_y,
                                  lm_ids,
                                  hter_label,
                                  eval=False,
                                  normalization=1.0,
                                  norm_by_words=False,
                                  return_hter_pre=False,
                                  requires_adapter=False,
                                  ):

    y_inp = seqs_y[:, :].contiguous()
    y_label = lm_ids[:, 2:-1].contiguous()
    words_norm = y_label.ne(Constants.PAD).float().sum(1)

    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            log_probs, hter_pre = model(seqs_x, y_inp, requires_adapter=requires_adapter)  #model返回tag和hter分别需要的部分
            
            loss = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization)
            loss_hter = critic_mse(hter_pre, hter_label)

            if norm_by_words:
                loss = loss.div(words_norm).sum()
            else:
                loss = loss.sum()

        torch.autograd.backward(loss + loss_hter, retain_graph=True)  # backward的就是两个loss加一起
        if return_hter_pre == True:
            return loss.item(), loss_hter.item(), hter_pre
        return loss.item(), loss_hter.item()
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs, hter_pre = model(seqs_x, y_inp, requires_adapter=requires_adapter)

            loss = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=True)
            loss_hter = critic_mse(hter_pre, hter_label)

        return loss.item(), loss_hter.item(), log_probs, hter_pre


def compute_forward_discriminator_margin(model,
                                    critic,
                                    critic_mse,
                                    seqs_x,
                                    seqs_y,
                                    lm_ids,
                                    hter_label,
                                    eval=False,
                                    normalization=1.0,
                                    norm_by_words=False,
                                    return_hter_pre=False,
                                    ):

    y_inp = seqs_y[:, :].contiguous()
    y_label = lm_ids[:, 2:-1].contiguous()
    words_norm = y_label.ne(Constants.PAD).float().sum(1)

    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            log_probs, hter_pre = model(seqs_x, y_inp, margin_loss=True)  #model返回tag和hter分别需要的部分

            loss = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization)
            loss_hter = critic_mse(hter_pre, hter_label)

            if norm_by_words:
                loss = loss.div(words_norm).sum()
            else:
                loss = loss.sum()

        torch.autograd.backward(loss + loss_hter, retain_graph=True)  # backward的就是两个loss加一起
        if return_hter_pre == True:
            return loss.item(), loss_hter.item(), hter_pre
        return loss.item(), loss_hter.item()
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs, hter_pre = model(seqs_x, y_inp, margin_loss=True)

            loss = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=True)
            loss_hter = critic_mse(hter_pre, hter_label)

        return loss.item(), loss_hter.item(), log_probs, hter_pre


def compute_forward_discriminator_one_class(model,
                                  critic,
                                  critic_mse,
                                  seqs_x,
                                  seqs_y,
                                  lm_ids,
                                  hter_label,
                                  eval=False,
                                  normalization=1.0,
                                  norm_by_words=False,
                                  return_hter_pre=False,
                                  no_sigmoid=False,
                                  ):

    y_inp = seqs_y[:, :].contiguous()
    y_label = lm_ids[:, 2:-1].contiguous()
    words_norm = y_label.ne(Constants.PAD).float().sum(1)

    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            log_probs, hter_pre = model(seqs_x, y_inp, one_class=True, no_sigmoid=no_sigmoid)  #model返回tag和hter分别需要的部分

            loss = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization)
            loss_hter = critic_mse(hter_pre, hter_label)

            if norm_by_words:
                loss = loss.div(words_norm).sum()
            else:
                loss = loss.sum()

        torch.autograd.backward(loss + loss_hter, retain_graph=True)  # backward的就是两个loss加一起
        if return_hter_pre == True:
            return loss.item(), loss_hter.item(), hter_pre
        return loss.item(), loss_hter.item()
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs, hter_pre = model(seqs_x, y_inp, one_class=True)

            loss = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=True, no_sigmoid=no_sigmoid)
            loss_hter = critic_mse(hter_pre, hter_label)

        return loss.item(), loss_hter.item(), log_probs, hter_pre


def compute_forward_discriminator_robust_aug_anti(model,
                                                critic,
                                                critic_mse,
                                                seqs_x,
                                                seqs_y,
                                                lm_ids,
                                                hter_label,
                                                eval=False,
                                                aug_lambda=0.1,
                                                normalization=1.0,
                                                norm_by_words=False,
                                                return_hter_pre=False,
                                                ):


    y_inp = seqs_y[:, :].contiguous()
    y_label = lm_ids[:, 2:-1].contiguous()
    words_norm = y_label.ne(Constants.PAD).float().sum(1)
    
    anti_aug_x = torch.zeros_like(seqs_x)
    for i in range(anti_aug_x.size(-1)):
        anti_aug_x[0, i] = Constants.PAD

    anti_aug_label = torch.zeros_like(y_label)
    for i in range(anti_aug_label.size(-1)):
        anti_aug_label[0, i] = random.randint(1, 2)


    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            log_probs, hter_pre = model(seqs_x, y_inp)  #model返回tag和hter分别需要的部分
            aug_log_probs, aug_hter_pre = model(anti_aug_x, y_inp)

            loss = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization)
            aug_loss = critic(inputs=aug_log_probs, labels=anti_aug_label, reduce=False, normalization=normalization)

            loss_hter = critic_mse(hter_pre, hter_label)

            loss = loss + aug_loss * aug_lambda

            if norm_by_words:
                loss = loss.div(words_norm).sum()
            else:
                loss = loss.sum()

        torch.autograd.backward(loss + loss_hter, retain_graph=True)  # backward的就是两个loss加一起
        if return_hter_pre == True:
            return loss.item(), loss_hter.item(), hter_pre
        return loss.item(), loss_hter.item()
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs, hter_pre = model(seqs_x, y_inp)
            #print(log_probs.size())
            #print(y_label.size())
            loss = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=True)
            loss_hter = critic_mse(hter_pre, hter_label)

        return loss.item(), loss_hter.item(), log_probs, hter_pre


def compute_forward_discriminator_align_v1(model,
                                    critic,
                                    critic_mse,
                                    seqs_x,
                                    seqs_y,
                                    lm_ids,
                                    hter_label,
                                    align_line,
                                    align_ratio=0.5,
                                    eval=False,
                                    normalization=1.0,
                                    norm_by_words=False,
                                    return_hter_pre=False,
                                    ):

    y_inp = seqs_y[:, :].contiguous()
    y_label = lm_ids[:, 2:-1].contiguous()
    words_norm = y_label.ne(Constants.PAD).float().sum(1)

    x_len = seqs_x.size(-1) - 2
    y_len = seqs_y.size(-1) - 3

    align_matrix = torch.zeros([y_len, x_len])

    for i in range(y_len):
        if str(i) in align_line:
            align_matrix[i] = align_matrix[i].index_fill(0, torch.tensor(align_line[str(i)]), True)

    align_matrix_pad = F.pad(align_matrix, pad = (1, 1, 2, 1)).cuda()
    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            log_probs, hter_pre = model(seqs_x, y_inp, align_matrix_pad = align_matrix_pad, align_ratio=align_ratio)  #model返回tag和hter分别需要的部分
            
            loss = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization)
            loss_hter = critic_mse(hter_pre, hter_label)

            if norm_by_words:
                loss = loss.div(words_norm).sum()
            else:
                loss = loss.sum()

        torch.autograd.backward(loss + loss_hter, retain_graph=True)  # backward的就是两个loss加一起
        if return_hter_pre == True:
            return loss.item(), loss_hter.item(), hter_pre
        return loss.item(), loss_hter.item()
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs, hter_pre = model(seqs_x, y_inp, align_matrix_pad = align_matrix_pad, align_ratio=align_ratio)
            loss = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=True)
            loss_hter = critic_mse(hter_pre, hter_label)

        return loss.item(), loss_hter.item(), log_probs, hter_pre


def compute_forward_discriminator_align_v2(model,
                                            critic,
                                            critic_mse,
                                            seqs_x,
                                            seqs_y,
                                            lm_ids,
                                            hter_label,
                                            align_line,
                                            eval=False,
                                            normalization=1.0,
                                            norm_by_words=False,
                                            return_hter_pre=False,
                                            attn_align_lambda=1,
                                            ):

    y_inp = seqs_y[:, :].contiguous()
    y_label = lm_ids[:, 2:-1].contiguous()
    words_norm = y_label.ne(Constants.PAD).float().sum(1)

    if not eval:
        model.train()
        critic.train()
        x_len = seqs_x.size(-1) - 2
        y_len = seqs_y.size(-1) - 3

        align_matrix = torch.zeros([y_len, x_len])

        for i in range(y_len):
            if str(i) in align_line:
                align_matrix[i] = align_matrix[i].index_fill(0, torch.tensor(align_line[str(i)]), True)

        align_matrix = align_matrix.cuda()

        # For training
        with torch.enable_grad():
            log_probs, hter_pre, ctx_attn = model(seqs_x, y_inp, return_attn = True)  #model返回tag和hter分别需要的部分

            # all
            #ctx_attn_all = torch.cat(ctx_attn, dim = 1)  # [batch_size, layer_num * head_num, seq_y_len + 3, seq_x_len + 2]
            #ctx_attn_real = ctx_attn_all[:, :, 2:-1, 1:-1]  # 去掉无关部分，<bos><eos> 这里把hter也去掉了，可能会有影响？

            # 2layer
            #ctx_attn_2layer = torch.cat(ctx_attn[3:5], dim = 1)  # [batch_size, choosed_layer_num * head_num, seq_y_len + 3, seq_x_len + 2]
            ctx_attn_1layer = torch.cat(ctx_attn[5:], dim = 1)  # [batch_size, choosed_layer_num * head_num, seq_y_len + 3, seq_x_len + 2]
            ctx_attn_real = ctx_attn_1layer[:, :, 2:-1, 1:-1]

            # random
            #ctx_attn_random = torch.stack((ctx_attn[5][:, 5, :, :], ctx_attn[4][:, 3, :, :]), dim = 1)  # [batch_size, choosed head_num, seq_y_len + 3, seq_x_len + 2]
            #ctx_attn_real = ctx_attn_random[:, :, 2:-1, 1:-1]

            # most
            #ctx_attn_most = torch.stack((ctx_attn[5][:, 3, :, :], ctx_attn[4][:, 2, :, :]), dim = 1)  # [batch_size, choosed head_num, seq_y_len + 3, seq_x_len + 2]
            #ctx_attn_real = ctx_attn_most[:, :, 2:-1, 1:-1]

            align_matrix_repeat = align_matrix.repeat(ctx_attn_real.size(0), ctx_attn_real.size(1), 1, 1)
            
            kl_loss = F.kl_div(ctx_attn_real.log(), align_matrix_repeat.softmax(dim=-1), reduction='sum') / ctx_attn_real.size(1)
            kl_loss *= attn_align_lambda

            loss = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization)
            loss_hter = critic_mse(hter_pre, hter_label)
            
            if norm_by_words:
                loss = loss.div(words_norm).sum()
            else:
                loss = loss.sum()
            
        torch.autograd.backward(loss + loss_hter + kl_loss, retain_graph=True)  # backward的就是两个loss加一起
        if return_hter_pre == True:
            return loss.item(), loss_hter.item(), hter_pre
        return loss.item(), loss_hter.item(), kl_loss.item()
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs, hter_pre = model(seqs_x, y_inp)
            loss = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=True)
            loss_hter = critic_mse(hter_pre, hter_label)

        return loss.item(), loss_hter.item(), log_probs, hter_pre


def compute_forward_discriminator_align_v3(model,
                                    critic,
                                    critic_mse,
                                    seqs_x,
                                    seqs_y,
                                    lm_ids,
                                    hter_label,
                                    align_line,
                                    eval=False,
                                    normalization=1.0,
                                    norm_by_words=False,
                                    return_hter_pre=False,
                                    attn_align_lambda=0.5,
                                    ):

    y_inp = seqs_y[:, :].contiguous()
    y_label = lm_ids[:, 2:-1].contiguous()
    words_norm = y_label.ne(Constants.PAD).float().sum(1)

    x_len = seqs_x.size(-1) - 2
    y_len = seqs_y.size(-1) - 3

    align_matrix = torch.zeros([y_len, x_len])

    for i in range(y_len):
        if str(i) in align_line:
            align_matrix[i] = align_matrix[i].index_fill(0, torch.tensor(align_line[str(i)]), True)

    align_matrix_pad = F.pad(align_matrix, pad = (1, 1, 2, 1)).cuda()
    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            log_probs, hter_pre = model(seqs_x, y_inp, align_matrix_pad = align_matrix_pad, add_align = True, align_ratio=attn_align_lambda)  #model返回tag和hter分别需要的部分
            
            loss = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization)
            loss_hter = critic_mse(hter_pre, hter_label)

            if norm_by_words:
                loss = loss.div(words_norm).sum()
            else:
                loss = loss.sum()

        torch.autograd.backward(loss + loss_hter, retain_graph=True)  # backward的就是两个loss加一起
        if return_hter_pre == True:
            return loss.item(), loss_hter.item(), hter_pre
        return loss.item(), loss_hter.item()
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs, hter_pre = model(seqs_x, y_inp, align_matrix_pad = align_matrix_pad, add_align = True, align_ratio=attn_align_lambda)
            loss = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=True)
            loss_hter = critic_mse(hter_pre, hter_label)

        return loss.item(), loss_hter.item(), log_probs, hter_pre


def compute_forward_discriminator_add_feature(model,
                                    critic,
                                    critic_mse,
                                    seqs_x,
                                    seqs_y,
                                    lm_ids,
                                    hter_label,
                                    align_line,
                                    eval=False,
                                    normalization=1.0,
                                    norm_by_words=False,
                                    return_hter_pre=False,
                                    ):

    y_inp = seqs_y[:, :].contiguous()
    y_label = lm_ids[:, 2:-1].contiguous()
    words_norm = y_label.ne(Constants.PAD).float().sum(1)

    x_len = seqs_x.size(-1) - 2
    y_len = seqs_y.size(-1) - 3

    align_matrix = torch.zeros([y_len, x_len])

    for i in range(y_len):
        if str(i) in align_line:
            align_matrix[i] = align_matrix[i].index_fill(0, torch.tensor(align_line[str(i)]), True)

    align_matrix_pad = F.pad(align_matrix, pad = (1, 1, 2, 1)).cuda()
    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            log_probs, hter_pre = model(seqs_x, y_inp, align_matrix_pad = align_matrix_pad, add_feature = True)  #model返回tag和hter分别需要的部分
            loss = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization)
            loss_hter = critic_mse(hter_pre, hter_label)

            if norm_by_words:
                loss = loss.div(words_norm).sum()
            else:
                loss = loss.sum()

        torch.autograd.backward(loss + loss_hter, retain_graph=True)  # backward的就是两个loss加一起
        if return_hter_pre == True:
            return loss.item(), loss_hter.item(), hter_pre
        return loss.item(), loss_hter.item()
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs, hter_pre = model(seqs_x, y_inp, align_matrix_pad = align_matrix_pad, add_feature = True)
            loss = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=True)
            loss_hter = critic_mse(hter_pre, hter_label)

        return loss.item(), loss_hter.item(), log_probs, hter_pre


def compute_forward_discriminator_align_v4(model,
                                            critic,
                                            critic_mse,
                                            seqs_x,
                                            seqs_y,
                                            lm_ids,
                                            hter_label,
                                            align_line,
                                            eval=False,
                                            normalization=1.0,
                                            norm_by_words=False,
                                            return_hter_pre=False,
                                            attn_align_lambda=0.5,
                                            ):

    y_inp = seqs_y[:, :].contiguous()
    y_label = lm_ids[:, 2:-1].contiguous()
    words_norm = y_label.ne(Constants.PAD).float().sum(1)

    if not eval:
        model.train()
        critic.train()
        x_len = seqs_x.size(-1) - 2
        y_len = seqs_y.size(-1) - 3
        align_matrix = torch.zeros([y_len, x_len])

        for i in range(y_len):
            if str(i) in align_line:
                align_matrix[i] = align_matrix[i].index_fill(0, torch.tensor(align_line[str(i)]), True)

        align_matrix = align_matrix.cuda()
        # For training
        with torch.enable_grad():
            log_probs, hter_pre, ctx_attn = model(seqs_x, y_inp, return_attn = True)  #model返回tag和hter分别需要的部分

            # ctx_attn 一个列表，包括所有layer的attn: [batch_size, head_num, query_len, key_len]
            ctx_attn_all = torch.stack(ctx_attn, dim = 1) # [batch_size, layer_num, head_num, query_len, key_len]
            
            # 选择前两个attn head，作为对齐训练attn
            ctx_attn_align_head = ctx_attn_all[:, :, 0:4, 2:-1, 1:-1].squeeze(0)   # 去掉无关部分，<bos><eos> 这里把hter也去掉了，可能会有影响？
            ctx_attn_align = ctx_attn_align_head.reshape(-1, y_len, x_len)   # [layer_num * choosed_head_num, seq_y, seq_x]
            
            align_matrix_repeat = align_matrix.repeat(ctx_attn_align.size(0), 1, 1)
            
            kl_loss = F.kl_div(ctx_attn_align.softmax(dim=-1).log(), align_matrix_repeat.softmax(dim=-1), reduction='sum') / ctx_attn_align.size(0)
            
            loss = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization)
            loss_hter = critic_mse(hter_pre, hter_label)

            if norm_by_words:
                loss = loss.div(words_norm).sum()
            else:
                loss = loss.sum()

        torch.autograd.backward((1 - attn_align_lambda) * (loss + loss_hter) + attn_align_lambda * kl_loss, retain_graph=True)  # backward的就是两个loss加一起
        if return_hter_pre == True:
            return loss.item(), loss_hter.item(), hter_pre
        return loss.item(), loss_hter.item(), kl_loss.item()
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs, hter_pre = model(seqs_x, y_inp)
            loss = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=True)
            loss_hter = critic_mse(hter_pre, hter_label)

        return loss.item(), loss_hter.item(), log_probs, hter_pre


def compute_attn_align_cossim(model,
                            seqs_x,
                            seqs_y,
                            lm_ids,
                            hter_label,
                            align_line,
                            ):
    # 计算attn 不同层不同head 和对齐矩阵的cosine相似度

    y_inp = seqs_y[:, :].contiguous()
    y_label = lm_ids[:, 2:-1].contiguous()
    words_norm = y_label.ne(Constants.PAD).float().sum(1)

    x_len = seqs_x.size(-1) - 2
    y_len = seqs_y.size(-1) - 3

    align_matrix = torch.zeros([y_len, x_len])

    for i in range(y_len):
        if str(i) in align_line:
            align_matrix[i] = align_matrix[i].index_fill(0, torch.tensor(align_line[str(i)]), True)

    align_matrix = align_matrix.cuda()

    cos = nn.CosineSimilarity(dim=-1)

    model.eval()
    with torch.no_grad():
        log_probs, hter_pre, ctx_attn = model(seqs_x, y_inp, return_attn = True)  #model返回tag和hter分别需要的部分

        ctx_attn_all = torch.stack(ctx_attn, dim = 1)  # [batch_size, layer_num, head_num, seq_y_len + 3, seq_x_len + 2]
        ctx_attn_real = ctx_attn_all[:, :, :, 2:-1, 1:-1]  # 去掉无关部分，<bos><eos> 这里把hter也去掉了
        
        align_matrix_repeat = align_matrix.repeat(ctx_attn_real.size(0), 1, 1).softmax(dim = -1)  # [batch_size, seq_y_len + 3, seq_x_len + 2]
        
        attn_align_cossim_one_sample = torch.zeros([ctx_attn_real.size(1), ctx_attn_real.size(2)])  # [layer_num, head_num]

        for layer_id in range(ctx_attn_real.size(1)):
            for head_id in range(ctx_attn_real.size(2)):
                ctx_attn_this = ctx_attn_real[:, layer_id, head_id, :, :]  # [batch_size, seq_y_len + 3, seq_x_len + 2]
                similarity = cos(ctx_attn_this, align_matrix_repeat)   # [batch_size, seq_y_len + 3]
                attn_align_cossim_one_sample[layer_id, head_id] = similarity.mean()

        return attn_align_cossim_one_sample


def compute_forward_discriminator_contrastive(model,
                                            critic,
                                            critic_mse,
                                            seqs_x,
                                            seqs_y,
                                            seqs_pe,
                                            seqs_align,
                                            lm_ids,
                                            hter_label,
                                            eval=False,
                                            loss_real_lambda=1.0,
                                            loss_parallel_lambda=1.0,
                                            loss_switch_lambda=1.0,
                                            cl_lambda_switch=1.0,
                                            cl_lambda_parallel = 1.0,
                                            normalization=1.0,
                                            norm_by_words=False,
                                            return_hter_pre=False,
                                            optim=None,
                                            ):

    batch_size = seqs_x.size(0)
    cos = nn.CosineSimilarity(dim=-1)

    y_inp = seqs_y[:, :].contiguous()
    pe_inp = seqs_pe[:, :].contiguous()
    y_label = lm_ids[:, 2:-1].contiguous() 
    y_label_parallel = [[1] * (seqs_pe[i].nonzero().size(0) - 3) for i in range(seqs_pe.size(0))]
    y_label_parallel = list(map(lambda s: [0, 0] + s + [0], y_label_parallel))
    y_label_parallel = np_pad_batch_2D(y_label_parallel, pad=Constants.PAD)
    y_label_parallel = y_label_parallel[:, 2:-1].contiguous() 
    y_label_switch = torch.where(y_label == 1, 2, y_label)
    words_norm = y_label.ne(Constants.PAD).float().sum(1)  # [batch_size]

    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            log_probs, hter_pre, dec_output = model(seqs_x, y_inp, get_result_and_representation=True)
            parallel_log_probs, parallel_hter_pre, parallel_dec_output = model(seqs_x, pe_inp, get_result_and_representation=True)

            loss = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization)
            loss_parallel = critic(inputs=parallel_log_probs, labels=y_label_parallel, reduce=False, normalization=normalization)
            loss_hter = critic_mse(hter_pre, hter_label)

            if norm_by_words:
                loss = loss.div(words_norm).sum()
                loss_parallel = loss_parallel.div(words_norm).sum()
            else:
                loss = loss.sum()
                loss_parallel = loss_parallel.sum()
            
            loss = loss * loss_real_lambda
            loss_parallel = loss_parallel * loss_parallel_lambda

            # switch语料
            all_cl_loss = []
            all_loss_switch = []
            for i in range(1, batch_size):
                order = list(range(i, batch_size)) + list(range(i))   # 调整的每种错位顺序
                switch_seqs_x = seqs_x[order]
                switch_log_probs, switch_hter_pre, switch_dec_output = model(switch_seqs_x, y_inp, get_result_and_representation=True)
                loss_switch = critic(inputs=switch_log_probs, labels=y_label_switch, reduce=False, normalization=normalization)
                if norm_by_words:
                    loss_switch = loss_switch.div(words_norm).sum()
                else:
                    loss_switch = loss_switch.sum()
                similarity = cos(dec_output, switch_dec_output)   # [batch_size, max_mt_len]
                sign = torch.where(y_label > 1.5, -1, y_label)    # [batch_size, max_mt_len]   ok1 bad-1
                cl_loss = torch.exp(similarity * sign).sum(-1).mean().unsqueeze(0)    # [1]
                all_cl_loss.append(cl_loss)
                all_loss_switch.append(loss_switch.unsqueeze(0))

            # 平行语料ok拉近
            all_cl_loss_parallel = []
            for bb in range(batch_size):
                for id_mt, id_pe in seqs_align[bb].items():
                    similarity = cos(dec_output[bb][id_mt], parallel_dec_output[bb][id_pe])  # 标量
                    cl_loss_parallel = torch.exp(-similarity).unsqueeze(0)    # [1]
                    all_cl_loss_parallel.append(cl_loss_parallel)

            if len(all_cl_loss_parallel) == 0:
                cl_loss_parallel = torch.tensor(0)
            else:
                all_cl_loss_parallel = torch.cat(all_cl_loss_parallel, dim = 0)
                cl_loss_parallel = (all_cl_loss_parallel.sum()) * cl_lambda_parallel

            if len(all_cl_loss) == 0:
                cl_loss = torch.tensor(0)
                loss_switch = torch.tensor(0)
            else:
                all_cl_loss = torch.cat(all_cl_loss, dim = 0)           # [batch_size - 1]
                all_loss_switch = torch.cat(all_loss_switch, dim = 0)   # [batch_size - 1]
                cl_loss = (all_cl_loss.mean()) * cl_lambda_switch
                loss_switch = (all_loss_switch.mean()) * loss_switch_lambda

            torch.autograd.backward(loss + loss_parallel + loss_switch + cl_loss + cl_loss_parallel + loss_hter, retain_graph=True)

        return loss.item(), loss_parallel.item(), loss_switch.item(), cl_loss.item(), cl_loss_parallel.item(), loss_hter.item()
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs, hter_pre, dec_output = model(seqs_x, y_inp, get_result_and_representation=True)
            parallel_log_probs, parallel_hter_pre, parallel_dec_output = model(seqs_x, pe_inp, get_result_and_representation=True)

            loss = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=True)
            loss_parallel = critic(inputs=parallel_log_probs, labels=y_label_parallel, reduce=True, normalization=normalization)
            loss_hter = critic_mse(hter_pre, hter_label)

            loss = loss * loss_real_lambda
            loss_parallel = loss_parallel * loss_parallel_lambda


            # switch语料
            all_cl_loss = []
            all_loss_switch = []
            for i in range(1, batch_size):
                order = list(range(i, batch_size)) + list(range(i))   # 调整的每种错位顺序
                switch_seqs_x = seqs_x[order]
                switch_log_probs, switch_hter_pre, switch_dec_output = model(switch_seqs_x, y_inp, get_result_and_representation=True)
                loss_switch = critic(inputs=switch_log_probs, labels=y_label_switch, reduce=True, normalization=normalization)
                similarity = cos(dec_output, switch_dec_output)   # [batch_size, max_mt_len]
                sign = torch.where(y_label > 1.5, -1, y_label)    # [batch_size, max_mt_len]
                cl_loss = torch.exp(similarity * sign).sum(-1).mean().unsqueeze(0)    # [1]
                all_cl_loss.append(cl_loss)
                all_loss_switch.append(loss_switch.unsqueeze(0))
            
            # 平行语料ok拉近
            all_cl_loss_parallel = []
            for bb in range(batch_size):
                for id_mt, id_pe in seqs_align[bb].items():
                    similarity = cos(dec_output[bb][id_mt], parallel_dec_output[bb][id_pe])  # 标量
                    cl_loss_parallel = torch.exp(-similarity).unsqueeze(0)    # [1]
                    all_cl_loss_parallel.append(cl_loss_parallel)

            if len(all_cl_loss_parallel) == 0:
                cl_loss_parallel = torch.tensor(0)
            else:
                all_cl_loss_parallel = torch.cat(all_cl_loss_parallel, dim = 0)
                cl_loss_parallel = (all_cl_loss_parallel.sum()) * cl_lambda_parallel

            if len(all_cl_loss) == 0:
                cl_loss = torch.tensor(0)
                loss_switch = torch.tensor(0)
            else:
                all_cl_loss = torch.cat(all_cl_loss, dim = 0)   # [batch_size - 1]
                all_loss_switch = torch.cat(all_loss_switch, dim = 0)   # [batch_size - 1]
                cl_loss = (all_cl_loss.mean()) * cl_lambda_switch
                loss_switch = (all_loss_switch.mean()) * loss_switch_lambda
            
        return loss.item(), loss_parallel.item(), loss_switch.item(), cl_loss.item(), cl_loss_parallel.item(), loss_hter.item(), log_probs, hter_pre


def compute_forward_discriminator_contrastive_src(model,
                                                critic,
                                                critic_mse,
                                                seqs_x,
                                                seqs_y,
                                                seqs_x_real,
                                                lm_ids,
                                                hter_label,
                                                eval=False,
                                                loss_real_lambda=1.0,
                                                loss_parallel_lambda=1.0,
                                                cl_lambda=1.0,
                                                normalization=1.0,
                                                norm_by_words=False,
                                                ):

    batch_size = seqs_x.size(0)
    cos = nn.CosineSimilarity(dim=-1)

    y_inp = seqs_y[:, :].contiguous()
    y_label = lm_ids[:, 2:-1].contiguous()                     # [batch_size, max_seq_len_of_this_batch]
    sign = torch.where(y_label < 1.5, 0, y_label)              # [batch_size, max_mt_len]   ok0 bad2
    y_label_parallel = torch.where(sign > 1.5, 1, sign)        # 只保留替换处 [batch_size, max_seq_len_of_this_batch]
    words_norm = y_label.ne(Constants.PAD).float().sum(1)      # [batch_size]
    words_norm_bad = y_label_parallel.sum(dim=-1)              # [batch_size]

    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            # 修改src的伪数据
            log_probs, hter_pre, dec_output = model(seqs_x, y_inp, get_result_and_representation=True)      
            # 正常src的平行语料
            parallel_log_probs, parallel_hter_pre, parallel_dec_output = model(seqs_x_real, y_inp, get_result_and_representation=True)  
            
            # 伪数据词级别loss
            loss = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization)      
            # 平行语料词级别loss
            loss_parallel = critic(inputs=parallel_log_probs, labels=y_label_parallel, reduce=False, normalization=normalization)
            # 伪数据句子级loss
            loss_hter = critic_mse(hter_pre, hter_label)    

            if norm_by_words:
                loss = loss.div(words_norm).sum()
                loss_parallel = loss_parallel.div(words_norm_bad).sum()
            else:
                loss = loss.sum()
                loss_parallel = loss_parallel.sum()
            
            loss = loss * loss_real_lambda
            loss_parallel = loss_parallel * loss_parallel_lambda

            # cl_loss
            similarity = cos(dec_output, parallel_dec_output)       # [batch_size, max_mt_len]
            cl_loss = (similarity * y_label_parallel).sum()   # 标量
            cl_loss *= cl_lambda

            torch.autograd.backward(loss + loss_parallel + cl_loss + loss_hter, retain_graph=True)

        return loss.item(), loss_parallel.item(), cl_loss.item(), loss_hter.item()
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs, hter_pre, dec_output = model(seqs_x, y_inp, get_result_and_representation=True)
            parallel_log_probs, parallel_hter_pre, parallel_dec_output = model(seqs_x_real, y_inp, get_result_and_representation=True)  # 正常src的平行语料

            loss = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=True)
            loss_parallel = critic(inputs=parallel_log_probs, labels=y_label_parallel, reduce=True, normalization=normalization)
            loss_hter = critic_mse(hter_pre, hter_label)

            loss = loss * loss_real_lambda
            loss_parallel = loss_parallel * loss_parallel_lambda

            # cl_loss
            similarity = cos(dec_output, parallel_dec_output)       # [batch_size, max_mt_len]
            cl_loss = (similarity * y_label_parallel).sum()   # 标量
            cl_loss *= cl_lambda
            
        return loss.item(), loss_parallel.item(), cl_loss.item(), loss_hter.item(), log_probs, hter_pre


def cal_contrast_group(model,
                        dataset,
                        critic,
                        critic_mse,
                        ):
    ss = 0
    right = 0

    for batch in dataset:
        seqs_x, seqs_y, xy_label, xy_hter, _, contrast_idx = batch

        x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)
            
        if Constants.USE_GPU:
            xy_hter = torch.tensor(xy_hter).cuda()
            contrast_idx = torch.tensor(contrast_idx).cuda().int().squeeze(0).squeeze(0)
        #print('contrast_idx')
        #print(contrast_idx)
        token_mask = torch.zeros_like(xy_label)[:, 2:-1]
        for idx in contrast_idx:
            token_mask[0, idx] = 1

        loss, loss_hter, logits, hter_pre = compute_forward_discriminator_token_mask(model=model,
                                                                            critic=critic,
                                                                            critic_mse=critic_mse,
                                                                            seqs_x=x,
                                                                            seqs_y=y,
                                                                            lm_ids=xy_label,
                                                                            hter_label=xy_hter,
                                                                            token_mask=token_mask,
                                                                            eval=True)
        xy_label = (xy_label[:, 2:-1].contiguous() * token_mask).view(-1).cpu().tolist()
        logits = logits.view(-1, 3).cpu().tolist()

        for e_l, e_o in zip(xy_label, logits):
            if e_l == 0:
                continue
            # label里是1和2。
            # 1原词，2噪声。
            if e_o[1] < e_o[2]:
                e_o = 2
            else:
                e_o = 1
            if e_o == e_l:
                right += 1
            ss += 1
    
    if ss == 0:
        return 1

    return float(right / ss)


def compute_forward_discriminator_token_mask(model,
                                  critic,
                                  critic_mse,
                                  seqs_x,
                                  seqs_y,
                                  lm_ids,
                                  hter_label,
                                  token_mask = None,
                                  eval=False,
                                  normalization=1.0,
                                  norm_by_words=False,
                                  return_hter_pre=False,
                                  margin=False,
                                  ):
    """
    只考虑训练样本中没被mask的token
    """
    y_inp = seqs_y[:, :].contiguous()
    y_label = lm_ids[:, 2:-1].contiguous()
    if token_mask is not None: y_label *= token_mask
    words_norm = y_label.ne(Constants.PAD).float().sum(1)

    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            log_probs, hter_pre = model(seqs_x, y_inp, margin_loss=margin)  #model返回tag和hter分别需要的部分

            # 这里传回来的loss就已经是 仅限于训练样本中没被mask的token的loss了

            loss = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization)

            loss_hter = critic_mse(hter_pre, hter_label)

            if norm_by_words:
                loss = loss.div(words_norm).sum()
            else:
                loss = loss.sum()

        # TODO 冻结参数debug
        loss.requires_grad_(True)
        loss_hter.requires_grad_(True)
        torch.autograd.backward(loss + loss_hter, retain_graph=True)  # backward的就是两个loss加一起
        #torch.autograd.backward(loss, retain_graph=True) # 只用token_loss更新梯度
        if return_hter_pre == True:
            return loss.item(), loss_hter.item(), hter_pre
        return loss.item(), loss_hter.item()
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs, hter_pre = model(seqs_x, y_inp, margin_loss=margin)
            loss = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=True)
            loss_hter = critic_mse(hter_pre, hter_label)

        return loss.item(), loss_hter.item(), log_probs, hter_pre


def compute_forward_discriminator_token_mask_add_feature(model,
                                                critic,
                                                critic_mse,
                                                seqs_x,
                                                seqs_y,
                                                lm_ids,
                                                hter_label,
                                                align_line,
                                                token_mask = None,
                                                eval=False,
                                                normalization=1.0,
                                                norm_by_words=False,
                                                return_hter_pre=False,
                                                margin=False,
                                                ):
    """
    只考虑训练样本中没被mask的token
    """
    y_inp = seqs_y[:, :].contiguous()
    y_label = lm_ids[:, 2:-1].contiguous()
    if token_mask is not None: y_label *= token_mask
    words_norm = y_label.ne(Constants.PAD).float().sum(1)

    x_len = seqs_x.size(-1) - 2
    y_len = seqs_y.size(-1) - 3

    align_matrix = torch.zeros([y_len, x_len])

    for i in range(y_len):
        if str(i) in align_line:
            align_matrix[i] = align_matrix[i].index_fill(0, torch.tensor(align_line[str(i)]), True)

    align_matrix_pad = F.pad(align_matrix, pad = (1, 1, 2, 1)).cuda()

    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            log_probs, hter_pre = model(seqs_x, y_inp, margin_loss=margin, align_matrix_pad = align_matrix_pad, add_feature = True)  #model返回tag和hter分别需要的部分

            # 这里传回来的loss就已经是 仅限于训练样本中没被mask的token的loss了
            loss = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization)
            loss_hter = critic_mse(hter_pre, hter_label)

            if norm_by_words:
                loss = loss.div(words_norm).sum()
            else:
                loss = loss.sum()

        # TODO 冻结参数debug
        loss.requires_grad_(True)
        loss_hter.requires_grad_(True)
        torch.autograd.backward(loss + loss_hter, retain_graph=True)  # backward的就是两个loss加一起
        #torch.autograd.backward(loss, retain_graph=True) # 只用token_loss更新梯度
        if return_hter_pre == True:
            return loss.item(), loss_hter.item(), hter_pre
        return loss.item(), loss_hter.item()
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs, hter_pre = model(seqs_x, y_inp, margin_loss=margin, align_matrix_pad = align_matrix_pad, add_feature = True)
            loss = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=True)
            loss_hter = critic_mse(hter_pre, hter_label)

        return loss.item(), loss_hter.item(), log_probs, hter_pre


def compute_forward_discriminator_token_mask_align_v1(model,
                                  critic,
                                  critic_mse,
                                  seqs_x,
                                  seqs_y,
                                  lm_ids,
                                  hter_label,
                                  align_line,
                                  token_mask = None,
                                  eval=False,
                                  normalization=1.0,
                                  norm_by_words=False,
                                  return_hter_pre=False,
                                  ):
    """
    只考虑训练样本中没被mask的token
    而且融入对齐信息
    """
    y_inp = seqs_y[:, :].contiguous()
    y_label = lm_ids[:, 2:-1].contiguous()
    if token_mask is not None: y_label *= token_mask
    words_norm = y_label.ne(Constants.PAD).float().sum(1)

    x_len = seqs_x.size(-1) - 2
    y_len = seqs_y.size(-1) - 3

    align_matrix = torch.zeros([y_len, x_len])

    for i in range(y_len):
        if str(i) in align_line:
            align_matrix[i] = align_matrix[i].index_fill(0, torch.tensor(align_line[str(i)]), True)

    align_matrix_pad = F.pad(align_matrix, pad = (1, 1, 2, 1)).cuda()

    if not eval:
        model.train()
        critic.train()
        # For training
        with torch.enable_grad():
            log_probs, hter_pre = model(seqs_x, y_inp, align_matrix_pad = align_matrix_pad, requires_adapter=False)  #model返回tag和hter分别需要的部分

            # 这里传回来的loss就已经是 仅限于训练样本中没被mask的token的loss了
            loss = critic(inputs=log_probs, labels=y_label, reduce=False, normalization=normalization)
            #print(loss)
            #assert 1==2
            loss_hter = critic_mse(hter_pre, hter_label)

            if norm_by_words:
                loss = loss.div(words_norm).sum()
            else:
                loss = loss.sum()

        # TODO 冻结参数debug
        loss.requires_grad_(True)
        loss_hter.requires_grad_(True)
        torch.autograd.backward(loss + loss_hter, retain_graph=True)  # backward的就是两个loss加一起
        #torch.autograd.backward(loss, retain_graph=True) # 只用token_loss更新梯度
        if return_hter_pre == True:
            return loss.item(), loss_hter.item(), hter_pre
        return loss.item(), loss_hter.item()
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs, hter_pre = model(seqs_x, y_inp, align_matrix_pad = align_matrix_pad, requires_adapter=False)
            loss = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=True)
            loss_hter = critic_mse(hter_pre, hter_label)

        return loss.item(), loss_hter.item(), log_probs, hter_pre


def compute_forward_discriminator_modify(model,
                                  critic,
                                  critic_mse,
                                  seqs_x,
                                  seqs_y,
                                  lm_ids,
                                  hter_label,
                                  ok_prob=None,
                                  eval=False,
                                  normalization=1.0,
                                  norm_by_words=False
                                  ):

    y_inp = seqs_y[:, :].contiguous()
    y_label = lm_ids[:, 2:-1].contiguous()
    words_norm = y_label.ne(Constants.PAD).float().sum(1)
    

    if not eval:
        model.train()
        critic.train()

        ok_prob = ok_prob.squeeze(0)
        true_prob = []
        for i in range(ok_prob.size(0)):
            if y_label.squeeze(0)[i] == 1:  # ok
                true_prob.append(ok_prob[i])
            else: # bad
                true_prob.append(1-ok_prob[i])
        true_prob = torch.tensor(true_prob).cuda()
        prob_weight = (1 - true_prob) * 10

        # For training
        with torch.enable_grad():
            log_probs, hter_pre = model(seqs_x, y_inp)  #model返回tag和hter分别需要的部分

            loss = critic(inputs=log_probs, labels=y_label, reduce=False, no_reduction=True, normalization=normalization)
            loss_hter = critic_mse(hter_pre, hter_label)

            # loss加权
            loss = torch.mul(loss, prob_weight)

            if norm_by_words:
                loss = loss.div(words_norm).sum()
            else:
                loss = loss.sum()

        torch.autograd.backward(loss + loss_hter)  # backward的就是两个loss加一起
        return loss.item(), loss_hter.item()
    else:
        model.eval()
        critic.eval()
        # For compute loss
        with torch.no_grad():
            log_probs, hter_pre = model(seqs_x, y_inp)
            loss = critic(inputs=log_probs, labels=y_label, normalization=normalization, reduce=True)
            loss_hter = critic_mse(hter_pre, hter_label)

        return loss.item(), loss_hter.item(), log_probs, hter_pre


def compute_forward_probe(qe_model,
                                  probe_model,
                                  critic,
                                  seqs_x,
                                  seqs_y,
                                  pos_line,
                                  normalization=1.0,
                                  return_acc=False,
                                  eval=False,
                                  representation_type="hidden"):

    y_inp = seqs_y[:, :].contiguous()
    pos_label = pos_line[:, 2:-1].contiguous()
    pos_label = pos_label.squeeze(0) - 6

    qe_model.eval()
    if representation_type == 'hidden':
        with torch.no_grad():
            # 提取表示
            representation = qe_model(seqs_x, seqs_y, get_representation=True).squeeze(0)[2:-1]
    elif representation_type == 'embedding':
        embed_seqs_mt = []
        for mt_token in seqs_y.squeeze(0)[2:-1]:
            embed_seq_mt = qe_model.decoder.embeddings.embeddings.weight[mt_token]
            embed_seqs_mt.append(embed_seq_mt)
        representation = torch.stack(embed_seqs_mt,0)     # [mt_seq_len, embed_dim]
    elif representation_type == 'random':
        seq_len = seqs_y.size(1)-3
        embed_dim = qe_model.decoder.embeddings.embeddings.weight.size(-1)
        representation = torch.randn(seq_len, embed_dim).cuda()

    if eval:
        # 测试模式
        probe_model.eval()
        critic.eval()
        with torch.no_grad():
            pos_log_probs = probe_model(representation)  # [seq_len, num_pos_class]

            loss_pos, n_correct = compute_loss_and_accuracy(
                pos_log_probs,
                pos_label,
                epsilon=0
            )
            loss = loss_pos    # 包含一个元素的tensor

    else:
        # 训练模式
        probe_model.train()
        critic.train()
        with torch.enable_grad():
            pos_log_probs = probe_model(representation)  # [seq_len, num_pos_class]

            loss_pos, n_correct = compute_loss_and_accuracy(
                pos_log_probs,
                pos_label,
                epsilon=0
            )
            loss = loss_pos    # 包含一个元素的tensor

        torch.autograd.backward(loss)

    if return_acc:
        return loss.item(), n_correct, pos_label.size(-1)

    return loss.item()


def loss_evaluation_probe(discriminator, 
                            probe, 
                            critic, 
                            valid_iterator,
                            representation='hidden'):

    n_sents = 0
    sum_tokens = 0
    sum_correct = 0
    sum_loss = 0.0
    valid_iter = valid_iterator.build_generator()

    for batch in valid_iter:
        seqs_x, seqs_y, xy_label = batch
        n_sents += len(seqs_x)

        x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)

        loss, n_correct, n_token = compute_forward_probe(qe_model=discriminator,
                                                            probe_model=probe,
                                                            critic=critic,
                                                            seqs_x=x,
                                                            seqs_y=y,
                                                            pos_line=xy_label,
                                                            return_acc=True,
                                                            eval=True,
                                                            representation_type=representation)

        sum_loss += loss
        sum_correct += n_correct
        sum_tokens += n_token
    
    return float(sum_loss / n_sents), float(sum_correct / sum_tokens)


def compute_forward_probe_seqlen(qe_model,
                                  probe_model,
                                  critic_mse,
                                  seqs_x,
                                  seqs_y,
                                  len_label,
                                  eval=False,
                                  return_len_pred=False,
                                  representation_type="hidden",
                                  amplify_time=10):
    
    y_inp = seqs_y[:, :].contiguous()

    qe_model.eval()
    if representation_type == 'hidden':
        with torch.no_grad():
            # 提取hter对应表示 [hidden_dim]
            representation = qe_model(seqs_x, seqs_y, get_representation=True).squeeze(0)[1]
    elif representation_type == 'random':
        embed_dim = qe_model.decoder.embeddings.embeddings.weight.size(-1)
        representation = torch.randn(embed_dim).cuda()

    if eval:
        # 测试模式
        probe_model.eval()
        critic_mse.eval()
        with torch.no_grad():
            len_pred = probe_model(representation, return_score=True)  # [1]
            len_pred *= amplify_time      # 加速收敛
            loss_len = critic_mse(len_pred, len_label)
            loss = loss_len    # 包含一个元素的tensor

    else:
        # 训练模式
        probe_model.train()
        critic_mse.train()
        with torch.enable_grad():
            len_pred = probe_model(representation, return_score=True)  # [1]
            len_pred *= amplify_time     # 加速收敛
            loss_len = critic_mse(len_pred, len_label)
            loss = loss_len    # 包含一个元素的tensor

        torch.autograd.backward(loss)

    if return_len_pred:
        return loss.item(), len_pred.item()
    return loss.item()


def loss_evaluation_probe_seqlen(discriminator, 
                            probe, 
                            critic_mse, 
                            valid_iterator,
                            representation='hidden'):
    # 返回loss和pearsonr
    n_sents = 0
    sum_loss_len = 0.0
    valid_iter = valid_iterator.build_generator()

    total_len_pred = []
    total_len_gold = []
    for batch in valid_iter:
        seqs_x, seqs_y, seqs_len = batch
        n_sents += len(seqs_x)

        x, y = prepare_data_qe(seqs_x, seqs_y, cuda=Constants.USE_GPU)
        if Constants.USE_GPU:
            seqs_len = torch.tensor(seqs_len).cuda()

        loss, len_pred = compute_forward_probe_seqlen(qe_model=discriminator,
                                        probe_model=probe,
                                        critic_mse=critic_mse,
                                        seqs_x=x,
                                        seqs_y=y,
                                        len_label=seqs_len,
                                        eval=True,
                                        return_len_pred=True,
                                        representation_type=representation)

        sum_loss_len += loss
        total_len_pred.append(len_pred)
        total_len_gold.append(seqs_len.item())

    p = pearsonr(total_len_pred, total_len_gold)[0]
    
    return float(sum_loss_len / n_sents), p


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        temp_target = target
        temp_target.masked_fill_(pad_mask, 0)
        nll_loss = -lprobs.gather(dim=-1, index=temp_target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss


def compute_loss_and_accuracy(log_probs, target, epsilon, ignore_index=-100, reduce=True):
    loss = label_smoothed_nll_loss(
        log_probs,
        target,
        epsilon,
        ignore_index,
        reduce,
    )
    mask = target.ne(ignore_index)
    n_correct = torch.sum(
        log_probs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
    )
    return loss, n_correct


def cal_prediction(model, critic, critic_mse, src_input, mt_input, tag, score):
    y_inp = mt_input[:, :].contiguous()
    y_label = tag[:, 2:-1].contiguous()
    model.eval()
    critic.eval()
    
    log_probs, hter_pre = model(src_input, y_inp)

    #log_probs, score_pre = qe_out
    log_probs = log_probs.squeeze(0)
    #print(log_probs)

    prediction = log_probs.argmax(-1)-1
    #print(prediction)   # [seq_len], 0 ok  1 bad
    # TODO:目前只是返回tag预测，之后也可以加上句子分数预测，多个分析
    return prediction


def get_embedding(model, src_input, mt_input):

    embed_seqs_mt = model.decoder.embeddings.embeddings(mt_input)[0][2:-1]
    embed_seqs_src = model.encoder.embeddings.embeddings(src_input)[0][1:-1]

    return embed_seqs_mt, embed_seqs_src


def hvp_memory(y, w, v):
    """
    此函数用于纪念这揪心的debug，注释舍不得删
    """
    #print("============================= in hvp")
    #if len(w) != len(v):
    #    raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = mello_autograd(y, w)
    # 需要做的事：复制一个不带梯度的v

    # 经过new测试，如果v真的没有梯度，这个是非常好算的。。。
    # 那么唯一的原因可能是，v没有梯度只代表v不用考虑梯度，但elemwise_products依然考虑v的计算过程
    
    # print(v[0].grad) #None
    new = []
    #for item in v:
    #    new.append(torch.ones_like(item))
    #print('newgrad')
    #print(new[0].grad) # None
    # new = v.copy()  # 试一下这种方法能不能抛弃计算过程  不行
    for item in v:
        aa = item.clone().detach()   # 终于可了！！！！！
        new.append(aa)
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, new):
        if grad_elem is not None and v_elem is not None:
            print('v_elem grad')
            print(v_elem.grad)
            elemwise_products += torch.sum(grad_elem * v_elem)
    
    """
    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        if grad_elem is not None and v_elem is not None:
            #print('v_elem grad')
            #print(v_elem.grad)  # 也是none啊！！！为什么越来越慢！！
            elemwise_products += torch.sum(grad_elem * v_elem)
    """

    # 问题一定出在，v的计算越来越复杂上；但是v这里已经梯度none了，怎么还慢？
    """
    test1 = mello_autograd(y, w)
    test2 = mello_autograd(y, w) # 这几个重复计算都是一样的
    test3 = mello_autograd(y, w)  # 这个非常快
    print(test1[0])
    print(test3[0])
    
    #assert test1[0] == test3[0]
    return test3
    """
    # return first_grads
    # Second backprop
    # 最慢在这里
    #print("============================= in cal return grads")
    return_grads = mello_autograd(elemwise_products, w)
    # 唯一的原因，这个计算正在变得越来越复杂
    #print("============================= out cal return grads")
    """
    print("**************in hvp")
    print(len(w))             # 263
    print(len(v))             # 260
    print(len(first_grads))   # 260
    print(len(return_grads))  # 260
    print("(((((((((((9a")
    """
    #print("============================= out hvp")
    return return_grads


def hvp(y, w, v):
    """
    hvp函数：
     - y: loss
     - w: 模型参数params
     - v: 海塞矩阵所乘的向量
    """

    first_grads = mello_autograd(y, w)

    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        if grad_elem is not None and v_elem is not None:
            elemwise_products += torch.sum(grad_elem * v_elem)
    
    # 之前的版本
    # return_grads = mello_autograd(elemwise_products, w)

    # 不保留计算图
    grad = list(torch.autograd.grad(elemwise_products, w, create_graph=False, allow_unused=True))
    return_grads = [g for g in grad if g is not None]

    return return_grads


def cal_hter_logits(model,
                    sample,
                    gpu=False,):
    """
    计算 一个样本经过模型的hter_logits, 维度：[batch_size, d_model]
    """
    seqs_x, seqs_y = sample[:2]
    src_input, mt_input = prepare_data_qe(seqs_x, seqs_y, cuda=gpu)

    y_inp = mt_input[:, :].contiguous()
    model.eval()
    
    hter_logits = model(src_input, y_inp, get_hter_logits=True)  # [1, 256]
    
    return hter_logits


def get_feature(model,
                sample,
                gpu=False):
    """
    计算 一个样本经过模型得到的句子表示和词表示
    """
    model.eval()
    seqs_x, seqs_y = sample[:2]
    src_input, mt_input = prepare_data_qe(seqs_x, seqs_y, cuda=gpu)
    y_inp = mt_input[:, :].contiguous()
    # 获取表示
    dec_output = model(src_input, y_inp, get_representation=True)
    # 句子表示
    hter_logits = dec_output[:, 1]      # [batch_size, hidden_dim]
    # 词表示
    tags_logits = dec_output[:, 2:-1]   # [batch_size, seq_len, hidden_dim]
    tags_logits = tags_logits.reshape(-1, tags_logits.size(-1)) # [batch_size * seq_len, hidden_dim]

    return hter_logits, tags_logits


def get_hter_feature_bert(tokenizer,
                    bert_model,
                    mt_sent,
                    gpu=False):
    encoding = tokenizer(mt_sent, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids'].cuda()
    attention_mask = encoding['attention_mask'].cuda()
    outputs = bert_model(input_ids, attention_mask=attention_mask) 
    feature = outputs[0][:,0,:]   # [batch_size, hidden_dim]

    return feature


def save_feature(model,
                    data_iterator,
                    save_path_prefix,
                    key,
                    gpu=False):
    """
    保存一个数据集的句子级和词级feature到文件
    """
    # 初始化faiss index
    # sent faiss
    sent_res = faiss.StandardGpuResources()
    sent_flat_config = faiss.GpuIndexFlatConfig()
    sent_flat_config.device = 0    # 指定gpu卡号 似乎是指能看见的卡号而不是真的机器卡号
    sent_faiss_index = faiss.GpuIndexFlatL2(sent_res, 256, sent_flat_config)
    # token faiss
    token_res = faiss.StandardGpuResources()
    token_flat_config = faiss.GpuIndexFlatConfig()
    token_flat_config.device = 0    # 指定gpu卡号 似乎是指能看见的卡号而不是真的机器卡号
    token_faiss_index = faiss.GpuIndexFlatL2(token_res, 256, token_flat_config)


    data_iter = data_iterator.build_generator()
        
    uidx=0
    #hter_logits_list = []  # 样本个数长度，存放每句的句向量
    #tags_logits_list = []   # 数据集中所有词数长度，存放每个词的向量
    tags_token_id_list = [] 
    tags_sent_id_list = []

    for batch in data_iter:
        print(uidx)
        seqs_x, seqs_y, xy_label, xy_hter = batch[:4]
        src_input, mt_input, tag = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=gpu)
        y_inp = mt_input[:, :].contiguous()
        # 获取表示
        dec_output = model(src_input, y_inp, get_representation=True)
        # 句子表示
        hter_logits = dec_output[:, 1]      # [batch_size, hidden_dim]
        sent_faiss_index.add(hter_logits.cpu().detach().numpy())
        #hter_logits_list.append(hter_logits)
        # 词表示
        tags_logits = dec_output[:, 2:-1]   # [batch_size, seq_len, hidden_dim]
        token_faiss_index.add(tags_logits.reshape(-1, tags_logits.size(-1)).cpu().detach().numpy())  # 这里需要换一下维度
        # 换维度，保存
        #tags_logits_list.append(tags_logits.reshape(-1, tags_logits.size(-1)))
        # 这句话内词id
        tags_token_id = torch.range(0,tags_logits.size(-2)-1)  # tensor([0., 1., 2., 3., 4., 5., 6., 7., 8.])
        tags_token_id_list.append(tags_token_id)
        # 这句话的id
        tags_sent_id = torch.ones_like(tags_token_id) * uidx   # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0.]) 
        tags_sent_id_list.append(tags_sent_id)

        uidx += 1
        #if uidx == 2: break

    # 写入hter faiss
    hter_index_cpu = faiss.index_gpu_to_cpu(sent_faiss_index)
    faiss.write_index(hter_index_cpu, save_path_prefix + key + "_hter.faiss")
    # 写入tag faiss
    tags_index_cpu = faiss.index_gpu_to_cpu(token_faiss_index)
    faiss.write_index(tags_index_cpu, save_path_prefix + key + "_tag.faiss")

    # 写入记录id的两个文件
    tags_token_id_list = torch.cat(tags_token_id_list, dim=0)  # [num_token_in_dataset]
    tags_sent_id_list = torch.cat(tags_sent_id_list, dim=0)    # [num_token_in_dataset]
    torch.save(tags_token_id_list, save_path_prefix + key + "_tag_id_token.tensor")
    torch.save(tags_sent_id_list, save_path_prefix + key + "_tag_id_sent.tensor")


def save_feature_hter(model,
                    data_iterator,
                    save_path_prefix,
                    key,
                    gpu=False):
    """
    保存一个数据集的句子级feature到文件
    """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    # 初始化faiss index
    # sent faiss
    sent_res = faiss.StandardGpuResources()
    sent_flat_config = faiss.GpuIndexFlatConfig()
    sent_flat_config.device = 0    # 指定gpu卡号 似乎是指能看见的卡号而不是真的机器卡号
    sent_faiss_index = faiss.GpuIndexFlatL2(sent_res, 256, sent_flat_config)
    
    data_iter = data_iterator.build_generator()
        
    uidx=0

    for batch in data_iter:
        print(uidx)
        seqs_x, seqs_y = batch[:2]
        src_input, mt_input = prepare_data_qe(seqs_x, seqs_y, cuda=gpu)
        y_inp = mt_input[:, :].contiguous()
        # 获取表示
        dec_output = model(src_input, y_inp, get_representation=True)
        # 句子表示
        hter_logits = dec_output[:, 1]      # [batch_size, hidden_dim]
        sent_faiss_index.add(hter_logits.cpu().detach().numpy())

        uidx += 1
        # if uidx == 2: break

    # 写入hter faiss
    hter_index_cpu = faiss.index_gpu_to_cpu(sent_faiss_index)
    faiss.write_index(hter_index_cpu, save_path_prefix + key + "_hter.faiss")


def save_feature_hter_bert(tokenizer,
                    bert_model,
                    data_path,
                    save_path_prefix,
                    key,
                    gpu=False):
    """
    保存一个数据集的句子级feature到文件  使用bert算feature
    """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    # 初始化faiss index
    # sent faiss
    sent_res = faiss.StandardGpuResources()
    sent_flat_config = faiss.GpuIndexFlatConfig()
    sent_flat_config.device = 0    # 指定gpu卡号 似乎是指能看见的卡号而不是真的机器卡号
    sent_faiss_index = faiss.GpuIndexFlatL2(sent_res, 768, sent_flat_config)
    
    uidx=0

    with open(data_path, 'r', encoding='utf-8') as f:
        mt_lines = f.readlines()

    for line in mt_lines:
        print(uidx)
        mt_sent = "".join(line.strip('\n').split())    # 删除空格的作弊思路
        encoding = tokenizer(mt_sent, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].cuda()
        attention_mask = encoding['attention_mask'].cuda()
        outputs = bert_model(input_ids, attention_mask=attention_mask) 
        feature = outputs[0][:,0,:]   # [batch_size, hidden_dim]
        sent_faiss_index.add(feature.cpu().detach().numpy())

        uidx += 1
        # if uidx == 2: break

    # 写入hter faiss
    hter_index_cpu = faiss.index_gpu_to_cpu(sent_faiss_index)
    faiss.write_index(hter_index_cpu, save_path_prefix + key + "_hter.faiss")


def save_id_list(data_iterator,
                    save_path_prefix,
                    key,
                    gpu=False):
    """
    保存一个数据集的词级id list到文件
    """
    data_iter = data_iterator.build_generator()

    uidx=0
    
    tags_token_id_list = [] 
    tags_sent_id_list = []

    for batch in data_iter:
        print(uidx)
        seqs_x, seqs_y, xy_label, xy_hter = batch[:4]
        src_input, mt_input = prepare_data_qe(seqs_x, seqs_y, cuda=gpu)
        seq_len = mt_input.size(-1)-3
        # 这句话内词id
        tags_token_id = torch.range(0, seq_len-1)  # tensor([0., 1., 2., 3., 4., 5., 6., 7., 8.])
        tags_token_id_list.append(tags_token_id)
        # 这句话的id
        tags_sent_id = torch.ones_like(tags_token_id) * uidx   # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0.]) 
        tags_sent_id_list.append(tags_sent_id)

        uidx += 1
        #if uidx == 2: break

    # 写入记录id的两个文件
    tags_token_id_list = torch.cat(tags_token_id_list, dim=0)  # [num_token_in_dataset]
    tags_sent_id_list = torch.cat(tags_sent_id_list, dim=0)    # [num_token_in_dataset]
    torch.save(tags_token_id_list, save_path_prefix + key + "_tag_id_token.tensor")
    torch.save(tags_sent_id_list, save_path_prefix + key + "_tag_id_sent.tensor")


def save_feature_tag(model,
                    data_iterator,
                    save_path_prefix,
                    key,
                    gpu=False):
    """
    保存一个数据集的词级feature到文件
    """
    # 初始化faiss index
    # token faiss
    token_res = faiss.StandardGpuResources()
    token_flat_config = faiss.GpuIndexFlatConfig()
    token_flat_config.device = 0    # 指定gpu卡号 似乎是指能看见的卡号而不是真的机器卡号
    token_faiss_index = faiss.GpuIndexFlatL2(token_res, 256, token_flat_config)


    data_iter = data_iterator.build_generator()
        
    uidx=0

    for batch in data_iter:
        print(uidx)
        seqs_x, seqs_y, xy_label, xy_hter = batch[:4]
        src_input, mt_input, tag = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=gpu)
        y_inp = mt_input[:, :].contiguous()
        # 获取表示
        dec_output = model(src_input, y_inp, get_representation=True)
        # 词表示
        tags_logits = dec_output[:, 2:-1]   # [batch_size, seq_len, hidden_dim]
        token_faiss_index.add(tags_logits.reshape(-1, tags_logits.size(-1)).cpu().detach().numpy())  # 这里需要换一下维度

        uidx += 1
        if uidx == 100000: break
        #assert 1==2

    # 写入tag faiss
    tags_index_cpu = faiss.index_gpu_to_cpu(token_faiss_index)
    faiss.write_index(tags_index_cpu, save_path_prefix + key + "_tag.faiss_10w")


def cal_loss_z_hter(model,
                    critic_mse,
                    sample,
                    gpu=False,
                    test_token_id=None,
                    result_instead_of_loss=False):
    """
    计算 一个样本的hter的loss（或是 用来计算loss的结果）
    """
    seqs_x, seqs_y, xy_label, xy_hter = sample
    src_input, mt_input, tag = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=gpu)
    if gpu:
        score = torch.tensor(xy_hter)
        score = score.cuda()

    y_inp = mt_input[:, :].contiguous()
    y_label = tag[:, 2:-1].contiguous()
    model.eval()
    critic_mse.eval()
    
    log_probs, hter_pre = model(src_input, y_inp)
    loss_score = critic_mse(hter_pre, score)   #[1]

    if result_instead_of_loss:
        loss_score = hter_pre.squeeze().squeeze()

    if test_token_id is not None:
        # 把test_token_id在句子中的id转化为在vocab中的id
        test_token_id = mt_input[0][test_token_id + 2]
        return loss_score, test_token_id

    return loss_score


def cal_loss_z_token(model,
                    critic,
                    sample,
                    token_id=None,
                    gpu=False,):
    """
    计算 token_loss
    如果 token_id=None，就计算这句话的每个token_loss，返回列表
    """
    seqs_x, seqs_y, xy_label, xy_hter = sample
    src_input, mt_input, tag = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=gpu)

    y_inp = mt_input[:, :].contiguous()
    y_label = tag[:, 2:-1].contiguous()
    model.eval()
    critic.eval()
    
    # log_probs [batch_size, seq_len, num_choice]   batch_size=1
    log_probs, hter_pre = model(src_input, y_inp)
    # loss_token  [1, seq_len]  是nllloss
    loss_token = critic(inputs=log_probs, labels=y_label, no_reduction=True, reduce=False)
    
    if token_id is not None:
        # 取该token位置对应的的loss_score
        loss_score = loss_token[0][token_id]
    else:
        loss_score = loss_token.squeeze(0)   # [seq_len]

    return loss_score


def cal_loss_z_both(model,
                    critic,
                    critic_mse,
                    sample,
                    gpu=False):
    """
    计算 一个样本的hter的loss（或是 用来计算loss的结果）
    """
    seqs_x, seqs_y, xy_label, xy_hter = sample
    src_input, mt_input, tag = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=gpu)
    if gpu:
        score = torch.tensor(xy_hter)
        score = score.cuda()

    y_inp = mt_input[:, :].contiguous()
    y_label = tag[:, 2:-1].contiguous()
    model.eval()
    critic.eval()
    critic_mse.eval()
    
    log_probs, hter_pre = model(src_input, y_inp)
    loss_token = critic(inputs=log_probs, labels=y_label, reduce=True) # [1]
    loss_hter = critic_mse(hter_pre, score)   #[1]
    loss_both =loss_token + loss_hter  #[1]

    return loss_both


def cal_grad_z_hter(model,
                    critic_mse,
                    sample,
                    gpu=False,
                    result_instead_of_loss=False):
    """
    计算 一个样本的hter的loss 对 模型参数θ的梯度
    """
    loss_score = cal_loss_z_hter(model,
                                    critic_mse,
                                    sample,
                                    gpu,
                                    result_instead_of_loss=result_instead_of_loss)

    params = [ p for p in model.parameters() if p.requires_grad ]
    return mello_autograd(loss_score, params)   


def cal_grad_z_token(model,
                    critic,
                    sample,
                    gpu=False,):
    """
    计算 一个样本的 [token_loss 对 模型参数θ的梯度]
    """
    loss_score_sentence = cal_loss_z_token(model,
                                        critic,
                                        sample,
                                        gpu=gpu,)
    params = [ p for p in model.parameters() if p.requires_grad ]
    # return [mello_autograd(loss_score, params) for loss_score in loss_score_sentence] # TODO  爆显存
    # 或许改一下呢 可能是这个要返回的太多了，应该只用loss就好，不用这个函数！
    """ 
    sent_grad = []
    for loss_score in loss_score_sentence:
        token_grad = mello_autograd(loss_score, params)
        sent_grad.append(token_grad)
    return sent_grad
    """
    # 测试，如果只需要一个token grad，会吗 就不会了！
    return [mello_autograd(loss_score_sentence[0], params)]
    

def cal_grad_z_both(model,
                    critic,
                    critic_mse,
                    sample,
                    gpu=False):
    """
    计算 一个样本的hter的loss 对 模型参数θ的梯度
    """
    loss_score = cal_loss_z_both(model,
                                    critic,
                                    critic_mse,
                                    sample,
                                    gpu)

    params = [ p for p in model.parameters() if p.requires_grad ]
    return mello_autograd(loss_score, params) 


def mello_autograd(loss, params):
    """
    包装autograd计算：排除模型中0值参数的影响，并转化为list
    """
    grad = list(torch.autograd.grad(loss, params, create_graph=True, allow_unused=True))
    grad = [g for g in grad if g is not None]

    return grad


def cal_s_test_hter(model,
                    critic_mse,
                    test_sample,
                    train_dataset,
                    test_token_id,
                    recursion_depth=10,
                    gpu=False,
                    damp=0.01,
                    scale=25.0,
                    result_instead_of_loss=False):
    """
    计算一个test token样本的s_test,（loss是hter的loss）
    公式：【hter_loss 对 一个test_token 的梯度 对 模型参数θ的梯度】×【loss对模型参数θ的海塞矩阵的逆】
    利用stochastic estimation随机采样
    """
    # hter_loss 对 一个test_token 的梯度w_e，计算方法：gradient*input
    loss_score, test_token_id = cal_loss_z_hter(model,
                                                critic_mse,
                                                test_sample,
                                                gpu,
                                                test_token_id,  # test_token_id从在本句中的id转化为vocab中的index
                                                result_instead_of_loss=result_instead_of_loss)  
    # 简单sum的方式
    #w_e = torch.autograd.grad(loss_score, model.decoder.embeddings.embeddings.weight, create_graph=True)[0][test_token_id].sum()
    
    # gi不给grad加权的方式
    
    grad = torch.autograd.grad(loss_score, model.decoder.embeddings.embeddings.weight, create_graph=True)[0][test_token_id]
    embed = model.decoder.embeddings.embeddings.weight[test_token_id]
    # gradient·input算内积的方式
    w_e = torch.dot(grad, embed)
    # gradient·input算L2norm的方式
    #w_e = torch.linalg.norm(grad * embed)
    

    # gi给grad加权100倍的方式
    """
    grad = torch.autograd.grad(loss_score, model.decoder.embeddings.embeddings.weight, create_graph=True)[0][test_token_id]
    embed = model.decoder.embeddings.embeddings.weight[test_token_id]
    w_e = torch.dot(grad*100, embed)
    """
    
    """
    # 试图优化w_e计算，没搞明白
    w_e = torch.autograd.grad(loss_score, model.decoder.embeddings.embeddings.weight, create_graph=True)[0][test_token_id]
    print(w_e[-10:])
    print(model.decoder.embeddings.embeddings.weight[test_token_id])
    print(model.decoder.embeddings.embeddings.weight[test_token_id].size())
    w_e = torch.autograd.grad(loss_score, model.decoder.embeddings.embeddings.weight[test_token_id], create_graph=True, allow_unused=True)
    print(w_e[-10:])
    """

    # w_e对模型参数θ的梯度v
    params = [ p for p in model.parameters() if p.requires_grad ]
    v = mello_autograd(w_e, params)
    
    # 近似计算公式，详见论文中的stochastic estimation部分
    # 随机采样recursion_depth个训练样本以代表整体，迭代计算直到h_estimate稳定
    # 【icml2017】Understanding Black-box Predictions via Influence Functions
    # TODO：采样多少样本比较合适？？
    h_estimate = v.copy()

    sample_ids = np.random.choice(range(len(train_dataset)), size=recursion_depth, replace=False)
    for sample_id in sample_ids:
        #new_h_estimate = []             # 要新建一个，无法原地修改
        #for item in h_estimate:
        #    new_h_estimate.append(item.clone().detach())
        #h_estimate = new_h_estimate
        print('sample id:', sample_id)
        train_sample = train_dataset[sample_id]
        # TODO 可能有梯度
        loss_score = cal_loss_z_hter(model,
                                        critic_mse,
                                        train_sample,
                                        gpu,
                                        result_instead_of_loss=result_instead_of_loss)
        grad_train = mello_autograd(loss_score, params)
        hv = hvp(loss_score, params, h_estimate)
        # 更新h_estimate
        # 调整版公式
        with torch.no_grad():
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)
            ]
        
        # 论文原版公式
        """
        h_estimate = [
            _v + _h_e - _hv / 10
            for _v, _h_e, _hv in zip(v, h_estimate, hv)
        ]
        """

            
    return h_estimate
    

def cal_s_test_hter_not_saliency(model,
                                critic_mse,
                                test_sample,
                                train_dataset,
                                test_token_id,
                                recursion_depth=10,
                                gpu=False,
                                damp=0.01,
                                scale=25.0,
                                result_instead_of_loss=False):
    """
    计算一个test token样本的s_test,（loss是hter的loss）
    这个directly函数，s_test版本是【训练集对结果的影响函数】，而不是【训练集对显著性的影响函数】
    公式：【hter_loss 对 模型参数θ的梯度】×【loss对模型参数θ的海塞矩阵的逆】
    利用stochastic estimation随机采样
    """
    # hter_loss 对 一个test样本 的梯度w_e，计算方法：gradient*input

    # 得到一个test样本的hter loss
    loss_score, test_token_id = cal_loss_z_hter(model,
                                                critic_mse,
                                                test_sample,
                                                gpu,
                                                test_token_id,  # test_token_id从在本句中的id转化为vocab中的index
                                                result_instead_of_loss=result_instead_of_loss)  

    # hter_loss 对模型参数θ的梯度v
    params = [ p for p in model.parameters() if p.requires_grad ]
    v = mello_autograd(loss_score, params) # test grad
    
    # 近似计算公式，详见论文中的stochastic estimation部分
    # 随机采样recursion_depth个训练样本以代表整体，迭代计算直到h_estimate稳定
    # 【icml2017】Understanding Black-box Predictions via Influence Functions
    # TODO：采样多少样本比较合适？？
    h_estimate = v.copy()

    sample_ids = np.random.choice(range(len(train_dataset)), size=recursion_depth, replace=False)
    for sample_id in sample_ids:
        #new_h_estimate = []             # 要新建一个，无法原地修改
        #for item in h_estimate:
        #    new_h_estimate.append(item.clone().detach())
        #h_estimate = new_h_estimate
        print('sample id:', sample_id)
        train_sample = train_dataset[sample_id]
        loss_score = cal_loss_z_hter(model,
                                        critic_mse,
                                        train_sample,
                                        gpu,
                                        result_instead_of_loss=result_instead_of_loss)
        grad_train = mello_autograd(loss_score, params)
        hv = hvp(loss_score, params, h_estimate)
        # 更新h_estimate
        # 调整版公式
        with torch.no_grad():
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)
            ]
            
    return h_estimate


def cal_s_test_token(model,
                    critic,
                    test_sample,
                    train_dataset,
                    test_token_id,
                    recursion_depth=10,
                    gpu=False,
                    damp=0.01,
                    scale=25.0,
                    saliency_token=-1):
    """
    计算一个test token样本的s_test,（loss是某个token预测的loss）
    公式：【one_token_loss 对 一个test_token 的梯度 对 模型参数θ的梯度】×【one_token_loss对模型参数θ的海塞矩阵的逆】
    利用stochastic estimation随机采样
    """
    # 此函数已作废
    if saliency_token == -1:
        saliency_token = test_token_id
    # one_token_loss 对 一个test_token 的梯度w_e，计算方法：gradient*input
    loss_score, test_token_id = cal_loss_z_token(model,
                                                critic,
                                                test_sample,
                                                saliency_token,
                                                gpu,
                                                test_token_id,)  # test_token_id从在本句中的id转化为vocab中的index
    print("here************")
    print(loss_score)
    # 简单sum的方式
    #w_e = torch.autograd.grad(loss_score, model.decoder.embeddings.embeddings.weight, create_graph=True)[0][test_token_id].sum()
    
    # gi不给grad加权的方式
    
    grad = torch.autograd.grad(loss_score, model.decoder.embeddings.embeddings.weight, create_graph=True)[0][test_token_id]
    embed = model.decoder.embeddings.embeddings.weight[test_token_id]
    w_e = torch.dot(grad, embed)
    

    # gi给grad加权100倍的方式
    """
    grad = torch.autograd.grad(loss_score, model.decoder.embeddings.embeddings.weight, create_graph=True)[0][test_token_id]
    embed = model.decoder.embeddings.embeddings.weight[test_token_id]
    w_e = torch.dot(grad*100, embed)
    """

    # w_e对模型参数θ的梯度v
    params = [ p for p in model.parameters() if p.requires_grad ]
    v = mello_autograd(w_e, params)
    
    # 近似计算公式，详见论文中的stochastic estimation部分
    # 随机采样recursion_depth个训练样本以代表整体，迭代计算直到h_estimate稳定
    # 【icml2017】Understanding Black-box Predictions via Influence Functions
    # TODO：采样多少样本比较合适？？
    h_estimate = v.copy()

    sample_ids = np.random.choice(range(len(train_dataset)), size=recursion_depth, replace=False)
    for sample_id in sample_ids:
        #new_h_estimate = []             # 要新建一个，无法原地修改
        #for item in h_estimate:
        #    new_h_estimate.append(item.clone().detach())
        #h_estimate = new_h_estimate
        print('sample id:', sample_id)
        train_sample = train_dataset[sample_id]
        loss_score = cal_loss_z_hter(model,
                                        critic_mse,
                                        train_sample,
                                        gpu,)
        grad_train = mello_autograd(loss_score, params)
        hv = hvp(loss_score, params, h_estimate)
        # 更新h_estimate
        # 调整版公式
        with torch.no_grad():
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)
            ]
   
    return h_estimate


def cal_s_test_token_not_saliency(model,
                                critic,
                                test_sample,
                                train_dataset,
                                test_token_id,
                                recursion_depth=10,
                                gpu=False,
                                damp=0.01,
                                scale=25.0,):
    """
    计算一个test token的s_test,（loss是该test_token的loss）
    这个directly函数，s_test版本是【训练集对结果的影响函数】，而不是【训练集对显著性的影响函数】
    公式：【test_token_loss 对 模型参数θ的梯度】×【loss对模型参数θ的海塞矩阵的逆】
    利用stochastic estimation随机采样
    """

    # 得到test_token_loss 标量
    loss_score = cal_loss_z_token(model,
                                    critic,
                                    test_sample,
                                    token_id=test_token_id,
                                    gpu=gpu,)
    #print("test_token_loss")
    #print(loss_score)

    # test_token_loss 对模型参数θ的梯度v
    params = [ p for p in model.parameters() if p.requires_grad ]
    v = mello_autograd(loss_score, params) # test_token_loss 对模型参数求梯度
    #hvp_log1 = "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/directqe_enzh_token_sample7000_scale500_test/hvp.log1"
    #hvp_log2 = "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/directqe_enzh_token_sample7000_scale500_test/hvp.log2"
    #hvp_log3 = "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/directqe_enzh_token_sample7000_scale500_test/hvp.log3"
    #hvp_log4 = "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/directqe_enzh_token_sample7000_scale500_test/hvp.log4"
    # 近似计算公式，详见论文中的stochastic estimation部分
    # 随机采样recursion_depth个训练样本以代表整体，迭代计算直到h_estimate稳定
    # 【icml2017】Understanding Black-box Predictions via Influence Functions
    # TODO：采样多少样本比较合适？？
    h_estimate = v.copy()
    #with open(hvp_log1, 'w', encoding='utf-8') as f:
    #    f.write(str(h_estimate))
    # 采样训练样本
    sample_ids = np.random.choice(range(len(train_dataset)), size=recursion_depth, replace=False)
    for sample_id in sample_ids:
        print('sample id:', sample_id)
        train_sample = train_dataset[sample_id]
        # 依次计算该训练样本中 每个token_loss，保留列表写入文件
        # loss_score_sentence  [sent_len]
        loss_score_sentence = cal_loss_z_token(model,
                                                critic,
                                                train_sample,
                                                gpu=gpu,)
        #print('loss_score_sentence')
        #print(loss_score_sentence)
        # 每个token_loss的平均值作为这一句的loss
        loss_score = sum(loss_score_sentence) / len(loss_score_sentence)
        grad_train = mello_autograd(loss_score, params)
        #with open(hvp_log4, 'w', encoding='utf-8') as f:
        #    f.write(str(grad_train))
        hv = hvp(loss_score, params, h_estimate)
        #with open(hvp_log2, 'w', encoding='utf-8') as f:
        #    f.write(str(hv))
        # 更新h_estimate
        # 调整版公式
        with torch.no_grad():
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)
            ]
        # 补除已改
        h_estimate = [h / scale for h in h_estimate]
        #with open(hvp_log3, 'w', encoding='utf-8') as f:
        #    f.write(str(h_estimate))
        #assert 1==2
    return h_estimate


def cal_s_test_both(model,
                    critic,
                    critic_mse,
                    test_sample,
                    train_dataset,
                    recursion_depth=10,
                    gpu=False,
                    damp=0.01,
                    scale=25.0):
    """
    计算一个test token样本的s_test, using both hter_loss and token_loss
    公式：【test_loss 对 模型参数θ的梯度】×【avg_train_loss对模型参数θ的海塞矩阵的逆】
    利用stochastic estimation随机采样，迭代求解上式
    """
    # cal test_loss = hter_loss + token_loss
    loss_score = cal_loss_z_both(model,
                                critic,
                                critic_mse,
                                test_sample,
                                gpu=gpu)  

    # test_loss 对模型参数θ的梯度v
    params = [ p for p in model.parameters() if p.requires_grad ]
    """
    param_log = "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/occupy/param_print2"
    model_log = "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/occupy/model_print"
    
    with open(param_log, 'w', encoding='utf-8') as p_f:
        for p in model.named_parameters():
            p_f.write(str(p))
    
    with open(model_log, 'w', encoding='utf-8') as m_f:
            m_f.write(str(model))
    """
    #v = mello_autograd(loss_score, params) # test grad
    v = list(torch.autograd.grad(loss_score, params, create_graph=False, allow_unused=True))
    v = [g for g,p in zip(v,params) if g is not None]
    
    
    #fff1 = "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/occupy/grad_check1"
    #fff2 = "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/occupy/grad_check2"
    """
    with open(fff1, 'w', encoding='utf-8') as ff:
        ff.write(str(v))
        
    v = list(torch.autograd.grad(loss_score, params, create_graph=False, allow_unused=True))
    v = [g for g,p in zip(v,params) if g is not None]

    with open(fff2, 'w', encoding='utf-8') as ff:
        ff.write(str(v))
    assert 1==2
    """
    
    # 近似计算公式，详见论文中的stochastic estimation部分
    # 随机采样recursion_depth个训练样本以代表整体，迭代计算直到h_estimate稳定
    # 【icml2017】Understanding Black-box Predictions via Influence Functions
    # TODO：采样多少样本比较合适？？
    h_estimate = v.copy()

    sample_ids = np.random.choice(range(len(train_dataset)), size=recursion_depth, replace=False)
    for sample_id in sample_ids:
        print('sample id:', sample_id)
        train_sample = train_dataset[sample_id]
        # TODO 可能有梯度
        loss_score = cal_loss_z_both(model,
                                    critic,
                                    critic_mse,
                                    train_sample,
                                    gpu=gpu)
        grad_train = mello_autograd(loss_score, params)
        hv = hvp(loss_score, params, h_estimate)
        # 更新h_estimate
        # 调整版公式
        with torch.no_grad():
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)
            ]

        # 补除已改
        h_estimate = [h / scale for h in h_estimate]
    """
    with open(fff2, 'w', encoding='utf-8') as ff:
        ff.write(str(h_estimate))
    assert 1==2
    """

    return h_estimate


def cal_gradients_param(model, critic, critic_mse, src_input, mt_input, tag, score, i_frac=1):
    # 这个对embedding求的梯度是针对模型参数的
    # 计算句子级分数、token级分数 对当前i_frac的mt embedding、src embedding的梯度
    y_inp = mt_input[:, :].contiguous()
    y_label = tag[:, 2:-1].contiguous()
    model.eval()
    critic.eval()
    critic_mse.eval()
    
    log_probs, hter_pre = model(src_input, y_inp, i_frac=i_frac)
    loss_score = critic_mse(hter_pre, score)   #[1]
    loss_tag = critic(inputs=log_probs, labels=y_label, normalization=1.0, reduce=False, no_reduction=True)
    loss_tag = loss_tag.squeeze(0)    # [seq_len]

    # ============================================= 句子级分数的梯度
    model.zero_grad()
    loss_score.backward(retain_graph=True) # TODO retain_graph=True 是做什么的？

    # 句子级分数对mt embedding的梯度
    embed_grads_sent_mt = []
    for mt_token in mt_input[0][2:-1]:
        embed_grad_sent_mt = model.decoder.embeddings.embeddings.weight.grad[mt_token] 
        embed_grads_sent_mt.append(embed_grad_sent_mt)
    embed_grads_sent_mt = torch.stack(embed_grads_sent_mt,0)     # [mt_seq_len, embed_dim]

    # 句子级分数对src embedding的梯度
    embed_grads_sent_src = []
    for src_token in src_input[0][1:-1]:
        embed_grad_sent_src = model.encoder.embeddings.embeddings.weight.grad[src_token] 
        embed_grads_sent_src.append(embed_grad_sent_src)
    embed_grads_sent_src = torch.stack(embed_grads_sent_src,0)     # [src_seq_len, embed_dim]

    # ============================================= token级分数的梯度
    embed_grads_tokens_mt = []
    embed_grads_tokens_src = []
    for i in range(loss_tag.size(-1)):   # 每个token位置
        model.zero_grad()   # 应该放哪里？可以试一下
        loss_tag[i].backward(retain_graph=True)     # 只能一个位置一个位置回传梯度
        embed_grads_token_mt = []   # 这个token位置的loss对mt每个embedding的梯度
        embed_grads_token_src = []  # 这个token位置的loss对src每个embedding的梯度
        for mt_token in mt_input[0][2:-1]:
            embed_grad = model.decoder.embeddings.embeddings.weight.grad[mt_token] 
            embed_grads_token_mt.append(embed_grad)
        for src_token in src_input[0][1:-1]:
            embed_grad = model.encoder.embeddings.embeddings.weight.grad[src_token] 
            embed_grads_token_src.append(embed_grad)
        embed_grads_token_mt = torch.stack(embed_grads_token_mt,0)     # [mt_seq_len, embed_dim]
        embed_grads_token_src = torch.stack(embed_grads_token_src,0)     # [src_seq_len, embed_dim]
        
        embed_grads_tokens_mt.append(embed_grads_token_mt)
        embed_grads_tokens_src.append(embed_grads_token_src)
    # 矩阵：每个mt(或src)token embedding 对每个词标签预测的贡献
    embed_grads_tokens_mt = torch.stack(embed_grads_tokens_mt,1)    # [mt_seq_len, mt_seq_len, embed_dim]
    embed_grads_tokens_src = torch.stack(embed_grads_tokens_src,1)  # [src_seq_len, mt_seq_len, embed_dim]

    return embed_grads_sent_mt, embed_grads_sent_src, embed_grads_tokens_mt, embed_grads_tokens_src


def cal_gradients_mid_loss(model, critic, critic_mse, src_input, mt_input, tag, score, i_frac=1):
    # 这个对embedding求的梯度是针对中间结果的
    # 计算句子分数、token标签预测 的loss 对当前i_frac的mt embedding、src embedding的梯度
    # 这个版本相对下一个的区别是，这个用loss回传梯度，下一个用预测结果回传梯度
    y_inp = mt_input[:, :].contiguous()
    y_label = tag[:, 2:-1].contiguous()
    model.eval()
    critic.eval()
    critic_mse.eval()
    
    log_probs, hter_pre, enc_grads, dec_grads = model(src_input, y_inp, i_frac=i_frac, return_grads=True)
    loss_score = critic_mse(hter_pre, score)   #[1]
    loss_tag = critic(inputs=log_probs, labels=y_label, normalization=1.0, reduce=False, no_reduction=True)
    loss_tag = loss_tag.squeeze(0)    # [seq_len]

    # ============================================= 句子级分数预测的梯度
    model.zero_grad()
    loss_score.backward(retain_graph=True) # TODO retain_graph=True 是做什么的？

    # 句子级分数对mt embedding的梯度
    embed_grads_sent_mt = dec_grads['embeddings'][0][2:-1]        # [mt_seq_len, embed_dim]

    # 句子级分数对src embedding的梯度
    embed_grads_sent_src = enc_grads['embeddings'][0][1:-1]       # [src_seq_len, embed_dim]

    # ============================================= token级标记预测的梯度
    embed_grads_tokens_mt = []
    embed_grads_tokens_src = []
    for i in range(loss_tag.size(-1)):   # 每个token位置
        model.zero_grad()   # 应该放哪里？可以试一下
        loss_tag[i].backward(retain_graph=True)     # 只能一个位置一个位置回传梯度
        embed_grads_token_mt = dec_grads['embeddings'][0][2:-1]  # 这个token位置的loss对mt每个embedding的梯度
        embed_grads_token_src = enc_grads['embeddings'][0][1:-1] # 这个token位置的loss对src每个embedding的梯度

        embed_grads_tokens_mt.append(embed_grads_token_mt)
        embed_grads_tokens_src.append(embed_grads_token_src)
    # 矩阵：每个mt(或src)token embedding 对每个词标签预测的贡献
    embed_grads_tokens_mt = torch.stack(embed_grads_tokens_mt,1)    # [mt_seq_len, mt_seq_len, embed_dim]
    embed_grads_tokens_src = torch.stack(embed_grads_tokens_src,1)  # [src_seq_len, mt_seq_len, embed_dim]

    return embed_grads_sent_mt, embed_grads_sent_src, embed_grads_tokens_mt, embed_grads_tokens_src


def cal_gradients_mid(model, src_input, mt_input, i_frac=1):
    # 这个对embedding求的梯度是针对中间结果的
    # 计算句子分数、token标签预测 对当前i_frac的mt embedding、src embedding的梯度
    y_inp = mt_input[:, :].contiguous()
    model.eval()
    log_probs, hter_pre, enc_grads, dec_grads = model(src_input, y_inp, i_frac=i_frac, return_grads=True)
    log_probs, _ = log_probs.squeeze(0).max(dim = -1)
    log_probs = - log_probs    # [mt_seq_len]

    # ============================================= 句子级分数预测的梯度
    model.zero_grad()
    hter_pre.squeeze(0).backward(retain_graph=True)  # TODO retain_graph=True 是做什么的？

    # 句子级分数对mt embedding的梯度
    embed_grads_sent_mt = dec_grads['embeddings'][0][2:-1]        # [mt_seq_len, embed_dim]

    # 句子级分数对src embedding的梯度
    embed_grads_sent_src = enc_grads['embeddings'][0][1:-1]       # [src_seq_len, embed_dim]

    # ============================================= token级标记预测的梯度
    embed_grads_tokens_mt = []
    embed_grads_tokens_src = []
    for i in range(log_probs.size(-1)):   # 每个token位置
        model.zero_grad()   # 应该放哪里？可以试一下
        log_probs[i].backward(retain_graph=True)   # 只能一个位置一个位置回传梯度
        embed_grads_token_mt = dec_grads['embeddings'][0][2:-1]  # 这个token位置的loss对mt每个embedding的梯度
        embed_grads_token_src = enc_grads['embeddings'][0][1:-1] # 这个token位置的loss对src每个embedding的梯度
        embed_grads_tokens_mt.append(embed_grads_token_mt)
        embed_grads_tokens_src.append(embed_grads_token_src)
    # 矩阵：每个mt(或src)token embedding 对每个词标签预测的贡献
    embed_grads_tokens_mt = torch.stack(embed_grads_tokens_mt,1)    # [mt_seq_len, mt_seq_len, embed_dim]
    embed_grads_tokens_src = torch.stack(embed_grads_tokens_src,1)  # [src_seq_len, mt_seq_len, embed_dim]

    return embed_grads_sent_mt, embed_grads_sent_src, embed_grads_tokens_mt, embed_grads_tokens_src


def one_gradients(model, 
                    src_input, 
                    mt_input):
    # 只算这一点的梯度，不加积分; gradients*input

    # embed_mt [mt_seq_len, embed_dim] 
    # embed_src [src_seq_len, embed_dim]
    embed_mt, embed_src = get_embedding(model, 
                                        src_input, 
                                        mt_input)

    # embed_grads_sent_mt [mt_seq_len, embed_dim] 
    # embed_grads_sent_src [src_seq_len, embed_dim]
    # embed_grads_tokens_mt [mt_seq_len, mt_seq_len, embed_dim]
    # embed_grads_tokens_src [src_seq_len, mt_seq_len, embed_dim]
    embed_grads_sent_mt, embed_grads_sent_src, \
        embed_grads_tokens_mt, embed_grads_tokens_src = cal_gradients_mid(model, 
                                                                        src_input, 
                                                                        mt_input, )

    # 手动放大梯度的影响
    """
    def amplify(x, n):
        return x*n
    embed_grads_sent_mt = amplify(embed_grads_sent_mt, 1000)
    embed_grads_sent_src = amplify(embed_grads_sent_src, 1000)
    embed_grads_tokens_mt = amplify(embed_grads_tokens_mt, 1000)
    embed_grads_tokens_src = amplify(embed_grads_tokens_src, 1000)
    """

    # 每个mt、src token对句子分数的贡献（softmax之前）
    # =========== 点积，加和
    #gi_sent_mt = torch.mul(embed_grads_sent_mt, embed_mt).sum(-1)   # [mt_seq_len]
    #gi_sent_src = torch.mul(embed_grads_sent_src, embed_src).sum(-1)   # [src_seq_len]
    # =========== L2
    #gi_sent_mt = torch.square(torch.mul(embed_grads_sent_mt, embed_mt)).sum(-1)   # [mt_seq_len]
    #gi_sent_src = torch.square(torch.mul(embed_grads_sent_src, embed_src)).sum(-1)   # [src_seq_len]
    # =========== L1
    gi_sent_mt = torch.abs(torch.mul(embed_grads_sent_mt, embed_mt)).sum(-1)   # [mt_seq_len]
    gi_sent_src = torch.abs(torch.mul(embed_grads_sent_src, embed_src)).sum(-1)   # [src_seq_len]

    gi_sent_mt = F.normalize(gi_sent_mt, dim=-1)
    gi_sent_src = F.normalize(gi_sent_src, dim=-1)

    embed_mt_matrix = embed_mt.unsqueeze(dim=1)
    embed_src_matrix = embed_src.unsqueeze(dim=1)
    # =========== 点积，加和
    #gi_token_mt = torch.mul(embed_grads_tokens_mt, embed_mt_matrix).sum(-1)      # [mt_seq_len, mt_seq_len] 
    #gi_token_src = torch.mul(embed_grads_tokens_src, embed_src_matrix).sum(-1)   # [src_seq_len, mt_seq_len] 

    # =========== L2
    #gi_token_mt = torch.square(torch.mul(embed_grads_tokens_mt, embed_mt_matrix)).sum(-1)      # [mt_seq_len, mt_seq_len] 
    #gi_token_src = torch.square(torch.mul(embed_grads_tokens_src, embed_src_matrix)).sum(-1)   # [src_seq_len, mt_seq_len] 
    # =========== L1
    gi_token_mt = torch.abs(torch.mul(embed_grads_tokens_mt, embed_mt_matrix)).sum(-1)      # [mt_seq_len, mt_seq_len] 
    gi_token_src = torch.abs(torch.mul(embed_grads_tokens_src, embed_src_matrix)).sum(-1)   # [src_seq_len, mt_seq_len]
    
    # 大小比较
    """
    print('embed**************')
    print(embed_grads_sent_mt.size())
    print(embed_grads_sent_mt)
    print(embed_mt.size())
    print(embed_mt)
    assert 1==2
    """

    gi_token_mt = F.normalize(gi_token_mt, dim=0)
    gi_token_src = F.normalize(gi_token_src, dim=0)

    # 看起来太平均了，可能是值太小的缘故？？

    return gi_sent_mt, gi_token_mt, \
                gi_sent_src, gi_token_src


def one_gradients_srcmt(model, 
                        src_input, 
                        mt_input):
    # 只算这一点的梯度，不加积分; gradients*input

    # embed_mt [mt_seq_len, embed_dim] 
    # embed_src [src_seq_len, embed_dim]
    embed_mt, embed_src = get_embedding(model, 
                                        src_input, 
                                        mt_input)
    embed_srcmt = torch.cat((embed_src, embed_mt), 0)   # [src_seq_len + mt_seq_len, embed_dim]

    # embed_grads_sent_mt [mt_seq_len, embed_dim] 
    # embed_grads_sent_src [src_seq_len, embed_dim]
    # embed_grads_tokens_mt [mt_seq_len, mt_seq_len, embed_dim]
    # embed_grads_tokens_src [src_seq_len, mt_seq_len, embed_dim]
    embed_grads_sent_mt, embed_grads_sent_src, \
        embed_grads_tokens_mt, embed_grads_tokens_src = cal_gradients_mid(model, 
                                                                        src_input, 
                                                                        mt_input, )
    embed_grads_sent_srcmt = torch.cat((embed_grads_sent_src, embed_grads_sent_mt), 0) # [src_seq_len + mt_seq_len, embed_dim]
    embed_grads_tokens_srcmt = torch.cat((embed_grads_tokens_src, embed_grads_tokens_mt), 0) # [src_seq_len + mt_seq_len, mt_seq_len, embed_dim]

    # =========== 每个mt、src token对句子分数的贡献
    gi_sent_srcmt = torch.abs(torch.mul(embed_grads_sent_srcmt, embed_srcmt)).sum(-1)   # [src_seq_len + mt_seq_len]
    #gi_sent_srcmt = torch.mul(embed_grads_sent_srcmt, embed_srcmt).sum(-1)   # [src_seq_len + mt_seq_len]
    gi_sent_srcmt = F.normalize(gi_sent_srcmt, dim=-1)
    #gi_sent_srcmt = (gi_sent_srcmt - torch.min(gi_sent_srcmt)) / (torch.max(gi_sent_srcmt) - torch.min(gi_sent_srcmt))

    # =========== 每个mt、src token对词标签的贡献
    embed_srcmt_matrix = embed_srcmt.unsqueeze(dim=1)
    gi_token_srcmt = torch.abs(torch.mul(embed_grads_tokens_srcmt, embed_srcmt_matrix)).sum(-1)      # [src_seq_len + mt_seq_len, mt_seq_len] 
    #gi_token_srcmt = torch.mul(embed_grads_tokens_srcmt, embed_srcmt_matrix).sum(-1)      # [src_seq_len + mt_seq_len, mt_seq_len] 
    gi_token_srcmt = F.normalize(gi_token_srcmt, dim=0)
    #gi_token_srcmt = (gi_token_srcmt - torch.min(gi_token_srcmt)) / (torch.max(gi_token_srcmt) - torch.min(gi_token_srcmt))


    return gi_sent_srcmt, gi_token_srcmt


def only_gradients(model, src_input, mt_input):
    # 只算这一点的梯度，不加积分; 不与embedding相乘，只考虑gradients，加绝对值，用normalization而不是softmax; 用预测结果而不是loss计算梯度

    # embed_grads_sent_mt [mt_seq_len, embed_dim] 
    # embed_grads_sent_src [src_seq_len, embed_dim]
    # embed_grads_tokens_mt [mt_seq_len, mt_seq_len, embed_dim]
    # embed_grads_tokens_src [src_seq_len, mt_seq_len, embed_dim]
    embed_grads_sent_mt, embed_grads_sent_src, \
        embed_grads_tokens_mt, embed_grads_tokens_src = cal_gradients_mid(model, src_input, mt_input)

    # 每个mt、src token对句子分数的贡献
    gi_sent_mt = torch.abs(embed_grads_sent_mt).sum(-1)   # [mt_seq_len]
    gi_sent_src = torch.abs(embed_grads_sent_src).sum(-1)   # [src_seq_len]

    gi_sent_mt = F.normalize(gi_sent_mt, dim=-1)
    gi_sent_src = F.normalize(gi_sent_src, dim=-1)

    # 每个mt、src token对词标签预测的贡献
    gi_token_mt = torch.abs(embed_grads_tokens_mt).sum(-1)      # [mt_seq_len, mt_seq_len] 
    gi_token_src = torch.abs(embed_grads_tokens_src).sum(-1)   # [src_seq_len, mt_seq_len]

    gi_token_mt = F.normalize(gi_token_mt, dim=0)
    gi_token_src = F.normalize(gi_token_src, dim = 0)

    return gi_sent_mt, gi_token_mt, \
                gi_sent_src, gi_token_src


def only_gradients_srcmt(model, src_input, mt_input):
    # 只算这一点的梯度，不加积分; 不与embedding相乘，只考虑gradients，加绝对值，用normalization而不是softmax; 用预测结果而不是loss计算梯度
    # 对src mt的梯度一起算

    # embed_grads_sent_mt [mt_seq_len, embed_dim]
    # embed_grads_sent_src [src_seq_len, embed_dim]
    # embed_grads_tokens_mt [mt_seq_len, mt_seq_len, embed_dim]
    # embed_grads_tokens_src [src_seq_len, mt_seq_len, embed_dim]
    embed_grads_sent_mt, embed_grads_sent_src, \
        embed_grads_tokens_mt, embed_grads_tokens_src = cal_gradients_mid(model, src_input, mt_input)

    embed_grads_sent_srcmt = torch.cat((embed_grads_sent_src, embed_grads_sent_mt), 0) # [src_seq_len + mt_seq_len, embed_dim]
    embed_grads_tokens_srcmt = torch.cat((embed_grads_tokens_src, embed_grads_tokens_mt), 0) # [src_seq_len + mt_seq_len, mt_seq_len, embed_dim]

    # ===================== 每个src+mt token对句子分数的贡献
    gi_sent_srcmt = torch.abs(embed_grads_sent_srcmt).sum(-1)   # [src_seq_len + mt_seq_len]
    gi_sent_srcmt = F.normalize(gi_sent_srcmt, dim=-1)
    
    # ===================== 每个src+mt token对词标签的贡献
    gi_token_srcmt = torch.abs(embed_grads_tokens_srcmt).sum(-1)   # [src_seq_len + mt_seq_len, mt_seq_len]
    gi_token_srcmt = F.normalize(gi_token_srcmt, dim=0)

    return gi_sent_srcmt, gi_token_srcmt


def integrated_gradients(model, 
                            critic,
                            critic_mse,
                            src_input, 
                            mt_input, 
                            tag, 
                            score, 
                            steps=50):
    
    # embed_mt [mt_seq_len, embed_dim] 
    # embed_src [src_seq_len, embed_dim]
    embed_mt, embed_src = get_embedding(model, 
                                        src_input, 
                                        mt_input)
    
    for i in range(0, steps + 1):
        i_frac = i / steps

        # embed_grads_sent_mt [mt_seq_len, embed_dim] 
        # embed_grads_sent_src [src_seq_len, embed_dim]
        # embed_grads_tokens_mt [mt_seq_len, mt_seq_len, embed_dim]
        # embed_grads_tokens_src [src_seq_len, mt_seq_len, embed_dim]
        embed_grads_sent_mt, embed_grads_sent_src, \
            embed_grads_tokens_mt, embed_grads_tokens_src = cal_gradients(model, 
                                                                            critic,
                                                                            critic_mse,
                                                                            src_input, 
                                                                            mt_input, 
                                                                            tag, 
                                                                            score,
                                                                            i_frac)
        # 到这应该没问题，得到embedding了，得到梯度了，然后就该乘起来了
        # 手动放大！感觉不放大就 太奇怪了！
        def amplify(x, n):
            return x*n
        embed_grads_sent_mt = amplify(embed_grads_sent_mt, 1000)
        embed_grads_sent_src = amplify(embed_grads_sent_src, 1000)
        embed_grads_tokens_mt = amplify(embed_grads_tokens_mt, 1000)
        embed_grads_tokens_src = amplify(embed_grads_tokens_src, 1000)

        if i==0:
            integrated_embed_grads_sent_mt = embed_grads_sent_mt
            integrated_embed_grads_sent_src = embed_grads_sent_src
            integrated_embed_grads_tokens_mt = embed_grads_tokens_mt
            integrated_embed_grads_tokens_src = embed_grads_tokens_src
        else:
            integrated_embed_grads_sent_mt += embed_grads_sent_mt
            integrated_embed_grads_sent_src += embed_grads_sent_src
            integrated_embed_grads_tokens_mt += embed_grads_tokens_mt
            integrated_embed_grads_tokens_src += embed_grads_tokens_src

    # 每个mt、src token对句子分数的贡献（softmax之前）
    ig_sent_mt = torch.mul(integrated_embed_grads_sent_mt, embed_mt).sum(-1)   # [mt_seq_len]
    ig_sent_src = torch.mul(integrated_embed_grads_sent_src, embed_src).sum(-1)   # [src_seq_len]
    # 一个问题：太小了！！还是应该softmax，不然没得看。。。。
    ig_sent_mt = F.softmax(ig_sent_mt)
    ig_sent_src = F.softmax(ig_sent_src)

    embed_mt_matrix = embed_mt.unsqueeze(dim=1)
    embed_src_matrix = embed_src.unsqueeze(dim=1)
    ig_token_mt = torch.mul(integrated_embed_grads_tokens_mt, embed_mt_matrix).sum(-1)      # [mt_seq_len, mt_seq_len] 
    ig_token_src = torch.mul(integrated_embed_grads_tokens_src, embed_src_matrix).sum(-1)   # [src_seq_len, mt_seq_len] 
    
    ig_token_mt = F.softmax(ig_token_mt, dim=0)
    ig_token_src = F.softmax(ig_token_src, dim=0)

    return ig_sent_mt, ig_token_mt, \
                ig_sent_src, ig_token_src


def cal_influence_single_hter(model,
                            critic_mse,
                            test_sample,
                            train_dataset,
                            test_token_id,
                            influence_path,
                            recursion_depth=10,
                            gpu=False,
                            damp=0.01,
                            scale=25.0,
                            result_instead_of_loss=False,
                            not_saliency=False):
    """
    计算【训练集所有样本】对【一个test_token对 句子级分数 的显著性】的影响
    """

    # 如果已经算过这个test token的influence，则报错，防止覆盖之前算的结果文件
    #assert os.path.exists(influence_path) == False

    train_dataset_size = len(train_dataset)

    # 先计算该test样本的该token的 s_test_vec [θ]
    if not_saliency:
        # 训练集对结果的直接影响
        s_test_vec = cal_s_test_hter_not_saliency(model,
                                                critic_mse,
                                                test_sample,
                                                train_dataset,
                                                test_token_id,
                                                recursion_depth,
                                                gpu,
                                                damp,
                                                scale,
                                                result_instead_of_loss=result_instead_of_loss)
    else:
        # 训练集对显著性的影响，也就是 训练集对 测试集对结果的影响 的影响
        s_test_vec = cal_s_test_hter(model,
                                        critic_mse,
                                        test_sample,
                                        train_dataset,
                                        test_token_id,
                                        recursion_depth,
                                        gpu,
                                        damp,
                                        scale,
                                        result_instead_of_loss=result_instead_of_loss)
    #print('s_test_vec=============================================================')
    #print(s_test_vec)
    #assert 1==2
    # 收集每个训练样本的影响
    train_influences = []
    # 按顺序load每个train样本
    for train_id in range(train_dataset_size):
        print("train id: ", train_id)
        train_sample = train_dataset[train_id]
        # 计算【一个训练样本的hter的loss】对【模型参数θ】的梯度：grad_z_vec [θ]
        grad_z_vec = cal_grad_z_hter(model,
                                    critic_mse,
                                    train_sample,
                                    gpu,
                                    result_instead_of_loss=result_instead_of_loss)
        
        # 影响函数：该训练样本的影响 tmp_influence [1]
        tmp_influence = -sum(
            [
                torch.sum(k * j).data
                for k, j in zip(grad_z_vec, s_test_vec)
            ]) / train_dataset_size # 做了个归一化 TODO 不做也可？

        train_influences.append(tmp_influence)
    
    infl_sort_id = sorted(enumerate(train_influences), key=lambda x:x[1], reverse=True)
    infl_sort_id = [x[0] for x in infl_sort_id]
    torch.save(train_influences, influence_path)
        
    return train_influences, infl_sort_id


def cal_influence_single_token(model,
                                critic,
                                test_sample,
                                train_dataset,
                                test_token_id,
                                influence_path,
                                recursion_depth=10,
                                gpu=False,
                                damp=0.01,
                                scale=25.0,
                                not_saliency=False,
                                log_path=None):
    """
    计算【训练集所有样本】对【一个test_token对 一个token标注 (的显著性)】的影响
    """

    train_dataset_size = len(train_dataset)

    # 先计算该test样本的该token的 s_test_vec [θ]
    if not_saliency:
        # 训练集对结果的直接影响，而不是对显著性的影响
        s_test_vec = cal_s_test_token_not_saliency(model,
                                                    critic,
                                                    test_sample,
                                                    train_dataset,
                                                    test_token_id,
                                                    recursion_depth,
                                                    gpu,
                                                    damp,
                                                    scale)
    else:
        # 训练集对显著性的影响，也就是 训练集对 测试集对结果的影响 的影响
        s_test_vec = cal_s_test_token(model,
                                        critic,
                                        test_sample,
                                        train_dataset,
                                        test_token_id,
                                        recursion_depth,
                                        gpu,
                                        damp,
                                        scale)

    # 收集每个训练样本的影响
    train_influences = []
    # 按顺序load每个train样本
    with open(log_path, 'w', encoding='utf-8') as logf:
        for train_id in range(train_dataset_size):
            print("train id: ", train_id)
            train_sample = train_dataset[train_id]
            # 计算【一个训练样本的token_loss们】对【模型参数θ】的梯度：grad_z_vec [θ]
            loss_score_sentence = cal_loss_z_token(model,
                                                critic,
                                                train_sample,
                                                gpu=gpu,)
            params = [ p for p in model.parameters() if p.requires_grad ]
            train_influence_sentence = []
            for loss_score in loss_score_sentence:
                grad_z_vec = mello_autograd(loss_score, params)
                # 影响函数：该训练样本的影响 tmp_influence [1]
                tmp_influence = sum(   # 负号已改
                    [
                        torch.sum(k * j).data
                        for k, j in zip(grad_z_vec, s_test_vec)
                    ]) / train_dataset_size
                train_influence_sentence.append(tmp_influence.item())
            
            # 记录这一个训练样本的一行影响力
            logf.write(' '.join([str(round(token_infl,6)) for token_infl in train_influence_sentence]) + '\n')

            # 其中最大的token影响力作为这个训练样本的影响力
            train_influences.append(max(train_influence_sentence))
    
    infl_sort_id = sorted(enumerate(train_influences), key=lambda x:x[1], reverse=True)
    infl_sort_id = [x[0] for x in infl_sort_id]
    torch.save(train_influences, influence_path)
        
    return train_influences, infl_sort_id


def cal_influence_single_both(model,
                            critic,
                            critic_mse,
                            test_sample,
                            train_dataset,
                            influence_path,
                            recursion_depth=10,
                            gpu=False,
                            damp=0.01,
                            scale=25.0):
    """
    计算【训练集所有样本】对【某test样本预测】的影响，both词标记和句子分数
    """

    train_dataset_size = len(train_dataset)

    # 先计算该test样本的该token的 s_test_vec [θ]
    s_test_vec = cal_s_test_both(model,
                                critic,
                                critic_mse,
                                test_sample,
                                train_dataset,
                                recursion_depth,
                                gpu,
                                damp,
                                scale)
    # print("caled s_test==========================")
    #print(s_test_vec)  # nan就变0了，最后有部分是0   # TODO 写到这里啦
    #assert 1==2
    # 收集每个训练样本的影响
    train_influences = []
    # 按顺序load每个train样本
    for train_id in range(train_dataset_size):
        print("train id: ", train_id)
        train_sample = train_dataset[train_id]
        # 计算【一个训练样本的hter的loss】对【模型参数θ】的梯度：grad_z_vec [θ]
        grad_z_vec = cal_grad_z_both(model,
                                    critic,
                                    critic_mse,
                                    train_sample,
                                    gpu)
        
        # 影响函数：该训练样本的影响 tmp_influence [1]
        tmp_influence = sum(   # 负号已改
            [
                torch.sum(k * j).data
                for k, j in zip(grad_z_vec, s_test_vec)
            ]) / train_dataset_size # 做了个归一化 TODO 不做也可？

        train_influences.append(tmp_influence)
    
    infl_sort_id = sorted(enumerate(train_influences), key=lambda x:x[1], reverse=True)
    infl_sort_id = [x[0] for x in infl_sort_id]
    torch.save(train_influences, influence_path)
        
    return train_influences, infl_sort_id


def cal_influence_single_detailed_hter(model,
                                    critic,
                                    critic_mse,
                                    test_sample,
                                    test_token_id,
                                    train_dataset,
                                    s_test_vec,
                                    train_sample_id,
                                    gpu=False,):
    """
    对一个test_token的显著性，计算某个训练样本内的影响分布
    """
    train_sample_this = train_dataset[train_sample_id]
    mt_this = train_sample_this[1][0]
    mt_len = len(mt_this)


    # 如果已经算过这个test token的influence，则报错，防止覆盖之前算的结果文件
    #assert os.path.exists(influence_path) == False

    train_dataset_size = len(train_dataset)
    

    # 这个原装训练样本的影响
    grad_z_vec = cal_grad_z_hter(model,
                                    critic_mse,
                                    train_sample_this,
                                    gpu,)
    this_influence = -sum(
        [
            torch.sum(k * j).data
            for k, j in zip(grad_z_vec, s_test_vec)
        ]) / train_dataset_size
    
    # 收集每个干扰样本的影响
    perturbed_influences = []
    for token_id in range(mt_len):
        train_sample_i = copy.deepcopy(train_sample_this) 
        train_sample_i[1][0][token_id] = 3
        grad_z_vec = cal_grad_z_hter(model,
                                    critic_mse,
                                    train_sample_i,
                                    gpu,)
        tmp_influence = -sum(
            [
                torch.sum(k * j).data
                for k, j in zip(grad_z_vec, s_test_vec)
            ]) / train_dataset_size

        perturbed_influences.append(tmp_influence)

    # perturb不加权
    # train_influence_saliency = [this_influence - perturb for perturb in perturbed_influences]
    # perturb加权100
    train_influence_saliency = [(this_influence - perturb)*100 for perturb in perturbed_influences]
    train_influence_saliency = torch.stack(train_influence_saliency)
    
        
    return train_influence_saliency


def plot_attribution1(train_influence_saliency, plot_path_prefix, \
                        mt_tokens, mt_dict, test_id, test_token_id, rank, train_id):
    # 对训练样本显著性分布生成1张图

    #mt_tokens = mt_tokens.cpu().detach().numpy().tolist()
    mt_sent = [mt_dict.id2token(w) for w in mt_tokens]
    mt_sent_len = len(mt_sent)
    mt_font_size = 240 // mt_sent_len

    # influence
    train_influence_saliency = train_influence_saliency.unsqueeze(0).cpu().detach().numpy()     # [1, mt_len]
    f, ax = plt.subplots(figsize = (30,30))
    sns.heatmap(train_influence_saliency, ax=ax, cmap="YlGnBu", square=True, \
        annot=train_influence_saliency, cbar=False, \
        yticklabels=False, annot_kws={'size':mt_font_size}, fmt='.4f')
    ax.set_xlabel('attribution', fontsize=mt_font_size)
    ax.set_xticklabels(mt_sent,rotation=0)
    plt.tick_params(labelsize=mt_font_size)
    plt.savefig(plot_path_prefix + "infl_test_{}_{}_train_{}_{}.jpg".format(test_id, test_token_id, rank, train_id))


def plot_attribution_src(attr_token_src, 
                        plot_path_prefix, mt_tokens, src_tokens, mt_dict, src_dict, 
                        id, gradients_type, lang='enzh'):
    # 每个样本生成6张attribution图

    if lang == 'enzh':
        # 支持中文
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    if lang == 'ende':
        font = {'family': 'serif'}
        matplotlib.rc('font', **font) 

    mt_tokens = mt_tokens.cpu().detach().numpy().tolist()
    mt_sent = [mt_dict.id2token(w) for w in mt_tokens]
    src_tokens = src_tokens.cpu().detach().numpy().tolist()
    src_sent = [src_dict.id2token(w) for w in src_tokens]
    mt_sent_len = len(mt_sent)
    src_sent_len = len(src_sent)
    mt_font_size = 240 // mt_sent_len
    src_font_size = 240 // src_sent_len
    srcmt_sent = src_sent + mt_sent
    srcmt_sent_len = len(srcmt_sent)
    srcmt_font_size = 240 // srcmt_sent_len
    
    # 4.attr_token_src: 每个src token对每个token标签预测的贡献
    attr_token_src = attr_token_src.cpu().detach().numpy()     # [src_len, mt_len]
    f, ax = plt.subplots(figsize = (30,30))
    sns.heatmap(attr_token_src, ax=ax, cmap="YlGnBu", square=True, annot=attr_token_src, cbar=False, \
        annot_kws={'size':src_font_size}, fmt='.4f')
    ax.set_xlabel('attribution', fontsize=src_font_size)
    ax.set_xticklabels(mt_sent,rotation=0)
    ax.set_yticklabels(src_sent,rotation=0)
    plt.tick_params(labelsize=src_font_size)
    plt.savefig(plot_path_prefix + "attr_{}_token_src_{}.jpg".format(gradients_type, id))



def plot_attribution_all(attr_sent_mt, attr_token_mt, attr_sent_src, attr_token_src, 
                        attr_sent_srcmt, attr_token_srcmt,
                        plot_path_prefix, mt_tokens, src_tokens, mt_dict, src_dict, 
                        id, gradients_type, lang='enzh'):
    # 每个样本生成6张attribution图

    if lang == 'enzh':
        # 支持中文
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    if lang == 'ende':
        font = {'family': 'serif'}
        matplotlib.rc('font', **font) 

    mt_tokens = mt_tokens.cpu().detach().numpy().tolist()
    mt_sent = [mt_dict.id2token(w) for w in mt_tokens]
    src_tokens = src_tokens.cpu().detach().numpy().tolist()
    src_sent = [src_dict.id2token(w) for w in src_tokens]
    mt_sent_len = len(mt_sent)
    src_sent_len = len(src_sent)
    mt_font_size = 240 // mt_sent_len
    src_font_size = 240 // src_sent_len
    srcmt_sent = src_sent + mt_sent
    srcmt_sent_len = len(srcmt_sent)
    srcmt_font_size = 240 // srcmt_sent_len
    
    # 1.attr_sent_mt: 每个mt token对句子级分数的贡献
    attr_sent_mt = attr_sent_mt.unsqueeze(0).cpu().detach().numpy()     # [1, mt_len]
    f, ax = plt.subplots(figsize = (30,30))
    sns.heatmap(attr_sent_mt, ax=ax, cmap="YlGnBu", square=True, annot=attr_sent_mt, cbar=False, \
        yticklabels=False, annot_kws={'size':mt_font_size}, fmt='.4f')
    ax.set_xlabel('attribution', fontsize=mt_font_size)
    ax.set_xticklabels(mt_sent,rotation=0)
    plt.tick_params(labelsize=mt_font_size)
    plt.savefig(plot_path_prefix + "attr_{}_sent_mt_{}.jpg".format(gradients_type, id))

    # 2.attr_token_mt: 每个mt token对每个token标签预测的贡献
    attr_token_mt = attr_token_mt.cpu().detach().numpy()     # [mt_len, mt_len]
    f, ax = plt.subplots(figsize = (30,30))
    sns.heatmap(attr_token_mt, ax=ax, cmap="YlGnBu", square=True, annot=attr_token_mt, cbar=False, \
        annot_kws={'size':mt_font_size}, fmt='.4f')
    ax.set_xlabel('attribution', fontsize=mt_font_size)
    ax.set_xticklabels(mt_sent,rotation=0)
    ax.set_yticklabels(mt_sent,rotation=0)
    plt.tick_params(labelsize=mt_font_size)
    plt.savefig(plot_path_prefix + "attr_{}_token_mt_{}.jpg".format(gradients_type, id))

    # 3.attr_sent_src: 每个src token对句子级分数的贡献
    attr_sent_src = attr_sent_src.unsqueeze(0).cpu().detach().numpy()     # [1, src_len]
    f, ax = plt.subplots(figsize = (30,30))
    sns.heatmap(attr_sent_src, ax=ax, cmap="YlGnBu", square=True, annot=attr_sent_src, cbar=False, \
        yticklabels=False, annot_kws={'size':src_font_size}, fmt='.4f')
    ax.set_xlabel('attribution', fontsize=src_font_size)
    ax.set_xticklabels(src_sent,rotation=0)
    plt.tick_params(labelsize=src_font_size)
    plt.savefig(plot_path_prefix + "attr_{}_sent_src_{}.jpg".format(gradients_type, id))

    # 4.attr_token_src: 每个src token对每个token标签预测的贡献
    attr_token_src = attr_token_src.cpu().detach().numpy()     # [src_len, mt_len]
    f, ax = plt.subplots(figsize = (30,30))
    sns.heatmap(attr_token_src, ax=ax, cmap="YlGnBu", square=True, annot=attr_token_src, cbar=False, \
        annot_kws={'size':src_font_size}, fmt='.4f')
    ax.set_xlabel('attribution', fontsize=src_font_size)
    ax.set_xticklabels(mt_sent,rotation=0)
    ax.set_yticklabels(src_sent,rotation=0)
    plt.tick_params(labelsize=src_font_size)
    plt.savefig(plot_path_prefix + "attr_{}_token_src_{}.jpg".format(gradients_type, id))

    # 5.attr_sent_srcmt: 每个srcmt token对句子级分数的贡献
    attr_sent_srcmt = attr_sent_srcmt.unsqueeze(0).cpu().detach().numpy()     # [1, src_len + mt_len]
    f, ax = plt.subplots(figsize = (30,30))
    sns.heatmap(attr_sent_srcmt, ax=ax, cmap="YlGnBu", square=True, annot=attr_sent_srcmt, cbar=False, \
        yticklabels=False, annot_kws={'size':srcmt_font_size}, fmt='.4f')
    ax.set_xlabel('attribution', fontsize=srcmt_font_size)
    ax.set_xticklabels(srcmt_sent,rotation=0)
    plt.tick_params(labelsize=srcmt_font_size)
    plt.savefig(plot_path_prefix + "attr_{}_sent_srcmt_{}.jpg".format(gradients_type, id))
    
    # 6.attr_token_srcmt: 每个srcmt token对每个token标签预测的贡献
    attr_token_srcmt = attr_token_srcmt.cpu().detach().numpy()     # [src_len + mt_len, mt_len]
    f, ax = plt.subplots(figsize = (30,30))
    sns.heatmap(attr_token_srcmt, ax=ax, cmap="YlGnBu", square=True, annot=attr_token_srcmt, cbar=False, \
        annot_kws={'size':srcmt_font_size}, fmt='.4f')
    ax.set_xlabel('attribution', fontsize=srcmt_font_size)
    ax.set_xticklabels(mt_sent,rotation=0)
    ax.set_yticklabels(srcmt_sent,rotation=0)
    plt.tick_params(labelsize=srcmt_font_size)
    plt.savefig(plot_path_prefix + "attr_{}_token_srcmt_{}.jpg".format(gradients_type, id))


def tag_x_att(att, tag, num_class):
    dim = num_class
    # x 重要性按类别加和
    # y 不同类别数量统计
    x = torch.zeros((dim)).cuda()
    y = torch.zeros((dim)).cuda()
    for i in range(tag.size(0)):
        x[tag[i]] += att[i]
        y[tag[i]] += 1
    return x, y


def cal_info(func, a, b, c=None):
    if c is not None:
        x = func(a, b)
        y = func(a, c)
        z = func(b, c)
        return x, y, z
    else:
        return func(a, b)


def loss_evaluation(model, critic, valid_iterator, LM=None, rank=0, world_size=1):

    n_sents = 0
    sum_loss = 0.0
    valid_iter = valid_iterator.build_generator()
    sum_correct = 0
    sum_word = 0
    for batch in valid_iter:
        seqs_x, seqs_y = batch

        n_sents += len(seqs_x)

        x, y = prepare_data(seqs_x, seqs_y, backward=LM == 'LM_backward', cuda=Constants.USE_GPU)

        loss, logits = compute_forward(model=model,
                                       critic=critic,
                                       seqs_x=x,
                                       seqs_y=y,
                                       lm=LM,
                                       eval=True)
        correct, word = precision_validation(logits, y)
        sum_correct += correct
        sum_word += word

        if np.isnan(loss):
            WARN("NaN detected!")

        sum_loss += float(loss)
    acc = sum_correct / sum_word
    if world_size > 1:
        sum_loss = dist.all_reduce_py(sum_loss)
        n_sents = dist.all_reduce_py(n_sents)
        acc = dist.all_reduce_py(acc)
    return float(sum_loss / n_sents), acc


def loss_evaluation_qe_origin(discriminator, critic, critic_mse, valid_iterator):

    n_sents = 0
    sum_loss = 0.0
    sum_loss_hter = 0.0
    valid_iter = valid_iterator.build_generator()

    TP = 0 + 1e-12
    FP = 0 + 1e-12
    FN = 0 + 1e-12
    TN = 0 + 1e-12
    #print("====in loss evaluation qe=========")
    total_hters = []
    total_hter_gold = []
    #ii = 0
    for batch in valid_iter:
        seqs_x, seqs_y, xy_label, xy_hter = batch
        n_sents += len(seqs_x)

        x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)
        if Constants.USE_GPU:
            xy_hter = torch.tensor(xy_hter)
            xy_hter = xy_hter.cuda()
        """
        print('------- one batch')
        print(x.size())
        print(y.size())
        print(xy_label.size())
        print(xy_hter.size())
        """
        #print(ii)
        loss, loss_hter, logits, hter_pre = compute_forward_discriminator(model=discriminator,
                                                                          critic=critic,
                                                                          critic_mse=critic_mse,
                                                                          seqs_x=x,
                                                                          seqs_y=y,
                                                                          lm_ids=xy_label,
                                                                          hter_label=xy_hter,
                                                                          eval=True)
        #print(loss)
        #print(loss_hter)
        hter_pre = hter_pre.view(-1).cpu().tolist()
        xy_hter = xy_hter.view(-1).cpu().tolist()
        for each in hter_pre:
            total_hters.append(each)
        for each in xy_hter:
            total_hter_gold.append(each)

        logits = logits.view(-1, 3).cpu().tolist()
        xy_label = xy_label[:, 2:-1].contiguous().view(-1).cpu().tolist()

        for e_l, e_o in zip(xy_label, logits):
            if e_l == 0:
                continue
            # label里是1和2。
            # 1原词，2噪声。
            if e_o[1] < e_o[2]:
                e_o = 2
            else:
                e_o = 1
            if e_o == e_l:
                if e_o == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if e_o == 1:
                    FP += 1
                else:
                    FN += 1

        if np.isnan(loss):
            WARN("NaN detected!")
        if np.isnan(loss_hter):
            WARN("NaN detected!")

        sum_loss += float(loss)
        sum_loss_hter += float(loss_hter)
        #ii += 1
        #if ii>=2: break

    precsion_1 = TP / (TP + FP)
    recall_1 = TP / (TP + FN)
    precsion_2 = TN / (FN + TN)
    recall_2 = TN / (FP + TN)
    f1_good = 2 * precsion_1 * recall_1 / (precsion_1 + recall_1)
    f1_bad = 2 * precsion_2 * recall_2 / (precsion_2 + recall_2)
    p = pearsonr(total_hters, total_hter_gold)[0]
    #print("====out of loss evaluation qe=========")
    return float(sum_loss / n_sents), float(sum_loss_hter / n_sents), f1_good, f1_bad, f1_bad * f1_good, p


def loss_evaluation_qe(discriminator, critic, critic_mse, valid_iterator, requires_adapter=False):

    n_sents = 0
    sum_loss = 0.0
    sum_loss_hter = 0.0
    valid_iter = valid_iterator.build_generator()

    TP = 0 + 1e-12
    FP = 0 + 1e-12
    FN = 0 + 1e-12
    TN = 0 + 1e-12
    #print("====in loss evaluation qe=========")
    total_hters = []
    total_hter_gold = []
    #ii = 0
    for batch in valid_iter:
        seqs_x, seqs_y, xy_label, xy_hter = batch
        n_sents += len(seqs_x)

        x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)
        if Constants.USE_GPU:
            xy_hter = torch.tensor(xy_hter)
            xy_hter = xy_hter.cuda()
        """
        print('------- one batch')
        print(x.size())
        print(y.size())
        print(xy_label.size())
        print(xy_hter.size())
        """
        #print(ii)
        loss, loss_hter, logits, hter_pre = compute_forward_discriminator(model=discriminator,
                                                                          critic=critic,
                                                                          critic_mse=critic_mse,
                                                                          seqs_x=x,
                                                                          seqs_y=y,
                                                                          lm_ids=xy_label,
                                                                          hter_label=xy_hter,
                                                                          eval=True,
                                                                          requires_adapter=requires_adapter)
        #print(loss)
        #print(loss_hter)
        hter_pre = hter_pre.view(-1).cpu().tolist()
        xy_hter = xy_hter.view(-1).cpu().tolist()
        for each in hter_pre:
            total_hters.append(each)
        for each in xy_hter:
            total_hter_gold.append(each)

        logits = logits.view(-1, 3).cpu().tolist()
        xy_label = xy_label[:, 2:-1].contiguous().view(-1).cpu().tolist()

        for e_l, e_o in zip(xy_label, logits):
            if e_l == 0:
                continue
            # label里是1和2。
            # 1原词，2噪声。
            if e_o[1] < e_o[2]:
                e_o = 2
            else:
                e_o = 1
            if e_o == e_l:
                if e_o == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if e_o == 1:
                    FP += 1
                else:
                    FN += 1

        if np.isnan(loss):
            WARN("NaN detected!")
        if np.isnan(loss_hter):
            WARN("NaN detected!")

        sum_loss += float(loss)
        sum_loss_hter += float(loss_hter)
        #ii += 1
        #if ii>=2: break

    precsion_1 = TP / (TP + FP)
    recall_1 = TP / (TP + FN)
    precsion_2 = TN / (FN + TN)
    recall_2 = TN / (FP + TN)
    f1_good = 2 * precsion_1 * recall_1 / (precsion_1 + recall_1)
    f1_bad = 2 * precsion_2 * recall_2 / (precsion_2 + recall_2)
    p = pearsonr(total_hters, total_hter_gold)[0]
    #print("====out of loss evaluation qe=========")
    return float(sum_loss / n_sents), float(sum_loss_hter / n_sents), f1_good, f1_bad, f1_bad * f1_good, p


def loss_evaluation_qe_margin(discriminator, critic, critic_mse, valid_iterator, requires_adapter=False):

    n_sents = 0
    sum_loss = 0.0
    sum_loss_hter = 0.0
    valid_iter = valid_iterator.build_generator()

    TP = 0 + 1e-12
    FP = 0 + 1e-12
    FN = 0 + 1e-12
    TN = 0 + 1e-12
    #print("====in loss evaluation qe=========")
    total_hters = []
    total_hter_gold = []
    #ii = 0
    for batch in valid_iter:
        seqs_x, seqs_y, xy_label, xy_hter = batch
        n_sents += len(seqs_x)

        x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)
        if Constants.USE_GPU:
            xy_hter = torch.tensor(xy_hter)
            xy_hter = xy_hter.cuda()
        """
        print('------- one batch')
        print(x.size())
        print(y.size())
        print(xy_label.size())
        print(xy_hter.size())
        """
        #print(ii)
        loss, loss_hter, logits, hter_pre = compute_forward_discriminator_margin(model=discriminator,
                                                                          critic=critic,
                                                                          critic_mse=critic_mse,
                                                                          seqs_x=x,
                                                                          seqs_y=y,
                                                                          lm_ids=xy_label,
                                                                          hter_label=xy_hter,
                                                                          eval=True,)
        #print(loss)
        #print(loss_hter)
        hter_pre = hter_pre.view(-1).cpu().tolist()
        xy_hter = xy_hter.view(-1).cpu().tolist()
        for each in hter_pre:
            total_hters.append(each)
        for each in xy_hter:
            total_hter_gold.append(each)

        logits = logits.view(-1, 3).cpu().tolist()
        xy_label = xy_label[:, 2:-1].contiguous().view(-1).cpu().tolist()

        for e_l, e_o in zip(xy_label, logits):
            if e_l == 0:
                continue
            # label里是1和2。
            # 1原词，2噪声。
            if e_o[1] < e_o[2]:
                e_o = 2
            else:
                e_o = 1
            if e_o == e_l:
                if e_o == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if e_o == 1:
                    FP += 1
                else:
                    FN += 1

        if np.isnan(loss):
            WARN("NaN detected!")
        if np.isnan(loss_hter):
            WARN("NaN detected!")

        sum_loss += float(loss)
        sum_loss_hter += float(loss_hter)
        #ii += 1
        #if ii>=2: break

    precsion_1 = TP / (TP + FP)
    recall_1 = TP / (TP + FN)
    precsion_2 = TN / (FN + TN)
    recall_2 = TN / (FP + TN)
    f1_good = 2 * precsion_1 * recall_1 / (precsion_1 + recall_1)
    f1_bad = 2 * precsion_2 * recall_2 / (precsion_2 + recall_2)
    p = pearsonr(total_hters, total_hter_gold)[0]
    #print("====out of loss evaluation qe=========")
    return float(sum_loss / n_sents), float(sum_loss_hter / n_sents), f1_good, f1_bad, f1_bad * f1_good, p


def loss_evaluation_qe_one_class(discriminator, critic, critic_mse, valid_iterator, no_sigmoid=False):

    n_sents = 0
    sum_loss = 0.0
    sum_loss_hter = 0.0
    valid_iter = valid_iterator.build_generator()

    TP = 0 + 1e-12
    FP = 0 + 1e-12
    FN = 0 + 1e-12
    TN = 0 + 1e-12
    #print("====in loss evaluation qe=========")
    total_hters = []
    total_hter_gold = []
    #ii = 0
    for batch in valid_iter:
        seqs_x, seqs_y, xy_label, xy_hter = batch
        n_sents += len(seqs_x)

        x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)
        if Constants.USE_GPU:
            xy_hter = torch.tensor(xy_hter)
            xy_hter = xy_hter.cuda()
        """
        print('------- one batch')
        print(x.size())
        print(y.size())
        print(xy_label.size())
        print(xy_hter.size())
        """
        #print(ii)
        loss, loss_hter, logits, hter_pre = compute_forward_discriminator_one_class(model=discriminator,
                                                                          critic=critic,
                                                                          critic_mse=critic_mse,
                                                                          seqs_x=x,
                                                                          seqs_y=y,
                                                                          lm_ids=xy_label,
                                                                          hter_label=xy_hter,
                                                                          eval=True,
                                                                          no_sigmoid=no_sigmoid)
        #print(loss)
        #print(loss_hter)
        hter_pre = hter_pre.view(-1).cpu().tolist()
        xy_hter = xy_hter.view(-1).cpu().tolist()
        for each in hter_pre:
            total_hters.append(each)
        for each in xy_hter:
            total_hter_gold.append(each)

        logits = logits.view(-1).cpu().tolist()
        xy_label = xy_label[:, 2:-1].contiguous().view(-1).cpu().tolist()

        for e_l, e_o in zip(xy_label, logits):
            if e_l == 0:
                continue
            # label里是1和2。
            # 1原词，2噪声。
            if e_o >= 0.5:
                e_o = 1
            else:
                e_o = 2
            if e_o == e_l:
                if e_o == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if e_o == 1:
                    FP += 1
                else:
                    FN += 1

        if np.isnan(loss):
            WARN("NaN detected!")
        if np.isnan(loss_hter):
            WARN("NaN detected!")

        sum_loss += float(loss)
        sum_loss_hter += float(loss_hter)
        #ii += 1
        #if ii>=2: break

    precsion_1 = TP / (TP + FP)
    recall_1 = TP / (TP + FN)
    precsion_2 = TN / (FN + TN)
    recall_2 = TN / (FP + TN)
    f1_good = 2 * precsion_1 * recall_1 / (precsion_1 + recall_1)
    f1_bad = 2 * precsion_2 * recall_2 / (precsion_2 + recall_2)
    p = pearsonr(total_hters, total_hter_gold)[0]
    #print("====out of loss evaluation qe=========")
    return float(sum_loss / n_sents), float(sum_loss_hter / n_sents), f1_good, f1_bad, f1_bad * f1_good, p


def loss_evaluation_qe_align_v1(discriminator, critic, critic_mse, valid_iterator, align_dict, align_ratio=0.5):

    n_sents = 0
    sum_loss = 0.0
    sum_loss_hter = 0.0
    valid_iter = valid_iterator.build_generator()

    TP = 0 + 1e-12
    FP = 0 + 1e-12
    FN = 0 + 1e-12
    TN = 0 + 1e-12
    #print("====in loss evaluation qe=========")
    total_hters = []
    total_hter_gold = []

    id = 0
    for batch in valid_iter:
        seqs_x, seqs_y, xy_label, xy_hter = batch
        n_sents += len(seqs_x)

        align_line = align_dict[str(id)]

        x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)
        if Constants.USE_GPU:
            xy_hter = torch.tensor(xy_hter)
            xy_hter = xy_hter.cuda()

        loss, loss_hter, logits, hter_pre = compute_forward_discriminator_align_v1(model=discriminator,
                                                                            critic=critic,
                                                                            critic_mse=critic_mse,
                                                                            seqs_x=x,
                                                                            seqs_y=y,
                                                                            lm_ids=xy_label,
                                                                            hter_label=xy_hter,
                                                                            align_line=align_line,
                                                                            align_ratio=align_ratio,
                                                                            eval=True)
        #print(loss)
        #print(loss_hter)
        hter_pre = hter_pre.view(-1).cpu().tolist()
        xy_hter = xy_hter.view(-1).cpu().tolist()
        for each in hter_pre:
            total_hters.append(each)
        for each in xy_hter:
            total_hter_gold.append(each)

        logits = logits.view(-1, 3).cpu().tolist()
        xy_label = xy_label[:, 2:-1].contiguous().view(-1).cpu().tolist()

        for e_l, e_o in zip(xy_label, logits):
            if e_l == 0:
                continue
            # label里是1和2。
            # 1原词，2噪声。
            if e_o[1] < e_o[2]:
                e_o = 2
            else:
                e_o = 1
            if e_o == e_l:
                if e_o == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if e_o == 1:
                    FP += 1
                else:
                    FN += 1

        if np.isnan(loss):
            WARN("NaN detected!")
        if np.isnan(loss_hter):
            WARN("NaN detected!")

        sum_loss += float(loss)
        sum_loss_hter += float(loss_hter)
        id += 1
        #if ii>=2: break

    precsion_1 = TP / (TP + FP)
    recall_1 = TP / (TP + FN)
    precsion_2 = TN / (FN + TN)
    recall_2 = TN / (FP + TN)
    f1_good = 2 * precsion_1 * recall_1 / (precsion_1 + recall_1)
    f1_bad = 2 * precsion_2 * recall_2 / (precsion_2 + recall_2)
    p = pearsonr(total_hters, total_hter_gold)[0]
    #print("====out of loss evaluation qe=========")
    return float(sum_loss / n_sents), float(sum_loss_hter / n_sents), f1_good, f1_bad, f1_bad * f1_good, p


def loss_evaluation_qe_align_v3(discriminator, critic, critic_mse, valid_iterator, align_dict, attn_align_lambda=0):

    n_sents = 0
    sum_loss = 0.0
    sum_loss_hter = 0.0
    valid_iter = valid_iterator.build_generator()

    TP = 0 + 1e-12
    FP = 0 + 1e-12
    FN = 0 + 1e-12
    TN = 0 + 1e-12
    #print("====in loss evaluation qe=========")
    total_hters = []
    total_hter_gold = []

    id = 0
    for batch in valid_iter:
        seqs_x, seqs_y, xy_label, xy_hter = batch
        n_sents += len(seqs_x)

        align_line = align_dict[str(id)]

        x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)
        if Constants.USE_GPU:
            xy_hter = torch.tensor(xy_hter)
            xy_hter = xy_hter.cuda()

        loss, loss_hter, logits, hter_pre = compute_forward_discriminator_align_v3(model=discriminator,
                                                                            critic=critic,
                                                                            critic_mse=critic_mse,
                                                                            seqs_x=x,
                                                                            seqs_y=y,
                                                                            lm_ids=xy_label,
                                                                            hter_label=xy_hter,
                                                                            align_line=align_line,
                                                                            attn_align_lambda=attn_align_lambda,
                                                                            eval=True)
        #print(loss)
        #print(loss_hter)
        hter_pre = hter_pre.view(-1).cpu().tolist()
        xy_hter = xy_hter.view(-1).cpu().tolist()
        for each in hter_pre:
            total_hters.append(each)
        for each in xy_hter:
            total_hter_gold.append(each)

        logits = logits.view(-1, 3).cpu().tolist()
        xy_label = xy_label[:, 2:-1].contiguous().view(-1).cpu().tolist()

        for e_l, e_o in zip(xy_label, logits):
            if e_l == 0:
                continue
            # label里是1和2。
            # 1原词，2噪声。
            if e_o[1] < e_o[2]:
                e_o = 2
            else:
                e_o = 1
            if e_o == e_l:
                if e_o == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if e_o == 1:
                    FP += 1
                else:
                    FN += 1

        if np.isnan(loss):
            WARN("NaN detected!")
        if np.isnan(loss_hter):
            WARN("NaN detected!")

        sum_loss += float(loss)
        sum_loss_hter += float(loss_hter)
        id += 1
        #if ii>=2: break

    precsion_1 = TP / (TP + FP)
    recall_1 = TP / (TP + FN)
    precsion_2 = TN / (FN + TN)
    recall_2 = TN / (FP + TN)
    f1_good = 2 * precsion_1 * recall_1 / (precsion_1 + recall_1)
    f1_bad = 2 * precsion_2 * recall_2 / (precsion_2 + recall_2)
    p = pearsonr(total_hters, total_hter_gold)[0]
    #print("====out of loss evaluation qe=========")
    return float(sum_loss / n_sents), float(sum_loss_hter / n_sents), f1_good, f1_bad, f1_bad * f1_good, p


def loss_evaluation_qe_add_feature(discriminator, critic, critic_mse, valid_iterator, align_dict, ):

    n_sents = 0
    sum_loss = 0.0
    sum_loss_hter = 0.0
    valid_iter = valid_iterator.build_generator()

    TP = 0 + 1e-12
    FP = 0 + 1e-12
    FN = 0 + 1e-12
    TN = 0 + 1e-12
    #print("====in loss evaluation qe=========")
    total_hters = []
    total_hter_gold = []

    id = 0
    for batch in valid_iter:
        seqs_x, seqs_y, xy_label, xy_hter = batch
        n_sents += len(seqs_x)

        align_line = align_dict[str(id)]

        x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)
        if Constants.USE_GPU:
            xy_hter = torch.tensor(xy_hter)
            xy_hter = xy_hter.cuda()

        loss, loss_hter, logits, hter_pre = compute_forward_discriminator_add_feature(model=discriminator,
                                                                            critic=critic,
                                                                            critic_mse=critic_mse,
                                                                            seqs_x=x,
                                                                            seqs_y=y,
                                                                            lm_ids=xy_label,
                                                                            hter_label=xy_hter,
                                                                            align_line=align_line,
                                                                            eval=True)

        hter_pre = hter_pre.view(-1).cpu().tolist()
        xy_hter = xy_hter.view(-1).cpu().tolist()
        for each in hter_pre:
            total_hters.append(each)
        for each in xy_hter:
            total_hter_gold.append(each)

        logits = logits.view(-1, 3).cpu().tolist()
        xy_label = xy_label[:, 2:-1].contiguous().view(-1).cpu().tolist()

        for e_l, e_o in zip(xy_label, logits):
            if e_l == 0:
                continue
            # label里是1和2。
            # 1原词，2噪声。
            if e_o[1] < e_o[2]:
                e_o = 2
            else:
                e_o = 1
            if e_o == e_l:
                if e_o == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if e_o == 1:
                    FP += 1
                else:
                    FN += 1

        if np.isnan(loss):
            WARN("NaN detected!")
        if np.isnan(loss_hter):
            WARN("NaN detected!")

        sum_loss += float(loss)
        sum_loss_hter += float(loss_hter)
        id += 1
        #if ii>=2: break

    precsion_1 = TP / (TP + FP)
    recall_1 = TP / (TP + FN)
    precsion_2 = TN / (FN + TN)
    recall_2 = TN / (FP + TN)
    f1_good = 2 * precsion_1 * recall_1 / (precsion_1 + recall_1)
    f1_bad = 2 * precsion_2 * recall_2 / (precsion_2 + recall_2)
    p = pearsonr(total_hters, total_hter_gold)[0]
    return float(sum_loss / n_sents), float(sum_loss_hter / n_sents), f1_good, f1_bad, f1_bad * f1_good, p


def loss_evaluation_qe_contrastive(discriminator, 
                                    critic, 
                                    critic_mse, 
                                    valid_iterator, 
                                    loss_real_lambda=1.0,
                                    loss_parallel_lambda=1.0,
                                    loss_switch_lambda=1.0,
                                    cl_lambda_switch=1.0,
                                    cl_lambda_parallel = 1.0,):
    # 还是对比学习
    n_sents = 0
    sum_loss = 0.0
    sum_loss_cl = 0.0
    sum_loss_hter = 0.0
    valid_iter = valid_iterator.build_generator()

    TP = 0 + 1e-12
    FP = 0 + 1e-12
    FN = 0 + 1e-12
    TN = 0 + 1e-12
    #print("====in loss evaluation qe=========")
    total_hters = []
    total_hter_gold = []
    #ii = 0
    for batch in valid_iter:
        seqs_x, seqs_y, xy_label, xy_hter, seqs_pe, seqs_align = batch
        n_sents += len(seqs_x)

        x, y, pe, xy_label = prepare_data_qe_with_pe(seqs_x, seqs_y, seqs_pe, xy_label, cuda=Constants.USE_GPU)
        
        if Constants.USE_GPU:
            xy_hter = torch.tensor(xy_hter).cuda()

        #print(ii)
        loss, loss_parallel, loss_switch, cl_loss, cl_loss_parallel, loss_hter, logits, hter_pre = \
                                    compute_forward_discriminator_contrastive(model=discriminator,
                                                                            critic=critic,
                                                                            critic_mse=critic_mse,
                                                                            seqs_x=x,
                                                                            seqs_y=y,
                                                                            seqs_pe=pe,
                                                                            seqs_align=seqs_align,
                                                                            lm_ids=xy_label,
                                                                            hter_label=xy_hter,
                                                                            eval=True,
                                                                            loss_real_lambda=loss_real_lambda,
                                                                            loss_parallel_lambda=loss_parallel_lambda,
                                                                            loss_switch_lambda=loss_switch_lambda,
                                                                            cl_lambda_switch=cl_lambda_switch,
                                                                            cl_lambda_parallel=cl_lambda_parallel)
        
        hter_pre = hter_pre.view(-1).cpu().tolist()
        xy_hter = xy_hter.view(-1).cpu().tolist()
        for each in hter_pre:
            total_hters.append(each)
        for each in xy_hter:
            total_hter_gold.append(each)

        logits = logits.view(-1, 3).cpu().tolist()
        xy_label = xy_label[:, 2:-1].contiguous().view(-1).cpu().tolist()

        for e_l, e_o in zip(xy_label, logits):
            if e_l == 0:
                continue
            # label里是1和2。
            # 1原词，2噪声。
            if e_o[1] < e_o[2]:
                e_o = 2
            else:
                e_o = 1
            if e_o == e_l:
                if e_o == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if e_o == 1:
                    FP += 1
                else:
                    FN += 1

        if np.isnan(loss):
            WARN("NaN detected!")
        if np.isnan(loss_hter):
            WARN("NaN detected!")

        sum_loss += float(loss)
        sum_loss_cl += float(cl_loss)
        sum_loss_hter += float(loss_hter)
        #ii += 1
        #if ii>=2: break

    precsion_1 = TP / (TP + FP)
    recall_1 = TP / (TP + FN)
    precsion_2 = TN / (FN + TN)
    recall_2 = TN / (FP + TN)
    f1_good = 2 * precsion_1 * recall_1 / (precsion_1 + recall_1)
    f1_bad = 2 * precsion_2 * recall_2 / (precsion_2 + recall_2)
    p = pearsonr(total_hters, total_hter_gold)[0]
    #print("====out of loss evaluation qe=========")
    return float(sum_loss / n_sents), float(sum_loss_cl / n_sents), float(sum_loss_hter / n_sents), f1_good, f1_bad, f1_bad * f1_good, p


def loss_evaluation_qe_v2(discriminator, critic, critic_mse, valid_iterator):

    n_sents = 0
    sum_loss = 0.0
    sum_loss_hter = 0.0
    valid_iter = valid_iterator.build_generator()

    TP = 0 + 1e-12
    FP = 0 + 1e-12
    FN = 0 + 1e-12
    TN = 0 + 1e-12
    #print("====in loss evaluation qe=========")
    total_hters = []
    total_hter_gold = []
    #ii = 0
    for batch in valid_iter:
        seqs_x, seqs_y, xy_label, xy_hter, seqs_x_real = batch
        n_sents += len(seqs_x)

        x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)
        if Constants.USE_GPU:
            xy_hter = torch.tensor(xy_hter)
            xy_hter = xy_hter.cuda()
        """
        print('------- one batch')
        print(x.size())
        print(y.size())
        print(xy_label.size())
        print(xy_hter.size())
        """
        #print(ii)
        loss, loss_hter, logits, hter_pre = compute_forward_discriminator(model=discriminator,
                                                                          critic=critic,
                                                                          critic_mse=critic_mse,
                                                                          seqs_x=x,
                                                                          seqs_y=y,
                                                                          lm_ids=xy_label,
                                                                          hter_label=xy_hter,
                                                                          eval=True)
        #print(loss)
        #print(loss_hter)
        hter_pre = hter_pre.view(-1).cpu().tolist()
        xy_hter = xy_hter.view(-1).cpu().tolist()
        for each in hter_pre:
            total_hters.append(each)
        for each in xy_hter:
            total_hter_gold.append(each)

        logits = logits.view(-1, 3).cpu().tolist()
        xy_label = xy_label[:, 2:-1].contiguous().view(-1).cpu().tolist()

        for e_l, e_o in zip(xy_label, logits):
            if e_l == 0:
                continue
            # label里是1和2。
            # 1原词，2噪声。
            if e_o[1] < e_o[2]:
                e_o = 2
            else:
                e_o = 1
            if e_o == e_l:
                if e_o == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if e_o == 1:
                    FP += 1
                else:
                    FN += 1

        if np.isnan(loss):
            WARN("NaN detected!")
        if np.isnan(loss_hter):
            WARN("NaN detected!")

        sum_loss += float(loss)
        sum_loss_hter += float(loss_hter)
        #ii += 1
        #if ii>=2: break

    precsion_1 = TP / (TP + FP)
    recall_1 = TP / (TP + FN)
    precsion_2 = TN / (FN + TN)
    recall_2 = TN / (FP + TN)
    f1_good = 2 * precsion_1 * recall_1 / (precsion_1 + recall_1)
    f1_bad = 2 * precsion_2 * recall_2 / (precsion_2 + recall_2)
    p = pearsonr(total_hters, total_hter_gold)[0]
    #print("====out of loss evaluation qe=========")
    return float(sum_loss / n_sents), float(sum_loss_hter / n_sents), f1_good, f1_bad, f1_bad * f1_good, p


def loss_evaluation_qe_contrastive_src(discriminator, 
                                    critic, 
                                    critic_mse, 
                                    valid_iterator,
                                    loss_real_lambda=1,
                                    loss_parallel_lambda=1,
                                    cl_lambda=1):

    n_sents = 0
    sum_loss = 0.0
    sum_loss_hter = 0.0
    valid_iter = valid_iterator.build_generator()

    TP = 0 + 1e-12
    FP = 0 + 1e-12
    FN = 0 + 1e-12
    TN = 0 + 1e-12
    total_hters = []
    total_hter_gold = []
    #ii = 0
    for batch in valid_iter:
        seqs_x, seqs_y, xy_label, xy_hter, seqs_x_real = batch
        n_sents += len(seqs_x)

        x, y, x_real, xy_label = prepare_data_qe_with_src_real(seqs_x, seqs_y, seqs_x_real, xy_label, cuda=Constants.USE_GPU)
        if Constants.USE_GPU:
            xy_hter = torch.tensor(xy_hter).cuda()

        loss_word, loss_word_parallel, cl_loss, loss_hter, logits, hter_pre = compute_forward_discriminator_contrastive_src(model=discriminator,
                                                        critic=critic,
                                                        critic_mse=critic_mse,
                                                        seqs_x=x,
                                                        seqs_y=y,
                                                        seqs_x_real=x_real,
                                                        lm_ids=xy_label,
                                                        hter_label=xy_hter,
                                                        eval=True,
                                                        loss_real_lambda=loss_real_lambda,
                                                        loss_parallel_lambda=loss_parallel_lambda,
                                                        cl_lambda=cl_lambda,)

        hter_pre = hter_pre.view(-1).cpu().tolist()
        xy_hter = xy_hter.view(-1).cpu().tolist()
        for each in hter_pre:
            total_hters.append(each)
        for each in xy_hter:
            total_hter_gold.append(each)

        logits = logits.view(-1, 3).cpu().tolist()
        xy_label = xy_label[:, 2:-1].contiguous().view(-1).cpu().tolist()

        for e_l, e_o in zip(xy_label, logits):
            if e_l == 0:
                continue
            # label里是1和2。
            # 1原词，2噪声。
            if e_o[1] < e_o[2]:
                e_o = 2
            else:
                e_o = 1
            if e_o == e_l:
                if e_o == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if e_o == 1:
                    FP += 1
                else:
                    FN += 1

        if np.isnan(loss_word):
            WARN("NaN detected!")
        if np.isnan(loss_hter):
            WARN("NaN detected!")

        sum_loss += (float(loss_word) + float(loss_word_parallel) + float(cl_loss))  # 词级与对比loss一起
        sum_loss_hter += float(loss_hter)

    precsion_1 = TP / (TP + FP)
    recall_1 = TP / (TP + FN)
    precsion_2 = TN / (FN + TN)
    recall_2 = TN / (FP + TN)
    f1_good = 2 * precsion_1 * recall_1 / (precsion_1 + recall_1)
    f1_bad = 2 * precsion_2 * recall_2 / (precsion_2 + recall_2)
    p = pearsonr(total_hters, total_hter_gold)[0]

    return float(sum_loss / n_sents), float(sum_loss_hter / n_sents), f1_good, f1_bad, f1_bad * f1_good, p


def loss_evaluation_qe_cl(discriminator, critic, critic_mse, valid_iterator):
    """
    对比学习 验证loss
    """

    n_sents = 0
    sum_loss = 0.0
    sum_loss_hter = 0.0
    valid_iter = valid_iterator.build_generator()

    TP = 0 + 1e-12
    FP = 0 + 1e-12
    FN = 0 + 1e-12
    TN = 0 + 1e-12

    total_hters = []
    total_hter_gold = []

    def func(seqs_x, seqs_y, xy_label, xy_hter):
        nonlocal n_sents
        nonlocal sum_loss
        nonlocal sum_loss_hter
        nonlocal TP
        nonlocal FP
        nonlocal FN
        nonlocal TN

        n_sents += len(seqs_x)

        x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)
        if Constants.USE_GPU:
            xy_hter = torch.tensor(xy_hter)
            xy_hter = xy_hter.cuda()
            
        loss, loss_hter, logits, hter_pre = compute_forward_discriminator(model=discriminator,
                                                                          critic=critic,
                                                                          critic_mse=critic_mse,
                                                                          seqs_x=x,
                                                                          seqs_y=y,
                                                                          lm_ids=xy_label,
                                                                          hter_label=xy_hter,
                                                                          eval=True)

        hter_pre = hter_pre.view(-1).cpu().tolist()
        xy_hter = xy_hter.view(-1).cpu().tolist()
        for each in hter_pre:
            total_hters.append(each)
        for each in xy_hter:
            total_hter_gold.append(each)

        logits = logits.view(-1, 3).cpu().tolist()
        xy_label = xy_label[:, 2:-1].contiguous().view(-1).cpu().tolist()

        for e_l, e_o in zip(xy_label, logits):
            if e_l == 0:
                continue
            # label里是1和2。
            # 1原词，2噪声。
            if e_o[1] < e_o[2]:
                e_o = 2
            else:
                e_o = 1
            if e_o == e_l:
                if e_o == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if e_o == 1:
                    FP += 1
                else:
                    FN += 1

        if np.isnan(loss):
            WARN("NaN detected!")
        if np.isnan(loss_hter):
            WARN("NaN detected!")

        sum_loss += float(loss)
        sum_loss_hter += float(loss_hter)

    for batch in valid_iter:
        parallel_seqs_x, parallel_seqs_y, parallel_xy_label, parallel_xy_hter, \
            pseudo_seqs_x, pseudo_seqs_y, pseudo_xy_label, pseudo_xy_hter = batch
        
        func(parallel_seqs_x, parallel_seqs_y, parallel_xy_label, parallel_xy_hter)
        func(pseudo_seqs_x, pseudo_seqs_y, pseudo_xy_label, pseudo_xy_hter)

    precsion_1 = TP / (TP + FP)
    recall_1 = TP / (TP + FN)
    precsion_2 = TN / (FN + TN)
    recall_2 = TN / (FP + TN)
    f1_good = 2 * precsion_1 * recall_1 / (precsion_1 + recall_1)
    f1_bad = 2 * precsion_2 * recall_2 / (precsion_2 + recall_2)
    p = pearsonr(total_hters, total_hter_gold)[0]

    return float(sum_loss / n_sents), float(sum_loss_hter / n_sents), f1_good, f1_bad, f1_bad * f1_good, p


def loss_evaluation_qe_many(discriminator, critic, critic_mse, valid_iterator, train_num_threshold):

    n_sents = 0
    sum_loss = 0.0
    sum_loss_hter = 0.0
    valid_iter = valid_iterator.build_generator()

    TP = 0 + 1e-12
    FP = 0 + 1e-12
    FN = 0 + 1e-12
    TN = 0 + 1e-12
    #print("====in loss evaluation qe=========")
    total_hters = []
    total_hter_gold = []
    #ii = 0
    for batch in valid_iter:
        seqs_x, seqs_y, xy_label, xy_hter, y_cnt = batch
        n_sents += len(seqs_x)

        y_cnt = torch.tensor(y_cnt).cuda().squeeze(0)
        many_mask = y_cnt >= train_num_threshold
        unk_mask = (torch.tensor(seqs_y).cuda()) != 3
        many_mask *= unk_mask

        x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)
        
        

        if Constants.USE_GPU:
            xy_hter = torch.tensor(xy_hter)
            xy_hter = xy_hter.cuda()

        loss, loss_hter, logits, hter_pre = compute_forward_discriminator_token_mask(model=discriminator,
                                                                          critic=critic,
                                                                          critic_mse=critic_mse,
                                                                          seqs_x=x,
                                                                          seqs_y=y,
                                                                          lm_ids=xy_label,
                                                                          hter_label=xy_hter,
                                                                          token_mask=many_mask,
                                                                          eval=True)
        #print(loss)
        #print(loss_hter)
        hter_pre = hter_pre.view(-1).cpu().tolist()
        xy_hter = xy_hter.view(-1).cpu().tolist()
        for each in hter_pre:
            total_hters.append(each)
        for each in xy_hter:
            total_hter_gold.append(each)

        logits = logits.view(-1, 3).cpu().tolist()
        #xy_label = xy_label[:, 2:-1].contiguous().view(-1).cpu().tolist()
        xy_label = (xy_label[:, 2:-1].contiguous() * many_mask).view(-1).cpu().tolist()
        #xy_label *= many_mask


        for e_l, e_o in zip(xy_label, logits):
            if e_l == 0:
                continue
            # label里是1和2。
            # 1原词，2噪声。
            if e_o[1] < e_o[2]:
                e_o = 2
            else:
                e_o = 1
            if e_o == e_l:
                if e_o == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if e_o == 1:
                    FP += 1
                else:
                    FN += 1

        if np.isnan(loss):
            WARN("NaN detected!")
        if np.isnan(loss_hter):
            WARN("NaN detected!")

        sum_loss += float(loss)
        sum_loss_hter += float(loss_hter)
        #ii += 1
        #if ii>=2: break

    precsion_1 = TP / (TP + FP)
    recall_1 = TP / (TP + FN)
    precsion_2 = TN / (FN + TN)
    recall_2 = TN / (FP + TN)
    f1_good = 2 * precsion_1 * recall_1 / (precsion_1 + recall_1)
    f1_bad = 2 * precsion_2 * recall_2 / (precsion_2 + recall_2)


    p = pearsonr(total_hters, total_hter_gold)[0]
    #print("====out of loss evaluation qe=========")
    return float(sum_loss / n_sents), float(sum_loss_hter / n_sents), f1_good, f1_bad, f1_bad * f1_good, p


def load_pretrained_model(model, pretrain_path, device, exclude_prefix=None):

    if exclude_prefix is None:
        exclude_prefix = []
    if pretrain_path is not None:
        INFO("Loading pretrained model from {}".format(pretrain_path))

        all_parameter_names = set([name for name in model.state_dict().keys()])

        pretrain_params = torch.load(pretrain_path, map_location=device)
        for name, params in pretrain_params.items():

            if name not in all_parameter_names:
                continue

            flag = False
            for pp in exclude_prefix:
                if name.startswith(pp):
                    flag = True
                    break
            if flag:
                continue

            INFO("Loading param: {}...".format(name))
            try:
                model.load_state_dict({name: params}, strict=False)
            except Exception as e:
                WARN("{}: {}".format(str(Exception), e))

        INFO("Pretrained model loaded.")


def train_lm(flags):
    """
    flags:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # ================================================================================== #
    # Initialization for training on different devices
    # - CPU/GPU
    # - Single/Distributed
    Constants.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if Constants.USE_GPU:
        torch.cuda.set_device(local_rank)
        Constants.CURRENT_DEVICE = "cuda:{0}".format(local_rank)
    else:
        Constants.CURRENT_DEVICE = "cpu"

    # If not root_rank, close logging
    # else write log of training to file.
    if rank == 0:
        write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    else:
        close_logging()

    # ================================================================================== #
    # Parsing configuration files
    # - Load default settings
    # - Load pre-defined settings
    # - Load user-defined settings

    configs = prepare_configs(flags.config_path, flags.predefined_config)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    INFO(pretty_configs(configs))

    Constants.SEED = training_configs['seed']

    set_seed(Constants.SEED)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    Constants.EOS = vocab_src.eos
    Constants.PAD = vocab_src.pad
    Constants.BOS = vocab_src.bos
    Constants.MASK = vocab_src.mask

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        )
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        )
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs["batch_size"],
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=training_configs['buffer_size'],
                                     batching_func=training_configs['batching_key'],
                                     world_size=world_size,
                                     rank=rank,
                                     shuffle=True)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=True, buffer_size=100000, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 0. Initial

    lrate = optimizer_configs['learning_rate']
    model_collections = Collections()

    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(flags.saveto, flags.model_name)),
                             num_max_keeping=training_configs['num_kept_checkpoints']
                             )
    best_model_prefix = os.path.join(flags.saveto, flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)

    # 1. Build Model & Criterion
    INFO('Building model...')
    timer.tic()
    model = build_model(n_src_vocab=vocab_src.max_n_words,
                        n_tgt_vocab=vocab_tgt.max_n_words, padding_idx=vocab_src.pad, **model_configs)
    INFO(model)

    critic = NMTCriterion(label_smoothing=model_configs['label_smoothing'], padding_idx=vocab_tgt.pad)
    INFO(critic)

    # 2. Move to GPU
    if Constants.USE_GPU:
        model = model.cuda()
        critic = critic.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(model, flags.pretrain_path, exclude_prefix=None, device=Constants.CURRENT_DEVICE)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')

    if not flags.multi_gpu:
        optim = Optimizer(name=optimizer_configs['optimizer'],
                          model=model,
                          lr=lrate,
                          grad_clip=optimizer_configs['grad_clip'],
                          optim_args=optimizer_configs['optimizer_params'],
                          update_cycle=training_configs['update_cycle']
                          )
    else:
        optim = dist.DistributedOptimizer(name=optimizer_configs['optimizer'],
                                          model=model,
                                          lr=lrate,
                                          grad_clip=optimizer_configs['grad_clip'],
                                          optim_args=optimizer_configs['optimizer_params'],
                                          device_id=local_rank
                                          )

    # 5. Build scheduler for optimizer if needed
    scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])

    # 6. build moving average

    if training_configs['moving_average_method'] is not None:
        ma = MovingAverage(moving_average_method=training_configs['moving_average_method'],
                           named_params=model.named_parameters(),
                           alpha=training_configs['moving_average_alpha'])
    else:
        ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # Reload from latest checkpoint
    if flags.reload:
        checkpoint_saver.load_latest(model=model,
                                     optim=optim,
                                     lr_scheduler=scheduler,
                                     collections=model_collections,
                                     ma=ma, device=Constants.CURRENT_DEVICE)

    # broadcast parameters and optimizer states
    if world_size > 1:
        INFO("Broadcasting model parameters...")
        dist.broadcast_parameters(params=model.state_dict())
        INFO("Broadcasting optimizer states...")
        dist.broadcast_optimizer_state(optimizer=optim.optim)
        INFO('Done.')

    # ================================================================================== #
    # Prepare training

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [1])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]
    is_early_stop = model_collections.get_collection("is_early_stop", [False, ])[-1]

    train_loss_meter = AverageMeter()
    sent_per_sec_meter = TimeMeter()
    tok_per_sec_meter = TimeMeter()

    update_cycle = training_configs['update_cycle']
    grad_denom = 0
    train_loss = 0.0
    cum_n_words = 0

    if rank == 0:
        summary_writer = SummaryWriter(log_dir=flags.log_path)
    else:
        summary_writer = None

    sent_per_sec_meter.start()
    tok_per_sec_meter.start()

    INFO('Begin training...')

    while True:

        if summary_writer is not None:
            summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()

        if rank == 0:
            training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                         total=len(training_iterator),
                                         unit="sents"
                                         )
        else:
            training_progress_bar = None

        for batch in training_iter:

            seqs_x, seqs_y = batch

            batch_size = len(seqs_x)
            cum_n_words += sum(len(s) for s in seqs_y)

            try:
                # Prepare data
                x, y = prepare_data(seqs_x, seqs_y, backward=flags.lm == 'LM_backward', cuda=Constants.USE_GPU)

                loss = compute_forward(model=model,
                                       critic=critic,
                                       seqs_x=x,
                                       seqs_y=y,
                                       lm=flags.lm,
                                       eval=False,
                                       normalization=1.0,
                                       norm_by_words=training_configs["norm_by_words"])

                update_cycle -= 1
                grad_denom += batch_size
                train_loss += loss

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom_count += 1
                else:
                    raise e

            # When update_cycle becomes 0, it means end of one batch. Several things will be done:
            # - update parameters
            # - reset update_cycle and grad_denom, update uidx
            # - learning rate scheduling
            # - update moving average

            if update_cycle == 0:

                # 0. reduce variables
                if world_size > 1:
                    grad_denom = dist.all_reduce_py(grad_denom)
                    train_loss = dist.all_reduce_py(train_loss)
                    cum_n_words = dist.all_reduce_py(cum_n_words)

                # 1. update parameters
                optim.step(denom=grad_denom)
                optim.zero_grad()

                if training_progress_bar is not None:
                    training_progress_bar.update(grad_denom)
                    training_progress_bar.set_description(' - (Epc {}, Upd {}) '.format(eidx, uidx))

                # 2. learning rate scheduling
                if scheduler is not None and optimizer_configs["schedule_method"] != "loss":
                    scheduler.step(global_step=uidx)

                # 3. update moving average
                if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                    ma.step()

                # 4. update meters
                train_loss_meter.update(train_loss, grad_denom)
                sent_per_sec_meter.update(grad_denom)
                tok_per_sec_meter.update(cum_n_words)

                # 5. reset accumulated variables, update uidx
                update_cycle = training_configs['update_cycle']
                grad_denom = 0
                uidx += 1
                cum_n_words = 0.0
                train_loss = 0.0

            else:
                continue

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):

                lrate = list(optim.get_lrate())[0]

                if summary_writer is not None:
                    summary_writer.add_scalar("Speed(sents/sec)", scalar_value=sent_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("Speed(words/sec)", scalar_value=tok_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss", scalar_value=train_loss_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                    summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)

                # Reset Meters
                sent_per_sec_meter.reset()
                tok_per_sec_meter.reset()
                train_loss_meter.reset()

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):
                with cache_parameters(model):

                    if ma is not None:
                        model.load_state_dict(ma.export_ma_params(), strict=False)

                    valid_loss, acc = loss_evaluation(model=model,
                                                      critic=critic,
                                                      valid_iterator=valid_iterator,
                                                      LM=flags.lm,
                                                      rank=rank,
                                                      world_size=world_size)

                if scheduler is not None and optimizer_configs["schedule_method"] == "loss":
                    scheduler.step(metric=valid_loss)

                model_collections.add_to_collection("history_losses", valid_loss)

                min_history_loss = np.array(model_collections.get_collection("history_losses")).min()

                if summary_writer is not None:
                    summary_writer.add_scalar("loss", valid_loss, global_step=uidx)
                    summary_writer.add_scalar("best_loss", min_history_loss, global_step=uidx)

                if valid_loss <= min_history_loss:
                    bad_count = 0
                    if is_early_stop is False and rank == 0:
                        torch.save(model.state_dict(), best_model_prefix + '.final')
                else:
                    bad_count += 1
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 1:
                        is_early_stop = True
                        WARN("Early Stop")

                INFO("{0} Loss: {1:.2f} Acc: {2:.2f} lrate: {3:6f} patience: {4}".format(
                    uidx, valid_loss, acc, lrate, bad_count
                ))
            if is_early_stop is True:
                break
            # ================================================================================== #
        if training_progress_bar is not None:
            training_progress_bar.close()

        if is_early_stop is True:
            break

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def mask_target_data_origin(y):
    # y:[batch_size,seq_len] , y_1_id , y_2_id ..y_i_id.. y_n_id
    # output:
    # y:[batch_size,seq_len], y_1_id , y_2_id ..mask.. y_n_id
    # lm_mask:[batch_size,seq_len] , [-1 , -1 ..y_i_id.. -1,-1]
    # type : int64

    MASK = Constants.MASK
    output_labels = []
    for j, each_sen in enumerate(y):
        output_label = []
        for i, token in enumerate(each_sen):
            id = token
            prob = random.random()
            if prob < 0.15:
                # 有一定几率见到原词
                prob = prob / 0.15
                if prob < 0.9:
                # if prob < 1:
                    y[j][i] = MASK
                output_label.append(id)
            else:
                output_label.append(0)

        output_labels.append(output_label)
    return y, output_labels


def mask_target_data(y):
    # y:[batch_size,seq_len] , y_1_id , y_2_id ..y_i_id.. y_n_id
    # output:
    # y:[batch_size,seq_len], y_1_id , y_2_id ..mask.. y_n_id
    # lm_mask:[batch_size,seq_len] , [-1 , -1 ..y_i_id.. -1,-1]
    # type : int64

    MASK = Constants.MASK
    output_labels = []
    for j, each_sen in enumerate(y):
        output_label = []
        for i, token in enumerate(each_sen):
            id = token
            prob = random.random()
            if prob < 0.7:
                # 有一定几率见到原词
                prob = prob / 0.7
                #if prob < 0.9:
                if prob < 1:
                    y[j][i] = MASK
                output_label.append(id)
            else:
                output_label.append(0)

        output_labels.append(output_label)
    return y, output_labels


def mask_target_data_wwm(y, y_str, mask_prob = 0.5):
    # whole word mask, not bpe
    # y:[batch_size,seq_len] , y_1_id , y_2_id ..y_i_id.. y_n_id
    # output:
    # y:[batch_size,seq_len], y_1_id , y_2_id ..mask.. y_n_id
    # lm_mask:[batch_size,seq_len] , [-1 , -1 ..y_i_id.. -1,-1]
    # type : int64

    MASK = Constants.MASK
    output_labels = []
    id_sent = 0
    for each_sen, each_sen_str in zip(y, y_str):
        output_label = []
        ii = 0
        while True:
            if ii >= len(each_sen): break
            id = each_sen[ii]               # 这个词的索引
            token = each_sen_str[ii]        # 这个词是什么
            prob = random.random()
            if prob < mask_prob:                  # 概率满足被mask掉的话
                y[id_sent][ii] = MASK
                y_str[id_sent][ii] = "[MASK]"
                output_label.append(id)
                while '@@' in token:
                    ii += 1
                    id = each_sen[ii]               # 这个词的索引
                    token = each_sen_str[ii]        # 这个词是什么
                    y[id_sent][ii] = MASK
                    y_str[id_sent][ii] = "[MASK]"
                    output_label.append(id)
            else:
                output_label.append(0)
                while '@@' in token:
                    ii += 1
                    id = each_sen[ii]               # 这个词的索引
                    token = each_sen_str[ii]        # 这个词是什么
                    output_label.append(0)
            ii += 1
        output_labels.append(output_label)
        id_sent += 1

    return y, y_str, output_labels


def mask_target_data_contrastive(y):
    # y:[batch_size,seq_len] , y_1_id , y_2_id ..y_i_id.. y_n_id
    # output:
    # y:[batch_size,seq_len], y_1_id , y_2_id ..mask.. y_n_id
    # lm_mask:[batch_size,seq_len] , [-1 , -1 ..y_i_id.. -1,-1]
    # type : int64

    MASK = Constants.MASK
    output_labels = []
    for j, each_sen in enumerate(y):
        output_label = []
        for i, token in enumerate(each_sen):
            id = token
            prob = random.random()
            if prob < 0.5:  # 一半mask
                y[j][i] = MASK
                output_label.append(id)
            else:
                output_label.append(0)

        output_labels.append(output_label)
    return y, output_labels


def mask_target_data_idx(y, align_position):
    # y:[batch_size,seq_len] , y_1_id , y_2_id ..y_i_id.. y_n_id
    # output:
    # y:[batch_size,seq_len], y_1_id , y_2_id ..mask.. y_n_id
    # lm_mask:[batch_size,seq_len] , [-1 , -1 ..y_i_id.. -1,-1]
    # type : int64

    MASK = Constants.MASK
    output_labels = []
    for j, each_sen in enumerate(y):
        output_label = []
        for i, token in enumerate(each_sen):
            id = token
            if i in align_position:
                y[j][i] = MASK
                output_label.append(id)
            else:
                output_label.append(0)

        output_labels.append(output_label)
    return y, output_labels


def train_mlm(flags):
    """
    flags:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # ================================================================================== #
    # Initialization for training on different devices
    # - CPU/GPU
    # - Single/Distributed
    Constants.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if Constants.USE_GPU:
        torch.cuda.set_device(local_rank)
        Constants.CURRENT_DEVICE = "cuda:{0}".format(local_rank)
    else:
        Constants.CURRENT_DEVICE = "cpu"

    # If not root_rank, close logging
    # else write log of training to file.
    if rank == 0:
        write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    else:
        close_logging()

    # ================================================================================== #
    # Parsing configuration files
    # - Load default settings
    # - Load pre-defined settings
    # - Load user-defined settings

    configs = prepare_configs(flags.config_path, flags.predefined_config)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    INFO(pretty_configs(configs))

    Constants.SEED = training_configs['seed']

    set_seed(Constants.SEED)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    Constants.EOS = vocab_src.eos
    Constants.PAD = vocab_src.pad
    Constants.BOS = vocab_src.bos
    Constants.MASK = vocab_src.mask

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        )
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        )
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs["batch_size"],
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=training_configs['buffer_size'],
                                     batching_func=training_configs['batching_key'],
                                     world_size=world_size,
                                     rank=rank,
                                     shuffle=True)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=True, buffer_size=100000, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 0. Initial

    lrate = optimizer_configs['learning_rate']
    model_collections = Collections()

    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(flags.saveto, flags.model_name)),
                             num_max_keeping=training_configs['num_kept_checkpoints']
                             )
    best_model_prefix = os.path.join(flags.saveto, flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)

    # 1. Build Model & Criterion
    INFO('Building model...')
    timer.tic()
    model = build_model(n_src_vocab=vocab_src.max_n_words,
                        n_tgt_vocab=vocab_tgt.max_n_words, padding_idx=vocab_src.pad, **model_configs)
    INFO(model)

    critic = NMTCriterion(padding_idx=vocab_tgt.pad, ignore_idx=0)
    INFO(critic)

    # 2. Move to GPU
    if Constants.USE_GPU:
        model = model.cuda()
        critic = critic.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(model, flags.pretrain_path, exclude_prefix=None, device=Constants.CURRENT_DEVICE)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')

    if not flags.multi_gpu:
        optim = Optimizer(name=optimizer_configs['optimizer'],
                          model=model,
                          lr=lrate,
                          grad_clip=optimizer_configs['grad_clip'],
                          optim_args=optimizer_configs['optimizer_params'],
                          update_cycle=training_configs['update_cycle']
                          )
    else:
        optim = dist.DistributedOptimizer(name=optimizer_configs['optimizer'],
                                          model=model,
                                          lr=lrate,
                                          grad_clip=optimizer_configs['grad_clip'],
                                          optim_args=optimizer_configs['optimizer_params'],
                                          device_id=local_rank
                                          )

    # 5. Build scheduler for optimizer if needed
    scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])

    # 6. build moving average

    if training_configs['moving_average_method'] is not None:
        ma = MovingAverage(moving_average_method=training_configs['moving_average_method'],
                           named_params=model.named_parameters(),
                           alpha=training_configs['moving_average_alpha'])
    else:
        ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # Reload from latest checkpoint
    if flags.reload:
        checkpoint_saver.load_latest(model=model,
                                     optim=optim,
                                     lr_scheduler=scheduler,
                                     collections=model_collections,
                                     ma=ma, device=Constants.CURRENT_DEVICE)

    # broadcast parameters and optimizer states
    if world_size > 1:
        INFO("Broadcasting model parameters...")
        dist.broadcast_parameters(params=model.state_dict())
        INFO("Broadcasting optimizer states...")
        dist.broadcast_optimizer_state(optimizer=optim.optim)
        INFO('Done.')

    # ================================================================================== #
    # Prepare training

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [1])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]
    is_early_stop = model_collections.get_collection("is_early_stop", [False, ])[-1]

    train_loss_meter = AverageMeter()
    sent_per_sec_meter = TimeMeter()
    tok_per_sec_meter = TimeMeter()

    update_cycle = training_configs['update_cycle']
    grad_denom = 0
    train_loss = 0.0
    cum_n_words = 0

    if rank == 0:
        summary_writer = SummaryWriter(log_dir=flags.log_path)
    else:
        summary_writer = None

    sent_per_sec_meter.start()
    tok_per_sec_meter.start()

    INFO('Begin training...')

    while True:

        if summary_writer is not None:
            summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()

        if rank == 0:
            training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                         total=len(training_iterator),
                                         unit="sents"
                                         )
        else:
            training_progress_bar = None

        for batch in training_iter:

            seqs_x, seqs_y = batch

            batch_size = len(seqs_x)
            cum_n_words += sum(len(s) for s in seqs_y)

            try:
                # Prepare data
                seqs_y, lm_ids = mask_target_data(seqs_y)
                x, y, lm_ids = prepare_data(seqs_x, seqs_y, lm_ids, cuda=Constants.USE_GPU)

                loss = compute_forward(model=model,
                                       critic=critic,
                                       seqs_x=x,
                                       seqs_y=y,
                                       lm_ids=lm_ids,
                                       lm='MLM',
                                       eval=False,
                                       normalization=1.0,
                                       norm_by_words=training_configs["norm_by_words"])

                update_cycle -= 1
                grad_denom += batch_size
                train_loss += loss

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom_count += 1
                else:
                    raise e

            # When update_cycle becomes 0, it means end of one batch. Several things will be done:
            # - update parameters
            # - reset update_cycle and grad_denom, update uidx
            # - learning rate scheduling
            # - update moving average

            if update_cycle == 0:

                # 0. reduce variables
                if world_size > 1:
                    grad_denom = dist.all_reduce_py(grad_denom)
                    train_loss = dist.all_reduce_py(train_loss)
                    cum_n_words = dist.all_reduce_py(cum_n_words)

                # 1. update parameters
                optim.step(denom=grad_denom)
                optim.zero_grad()

                if training_progress_bar is not None:
                    training_progress_bar.update(grad_denom)
                    training_progress_bar.set_description(' - (Epc {}, Upd {}) '.format(eidx, uidx))

                # 2. learning rate scheduling
                if scheduler is not None and optimizer_configs["schedule_method"] != "loss":
                    scheduler.step(global_step=uidx)

                # 3. update moving average
                if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                    ma.step()

                # 4. update meters
                train_loss_meter.update(train_loss, grad_denom)
                sent_per_sec_meter.update(grad_denom)
                tok_per_sec_meter.update(cum_n_words)

                # 5. reset accumulated variables, update uidx
                update_cycle = training_configs['update_cycle']
                grad_denom = 0
                uidx += 1
                cum_n_words = 0.0
                train_loss = 0.0

            else:
                continue

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):

                lrate = list(optim.get_lrate())[0]

                if summary_writer is not None:
                    summary_writer.add_scalar("Speed(sents/sec)", scalar_value=sent_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("Speed(words/sec)", scalar_value=tok_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss", scalar_value=train_loss_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                    summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)

                # Reset Meters
                sent_per_sec_meter.reset()
                tok_per_sec_meter.reset()
                train_loss_meter.reset()

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):
                with cache_parameters(model):

                    if ma is not None:
                        model.load_state_dict(ma.export_ma_params(), strict=False)

                    valid_loss, acc = loss_evaluation(model=model,
                                                      critic=critic,
                                                      valid_iterator=valid_iterator,
                                                      LM='MLM',
                                                      rank=rank,
                                                      world_size=world_size)

                if scheduler is not None and optimizer_configs["schedule_method"] == "loss":
                    scheduler.step(metric=valid_loss)

                model_collections.add_to_collection("history_losses", valid_loss)

                min_history_loss = np.array(model_collections.get_collection("history_losses")).min()

                if summary_writer is not None:
                    summary_writer.add_scalar("loss", valid_loss, global_step=uidx)
                    summary_writer.add_scalar("best_loss", min_history_loss, global_step=uidx)

                if valid_loss <= min_history_loss:
                    bad_count = 0
                    if is_early_stop is False and rank == 0:
                        torch.save(model.state_dict(), best_model_prefix + '.final')
                else:
                    bad_count += 1
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 1:
                        is_early_stop = True
                        WARN("Early Stop")

                INFO("{0} Loss: {1:.2f} Acc: {2:.2f} lrate: {3:6f} patience: {4}".format(
                    uidx, valid_loss, acc, lrate, bad_count
                ))
            if is_early_stop is True:
                break
            # ================================================================================== #
        if training_progress_bar is not None:
            training_progress_bar.close()

        if is_early_stop is True:
            break

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def train_mlm_wwm(flags):
    """
    whole word mask 版本
    flags:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # ================================================================================== #
    # Initialization for training on different devices
    # - CPU/GPU
    # - Single/Distributed
    Constants.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if Constants.USE_GPU:
        torch.cuda.set_device(local_rank)
        Constants.CURRENT_DEVICE = "cuda:{0}".format(local_rank)
    else:
        Constants.CURRENT_DEVICE = "cpu"

    # If not root_rank, close logging
    # else write log of training to file.
    if rank == 0:
        write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    else:
        close_logging()

    # ================================================================================== #
    # Parsing configuration files
    # - Load default settings
    # - Load pre-defined settings
    # - Load user-defined settings

    configs = prepare_configs(flags.config_path, flags.predefined_config)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    #INFO(pretty_configs(configs))

    Constants.SEED = training_configs['seed']

    set_seed(Constants.SEED)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    Constants.EOS = vocab_src.eos
    Constants.PAD = vocab_src.pad
    Constants.BOS = vocab_src.bos
    Constants.MASK = vocab_src.mask

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        TextLineStrDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        )
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        )
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs["batch_size"],
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=training_configs['buffer_size'],
                                     batching_func=training_configs['batching_key'],
                                     world_size=world_size,
                                     rank=rank,
                                     shuffle=False)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=True, buffer_size=100000, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 0. Initial

    lrate = optimizer_configs['learning_rate']
    model_collections = Collections()

    checkpoint_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(flags.saveto, flags.model_name)),
                             num_max_keeping=training_configs['num_kept_checkpoints']
                             )
    best_model_prefix = os.path.join(flags.saveto, flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)

    # 1. Build Model & Criterion
    INFO('Building model...')
    timer.tic()
    model = build_model(n_src_vocab=vocab_src.max_n_words,
                        n_tgt_vocab=vocab_tgt.max_n_words, padding_idx=vocab_src.pad, **model_configs)
    #INFO(model)

    critic = NMTCriterion(padding_idx=vocab_tgt.pad, ignore_idx=0)
    INFO(critic)

    # 2. Move to GPU
    if Constants.USE_GPU:
        model = model.cuda()
        critic = critic.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(model, flags.pretrain_path, exclude_prefix=None, device=Constants.CURRENT_DEVICE)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')

    if not flags.multi_gpu:
        optim = Optimizer(name=optimizer_configs['optimizer'],
                          model=model,
                          lr=lrate,
                          grad_clip=optimizer_configs['grad_clip'],
                          optim_args=optimizer_configs['optimizer_params'],
                          update_cycle=training_configs['update_cycle']
                          )
    else:
        optim = dist.DistributedOptimizer(name=optimizer_configs['optimizer'],
                                          model=model,
                                          lr=lrate,
                                          grad_clip=optimizer_configs['grad_clip'],
                                          optim_args=optimizer_configs['optimizer_params'],
                                          device_id=local_rank
                                          )

    # 5. Build scheduler for optimizer if needed
    scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])

    # 6. build moving average

    if training_configs['moving_average_method'] is not None:
        ma = MovingAverage(moving_average_method=training_configs['moving_average_method'],
                           named_params=model.named_parameters(),
                           alpha=training_configs['moving_average_alpha'])
    else:
        ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # Reload from latest checkpoint
    if flags.reload:
        checkpoint_saver.load_latest(model=model,
                                     optim=optim,
                                     lr_scheduler=scheduler,
                                     collections=model_collections,
                                     ma=ma, device=Constants.CURRENT_DEVICE)

    # broadcast parameters and optimizer states
    if world_size > 1:
        INFO("Broadcasting model parameters...")
        dist.broadcast_parameters(params=model.state_dict())
        INFO("Broadcasting optimizer states...")
        dist.broadcast_optimizer_state(optimizer=optim.optim)
        INFO('Done.')

    # ================================================================================== #
    # Prepare training

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [1])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]
    is_early_stop = model_collections.get_collection("is_early_stop", [False, ])[-1]

    train_loss_meter = AverageMeter()
    sent_per_sec_meter = TimeMeter()
    tok_per_sec_meter = TimeMeter()

    update_cycle = training_configs['update_cycle']
    grad_denom = 0
    train_loss = 0.0
    cum_n_words = 0

    if rank == 0:
        summary_writer = SummaryWriter(log_dir=flags.log_path)
    else:
        summary_writer = None

    sent_per_sec_meter.start()
    tok_per_sec_meter.start()

    INFO('Begin training...')

    while True:

        if summary_writer is not None:
            summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()

        if rank == 0:
            training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                         total=len(training_iterator),
                                         unit="sents"
                                         )
        else:
            training_progress_bar = None

        for batch in training_iter:

            seqs_x, seqs_y, seqs_y_str = batch

            batch_size = len(seqs_x)
            cum_n_words += sum(len(s) for s in seqs_y)

            try:
                # Prepare data
                seqs_y, seqs_y_str, lm_ids = mask_target_data_wwm(seqs_y, seqs_y_str, mask_prob = 0.5)
                x, y, lm_ids = prepare_data(seqs_x, seqs_y, lm_ids, cuda=Constants.USE_GPU)

                loss = compute_forward(model=model,
                                       critic=critic,
                                       seqs_x=x,
                                       seqs_y=y,
                                       lm_ids=lm_ids,
                                       lm='MLM',
                                       eval=False,
                                       normalization=1.0,
                                       norm_by_words=training_configs["norm_by_words"])

                update_cycle -= 1
                grad_denom += batch_size
                train_loss += loss

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom_count += 1
                else:
                    raise e

            # When update_cycle becomes 0, it means end of one batch. Several things will be done:
            # - update parameters
            # - reset update_cycle and grad_denom, update uidx
            # - learning rate scheduling
            # - update moving average

            if update_cycle == 0:

                # 0. reduce variables
                if world_size > 1:
                    grad_denom = dist.all_reduce_py(grad_denom)
                    train_loss = dist.all_reduce_py(train_loss)
                    cum_n_words = dist.all_reduce_py(cum_n_words)

                # 1. update parameters
                optim.step(denom=grad_denom)
                optim.zero_grad()

                if training_progress_bar is not None:
                    training_progress_bar.update(grad_denom)
                    training_progress_bar.set_description(' - (Epc {}, Upd {}) '.format(eidx, uidx))

                # 2. learning rate scheduling
                if scheduler is not None and optimizer_configs["schedule_method"] != "loss":
                    scheduler.step(global_step=uidx)

                # 3. update moving average
                if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                    ma.step()

                # 4. update meters
                train_loss_meter.update(train_loss, grad_denom)
                sent_per_sec_meter.update(grad_denom)
                tok_per_sec_meter.update(cum_n_words)

                # 5. reset accumulated variables, update uidx
                update_cycle = training_configs['update_cycle']
                grad_denom = 0
                uidx += 1
                cum_n_words = 0.0
                train_loss = 0.0

            else:
                continue

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):

                lrate = list(optim.get_lrate())[0]

                if summary_writer is not None:
                    summary_writer.add_scalar("Speed(sents/sec)", scalar_value=sent_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("Speed(words/sec)", scalar_value=tok_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss", scalar_value=train_loss_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                    summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)

                # Reset Meters
                sent_per_sec_meter.reset()
                tok_per_sec_meter.reset()
                train_loss_meter.reset()

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):
                with cache_parameters(model):

                    if ma is not None:
                        model.load_state_dict(ma.export_ma_params(), strict=False)

                    valid_loss, acc = loss_evaluation(model=model,
                                                        critic=critic,
                                                        valid_iterator=valid_iterator,
                                                        LM='MLM',
                                                        rank=rank,
                                                        world_size=world_size)

                if scheduler is not None and optimizer_configs["schedule_method"] == "loss":
                    scheduler.step(metric=valid_loss)

                model_collections.add_to_collection("history_losses", valid_loss)

                min_history_loss = np.array(model_collections.get_collection("history_losses")).min()

                if summary_writer is not None:
                    summary_writer.add_scalar("loss", valid_loss, global_step=uidx)
                    summary_writer.add_scalar("best_loss", min_history_loss, global_step=uidx)

                if valid_loss <= min_history_loss:
                    bad_count = 0
                    if is_early_stop is False and rank == 0:
                        torch.save(model.state_dict(), best_model_prefix + '.final')
                else:
                    bad_count += 1
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 1:
                        is_early_stop = True
                        WARN("Early Stop")

                INFO("{0} Loss: {1:.2f} Acc: {2:.2f} lrate: {3:6f} patience: {4}".format(
                    uidx, valid_loss, acc, lrate, bad_count
                ))
            if is_early_stop is True:
                break
            # ================================================================================== #
        if training_progress_bar is not None:
            training_progress_bar.close()

        if is_early_stop is True:
            break

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def gen_noise_data(gold, pre, prob, k):
    # gold: reference logits
    # pre: generated logits
    # prob: the prob to choose noise
    # k: the quality of noise data
    # mix the gold logits and pre logits to gen noise data which includes word-level noise

    assert k > 1
    pre = pre.topk(k=k, dim=2)
    batch_size, seq_len = pre[1][:, :, 0].size()
    pre_prev = pre[1][:, :, k - 1].cpu().tolist()
    pre_kth = pre[1][:, :, k - 2].cpu().tolist()

    gold = gold.cpu().tolist()
    noise_data = []
    noise_label = []
    hter_label = []
    for each_sen_1st, each_sen_2nd, each_sen_gold in zip(pre_prev, pre_kth, gold):
        bad_count = 0.0 + 1e-12
        all_count = 0.0 + 1e-12
        for each_1st, each_2nd, each_gold in zip(each_sen_1st, each_sen_2nd, each_sen_gold):
            if each_gold == Constants.PAD or each_gold == Constants.EOS:
                noise_data.append(each_gold)
                noise_label.append(0)
            elif each_gold == Constants.BOS:
                noise_data.append(Constants.BOS)
                noise_data.append(Constants.HTER)
                noise_label.append(0)
                noise_label.append(0)
            else:
                all_count += 1
                p = random.random()
                if p < prob:
                    noise_data.append(each_1st if each_1st != each_gold else each_2nd)
                    bad_count += 1
                    noise_label.append(2)
                else:
                    noise_data.append(each_gold)
                    noise_label.append(1)
        hter_label.append(bad_count / all_count)

    noise_data = torch.tensor(noise_data).view((batch_size, seq_len + 1)).cuda()
    noise_label = torch.tensor(noise_label).view((batch_size, seq_len + 1)).cuda()
    hter_label = torch.tensor(hter_label).view((batch_size, 1)).cuda()

    return noise_data, noise_label, hter_label


def gen_noise_data2(gold, pre, lm_ids):
    # 不按照prob选，而是只选mask位置的那些词；
    # 直接选生成概率最大的词，就是ELE...的做法

    pre = pre.topk(k=1, dim=2)
    batch_size, seq_len = pre[1][:, :, 0].size()
    pre = pre[1][:, :, 0].cpu().tolist()

    lm_ids = lm_ids.cpu().tolist()
    gold = gold.cpu().tolist()
    noise_data = []
    noise_label = []
    hter_label = []
    for each_sen_1st, each_sen_gold, each_sen_lm in zip(pre, gold, lm_ids):
        bad_count = 0.0 + 1e-12
        all_count = 0.0 + 1e-12
        for each_1st, each_gold, each_id in zip(each_sen_1st, each_sen_gold, each_sen_lm):
            if each_gold == Constants.PAD or each_gold == Constants.EOS:
                noise_data.append(each_gold)
                noise_label.append(0)
            elif each_gold == Constants.BOS:
                noise_data.append(Constants.BOS)
                noise_data.append(Constants.HTER)
                noise_label.append(0)
                noise_label.append(0)
            else:
                all_count += 1
                if each_id != 0 and each_1st != each_gold:
                    noise_data.append(each_1st)
                    bad_count += 1
                    noise_label.append(2)
                else:
                    noise_data.append(each_gold)
                    noise_label.append(1)
        hter_label.append(bad_count / all_count)

    noise_data = torch.tensor(noise_data).view((batch_size, seq_len + 1)).cuda()
    noise_label = torch.tensor(noise_label).view((batch_size, seq_len + 1)).cuda()
    hter_label = torch.tensor(hter_label).view((batch_size, 1)).cuda()

    return noise_data, noise_label, hter_label


def gen_noise_data3(gold, pre, lm_ids):
    # 不按照prob选，而是只选mask位置的那些词；
    # 直接选生成概率最大的词，就是ELE...的做法

    pre = pre.topk(k=10, dim=2)
    batch_size, seq_len = pre[1][:, :, 0].size()
    pre = pre[1][:, :].cpu().tolist()
    # 这个时候each_1st就是一个长度为10的列表

    lm_ids = lm_ids.cpu().tolist()
    gold = gold.cpu().tolist()
    noise_data = []
    noise_label = []
    hter_label = []
    for each_sen_1st, each_sen_gold, each_sen_lm in zip(pre, gold, lm_ids):
        bad_count = 0.0 + 1e-12
        all_count = 0.0 + 1e-12
        for each_1st, each_gold, each_id in zip(each_sen_1st, each_sen_gold, each_sen_lm):
            if each_gold == Constants.PAD or each_gold == Constants.EOS:
                noise_data.append(each_gold)
                noise_label.append(0)
            elif each_gold == Constants.BOS:
                noise_data.append(Constants.BOS)
                noise_data.append(Constants.HTER)
                noise_label.append(0)
                noise_label.append(0)
            else:
                all_count += 1
                choose_one = random.choice(each_1st)
                if each_id != 0 and choose_one != each_gold:
                    noise_data.append(choose_one)
                    bad_count += 1
                    noise_label.append(2)
                else:
                    noise_data.append(each_gold)
                    noise_label.append(1)
        hter_label.append(bad_count / all_count)

    noise_data = torch.tensor(noise_data).view((batch_size, seq_len + 1)).cuda()
    noise_label = torch.tensor(noise_label).view((batch_size, seq_len + 1)).cuda()
    hter_label = torch.tensor(hter_label).view((batch_size, 1)).cuda()

    return noise_data, noise_label, hter_label


def gen_noise_data4(gold, pre, lm_ids):
    # 不按照prob选，而是只选mask位置的那些词；
    # 就选生成概率第五大的词。

    pre = pre.topk(k=10, dim=2)
    batch_size, seq_len = pre[1][:, :, 0].size()

    pre = pre[1][:, :, 4].cpu().tolist()
    # 这个时候each_1st就是一个id

    lm_ids = lm_ids.cpu().tolist()
    gold = gold.cpu().tolist()
    noise_data = []
    noise_label = []
    hter_label = []
    for each_sen_1st, each_sen_gold, each_sen_lm in zip(pre, gold, lm_ids):
        bad_count = 0.0 + 1e-12
        all_count = 0.0 + 1e-12
        for each_1st, each_gold, each_id in zip(each_sen_1st, each_sen_gold, each_sen_lm):
            if each_gold == Constants.PAD or each_gold == Constants.EOS:
                noise_data.append(each_gold)
                noise_label.append(0)
            elif each_gold == Constants.BOS:
                noise_data.append(Constants.BOS)
                noise_data.append(Constants.HTER)
                noise_label.append(0)
                noise_label.append(0)
            else:
                all_count += 1
                choose_one = each_1st
                if each_id != 0 and choose_one != each_gold:
                    noise_data.append(choose_one)
                    bad_count += 1
                    noise_label.append(2)
                else:
                    noise_data.append(each_gold)
                    noise_label.append(1)
        hter_label.append(bad_count / all_count)

    noise_data = torch.tensor(noise_data).view((batch_size, seq_len + 1)).cuda()
    noise_label = torch.tensor(noise_label).view((batch_size, seq_len + 1)).cuda()
    hter_label = torch.tensor(hter_label).view((batch_size, 1)).cuda()

    return noise_data, noise_label, hter_label


def gen_noise_data5(gold, pre, lm_ids, vocab_size):
    # 不按照prob选，而是只选mask位置的那些词；
    # 在除了最大生辰概率的词之外的整个词表上按照概率随机采样，同时，需要注意，这里的softmax是带温度的。
    pre = pre.topk(k=50, dim=2)
    #print(pre)
    batch_size, seq_len = pre[1][:, :, 0].size()

    logits = pre[0].cpu().tolist()
    pre = pre[1].cpu().tolist()
    # 这个时候each_1st就是一个词表大小的序列

    lm_ids = lm_ids.cpu().tolist()
    gold = gold.cpu().tolist()
    noise_data = []
    noise_label = []
    hter_label = []

    for i, each_batch in enumerate(logits):
        for j, each_token in enumerate(each_batch):
            # each_token[np.argmax(each_token)] = 0
            choose_one = random.choices(pre[i][j], weights=each_token)[0]
            pre[i][j] = choose_one

    for each_sen_1st, each_sen_gold, each_sen_lm in zip(pre, gold, lm_ids):
        bad_count = 0.0 + 1e-12
        all_count = 0.0 + 1e-12
        for each_1st, each_gold, each_id in zip(each_sen_1st, each_sen_gold, each_sen_lm):
            if each_gold == Constants.PAD or each_gold == Constants.EOS:
                noise_data.append(each_gold)
                noise_label.append(0)
            elif each_gold == Constants.BOS:
                noise_data.append(Constants.BOS)
                noise_data.append(Constants.HTER)
                noise_label.append(0)
                noise_label.append(0)
            else:
                all_count += 1
                choose_one = each_1st
                if each_id != 0 and choose_one != each_gold:  # 是被mask的token，而且模型选的不一样，才bad
                    noise_data.append(choose_one)
                    bad_count += 1
                    noise_label.append(2)
                else:
                    noise_data.append(each_gold)
                    noise_label.append(1)
        hter_label.append(bad_count / all_count)

    noise_data = torch.tensor(noise_data).view((batch_size, seq_len + 1)).cuda()
    noise_label = torch.tensor(noise_label).view((batch_size, seq_len + 1)).cuda()
    hter_label = torch.tensor(hter_label).view((batch_size, 1)).cuda()

    return noise_data, noise_label, hter_label


def gen_noise_data5_y2x(seqs_x, gold, pre, lm_ids, seqs_index, yx_align_dict):
    # 不按照prob选，而是只选mask位置的那些词；
    # 在除了最大生辰概率的词之外的整个词表上按照概率随机采样，同时，需要注意，这里的softmax是带温度的。
    pre = pre.topk(k=50, dim=2)
    #print(pre)
    batch_size, seq_len = pre[1][:, :, 0].size()

    logits = pre[0].cpu().tolist()
    pre = pre[1].cpu().tolist()
    # 这个时候each_1st就是一个词表大小的序列

    lm_ids = lm_ids.cpu().tolist()
    gold = gold.cpu().tolist()
    noise_data = []
    noise_label = []
    hter_label = []

    for i, each_batch in enumerate(logits):
        for j, each_token in enumerate(each_batch):
            # each_token[np.argmax(each_token)] = 0
            choose_one = random.choices(pre[i][j], weights=each_token)[0]
            pre[i][j] = choose_one

    sent_id = 0
    for each_sen_1st, each_sen_gold, each_sen_lm in zip(pre, gold, lm_ids):
        sent_index = int(seqs_index[sent_id].squeeze().item())
        noise_label_line_y = []
        yx_align_line = yx_align_dict[str(sent_index)]
        for each_1st, each_gold, each_id in zip(each_sen_1st, each_sen_gold, each_sen_lm): # 分别是一个元素
            if each_gold == Constants.PAD or each_gold == Constants.EOS:
                noise_data.append(each_gold)
            elif each_gold == Constants.BOS:
                noise_data.append(Constants.BOS)  # TODO noiseDATA不大对劲，是x还是y，要不要加hter？？
            else:
                choose_one = each_1st
                if each_id != 0 and choose_one != each_gold:  # 是被mask的token，而且模型选的不一样，才bad
                    noise_data.append(choose_one)
                    noise_label_line_y.append(2)
                else:
                    noise_data.append(each_gold)
                    noise_label_line_y.append(1)
        # 这句话的y的noise_label转x的noise_label
        # 直接每个bad en token映射到bad zh token
        noise_label_x = [1] * len(seqs_x[sent_id])  # x即zh的bpe句长度，暂且都ok
        for i, y_label in enumerate(noise_label_line_y):
            if y_label == 2:
                if str(i) in yx_align_line:
                    bad_x_list = yx_align_line[str(i)]
                    noise_label_x = [2 if id in bad_x_list else noise_label_x[id] for id in range(len(noise_label_x))]
        hter_label.append(noise_label_x.count(2) / len(noise_label_x))
        sent_id += 1
        noise_label.append(noise_label_x)
    noise_data = torch.tensor(noise_data).view((batch_size, seq_len)).cuda()  # 两侧各加1
    hter_label = torch.tensor(hter_label).view((batch_size, 1)).cuda()

    return noise_data, noise_label, hter_label


def gen_noise_data5_y2x_robust(gold, pre, lm_ids):
    # 不按照prob选，而是只选mask位置的那些词；
    # 在除了最大生辰概率的词之外的整个词表上按照概率随机采样，同时，需要注意，这里的softmax是带温度的。
    pre = pre.topk(k=50, dim=2)
    #print(pre)
    batch_size, seq_len = pre[1][:, :, 0].size()

    logits = pre[0].cpu().tolist()
    pre = pre[1].cpu().tolist()
    # 这个时候each_1st就是一个词表大小的序列

    lm_ids = lm_ids.cpu().tolist()
    gold = gold.cpu().tolist()
    noise_data = []

    for i, each_batch in enumerate(logits):
        for j, each_token in enumerate(each_batch):
            # each_token[np.argmax(each_token)] = 0
            choose_one = random.choices(pre[i][j], weights=each_token)[0]
            pre[i][j] = choose_one

    for each_sen_1st, each_sen_gold, each_sen_lm in zip(pre, gold, lm_ids):
        flag_change = False
        for each_1st, each_gold, each_id in zip(each_sen_1st, each_sen_gold, each_sen_lm): # 分别是一个元素
            if each_gold == Constants.PAD or each_gold == Constants.EOS: 
                noise_data.append(each_gold)
            elif each_gold == Constants.BOS: 
                noise_data.append(Constants.BOS)
            else:
                choose_one = each_1st
                if each_id != 0 and choose_one != each_gold:  # 是被mask的token，而且模型选的不一样，就说明标签改变
                    noise_data.append(choose_one)
                    flag_change = True
                else:
                    noise_data.append(each_gold)
        
    noise_data = torch.tensor(noise_data).view((batch_size, seq_len)).cuda()  # 两侧各加1

    return noise_data, flag_change


def gen_noise_data5_y2x_robust_must(gold, pre, lm_ids):
    # 必须生成和原来不一样的！
    pre = pre.topk(k=50, dim=2)
    batch_size, seq_len = pre[1][:, :, 0].size()

    logits = pre[0].cpu().tolist()
    pre = pre[1].cpu().tolist()
    # 这个时候each_1st就是一个词表大小的序列

    lm_ids = lm_ids.cpu().tolist()
    gold = gold.cpu().tolist()
    noise_data = []

    sent_id = 0
    for each_sen_gold, each_sen_lm in zip(gold, lm_ids):
        token_id = 0
        for each_gold, each_id in zip(each_sen_gold, each_sen_lm): # 分别是一个元素
            if each_gold == Constants.PAD or each_gold == Constants.EOS: 
                noise_data.append(each_gold)
            elif each_gold == Constants.BOS: 
                noise_data.append(Constants.BOS)
            else:
                choose_3 = random.choices(pre[sent_id][token_id], weights=logits[sent_id][token_id], k = 3)
                if each_id != 0:
                    # 需要改变
                    choose_one = random.choices(pre[sent_id][token_id], weights=logits[sent_id][token_id])[0]
                    while(choose_one == each_gold):
                        choose_one = random.choices(pre[sent_id][token_id], weights=logits[sent_id][token_id])[0]
                    noise_data.append(choose_one)
                    flag_change = True
                else:
                    # 无需改变
                    noise_data.append(each_gold)
            token_id += 1
        sent_id += 1
        
    noise_data = torch.tensor(noise_data).view((batch_size, seq_len)).cuda()  # 两侧各加1

    return noise_data


def rand_noise_data(gold, prob, vocab_size):

    batch_size, seq_len = gold.size()
    gold = gold.cpu().tolist()
    noise_data = []
    noise_label = []
    hter_label = []

    for each_sen_gold in gold:
        bad_count = 0.0 + 1e-12
        all_count = 0.0 + 1e-12
        for each_gold in each_sen_gold:
            if each_gold == Constants.PAD or each_gold == Constants.EOS:
                noise_data.append(each_gold)
                noise_label.append(0)
            elif each_gold == Constants.BOS:
                noise_data.append(Constants.BOS)
                noise_data.append(Constants.HTER)
                noise_label.append(0)
                noise_label.append(0)
            else:
                all_count += 1
                p = random.random()
                if p < prob:
                    noise_data.append(random.randint(10, vocab_size-10))
                    bad_count += 1
                    noise_label.append(2)
                else:
                    noise_data.append(each_gold)
                    noise_label.append(1)
        hter_label.append(bad_count / all_count)

    noise_data = torch.tensor(noise_data).view((batch_size, seq_len + 1)).cuda()
    noise_label = torch.tensor(noise_label).view((batch_size, seq_len + 1)).cuda()
    hter_label = torch.tensor(hter_label).view((batch_size, 1)).cuda()

    return noise_data, noise_label, hter_label


def generate_dev(flags):
    """
    flags:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # ================================================================================== #
    # Initialization for training on different devices
    # - CPU/GPU
    # - Single/Distributed
    Constants.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if Constants.USE_GPU:
        torch.cuda.set_device(local_rank)
        Constants.CURRENT_DEVICE = "cuda:{0}".format(local_rank)
    else:
        Constants.CURRENT_DEVICE = "cpu"

    # If not root_rank, close logging
    # else write log of training to file.
    if rank == 0:
        write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    else:
        close_logging()

    # ================================================================================== #
    # Parsing configuration files
    # - Load default settings
    # - Load pre-defined settings
    # - Load user-defined settings

    configs = prepare_configs(flags.config_path, flags.predefined_config)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    training_configs = configs['training_configs']

    # INFO(pretty_configs(configs))

    Constants.SEED = training_configs['seed']

    set_seed(Constants.SEED)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    Constants.EOS = vocab_src.eos
    Constants.PAD = vocab_src.pad
    Constants.BOS = vocab_src.bos
    Constants.MASK = vocab_src.mask

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        )
    )
    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs["batch_size"],
                                     use_bucket=False,
                                     buffer_size=training_configs['buffer_size'],
                                     batching_func=training_configs['batching_key'],
                                     world_size=world_size,
                                     rank=rank,
                                     shuffle=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Building model...')
    timer.tic()
    model = build_model(n_src_vocab=vocab_src.max_n_words,
                        n_tgt_vocab=vocab_tgt.max_n_words, padding_idx=vocab_src.pad, **model_configs)
    # INFO(model)
    if Constants.USE_GPU:
        model = model.cuda()
    load_pretrained_model(model, flags.pretrain_path, exclude_prefix=None, device=Constants.CURRENT_DEVICE)

    INFO('Begin training...')
    training_iter = training_iterator.build_generator()
    f_x = open(flags.saveto + 'mlm_dev.src.BPE', 'w', encoding='utf-8')
    f_y = open(flags.saveto + 'mlm_dev.mt.BPE', 'w', encoding='utf-8')
    f_l = open(flags.saveto + 'mlm_dev.tag', 'w', encoding='utf-8')
    f_h = open(flags.saveto + 'mlm_dev.hter', 'w', encoding='utf-8')
    hter = []
    for batch in training_iter:
        seqs_x, seqs_y = batch
        # Prepare data
        # seqs_y, lm_ids = mask_target_data(seqs_y)
        x, y, y_mask, y_reverse, lm_ids = prepare_data_all(seqs_x, seqs_y, cuda=Constants.USE_GPU)
        # x, y, lm_ids = prepare_data(seqs_x, seqs_y, lm_ids, cuda=Constants.USE_GPU)
        with torch.no_grad():
            mlm_logits = model(x, y_mask, 'MLM', log_probs=False)
        word_noise_data, word_noise_label, word_hter_label = gen_noise_data5(y, mlm_logits, lm_ids, 100)
        for line_src, line_tgt, line_tag, line_hter in zip(x, word_noise_data, word_noise_label, word_hter_label):
            line_src = line_src.cpu().tolist()
            line_tgt = line_tgt.cpu().tolist()
            line_tag = line_tag.cpu().tolist()
            line_hter = line_hter.cpu().tolist()
            line_src = vocab_src.ids2sent(line_src)
            line_tgt = vocab_tgt.ids2sent(line_tgt[2:])
            f_x.write(line_src + '\n')
            f_y.write(line_tgt + '\n')
            new_line_tag = []
            for each_tag in line_tag:
                if each_tag == 1:
                    new_line_tag.append('OK')
                elif each_tag == 2:
                    new_line_tag.append('BAD')
            f_l.write(' '.join(new_line_tag) + '\n')
            if line_hter[0] < 0.001:
                line_hter[0] = 0
            f_h.write(str(line_hter[0])[:6] + '\n')
            hter.append(line_hter[0])
        continue
    print('mean_hter:%lf'%(sum(hter)/len(hter)))


def generate_dev_src(flags):
    # mask掉src端数据，需要根据对齐文件，知道mt哪里标bad
    """
    flags:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # ================================================================================== #
    # Initialization for training on different devices
    # - CPU/GPU
    # - Single/Distributed
    Constants.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if Constants.USE_GPU:
        torch.cuda.set_device(local_rank)
        Constants.CURRENT_DEVICE = "cuda:{0}".format(local_rank)
    else:
        Constants.CURRENT_DEVICE = "cpu"

    # If not root_rank, close logging
    # else write log of training to file.
    if rank == 0:
        write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    else:
        close_logging()

    # ================================================================================== #
    # Parsing configuration files
    # - Load default settings
    # - Load pre-defined settings
    # - Load user-defined settings

    configs = prepare_configs(flags.config_path, flags.predefined_config)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    training_configs = configs['training_configs']

    # INFO(pretty_configs(configs))

    Constants.SEED = training_configs['seed']

    set_seed(Constants.SEED)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])
    with open(flags.align_bpe_path, 'r', encoding='utf-8') as f:
        yx_align_dict = json.load(f)

    Constants.EOS = vocab_src.eos
    Constants.PAD = vocab_src.pad
    Constants.BOS = vocab_src.bos
    Constants.MASK = vocab_src.mask

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        TextLineStrDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QEHterDataset(data_path=data_configs['train_data'][2]),
    )
    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs["batch_size"],
                                     use_bucket=False,
                                     buffer_size=training_configs['buffer_size'],
                                     batching_func=training_configs['batching_key'],
                                     world_size=world_size,
                                     rank=rank,
                                     shuffle=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Building model...')
    timer.tic()
    model = build_model(n_src_vocab=vocab_src.max_n_words,
                        n_tgt_vocab=vocab_tgt.max_n_words, padding_idx=vocab_src.pad, **model_configs)
    # INFO(model)
    if Constants.USE_GPU:
        model = model.cuda()
    load_pretrained_model(model, flags.pretrain_path, exclude_prefix=None, device=Constants.CURRENT_DEVICE)

    INFO('Begin training...')
    training_iter = training_iterator.build_generator()
    split = flags.split
    f_x = open(flags.saveto + 'pseudo_' + split + '.src.BPE', 'w', encoding='utf-8')
    f_y = open(flags.saveto + 'pseudo_' + split + '.mt.BPE', 'w', encoding='utf-8')
    f_l = open(flags.saveto + 'pseudo_' + split + '.tag.BPE', 'w', encoding='utf-8')
    f_h = open(flags.saveto + 'pseudo_' + split + '.hter', 'w', encoding='utf-8')
    f_x_parallel = open(flags.saveto + 'parallel_' + split + '.src.BPE', 'w', encoding='utf-8')
    f_y_parallel = open(flags.saveto + 'parallel_' + split + '.mt.BPE', 'w', encoding='utf-8')
    f_l_parallel = open(flags.saveto + 'parallel_' + split + '.tag.BPE', 'w', encoding='utf-8')
    f_h_parallel = open(flags.saveto + 'parallel_' + split + '.hter', 'w', encoding='utf-8')
    hter = []
    for batch in training_iter:
        seqs_x, seqs_y, seqs_y_str, seqs_index = batch
        # Prepare data
        # seqs_y, lm_ids = mask_target_data(seqs_y)
        x, y, y_mask, y_reverse, lm_ids = prepare_data_all(seqs_x, seqs_y, wwm=True, seqs_y_str=seqs_y_str, cuda=Constants.USE_GPU)
        
        if Constants.USE_GPU:
            seqs_index = torch.tensor(seqs_index).squeeze(-1).cuda()

        # 保存平行语料
        for line_src, line_tgt in zip(y, x):
            line_src = line_src.cpu().tolist()
            line_tgt = line_tgt.cpu().tolist()
            #print(line_tgt)
            new_line_tag = []
            for i in range(sum(e>2 for e in line_tgt)):
                new_line_tag.append('OK')
            f_l_parallel.write(' '.join(new_line_tag) + '\n')
            line_src = vocab_tgt.ids2sent(line_src)
            line_tgt = vocab_src.ids2sent(line_tgt[1:])
            f_x_parallel.write(line_src + '\n')
            f_y_parallel.write(line_tgt + '\n')
            f_h_parallel.write(str(0) + '\n')

        # 生成并保存伪数据
        with torch.no_grad():
            mlm_logits = model(x, y_mask, 'MLM', log_probs=False)  # [batch_size, seq_len, hidden_dim]
        word_noise_data_y, word_noise_label_x, word_hter_label_x = gen_noise_data5_y2x(seqs_x, y, mlm_logits, lm_ids, seqs_index, yx_align_dict)

        for line_y, line_x, line_tag, line_hter in zip(word_noise_data_y, x, word_noise_label_x, word_hter_label_x):
            line_y = line_y.cpu().tolist()[1:]
            line_x = line_x.cpu().tolist()
            line_hter = line_hter.cpu().tolist()
            line_y = vocab_tgt.ids2sent(line_y)
            line_x = vocab_src.ids2sent(line_x)
            f_x.write(line_y + '\n')
            f_y.write(line_x + '\n')
            new_line_tag = []
            for each_tag in line_tag:
                if each_tag == 1:
                    new_line_tag.append('OK')
                elif each_tag == 2:
                    new_line_tag.append('BAD')
            f_l.write(' '.join(new_line_tag) + '\n')
            if line_hter[0] < 0.001:
                line_hter[0] = 0
            f_h.write(str(line_hter[0])[:6] + '\n')
            hter.append(line_hter[0])
        continue
    print('mean_hter:%lf'%(sum(hter)/len(hter)))


def generate_robust(flags):
    # mask掉src端数据，需要根据对齐文件，知道mt哪里标bad
    # 基于qe测试集生成鲁棒性测试集，对每个ok token，都生成一条相应的bad
    """
    flags:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # ================================================================================== #
    # Initialization for training on different devices
    # - CPU/GPU
    # - Single/Distributed
    Constants.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if Constants.USE_GPU:
        torch.cuda.set_device(local_rank)
        Constants.CURRENT_DEVICE = "cuda:{0}".format(local_rank)
    else:
        Constants.CURRENT_DEVICE = "cpu"

    # If not root_rank, close logging
    # else write log of training to file.
    if rank == 0:
        write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    else:
        close_logging()

    # ================================================================================== #
    # Parsing configuration files
    # - Load default settings
    # - Load pre-defined settings
    # - Load user-defined settings

    configs = prepare_configs(flags.config_path, flags.predefined_config)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    training_configs = configs['training_configs']

    # INFO(pretty_configs(configs))

    Constants.SEED = training_configs['seed']

    set_seed(Constants.SEED)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])
    with open(flags.align_bpe_path_mt2src, 'r', encoding='utf-8') as f:
        align_bpe_dict_mt2src = json.load(f)

    Constants.EOS = vocab_src.eos
    Constants.PAD = vocab_src.pad
    Constants.BOS = vocab_src.bos
    Constants.MASK = vocab_src.mask

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2]),
        QEHterDataset(data_path=data_configs['train_data'][3]),
        TextLineStrDataset(data_path=data_configs['train_data'][0],
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineStrDataset(data_path=data_configs['train_data'][1],
                        max_len=data_configs['max_len'][1],
                        ),
    )
    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs["batch_size"],    # 1
                                     use_bucket=False,
                                     buffer_size=training_configs['buffer_size'],
                                     batching_func=training_configs['batching_key'],
                                     world_size=world_size,
                                     rank=rank,
                                     shuffle=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Building model...')
    timer.tic()
    model = build_model(n_src_vocab=vocab_tgt.max_n_words,
                        n_tgt_vocab=vocab_src.max_n_words, padding_idx=vocab_tgt.pad, **model_configs)
    # INFO(model)
    if Constants.USE_GPU:
        model = model.cuda()
    load_pretrained_model(model, flags.pretrain_path, exclude_prefix=None, device=Constants.CURRENT_DEVICE)

    INFO('Begin training...')
    training_iter = training_iterator.build_generator()

    split = flags.split
    f_x = open(flags.saveto + 'robust_' + split + '.src.BPE', 'w', encoding='utf-8')
    f_y = open(flags.saveto + 'robust_' + split + '.mt.BPE', 'w', encoding='utf-8')
    f_label = open(flags.saveto + 'robust_' + split + '.tag.BPE', 'w', encoding='utf-8')
    f_hter = open(flags.saveto + 'robust_' + split + '.hter', 'w', encoding='utf-8')
    f_idx = open(flags.saveto + 'robust_' + split + '.idx.BPE', 'w', encoding='utf-8')

    id = 0
    for batch in training_iter:   # batch = 1
        print(id)
        seqs_x, seqs_y, xy_label, xy_hter, seqs_x_str, seqs_y_str = batch
        xy_hter = torch.tensor(xy_hter).squeeze()
        align_bpe_dict_line = align_bpe_dict_mt2src[str(id)]  # 这句话中每个mt token对齐的src token们

        # 收集这个样本衍生的伪数据
        robust_x_list = []
        robust_y_list = []
        robust_label_list = []
        robust_hter_list = []
        robust_idx_list = []
        
        robust_idx_list_real = []

        y_len = len(seqs_y[-1])    # mt句的长度
        for i in range(y_len):   # 对mt每个位置
            if xy_label[0][i] == 2: continue     # bad情况不考虑
            if seqs_y[0][i] == 3: continue   # 跳过unk
            if i and "@@" in seqs_y_str[0][i - 1]: continue    # 不考虑bpe的非第一个token
            if str(i) not in align_bpe_dict_line: continue     # 不考虑没有对齐的src token的mt token
            align_position = align_bpe_dict_line[str(i)]
            x, y, x_mask, lm_ids_x = prepare_data_all_robust(seqs_x, seqs_y, align_position=align_position, cuda=Constants.USE_GPU)

            # 生成伪数据
            with torch.no_grad():
                mlm_logits_x = model(y, x_mask, 'MLM', log_probs=False)  # [batch_size, seq_len, hidden_dim]
            
            # word_noise_data_x, flag_change = gen_noise_data5_y2x_robust(x, mlm_logits_x, lm_ids_x)
            word_noise_data_x = gen_noise_data5_y2x_robust_must(x, mlm_logits_x, lm_ids_x)
            flag_change = True
            if flag_change:
                robust_idx_list_this = []
                word_noise_label_y = xy_label[0][:]
                word_noise_label_y[i] = 2          # 标签变为bad
                robust_idx_list_this.append(i)
                while "@@" in seqs_y_str[0][i]:
                    word_noise_label_y[i + 1] = 2
                    robust_idx_list_this.append(i + 1)
                    i += 1
                robust_idx_list_real.extend(robust_idx_list_this)
                robust_x_list.append(vocab_src.ids2sent(word_noise_data_x[0].cpu().tolist()))
                robust_y_list.append(seqs_y_str[0])
                robust_label_list.append(word_noise_label_y)
                robust_hter_list.append(word_noise_label_y.count(2) / len(word_noise_label_y))
                robust_idx_list.append(robust_idx_list_this)
            
        if len(robust_idx_list):
            f_x.write(' '.join(seqs_x_str[0]) + '\n')
            f_y.write(' '.join(seqs_y_str[0]) + '\n')
            temp_label = ['OK' if item == 1 else 'BAD' for item in xy_label[0]]
            f_label.write(' '.join(temp_label) + '\n')
            f_hter.write('%.6f\n' % xy_hter.item())  # TODO
            f_idx.write(' '.join(list(map(str, robust_idx_list_real))) + '\n')

            for x, y, label, hter, idx in zip(robust_x_list, robust_y_list, robust_label_list, robust_hter_list, robust_idx_list):
                f_x.write(x + '\n')
                f_y.write(' '.join(y) + '\n')
                temp_label = ['OK' if item == 1 else 'BAD' for item in label]
                f_label.write(' '.join(temp_label) + '\n')
                f_hter.write('%.6f\n' % hter)
                f_idx.write(' '.join(list(map(str, idx))) + '\n')
        
        id += 1
        #if id >= 3: break

def generate_robust_50(flags):
    # mask掉src端数据，需要根据对齐文件，知道mt哪里标bad
    # 基于qe训练集生成一个，把ok的50% mask掉换bad的版本，每条数据只生成一个对应的伪数据，但是mask比例更大些
    """
    flags:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # ================================================================================== #
    # Initialization for training on different devices
    # - CPU/GPU
    # - Single/Distributed
    Constants.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if Constants.USE_GPU:
        torch.cuda.set_device(local_rank)
        Constants.CURRENT_DEVICE = "cuda:{0}".format(local_rank)
    else:
        Constants.CURRENT_DEVICE = "cpu"

    # If not root_rank, close logging
    # else write log of training to file.
    if rank == 0:
        write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    else:
        close_logging()

    # ================================================================================== #
    # Parsing configuration files
    # - Load default settings
    # - Load pre-defined settings
    # - Load user-defined settings

    configs = prepare_configs(flags.config_path, flags.predefined_config)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    training_configs = configs['training_configs']

    # INFO(pretty_configs(configs))

    Constants.SEED = training_configs['seed']

    set_seed(Constants.SEED)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])
    with open(flags.align_bpe_path_mt2src, 'r', encoding='utf-8') as f:
        align_bpe_dict_mt2src = json.load(f)

    Constants.EOS = vocab_src.eos
    Constants.PAD = vocab_src.pad
    Constants.BOS = vocab_src.bos
    Constants.MASK = vocab_src.mask

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2]),
        QEHterDataset(data_path=data_configs['train_data'][3]),
        TextLineStrDataset(data_path=data_configs['train_data'][0],
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineStrDataset(data_path=data_configs['train_data'][1],
                        max_len=data_configs['max_len'][1],
                        ),
    )
    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs["batch_size"],    # 1
                                     use_bucket=False,
                                     buffer_size=training_configs['buffer_size'],
                                     batching_func=training_configs['batching_key'],
                                     world_size=world_size,
                                     rank=rank,
                                     shuffle=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Building model...')
    timer.tic()
    model = build_model(n_src_vocab=vocab_tgt.max_n_words,
                        n_tgt_vocab=vocab_src.max_n_words, padding_idx=vocab_tgt.pad, **model_configs)
    # INFO(model)
    if Constants.USE_GPU:
        model = model.cuda()
    load_pretrained_model(model, flags.pretrain_path, exclude_prefix=None, device=Constants.CURRENT_DEVICE)

    INFO('Begin training...')
    training_iter = training_iterator.build_generator()

    split = flags.split
    f_x = open(flags.saveto + 'robust_' + split + '.src.BPE', 'w', encoding='utf-8')
    f_y = open(flags.saveto + 'robust_' + split + '.mt.BPE', 'w', encoding='utf-8')
    f_label = open(flags.saveto + 'robust_' + split + '.tag.BPE', 'w', encoding='utf-8')
    f_hter = open(flags.saveto + 'robust_' + split + '.hter', 'w', encoding='utf-8')
    f_idx = open(flags.saveto + 'robust_' + split + '.idx.BPE', 'w', encoding='utf-8')

    id = 0
    for batch in training_iter:   # batch = 1
        print(id)
        seqs_x, seqs_y, xy_label, xy_hter, seqs_x_str, seqs_y_str = batch
        xy_hter = torch.tensor(xy_hter).squeeze()

        align_bpe_dict_line = align_bpe_dict_mt2src[str(id)]  # 这句话中每个mt token对齐的src token们

        y_len = len(seqs_y[-1])    # mt句的长度
        
        ok_ids = [i for i in range(y_len) if xy_label[0][i] == 1]
        align_position = []
        finally_ok_id = []
        finally_robust_id = []
        
        for i in ok_ids:
            if str(i) in align_bpe_dict_line:
                align_position.extend(align_bpe_dict_line[str(i)])
                finally_robust_id.append(i)
            else:
                finally_ok_id.append(i)
        
        if len(align_position) == 0:   #没有的话，干脆舍弃这条样本
            continue

        ok_align_position = list(set(align_position))  # 去重，得到全部mt ok token对应的src token
        x, y, x_mask, lm_ids_x = prepare_data_all_robust(seqs_x, seqs_y, align_position=ok_align_position, cuda=Constants.USE_GPU)

        # 生成伪数据
        with torch.no_grad():
            mlm_logits_x = model(y, x_mask, 'MLM', log_probs=False)  # [batch_size, seq_len, hidden_dim]

        word_noise_data_x = gen_noise_data5_y2x_robust_must(x, mlm_logits_x, lm_ids_x)

        # 写入平行语料
        f_x.write(' '.join(seqs_x_str[0]) + '\n')
        f_y.write(' '.join(seqs_y_str[0]) + '\n')
        temp_label = ['OK' if item == 1 else 'BAD' for item in xy_label[0]]
        f_label.write(' '.join(temp_label) + '\n')
        f_hter.write('%.6f\n' % xy_hter.item()) 
        f_idx.write(' '.join(list(map(str, finally_robust_id))) + '\n')

        # 写入对应的伪数据robust
        robust_x_line = vocab_src.ids2sent(word_noise_data_x[0].cpu().tolist())
        f_x.write(robust_x_line + '\n')
        f_y.write(' '.join(seqs_y_str[0]) + '\n')
        temp_label = ['OK' if i in finally_ok_id else 'BAD' for i in range(len(xy_label[0]))]
        f_label.write(' '.join(temp_label) + '\n')
        f_hter.write('%.6f\n' % 1) 
        f_idx.write(' '.join(list(map(str, finally_robust_id))) + '\n')
        
        id += 1
        #if id >= 5: break


def generate_dev_with_parallel(flags):
    """
    由于生成伪数据的时候，过滤掉了超长的样本；所以为了得到和平行语料一一对应的关系，
    此处把平行语料也做一份长度过滤，保存起来。
    flags:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # ================================================================================== #
    # Initialization for training on different devices
    # - CPU/GPU
    # - Single/Distributed
    Constants.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if Constants.USE_GPU:
        torch.cuda.set_device(local_rank)
        Constants.CURRENT_DEVICE = "cuda:{0}".format(local_rank)
    else:
        Constants.CURRENT_DEVICE = "cpu"

    # If not root_rank, close logging
    # else write log of training to file.
    if rank == 0:
        write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    else:
        close_logging()

    # ================================================================================== #
    # Parsing configuration files
    # - Load default settings
    # - Load pre-defined settings
    # - Load user-defined settings

    configs = prepare_configs(flags.config_path, flags.predefined_config)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    training_configs = configs['training_configs']

    # INFO(pretty_configs(configs))

    Constants.SEED = training_configs['seed']

    set_seed(Constants.SEED)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    Constants.EOS = vocab_src.eos
    Constants.PAD = vocab_src.pad
    Constants.BOS = vocab_src.bos
    Constants.MASK = vocab_src.mask

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        )
    )
    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs["batch_size"],
                                     use_bucket=False,
                                     buffer_size=training_configs['buffer_size'],
                                     batching_func=training_configs['batching_key'],
                                     world_size=world_size,
                                     rank=rank,
                                     shuffle=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Building model...')
    timer.tic()
    model = build_model(n_src_vocab=vocab_src.max_n_words,
                        n_tgt_vocab=vocab_tgt.max_n_words, padding_idx=vocab_src.pad, **model_configs)
    # INFO(model)
    if Constants.USE_GPU:
        model = model.cuda()
    load_pretrained_model(model, flags.pretrain_path, exclude_prefix=None, device=Constants.CURRENT_DEVICE)

    INFO('Begin training...')
    training_iter = training_iterator.build_generator()
    split = flags.split
    f_x = open(flags.saveto + 'pseudo_' + split + '.src.BPE', 'w', encoding='utf-8')
    f_y = open(flags.saveto + 'pseudo_' + split + '.mt.BPE', 'w', encoding='utf-8')
    f_l = open(flags.saveto + 'pseudo_' + split + '.tag', 'w', encoding='utf-8')
    f_h = open(flags.saveto + 'pseudo_' + split + '.hter', 'w', encoding='utf-8')
    f_x_parallel = open(flags.saveto + 'parallel_' + split + '.src.BPE', 'w', encoding='utf-8')
    f_y_parallel = open(flags.saveto + 'parallel_' + split + '.mt.BPE', 'w', encoding='utf-8')
    f_l_parallel = open(flags.saveto + 'parallel_' + split + '.tag', 'w', encoding='utf-8')
    f_h_parallel = open(flags.saveto + 'parallel_' + split + '.hter', 'w', encoding='utf-8')

    hter = []
    for batch in training_iter:
        seqs_x, seqs_y = batch
        # Prepare data
        # seqs_y, lm_ids = mask_target_data(seqs_y)
        x, y, y_mask, y_reverse, lm_ids = prepare_data_all(seqs_x, seqs_y, cuda=Constants.USE_GPU)
        
        # 保存平行语料
        for line_src, line_tgt in zip(x, y):
            line_src = line_src.cpu().tolist()
            line_tgt = line_tgt.cpu().tolist()
            #print(line_tgt)
            new_line_tag = []
            for i in range(sum(e>2 for e in line_tgt)):
                new_line_tag.append('OK')
            f_l_parallel.write(' '.join(new_line_tag) + '\n')
            line_src = vocab_src.ids2sent(line_src)
            line_tgt = vocab_tgt.ids2sent(line_tgt[1:])
            f_x_parallel.write(line_src + '\n')
            f_y_parallel.write(line_tgt + '\n')
            f_h_parallel.write(str(0) + '\n')

        # 生成并保存伪数据
        with torch.no_grad():
            mlm_logits = model(x, y_mask, 'MLM', log_probs=False)
        word_noise_data, word_noise_label, word_hter_label = gen_noise_data5(y, mlm_logits, lm_ids, 100)
        for line_src, line_tgt, line_tag, line_hter in zip(x, word_noise_data, word_noise_label, word_hter_label):
            line_src = line_src.cpu().tolist()
            line_tgt = line_tgt.cpu().tolist()
            line_tag = line_tag.cpu().tolist()
            line_hter = line_hter.cpu().tolist()
            line_src = vocab_src.ids2sent(line_src)
            line_tgt = vocab_tgt.ids2sent(line_tgt[2:])
            f_x.write(line_src + '\n')
            f_y.write(line_tgt + '\n')
            new_line_tag = []
            for each_tag in line_tag:
                if each_tag == 1:
                    new_line_tag.append('OK')
                elif each_tag == 2:
                    new_line_tag.append('BAD')
            f_l.write(' '.join(new_line_tag) + '\n')
            if line_hter[0] < 0.001:
                line_hter[0] = 0
            f_h.write(str(line_hter[0])[:6] + '\n')
            hter.append(line_hter[0])
        continue
    print('mean_hter:%lf'%(sum(hter)/len(hter)))


def train_discriminator(flags):
    """
    flags:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # ================================================================================== #
    # Initialization for training on different devices
    # - CPU/GPU
    # - Single/Distributed
    Constants.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if Constants.USE_GPU:
        torch.cuda.set_device(local_rank)
        Constants.CURRENT_DEVICE = "cuda:{0}".format(local_rank)
    else:
        Constants.CURRENT_DEVICE = "cpu"

    # If not root_rank, close logging
    # else write log of training to file.
    if rank == 0:
        write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    else:
        close_logging()

    # ================================================================================== #
    # Parsing configuration files
    # - Load default settings
    # - Load pre-defined settings
    # - Load user-defined settings

    configs = prepare_configs(flags.config_path, flags.predefined_config)

    data_configs = configs['data_configs']
    discriminator_configs = configs['discriminator_configs']
    generator_configs = configs['generator_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    INFO(pretty_configs(configs))

    Constants.SEED = training_configs['seed']

    set_seed(Constants.SEED)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    Constants.EOS = vocab_src.eos
    Constants.PAD = vocab_src.pad
    Constants.BOS = vocab_src.bos
    Constants.MASK = vocab_src.mask

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        )
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs["batch_size"],
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=training_configs['buffer_size'],
                                     batching_func=training_configs['batching_key'],
                                     world_size=world_size,
                                     rank=rank,
                                     shuffle=True)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, buffer_size=100000, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 0. Initial

    lrate = optimizer_configs['learning_rate']
    model_collections = Collections()

    best_model_prefix = os.path.join(flags.saveto, flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)

    # 1. Build Model & Criterion
    INFO('Building model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words,
                                padding_idx=vocab_src.pad, **discriminator_configs)

    generator_mlm = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words,
                                padding_idx=vocab_src.pad, **generator_configs)
    '''
    generator_lm_forward = build_model(n_src_vocab=vocab_src.max_n_words,
                                       n_tgt_vocab=vocab_tgt.max_n_words,
                                       padding_idx=vocab_src.pad, **generator_configs)

    generator_lm_backward = build_model(n_src_vocab=vocab_src.max_n_words,
                                        n_tgt_vocab=vocab_tgt.max_n_words,
                                        padding_idx=vocab_src.pad, **generator_configs)
    '''
    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)
    # 2. Move to GPU
    if Constants.USE_GPU:
        discriminator = discriminator.cuda()
        generator_mlm = generator_mlm.cuda()
        # generator_lm_forward = generator_lm_forward.cuda()
        # generator_lm_backward = generator_lm_backward.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(generator_mlm, flags.pretrain_generator_mlm_path, exclude_prefix=None,
                          device=Constants.CURRENT_DEVICE)
    # load_pretrained_model(generator_lm_forward, flags.pretrain_generator_lm_forward_path, exclude_prefix=None,
    #                      device=Constants.CURRENT_DEVICE)
    # load_pretrained_model(generator_lm_backward, flags.pretrain_generator_lm_backward_path, exclude_prefix=None,
    #                      device=Constants.CURRENT_DEVICE)

    # 加载MLM的embedding层到判别器中
    INFO('load embedding')
    times_bigger = int(discriminator_configs['d_model'] / generator_configs['d_model'])
    for key in discriminator.encoder.embeddings.embeddings.state_dict().keys():
        discriminator.encoder.embeddings.embeddings.state_dict()[key].copy_(
            generator_mlm.encoder.embeddings.embeddings.state_dict()[key].repeat(1, times_bigger))
        discriminator.decoder.embeddings.embeddings.state_dict()[key].copy_(
            generator_mlm.decoder.embeddings.embeddings.state_dict()[key].repeat(1, times_bigger))

    INFO('Done. Elapsed time {0}'.format(timer.toc()))
    load_pretrained_model(discriminator, flags.pretrain_discriminator_path, exclude_prefix=None,device=Constants.CURRENT_DEVICE)
    # 4. Build optimizer
    INFO('Building Optimizer...')

    if not flags.multi_gpu:
        optim = Optimizer(name=optimizer_configs['optimizer'],
                          model=discriminator,
                          lr=lrate,
                          grad_clip=optimizer_configs['grad_clip'],
                          optim_args=optimizer_configs['optimizer_params'],
                          update_cycle=training_configs['update_cycle']
                          )
    else:
        optim = dist.DistributedOptimizer(name=optimizer_configs['optimizer'],
                                          model=discriminator,
                                          lr=lrate,
                                          grad_clip=optimizer_configs['grad_clip'],
                                          optim_args=optimizer_configs['optimizer_params'],
                                          device_id=local_rank
                                          )

    # 加载embedding后是否要更新embedding参数。似乎embedding矩阵没有bias。
    # discriminator.encoder.embeddings.embeddings.weight.requires_grad = False
    # discriminator.decoder.embeddings.embeddings.weight.requires_grad = False

    # 5. Build scheduler for optimizer if needed
    scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])

    # 6. build moving average
    if training_configs['moving_average_method'] is not None:
        ma = MovingAverage(moving_average_method=training_configs['moving_average_method'],
                           named_params=discriminator.named_parameters(),
                           alpha=training_configs['moving_average_alpha'])
    else:
        ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # broadcast parameters and optimizer states
    if world_size > 1:
        INFO("Broadcasting model parameters...")
        dist.broadcast_parameters(params=discriminator.state_dict())
        dist.broadcast_parameters(params=generator_mlm.state_dict())
        # dist.broadcast_parameters(params=generator_lm_forward.state_dict())
        # dist.broadcast_parameters(params=generator_lm_backward.state_dict())
        INFO("Broadcasting optimizer states...")
        dist.broadcast_optimizer_state(optimizer=optim.optim)
        INFO('Done.')

    # ================================================================================== #
    # Prepare training

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [1])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]
    is_early_stop = model_collections.get_collection("is_early_stop", [False, ])[-1]

    train_loss_sen_forward_meter = AverageMeter()
    train_loss_sen_backward_meter = AverageMeter()
    train_loss_word_meter = AverageMeter()
    train_loss_sen_forward_hter_meter = AverageMeter()
    train_loss_sen_backward_hter_meter = AverageMeter()
    train_loss_word_hter_meter = AverageMeter()
    sent_per_sec_meter = TimeMeter()
    tok_per_sec_meter = TimeMeter()

    update_cycle = training_configs['update_cycle']
    grad_denom = 0
    train_loss_sen_forward = 0.0
    train_loss_sen_backward = 0.0
    train_loss_word = 0.0
    train_loss_sen_forward_hter = 0.0
    train_loss_sen_backward_hter = 0.0
    train_loss_word_hter = 0.0
    cum_n_words = 0

    if rank == 0:
        summary_writer = SummaryWriter(log_dir=flags.log_path)
    else:
        summary_writer = None

    sent_per_sec_meter.start()
    tok_per_sec_meter.start()

    INFO('Begin training...')

    while True:

        if summary_writer is not None:
            summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()

        if rank == 0:
            training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                         total=len(training_iterator),
                                         unit="sents"
                                         )
        else:
            training_progress_bar = None
        for batch in training_iter:

            seqs_x, seqs_y = batch
            batch_size = len(seqs_x)
            cum_n_words += sum(len(s) for s in seqs_y)

            # len_seqs_y = []
            # for s in seqs_y:
            #     len_seqs_y.append(len(s))
            # min_len_seqs_y = min(len_seqs_y)

            try:
                # Prepare data, 同时得到三种模型的输入数据
                x, y, y_mask, y_reverse, lm_ids = prepare_data_all(seqs_x, seqs_y, cuda=Constants.USE_GPU)
                # lm_ids 标记出哪些位置是被mask的，这些词是我需要拿来用的，非0表示被mask的。

                # 根据正向和反向LM的来计算句子级别的带噪数据
                # ini_tgt_seq_len = random.randint(1, max(1, min_len_seqs_y))
                # +1 是因为有bos。
                # ini_tgt_seq = y[:, :ini_tgt_seq_len + 1]
                # ini_tgt_seq_reverse = y_reverse[:, :ini_tgt_seq_len + 1]

                # 根据MLM生成词级别的带噪数据
                with torch.no_grad():
                    # 输入是否带有mask并不关键
                    # mlm_logits = generator_mlm(x, y, 'MLM')
                    mlm_logits = generator_mlm(x, y_mask, 'MLM', log_probs=False)
                '''
                with torch.no_grad():
                    word_ids = beam_search(nmt_model=generator_lm_forward, beam_size=2,
                                           max_steps=y.size()[1]-ini_tgt_seq_len,
                                           src_seqs=x, ini_tgt_seqs=ini_tgt_seq, alpha=0.0)
                    word_ids_reverse = beam_search(nmt_model=generator_lm_backward, beam_size=2,
                                                   max_steps=y.size()[1]-ini_tgt_seq_len,
                                                   src_seqs=x, ini_tgt_seqs=ini_tgt_seq_reverse, alpha=0.0)
                '''

                # 按照pk选
                # word_noise_data, word_noise_label, word_hter_label = gen_noise_data(y, mlm_logits, 0.15, k=5)
                # 随机选
                # word_noise_data, word_noise_label, word_hter_label = rand_noise_data(y, prob, vocab_tgt.max_n_words)
                # 按照mask位置的词，生成概率最大的
                # word_noise_data, word_noise_label, word_hter_label = gen_noise_data2(y, mlm_logits, lm_ids)
                # 按照mask位置的词，生成概率top10中随机的一个
                word_noise_data, word_noise_label, word_hter_label = gen_noise_data5(y, mlm_logits, lm_ids, 100)
                '''
                # 取出第一个beam里的输出
                word_ids = word_ids[:, 0, :]
                word_ids_reverse = word_ids_reverse[:, 0, :]

                word_ids = word_ids.cpu().tolist()
                word_ids_reverse = word_ids_reverse.cpu().tolist()

                noise_sentences = []
                for sen in word_ids:
                    noise_sentence = []
                    for word in sen:
                        if word not in [Constants.BOS, Constants.EOS, Constants.PAD]:
                            noise_sentence.append(word)
                        else:
                            break
                    noise_sentences.append(noise_sentence)

                noise_sentences_reverse = []
                for sen in word_ids_reverse:
                    noise_sentence_reverse = []
                    for word in sen:
                        if word not in [Constants.BOS, Constants.EOS, Constants.PAD]:
                            noise_sentence_reverse.append(word)
                        else:
                            break
                    noise_sentences_reverse.append(noise_sentence_reverse)
            
                sen_noise_data, sen_noise_label, sen_hter_label = prepare_data_fake(
                    noise_sentences, seqs_y, cuda=Constants.USE_GPU)

                sen_noise_data_reverse, sen_noise_label_reverse, sen_hter_label_reverse = \
                    prepare_data_fake(noise_sentences_reverse, seqs_y, backward=True, cuda=Constants.USE_GPU)
                '''
                loss_word, loss_word_hter = \
                    compute_forward_discriminator(model=discriminator,
                                                  critic=critic,
                                                  critic_mse=critic_mse,
                                                  seqs_x=x,
                                                  seqs_y=word_noise_data,
                                                  lm_ids=word_noise_label,
                                                  hter_label=word_hter_label,
                                                  eval=False,
                                                  normalization=1.0,
                                                  norm_by_words=training_configs["norm_by_words"]
                                                  )
                '''
                loss_sen_forward, loss_sen_forward_hter = \
                    compute_forward_discriminator(model=discriminator,
                                                  critic=critic,
                                                  critic_mse=critic_mse,
                                                  seqs_x=x,
                                                  seqs_y=sen_noise_data,
                                                  lm_ids=sen_noise_label,
                                                  hter_label=sen_hter_label,
                                                  eval=False,
                                                  normalization=1.0,
                                                  norm_by_words=training_configs["norm_by_words"]
                                                  )

                loss_sen_backward, loss_sen_backward_hter = \
                    compute_forward_discriminator(model=discriminator,
                                                  critic=critic,
                                                  critic_mse=critic_mse,
                                                  seqs_x=x,
                                                  seqs_y=sen_noise_data_reverse,
                                                  lm_ids=sen_noise_label_reverse,
                                                  hter_label=sen_hter_label_reverse,
                                                  eval=False,
                                                  normalization=1.0,
                                                  norm_by_words=training_configs["norm_by_words"]
                                                  )
                '''
                loss_sen_forward = 0
                loss_sen_forward_hter = 0
                loss_sen_backward = 0
                loss_sen_backward_hter = 0
                if np.isnan(loss_word) or np.isnan(loss_word_hter) or np.isnan(loss_sen_forward) or \
                    np.isnan(loss_sen_forward_hter) or np.isnan(loss_sen_backward) or np.isnan(loss_sen_backward_hter):
                    WARN("NaN detected!")
                update_cycle -= 1
                grad_denom += batch_size

                train_loss_sen_forward += loss_sen_forward
                train_loss_sen_backward += loss_sen_backward
                train_loss_word += loss_word

                train_loss_sen_forward_hter += loss_sen_forward_hter
                train_loss_sen_backward_hter += loss_sen_backward_hter
                train_loss_word_hter += loss_word_hter

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom_count += 1
                else:
                    raise e

            # When update_cycle becomes 0, it means end of one batch. Several things will be done:
            # - update parameters
            # - reset update_cycle and grad_denom, update uidx
            # - learning rate scheduling
            # - update moving average

            if update_cycle == 0:

                # 0. reduce variables
                if world_size > 1:
                    grad_denom = dist.all_reduce_py(grad_denom)
                    train_loss_sen_forward = dist.all_reduce_py(train_loss_sen_forward)
                    train_loss_sen_backward = dist.all_reduce_py(train_loss_sen_backward)
                    train_loss_word = dist.all_reduce_py(train_loss_word)
                    train_loss_sen_forward_hter = dist.all_reduce_py(train_loss_sen_forward_hter)
                    train_loss_sen_backward_hter = dist.all_reduce_py(train_loss_sen_backward_hter)
                    train_loss_word_hter = dist.all_reduce_py(train_loss_word_hter)
                    cum_n_words = dist.all_reduce_py(cum_n_words)

                # 1. update parameters
                optim.step(denom=grad_denom)
                optim.zero_grad()

                if training_progress_bar is not None:
                    training_progress_bar.update(grad_denom)
                    training_progress_bar.set_description(' - (Epc {}, Upd {}) '.format(eidx, uidx))

                # 2. learning rate scheduling
                if scheduler is not None and optimizer_configs["schedule_method"] != "loss":
                    scheduler.step(global_step=uidx)

                # 3. update moving average
                if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                    ma.step()

                # 4. update meters
                train_loss_sen_forward_meter.update(train_loss_sen_forward, grad_denom)
                train_loss_sen_backward_meter.update(train_loss_sen_backward, grad_denom)
                train_loss_word_meter.update(train_loss_word, grad_denom)
                train_loss_sen_forward_hter_meter.update(train_loss_sen_forward_hter, grad_denom)
                train_loss_sen_backward_hter_meter.update(train_loss_sen_backward_hter, grad_denom)
                train_loss_word_hter_meter.update(train_loss_word_hter, grad_denom)
                sent_per_sec_meter.update(grad_denom)
                tok_per_sec_meter.update(cum_n_words)

                # 5. reset accumulated variables, update uidx
                update_cycle = training_configs['update_cycle']
                grad_denom = 0
                uidx += 1
                cum_n_words = 0.0
                train_loss_sen_forward = 0.0
                train_loss_sen_backward = 0.0
                train_loss_word = 0.0

                train_loss_sen_forward_hter = 0.0
                train_loss_sen_backward_hter = 0.0
                train_loss_word_hter = 0.0

            else:
                continue
            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):

                lrate = list(optim.get_lrate())[0]

                if summary_writer is not None:
                    summary_writer.add_scalar("Speed(sents/sec)", scalar_value=sent_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("Speed(words/sec)", scalar_value=tok_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_forward", scalar_value=train_loss_sen_forward_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_backward", scalar_value=train_loss_sen_backward_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_word", scalar_value=train_loss_word_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_forward_hter",
                                              scalar_value=train_loss_sen_forward_hter_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_backward_hter",
                                              scalar_value=train_loss_sen_backward_hter_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_word_hter", scalar_value=train_loss_word_hter_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                    summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)

                # Reset Meters
                sent_per_sec_meter.reset()
                tok_per_sec_meter.reset()
                train_loss_sen_forward_meter.reset()
                train_loss_sen_backward_meter.reset()
                train_loss_word_meter.reset()
                train_loss_sen_forward_hter_meter.reset()
                train_loss_sen_backward_hter_meter.reset()
                train_loss_word_hter_meter.reset()

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):
                with cache_parameters(discriminator):

                    if ma is not None:
                        discriminator.load_state_dict(ma.export_ma_params(), strict=False)

                    valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                        loss_evaluation_qe(discriminator=discriminator,
                                           critic=critic,
                                           critic_mse=critic_mse,
                                           valid_iterator=valid_iterator)

                    # 保存ckpt
                    torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                
                if scheduler is not None and optimizer_configs["schedule_method"] == "loss":
                    scheduler.step(metric=valid_loss)

                model_collections.add_to_collection("history_loss", valid_loss)

                min_valid_loss = np.array(model_collections.get_collection("history_loss")).min()

                if summary_writer is not None:
                    summary_writer.add_scalar("loss_word", valid_loss, global_step=uidx)
                    summary_writer.add_scalar('loss_hter', valid_loss_hter, global_step=uidx)
                    summary_writer.add_scalar('f1good', f1good, global_step=uidx)
                    summary_writer.add_scalar('f1bad', f1bad, global_step=uidx)
                    summary_writer.add_scalar('f1multi', f1multi, global_step=uidx)
                    summary_writer.add_scalar('pearsonr', p, global_step=uidx)

                if eidx >= 0:
                    if valid_loss <= min_valid_loss:
                        bad_count = 0
                        if is_early_stop is False and rank == 0:
                            INFO("save")
                            torch.save(discriminator.state_dict(), best_model_prefix + '.final')
                    else:
                        bad_count += 1
                        if bad_count >= training_configs['early_stop_patience']:
                            is_early_stop = True
                            WARN("Early Stop")

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.6f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.6f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
    
            if is_early_stop is True:
                break
            # ================================================================================== #
        if training_progress_bar is not None:
            training_progress_bar.close()

        if is_early_stop is True:
            break

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def train_discriminator_one_class(flags):
    # 单分类替代多分类，词级别预测最后线性层映射到一维
    """
    flags:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # ================================================================================== #
    # Initialization for training on different devices
    # - CPU/GPU
    # - Single/Distributed
    Constants.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if Constants.USE_GPU:
        torch.cuda.set_device(local_rank)
        Constants.CURRENT_DEVICE = "cuda:{0}".format(local_rank)
    else:
        Constants.CURRENT_DEVICE = "cpu"

    # If not root_rank, close logging
    # else write log of training to file.
    if rank == 0:
        write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    else:
        close_logging()

    # ================================================================================== #
    # Parsing configuration files
    # - Load default settings
    # - Load pre-defined settings
    # - Load user-defined settings

    configs = prepare_configs(flags.config_path, flags.predefined_config)

    data_configs = configs['data_configs']
    discriminator_configs = configs['discriminator_configs']
    generator_configs = configs['generator_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    INFO(pretty_configs(configs))

    Constants.SEED = training_configs['seed']

    set_seed(Constants.SEED)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    Constants.EOS = vocab_src.eos
    Constants.PAD = vocab_src.pad
    Constants.BOS = vocab_src.bos
    Constants.MASK = vocab_src.mask

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        )
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs["batch_size"],
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=training_configs['buffer_size'],
                                     batching_func=training_configs['batching_key'],
                                     world_size=world_size,
                                     rank=rank,
                                     shuffle=True)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, buffer_size=100000, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 0. Initial

    lrate = optimizer_configs['learning_rate']
    model_collections = Collections()

    best_model_prefix = os.path.join(flags.saveto, flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)

    # 1. Build Model & Criterion
    INFO('Building model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words,
                                padding_idx=vocab_src.pad, **discriminator_configs)

    generator_mlm = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words,
                                padding_idx=vocab_src.pad, **generator_configs)
    '''
    generator_lm_forward = build_model(n_src_vocab=vocab_src.max_n_words,
                                       n_tgt_vocab=vocab_tgt.max_n_words,
                                       padding_idx=vocab_src.pad, **generator_configs)

    generator_lm_backward = build_model(n_src_vocab=vocab_src.max_n_words,
                                        n_tgt_vocab=vocab_tgt.max_n_words,
                                        padding_idx=vocab_src.pad, **generator_configs)
    '''

    if flags.criterion == "focal_loss":
        critic = NMTCriterionFocalLoss(padding_idx=0, ignore_idx=0)
    elif flags.criterion == "nll_loss":
        critic = NMTCriterion(padding_idx=0, ignore_idx=0, one_class=True)
    elif flags.criterion == "hinge_loss":
        critic = NMTCriterionHingeLoss(padding_idx=0, ignore_idx=0, one_class=True)

    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)
    # 2. Move to GPU
    if Constants.USE_GPU:
        discriminator = discriminator.cuda()
        generator_mlm = generator_mlm.cuda()
        # generator_lm_forward = generator_lm_forward.cuda()
        # generator_lm_backward = generator_lm_backward.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(generator_mlm, flags.pretrain_generator_mlm_path, exclude_prefix=None,
                          device=Constants.CURRENT_DEVICE)
    # load_pretrained_model(generator_lm_forward, flags.pretrain_generator_lm_forward_path, exclude_prefix=None,
    #                      device=Constants.CURRENT_DEVICE)
    # load_pretrained_model(generator_lm_backward, flags.pretrain_generator_lm_backward_path, exclude_prefix=None,
    #                      device=Constants.CURRENT_DEVICE)

    # 加载MLM的embedding层到判别器中
    INFO('load embedding')
    times_bigger = int(discriminator_configs['d_model'] / generator_configs['d_model'])
    for key in discriminator.encoder.embeddings.embeddings.state_dict().keys():
        discriminator.encoder.embeddings.embeddings.state_dict()[key].copy_(
            generator_mlm.encoder.embeddings.embeddings.state_dict()[key].repeat(1, times_bigger))
        discriminator.decoder.embeddings.embeddings.state_dict()[key].copy_(
            generator_mlm.decoder.embeddings.embeddings.state_dict()[key].repeat(1, times_bigger))

    INFO('Done. Elapsed time {0}'.format(timer.toc()))
    load_pretrained_model(discriminator, flags.pretrain_discriminator_path, exclude_prefix=None,device=Constants.CURRENT_DEVICE)
    
    # 4. Build optimizer
    INFO('Building Optimizer...')

    if not flags.multi_gpu:
        optim = Optimizer(name=optimizer_configs['optimizer'],
                          model=discriminator,
                          lr=lrate,
                          grad_clip=optimizer_configs['grad_clip'],
                          optim_args=optimizer_configs['optimizer_params'],
                          update_cycle=training_configs['update_cycle']
                          )
    else:
        optim = dist.DistributedOptimizer(name=optimizer_configs['optimizer'],
                                          model=discriminator,
                                          lr=lrate,
                                          grad_clip=optimizer_configs['grad_clip'],
                                          optim_args=optimizer_configs['optimizer_params'],
                                          device_id=local_rank
                                          )

    # 加载embedding后是否要更新embedding参数。似乎embedding矩阵没有bias。
    # discriminator.encoder.embeddings.embeddings.weight.requires_grad = False
    # discriminator.decoder.embeddings.embeddings.weight.requires_grad = False

    # 5. Build scheduler for optimizer if needed
    scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])

    # 6. build moving average
    if training_configs['moving_average_method'] is not None:
        ma = MovingAverage(moving_average_method=training_configs['moving_average_method'],
                           named_params=discriminator.named_parameters(),
                           alpha=training_configs['moving_average_alpha'])
    else:
        ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # broadcast parameters and optimizer states
    if world_size > 1:
        INFO("Broadcasting model parameters...")
        dist.broadcast_parameters(params=discriminator.state_dict())
        dist.broadcast_parameters(params=generator_mlm.state_dict())
        # dist.broadcast_parameters(params=generator_lm_forward.state_dict())
        # dist.broadcast_parameters(params=generator_lm_backward.state_dict())
        INFO("Broadcasting optimizer states...")
        dist.broadcast_optimizer_state(optimizer=optim.optim)
        INFO('Done.')

    # ================================================================================== #
    # Prepare training

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [1])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]
    is_early_stop = model_collections.get_collection("is_early_stop", [False, ])[-1]

    train_loss_sen_forward_meter = AverageMeter()
    train_loss_sen_backward_meter = AverageMeter()
    train_loss_word_meter = AverageMeter()
    train_loss_sen_forward_hter_meter = AverageMeter()
    train_loss_sen_backward_hter_meter = AverageMeter()
    train_loss_word_hter_meter = AverageMeter()
    sent_per_sec_meter = TimeMeter()
    tok_per_sec_meter = TimeMeter()

    update_cycle = training_configs['update_cycle']
    grad_denom = 0
    train_loss_sen_forward = 0.0
    train_loss_sen_backward = 0.0
    train_loss_word = 0.0
    train_loss_sen_forward_hter = 0.0
    train_loss_sen_backward_hter = 0.0
    train_loss_word_hter = 0.0
    cum_n_words = 0

    if rank == 0:
        summary_writer = SummaryWriter(log_dir=flags.log_path)
    else:
        summary_writer = None

    sent_per_sec_meter.start()
    tok_per_sec_meter.start()

    no_sigmoid = False
    if flags.criterion == "hinge_loss":
        no_sigmoid = True

    INFO('Begin training...')

    while True:

        if summary_writer is not None:
            summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()

        if rank == 0:
            training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                         total=len(training_iterator),
                                         unit="sents"
                                         )
        else:
            training_progress_bar = None
        for batch in training_iter:

            seqs_x, seqs_y = batch
            batch_size = len(seqs_x)
            cum_n_words += sum(len(s) for s in seqs_y)

            # len_seqs_y = []
            # for s in seqs_y:
            #     len_seqs_y.append(len(s))
            # min_len_seqs_y = min(len_seqs_y)

            try:
                # Prepare data, 同时得到三种模型的输入数据
                x, y, y_mask, y_reverse, lm_ids = prepare_data_all(seqs_x, seqs_y, cuda=Constants.USE_GPU)
                # lm_ids 标记出哪些位置是被mask的，这些词是我需要拿来用的，非0表示被mask的。

                # 根据正向和反向LM的来计算句子级别的带噪数据
                # ini_tgt_seq_len = random.randint(1, max(1, min_len_seqs_y))
                # +1 是因为有bos。
                # ini_tgt_seq = y[:, :ini_tgt_seq_len + 1]
                # ini_tgt_seq_reverse = y_reverse[:, :ini_tgt_seq_len + 1]

                # 根据MLM生成词级别的带噪数据
                with torch.no_grad():
                    # 输入是否带有mask并不关键
                    # mlm_logits = generator_mlm(x, y, 'MLM')
                    mlm_logits = generator_mlm(x, y_mask, 'MLM', log_probs=False)
                '''
                with torch.no_grad():
                    word_ids = beam_search(nmt_model=generator_lm_forward, beam_size=2,
                                           max_steps=y.size()[1]-ini_tgt_seq_len,
                                           src_seqs=x, ini_tgt_seqs=ini_tgt_seq, alpha=0.0)
                    word_ids_reverse = beam_search(nmt_model=generator_lm_backward, beam_size=2,
                                                   max_steps=y.size()[1]-ini_tgt_seq_len,
                                                   src_seqs=x, ini_tgt_seqs=ini_tgt_seq_reverse, alpha=0.0)
                '''

                word_noise_data, word_noise_label, word_hter_label = gen_noise_data5(y, mlm_logits, lm_ids, 100)
                
                loss_word, loss_word_hter = \
                    compute_forward_discriminator_one_class(model=discriminator,
                                                  critic=critic,
                                                  critic_mse=critic_mse,
                                                  seqs_x=x,
                                                  seqs_y=word_noise_data,
                                                  lm_ids=word_noise_label,
                                                  hter_label=word_hter_label,
                                                  eval=False,
                                                  normalization=1.0,
                                                  norm_by_words=training_configs["norm_by_words"],
                                                  no_sigmoid=no_sigmoid
                                                  )
                
                loss_sen_forward = 0
                loss_sen_forward_hter = 0
                loss_sen_backward = 0
                loss_sen_backward_hter = 0
                if np.isnan(loss_word) or np.isnan(loss_word_hter) or np.isnan(loss_sen_forward) or \
                    np.isnan(loss_sen_forward_hter) or np.isnan(loss_sen_backward) or np.isnan(loss_sen_backward_hter):
                    WARN("NaN detected!")
                update_cycle -= 1
                grad_denom += batch_size

                train_loss_sen_forward += loss_sen_forward
                train_loss_sen_backward += loss_sen_backward
                train_loss_word += loss_word

                train_loss_sen_forward_hter += loss_sen_forward_hter
                train_loss_sen_backward_hter += loss_sen_backward_hter
                train_loss_word_hter += loss_word_hter

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom_count += 1
                else:
                    raise e

            # When update_cycle becomes 0, it means end of one batch. Several things will be done:
            # - update parameters
            # - reset update_cycle and grad_denom, update uidx
            # - learning rate scheduling
            # - update moving average

            if update_cycle == 0:

                # 0. reduce variables
                if world_size > 1:
                    grad_denom = dist.all_reduce_py(grad_denom)
                    train_loss_sen_forward = dist.all_reduce_py(train_loss_sen_forward)
                    train_loss_sen_backward = dist.all_reduce_py(train_loss_sen_backward)
                    train_loss_word = dist.all_reduce_py(train_loss_word)
                    train_loss_sen_forward_hter = dist.all_reduce_py(train_loss_sen_forward_hter)
                    train_loss_sen_backward_hter = dist.all_reduce_py(train_loss_sen_backward_hter)
                    train_loss_word_hter = dist.all_reduce_py(train_loss_word_hter)
                    cum_n_words = dist.all_reduce_py(cum_n_words)

                # 1. update parameters
                optim.step(denom=grad_denom)
                optim.zero_grad()

                if training_progress_bar is not None:
                    training_progress_bar.update(grad_denom)
                    training_progress_bar.set_description(' - (Epc {}, Upd {}) '.format(eidx, uidx))

                # 2. learning rate scheduling
                if scheduler is not None and optimizer_configs["schedule_method"] != "loss":
                    scheduler.step(global_step=uidx)

                # 3. update moving average
                if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                    ma.step()

                # 4. update meters
                train_loss_sen_forward_meter.update(train_loss_sen_forward, grad_denom)
                train_loss_sen_backward_meter.update(train_loss_sen_backward, grad_denom)
                train_loss_word_meter.update(train_loss_word, grad_denom)
                train_loss_sen_forward_hter_meter.update(train_loss_sen_forward_hter, grad_denom)
                train_loss_sen_backward_hter_meter.update(train_loss_sen_backward_hter, grad_denom)
                train_loss_word_hter_meter.update(train_loss_word_hter, grad_denom)
                sent_per_sec_meter.update(grad_denom)
                tok_per_sec_meter.update(cum_n_words)

                # 5. reset accumulated variables, update uidx
                update_cycle = training_configs['update_cycle']
                grad_denom = 0
                uidx += 1
                cum_n_words = 0.0
                train_loss_sen_forward = 0.0
                train_loss_sen_backward = 0.0
                train_loss_word = 0.0

                train_loss_sen_forward_hter = 0.0
                train_loss_sen_backward_hter = 0.0
                train_loss_word_hter = 0.0

            else:
                continue
            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):

                lrate = list(optim.get_lrate())[0]

                if summary_writer is not None:
                    summary_writer.add_scalar("Speed(sents/sec)", scalar_value=sent_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("Speed(words/sec)", scalar_value=tok_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_forward", scalar_value=train_loss_sen_forward_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_backward", scalar_value=train_loss_sen_backward_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_word", scalar_value=train_loss_word_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_forward_hter",
                                              scalar_value=train_loss_sen_forward_hter_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_backward_hter",
                                              scalar_value=train_loss_sen_backward_hter_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_word_hter", scalar_value=train_loss_word_hter_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                    summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)

                # Reset Meters
                sent_per_sec_meter.reset()
                tok_per_sec_meter.reset()
                train_loss_sen_forward_meter.reset()
                train_loss_sen_backward_meter.reset()
                train_loss_word_meter.reset()
                train_loss_sen_forward_hter_meter.reset()
                train_loss_sen_backward_hter_meter.reset()
                train_loss_word_hter_meter.reset()

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):
                with cache_parameters(discriminator):

                    if ma is not None:
                        discriminator.load_state_dict(ma.export_ma_params(), strict=False)

                    valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                        loss_evaluation_qe_one_class(discriminator=discriminator,
                                           critic=critic,
                                           critic_mse=critic_mse,
                                           valid_iterator=valid_iterator,
                                           no_sigmoid=no_sigmoid)

                    # 保存ckpt
                    torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                
                if scheduler is not None and optimizer_configs["schedule_method"] == "loss":
                    scheduler.step(metric=valid_loss)

                model_collections.add_to_collection("history_loss", valid_loss)

                min_valid_loss = np.array(model_collections.get_collection("history_loss")).min()

                if summary_writer is not None:
                    summary_writer.add_scalar("loss_word", valid_loss, global_step=uidx)
                    summary_writer.add_scalar('loss_hter', valid_loss_hter, global_step=uidx)
                    summary_writer.add_scalar('f1good', f1good, global_step=uidx)
                    summary_writer.add_scalar('f1bad', f1bad, global_step=uidx)
                    summary_writer.add_scalar('f1multi', f1multi, global_step=uidx)
                    summary_writer.add_scalar('pearsonr', p, global_step=uidx)

                if eidx >= 0:
                    if valid_loss <= min_valid_loss:
                        bad_count = 0
                        if is_early_stop is False and rank == 0:
                            INFO("save")
                            torch.save(discriminator.state_dict(), best_model_prefix + '.final')
                    else:
                        bad_count += 1
                        if bad_count >= training_configs['early_stop_patience']:
                            is_early_stop = True
                            WARN("Early Stop")

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.6f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.6f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
    
            if is_early_stop is True:
                break
            # ================================================================================== #
        if training_progress_bar is not None:
            training_progress_bar.close()

        if is_early_stop is True:
            break

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def train_discriminator_contrastive_stat(flags):
    """
    对比预训练
    直接用存好的训练数据，而不是一边从平行语料生成一边训
    flags:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # ================================================================================== #
    # Initialization for training on different devices
    # - CPU/GPU
    # - Single/Distributed
    Constants.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if Constants.USE_GPU:
        torch.cuda.set_device(local_rank)
        Constants.CURRENT_DEVICE = "cuda:{0}".format(local_rank)
    else:
        Constants.CURRENT_DEVICE = "cpu"

    # If not root_rank, close logging
    # else write log of training to file.
    if rank == 0:
        write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    else:
        close_logging()

    # ================================================================================== #
    # Parsing configuration files
    # - Load default settings
    # - Load pre-defined settings
    # - Load user-defined settings

    configs = prepare_configs(flags.config_path, flags.predefined_config)

    data_configs = configs['data_configs']
    discriminator_configs = configs['discriminator_configs']
    generator_configs = configs['generator_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    valid_cl = training_configs["valid_cl"]
    loss_real_lambda = training_configs["loss_real_lambda"]
    loss_parallel_lambda = training_configs["loss_parallel_lambda"]
    cl_lambda = training_configs["cl_lambda"]

    INFO(pretty_configs(configs))
  
    Constants.SEED = training_configs['seed']

    set_seed(Constants.SEED)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    Constants.EOS = vocab_src.eos
    Constants.PAD = vocab_src.pad
    Constants.BOS = vocab_src.bos
    Constants.MASK = vocab_src.mask

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2]),
        QEHterDataset(data_path=data_configs['train_data'][3]),
        TextLineDataset(data_path=data_configs['train_data'][4],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3]),
        TextLineDataset(data_path=data_configs['valid_data'][4],
                        vocabulary=vocab_src,
                        ),
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs["batch_size"],
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=training_configs['buffer_size'],
                                     batching_func=training_configs['batching_key'],
                                     world_size=world_size,
                                     rank=rank,
                                     shuffle=True)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, buffer_size=100000, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 0. Initial

    lrate = optimizer_configs['learning_rate']
    model_collections = Collections()

    best_model_prefix = os.path.join(flags.saveto, flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)

    # 1. Build Model & Criterion
    INFO('Building model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words,
                                padding_idx=vocab_src.pad, **discriminator_configs)

    generator_mlm = build_model(n_src_vocab=vocab_tgt.max_n_words,   # 注意，按照反向改动了
                                n_tgt_vocab=vocab_src.max_n_words,
                                padding_idx=vocab_tgt.pad, **generator_configs)

    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)
    # 2. Move to GPU
    if Constants.USE_GPU:
        discriminator = discriminator.cuda()
        generator_mlm = generator_mlm.cuda()
        # generator_lm_forward = generator_lm_forward.cuda()
        # generator_lm_backward = generator_lm_backward.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(generator_mlm, flags.pretrain_generator_mlm_path, exclude_prefix=None,
                          device=Constants.CURRENT_DEVICE)

    # 加载MLM的embedding层到判别器中
    INFO('load embedding')
    times_bigger = int(discriminator_configs['d_model'] / generator_configs['d_model'])
    for key in discriminator.encoder.embeddings.embeddings.state_dict().keys():
        discriminator.encoder.embeddings.embeddings.state_dict()[key].copy_(
            generator_mlm.decoder.embeddings.embeddings.state_dict()[key].repeat(1, times_bigger))
        discriminator.decoder.embeddings.embeddings.state_dict()[key].copy_(
            generator_mlm.encoder.embeddings.embeddings.state_dict()[key].repeat(1, times_bigger))

    INFO('Done. Elapsed time {0}'.format(timer.toc()))
    load_pretrained_model(discriminator, flags.pretrain_discriminator_path, exclude_prefix=None,device=Constants.CURRENT_DEVICE)
    # 4. Build optimizer
    INFO('Building Optimizer...')

    if not flags.multi_gpu:
        optim = Optimizer(name=optimizer_configs['optimizer'],
                          model=discriminator,
                          lr=lrate,
                          grad_clip=optimizer_configs['grad_clip'],
                          optim_args=optimizer_configs['optimizer_params'],
                          update_cycle=training_configs['update_cycle']
                          )
    else:
        optim = dist.DistributedOptimizer(name=optimizer_configs['optimizer'],
                                          model=discriminator,
                                          lr=lrate,
                                          grad_clip=optimizer_configs['grad_clip'],
                                          optim_args=optimizer_configs['optimizer_params'],
                                          device_id=local_rank
                                          )

    # 加载embedding后是否要更新embedding参数。似乎embedding矩阵没有bias。
    # discriminator.encoder.embeddings.embeddings.weight.requires_grad = False
    # discriminator.decoder.embeddings.embeddings.weight.requires_grad = False

    # 5. Build scheduler for optimizer if needed
    scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])

    # 6. build moving average
    if training_configs['moving_average_method'] is not None:
        ma = MovingAverage(moving_average_method=training_configs['moving_average_method'],
                           named_params=discriminator.named_parameters(),
                           alpha=training_configs['moving_average_alpha'])
    else:
        ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # broadcast parameters and optimizer states
    if world_size > 1:
        INFO("Broadcasting model parameters...")
        dist.broadcast_parameters(params=discriminator.state_dict())
        dist.broadcast_parameters(params=generator_mlm.state_dict())
        # dist.broadcast_parameters(params=generator_lm_forward.state_dict())
        # dist.broadcast_parameters(params=generator_lm_backward.state_dict())
        INFO("Broadcasting optimizer states...")
        dist.broadcast_optimizer_state(optimizer=optim.optim)
        INFO('Done.')

    # ================================================================================== #
    # Prepare training

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [1])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]
    is_early_stop = model_collections.get_collection("is_early_stop", [False, ])[-1]

    train_loss_sen_forward_meter = AverageMeter()
    train_loss_sen_backward_meter = AverageMeter()
    train_loss_word_meter = AverageMeter()
    train_loss_sen_forward_hter_meter = AverageMeter()
    train_loss_sen_backward_hter_meter = AverageMeter()
    train_loss_word_hter_meter = AverageMeter()
    sent_per_sec_meter = TimeMeter()
    tok_per_sec_meter = TimeMeter()

    update_cycle = training_configs['update_cycle']
    grad_denom = 0
    train_loss_sen_forward = 0.0
    train_loss_sen_backward = 0.0
    train_loss_word = 0.0
    train_loss_sen_forward_hter = 0.0
    train_loss_sen_backward_hter = 0.0
    train_loss_word_hter = 0.0
    cum_n_words = 0

    if rank == 0:
        summary_writer = SummaryWriter(log_dir=flags.log_path)
    else:
        summary_writer = None

    sent_per_sec_meter.start()
    tok_per_sec_meter.start()

    INFO('Begin training...')

    # 保存initial ckpt
    torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.0')

    while True:

        if summary_writer is not None:
            summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()

        if rank == 0:
            training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                         total=len(training_iterator),
                                         unit="sents"
                                         )
        else:
            training_progress_bar = None
        
        

        for batch in training_iter:

            seqs_x, seqs_y, xy_label, xy_hter, seqs_x_real = batch
            

            batch_size = len(seqs_x)
            cum_n_words += sum(len(s) for s in seqs_y)

            
            try:
                x, y, x_real, xy_label = prepare_data_qe_with_src_real(seqs_x, seqs_y, seqs_x_real, xy_label, cuda=Constants.USE_GPU)

                if Constants.USE_GPU:
                    xy_hter = torch.tensor(xy_hter).squeeze(-1)
                    xy_hter = xy_hter.cuda()

                loss_word, loss_word_parallel, cl_loss, loss_hter = compute_forward_discriminator_contrastive_src(model=discriminator,
                                                                critic=critic,
                                                                critic_mse=critic_mse,
                                                                seqs_x=x,
                                                                seqs_y=y,
                                                                seqs_x_real=x_real,
                                                                lm_ids=xy_label,
                                                                hter_label=xy_hter,
                                                                loss_real_lambda=loss_real_lambda,
                                                                loss_parallel_lambda=loss_parallel_lambda,
                                                                cl_lambda=cl_lambda,)

                loss_sen_forward = 0
                loss_sen_forward_hter = 0
                loss_sen_backward = 0
                loss_sen_backward_hter = 0
                if np.isnan(loss_word) or np.isnan(loss_hter) or np.isnan(loss_sen_forward) or \
                    np.isnan(loss_sen_forward_hter) or np.isnan(loss_sen_backward) or np.isnan(loss_sen_backward_hter):
                    WARN("NaN detected!")
                update_cycle -= 1
                grad_denom += batch_size

                train_loss_sen_forward += loss_sen_forward
                train_loss_sen_backward += loss_sen_backward
                train_loss_word += loss_word

                train_loss_sen_forward_hter += loss_sen_forward_hter
                train_loss_sen_backward_hter += loss_sen_backward_hter
                train_loss_word_hter += loss_hter

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom_count += 1
                else:
                    raise e

            # When update_cycle becomes 0, it means end of one batch. Several things will be done:
            # - update parameters
            # - reset update_cycle and grad_denom, update uidx
            # - learning rate scheduling
            # - update moving average

            if update_cycle == 0:

                # 0. reduce variables
                if world_size > 1:
                    grad_denom = dist.all_reduce_py(grad_denom)
                    train_loss_sen_forward = dist.all_reduce_py(train_loss_sen_forward)
                    train_loss_sen_backward = dist.all_reduce_py(train_loss_sen_backward)
                    train_loss_word = dist.all_reduce_py(train_loss_word)
                    train_loss_sen_forward_hter = dist.all_reduce_py(train_loss_sen_forward_hter)
                    train_loss_sen_backward_hter = dist.all_reduce_py(train_loss_sen_backward_hter)
                    train_loss_word_hter = dist.all_reduce_py(train_loss_word_hter)
                    cum_n_words = dist.all_reduce_py(cum_n_words)

                # 1. update parameters
                optim.step(denom=grad_denom)
                optim.zero_grad()

                if training_progress_bar is not None:
                    training_progress_bar.update(grad_denom)
                    training_progress_bar.set_description(' - (Epc {}, Upd {}) '.format(eidx, uidx))

                # 2. learning rate scheduling
                if scheduler is not None and optimizer_configs["schedule_method"] != "loss":
                    scheduler.step(global_step=uidx)

                # 3. update moving average
                if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                    ma.step()

                # 4. update meters
                train_loss_sen_forward_meter.update(train_loss_sen_forward, grad_denom)
                train_loss_sen_backward_meter.update(train_loss_sen_backward, grad_denom)
                train_loss_word_meter.update(train_loss_word, grad_denom)
                train_loss_sen_forward_hter_meter.update(train_loss_sen_forward_hter, grad_denom)
                train_loss_sen_backward_hter_meter.update(train_loss_sen_backward_hter, grad_denom)
                train_loss_word_hter_meter.update(train_loss_word_hter, grad_denom)
                sent_per_sec_meter.update(grad_denom)
                tok_per_sec_meter.update(cum_n_words)

                # 5. reset accumulated variables, update uidx
                update_cycle = training_configs['update_cycle']
                grad_denom = 0
                uidx += 1
                cum_n_words = 0.0
                train_loss_sen_forward = 0.0
                train_loss_sen_backward = 0.0
                train_loss_word = 0.0

                train_loss_sen_forward_hter = 0.0
                train_loss_sen_backward_hter = 0.0
                train_loss_word_hter = 0.0

            else:
                continue
            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):

                lrate = list(optim.get_lrate())[0]

                if summary_writer is not None:
                    summary_writer.add_scalar("Speed(sents/sec)", scalar_value=sent_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("Speed(words/sec)", scalar_value=tok_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_forward", scalar_value=train_loss_sen_forward_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_backward", scalar_value=train_loss_sen_backward_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_word", scalar_value=train_loss_word_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_forward_hter",
                                              scalar_value=train_loss_sen_forward_hter_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_backward_hter",
                                              scalar_value=train_loss_sen_backward_hter_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_word_hter", scalar_value=train_loss_word_hter_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                    summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)

                # Reset Meters
                sent_per_sec_meter.reset()
                tok_per_sec_meter.reset()
                train_loss_sen_forward_meter.reset()
                train_loss_sen_backward_meter.reset()
                train_loss_word_meter.reset()
                train_loss_sen_forward_hter_meter.reset()
                train_loss_sen_backward_hter_meter.reset()
                train_loss_word_hter_meter.reset()

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):
                with cache_parameters(discriminator):

                    if ma is not None:
                        discriminator.load_state_dict(ma.export_ma_params(), strict=False)

                    if not valid_cl:
                        # 使用普通词级别loss验证，训练验证目标不一致
                        valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                            loss_evaluation_qe_v2(discriminator=discriminator,
                                            critic=critic,
                                            critic_mse=critic_mse,
                                            valid_iterator=valid_iterator)
                    else:
                        # 训练验证目标一致，验证也是词级loss+对比loss
                        valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                            loss_evaluation_qe_contrastive_src(discriminator=discriminator,
                                                            critic=critic,
                                                            critic_mse=critic_mse,
                                                            valid_iterator=valid_iterator,
                                                            loss_real_lambda=loss_real_lambda,
                                                            loss_parallel_lambda=loss_parallel_lambda,
                                                            cl_lambda=cl_lambda,)
         
                    # 保存ckpt
                    torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                
                if scheduler is not None and optimizer_configs["schedule_method"] == "loss":
                    scheduler.step(metric=valid_loss)

                model_collections.add_to_collection("history_loss", valid_loss)

                min_valid_loss = np.array(model_collections.get_collection("history_loss")).min()

                if summary_writer is not None:
                    summary_writer.add_scalar("loss_word", valid_loss, global_step=uidx)
                    summary_writer.add_scalar('loss_hter', valid_loss_hter, global_step=uidx)
                    summary_writer.add_scalar('f1good', f1good, global_step=uidx)
                    summary_writer.add_scalar('f1bad', f1bad, global_step=uidx)
                    summary_writer.add_scalar('f1multi', f1multi, global_step=uidx)
                    summary_writer.add_scalar('pearsonr', p, global_step=uidx)

                if eidx >= 0:
                    if valid_loss <= min_valid_loss:
                        bad_count = 0
                        if is_early_stop is False and rank == 0:
                            INFO("save")
                            torch.save(discriminator.state_dict(), best_model_prefix + '.final')
                    else:
                        bad_count += 1
                        if bad_count >= training_configs['early_stop_patience']:
                            is_early_stop = True
                            WARN("Early Stop")

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.6f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.6f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
    
            if is_early_stop is True:
                break
            # ================================================================================== #
        if training_progress_bar is not None:
            training_progress_bar.close()

        if is_early_stop is True:
            break

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def train_discriminator_contrastive(flags):
    """
    对比预训练
    读取平行语料，生成动态对比伪数据用于训练
    flags:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # ================================================================================== #
    # Initialization for training on different devices
    # - CPU/GPU
    # - Single/Distributed
    Constants.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if Constants.USE_GPU:
        torch.cuda.set_device(local_rank)
        Constants.CURRENT_DEVICE = "cuda:{0}".format(local_rank)
    else:
        Constants.CURRENT_DEVICE = "cpu"

    # If not root_rank, close logging
    # else write log of training to file.
    if rank == 0:
        write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    else:
        close_logging()

    # ================================================================================== #
    # Parsing configuration files
    # - Load default settings
    # - Load pre-defined settings
    # - Load user-defined settings

    configs = prepare_configs(flags.config_path, flags.predefined_config)

    data_configs = configs['data_configs']
    discriminator_configs = configs['discriminator_configs']
    generator_configs = configs['generator_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    valid_cl = training_configs["valid_cl"]
    loss_real_lambda = training_configs["loss_real_lambda"]
    loss_parallel_lambda = training_configs["loss_parallel_lambda"]
    cl_lambda = training_configs["cl_lambda"]

    INFO(pretty_configs(configs))
  
    Constants.SEED = training_configs['seed']

    set_seed(Constants.SEED)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])
    with open(flags.align_bpe_path, 'r', encoding='utf-8') as f:
        xy_align_dict = json.load(f)

    Constants.EOS = vocab_src.eos
    Constants.PAD = vocab_src.pad
    Constants.BOS = vocab_src.bos
    Constants.MASK = vocab_src.mask

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        TextLineStrDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        QEHterDataset(data_path=data_configs['train_data'][2]),
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3]),
        TextLineDataset(data_path=data_configs['valid_data'][4],
                        vocabulary=vocab_src,
                        ),
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs["batch_size"],
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=training_configs['buffer_size'],
                                     batching_func=training_configs['batching_key'],
                                     world_size=world_size,
                                     rank=rank,
                                     shuffle=False)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, buffer_size=100000, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 0. Initial

    lrate = optimizer_configs['learning_rate']
    model_collections = Collections()

    best_model_prefix = os.path.join(flags.saveto, flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)

    # 1. Build Model & Criterion
    INFO('Building model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words,
                                padding_idx=vocab_src.pad, **discriminator_configs)

    generator_mlm = build_model(n_src_vocab=vocab_tgt.max_n_words,   # 注意，按照反向改动了
                                n_tgt_vocab=vocab_src.max_n_words,
                                padding_idx=vocab_tgt.pad, **generator_configs)

    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)
    # 2. Move to GPU
    if Constants.USE_GPU:
        discriminator = discriminator.cuda()
        generator_mlm = generator_mlm.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(generator_mlm, flags.pretrain_generator_mlm_path, exclude_prefix=None,
                          device=Constants.CURRENT_DEVICE)

    # 加载MLM的embedding层到判别器中
    INFO('load embedding')
    times_bigger = int(discriminator_configs['d_model'] / generator_configs['d_model'])
    for key in discriminator.encoder.embeddings.embeddings.state_dict().keys():
        discriminator.encoder.embeddings.embeddings.state_dict()[key].copy_(
            generator_mlm.decoder.embeddings.embeddings.state_dict()[key].repeat(1, times_bigger))
        discriminator.decoder.embeddings.embeddings.state_dict()[key].copy_(
            generator_mlm.encoder.embeddings.embeddings.state_dict()[key].repeat(1, times_bigger))

    INFO('Done. Elapsed time {0}'.format(timer.toc()))
    load_pretrained_model(discriminator, flags.pretrain_discriminator_path, exclude_prefix=None,device=Constants.CURRENT_DEVICE)
    # 4. Build optimizer
    INFO('Building Optimizer...')

    if not flags.multi_gpu:
        optim = Optimizer(name=optimizer_configs['optimizer'],
                          model=discriminator,
                          lr=lrate,
                          grad_clip=optimizer_configs['grad_clip'],
                          optim_args=optimizer_configs['optimizer_params'],
                          update_cycle=training_configs['update_cycle']
                          )
    else:
        optim = dist.DistributedOptimizer(name=optimizer_configs['optimizer'],
                                          model=discriminator,
                                          lr=lrate,
                                          grad_clip=optimizer_configs['grad_clip'],
                                          optim_args=optimizer_configs['optimizer_params'],
                                          device_id=local_rank
                                          )

    # 5. Build scheduler for optimizer if needed
    scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])

    # 6. build moving average
    if training_configs['moving_average_method'] is not None:
        ma = MovingAverage(moving_average_method=training_configs['moving_average_method'],
                           named_params=discriminator.named_parameters(),
                           alpha=training_configs['moving_average_alpha'])
    else:
        ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # broadcast parameters and optimizer states
    if world_size > 1:
        INFO("Broadcasting model parameters...")
        dist.broadcast_parameters(params=discriminator.state_dict())
        dist.broadcast_parameters(params=generator_mlm.state_dict())
        INFO("Broadcasting optimizer states...")
        dist.broadcast_optimizer_state(optimizer=optim.optim)
        INFO('Done.')

    # ================================================================================== #
    # Prepare training

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [1])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]
    is_early_stop = model_collections.get_collection("is_early_stop", [False, ])[-1]

    train_loss_sen_forward_meter = AverageMeter()
    train_loss_sen_backward_meter = AverageMeter()
    train_loss_word_meter = AverageMeter()
    train_loss_sen_forward_hter_meter = AverageMeter()
    train_loss_sen_backward_hter_meter = AverageMeter()
    train_loss_word_hter_meter = AverageMeter()
    sent_per_sec_meter = TimeMeter()
    tok_per_sec_meter = TimeMeter()

    update_cycle = training_configs['update_cycle']
    grad_denom = 0
    train_loss_sen_forward = 0.0
    train_loss_sen_backward = 0.0
    train_loss_word = 0.0
    train_loss_sen_forward_hter = 0.0
    train_loss_sen_backward_hter = 0.0
    train_loss_word_hter = 0.0
    cum_n_words = 0

    if rank == 0:
        summary_writer = SummaryWriter(log_dir=flags.log_path)
    else:
        summary_writer = None

    sent_per_sec_meter.start()
    tok_per_sec_meter.start()

    INFO('Begin training...')

    # 保存initial ckpt
    #torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.0')

    while True:

        if summary_writer is not None:
            summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()

        if rank == 0:
            training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                         total=len(training_iterator),
                                         unit="sents"
                                         )
        else:
            training_progress_bar = None
        
        for batch in training_iter:

            seqs_x, seqs_y, seqs_x_str, seqs_index = batch
            batch_size = len(seqs_x)
            cum_n_words += sum(len(s) for s in seqs_y)

            try:
                # x是英文，mask掉英文
                y, x, x_mask, x_reverse, x_lm_ids, y_with_hter = prepare_data_all_v2(seqs_y, seqs_x, wwm=True, seqs_y_str=seqs_x_str, cuda=Constants.USE_GPU)
                #x, y, x_real, xy_label = prepare_data_qe_with_src_real(seqs_x, seqs_y, cuda=Constants.USE_GPU)
                # prepare_data_all 确实只加1个

                if Constants.USE_GPU:
                    seqs_index = torch.tensor(seqs_index).squeeze(-1).cuda()
                
                # 生成伪数据
                with torch.no_grad():
                    # 英文的logits
                    mlm_logits = generator_mlm(y, x_mask, 'MLM', log_probs=False)  # [batch_size, seq_len, hidden_dim]
                word_noise_data_x, word_noise_label_y, word_hter_label_y = gen_noise_data5_y2x(seqs_y, x, mlm_logits, x_lm_ids, seqs_index, xy_align_dict)
                # 处理label，pad、转tensor并到gpu
                word_noise_label_y = list(map(lambda s: [0, 0] + s + [0], word_noise_label_y))
                word_noise_label_y = np_pad_batch_2D(word_noise_label_y, pad=Constants.PAD,
                              cuda=Constants.USE_GPU)
                
                #print('debug=====================')

                for line_x, line_x_noise, line_y, line_tag in zip(x, word_noise_data_x, y_with_hter, word_noise_label_y):
                    line_y = line_y.cpu().tolist()[2:]
                    line_x = line_x.cpu().tolist()
                    line_x_noise = line_x_noise.cpu().tolist()
                    line_y = vocab_tgt.ids2sent(line_y)
                    line_x = vocab_src.ids2sent(line_x)
                    line_x_noise = vocab_src.ids2sent(line_x_noise)
                    
                    new_line_tag = []
                    for each_tag in line_tag:
                        if each_tag == 1:
                            new_line_tag.append('OK')
                        elif each_tag == 2:
                            new_line_tag.append('BAD')
                    #print(line_x)
                    #print(line_x_noise)
                    #print(line_y)
                    #print(new_line_tag)
                    #assert 1==2

                loss_word, loss_word_parallel, cl_loss, loss_hter = compute_forward_discriminator_contrastive_src(model=discriminator,
                                                                critic=critic,
                                                                critic_mse=critic_mse,
                                                                seqs_x=word_noise_data_x,  # 前后各加1
                                                                seqs_y=y_with_hter,        # 前2后1
                                                                seqs_x_real=x,             # 前后各加1
                                                                lm_ids=word_noise_label_y, # 前2后1
                                                                hter_label=word_hter_label_y,
                                                                eval=False,
                                                                loss_real_lambda=loss_real_lambda,
                                                                loss_parallel_lambda=loss_parallel_lambda,
                                                                cl_lambda=cl_lambda,)

                loss_sen_forward = 0
                loss_sen_forward_hter = 0
                loss_sen_backward = 0
                loss_sen_backward_hter = 0
                if np.isnan(loss_word) or np.isnan(loss_hter) or np.isnan(loss_sen_forward) or \
                    np.isnan(loss_sen_forward_hter) or np.isnan(loss_sen_backward) or np.isnan(loss_sen_backward_hter):
                    WARN("NaN detected!")
                update_cycle -= 1
                grad_denom += batch_size

                train_loss_sen_forward += loss_sen_forward
                train_loss_sen_backward += loss_sen_backward
                train_loss_word += loss_word

                train_loss_sen_forward_hter += loss_sen_forward_hter
                train_loss_sen_backward_hter += loss_sen_backward_hter
                train_loss_word_hter += loss_hter

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom_count += 1
                else:
                    raise e

            # When update_cycle becomes 0, it means end of one batch. Several things will be done:
            # - update parameters
            # - reset update_cycle and grad_denom, update uidx
            # - learning rate scheduling
            # - update moving average

            if update_cycle == 0:

                # 0. reduce variables
                if world_size > 1:
                    grad_denom = dist.all_reduce_py(grad_denom)
                    train_loss_sen_forward = dist.all_reduce_py(train_loss_sen_forward)
                    train_loss_sen_backward = dist.all_reduce_py(train_loss_sen_backward)
                    train_loss_word = dist.all_reduce_py(train_loss_word)
                    train_loss_sen_forward_hter = dist.all_reduce_py(train_loss_sen_forward_hter)
                    train_loss_sen_backward_hter = dist.all_reduce_py(train_loss_sen_backward_hter)
                    train_loss_word_hter = dist.all_reduce_py(train_loss_word_hter)
                    cum_n_words = dist.all_reduce_py(cum_n_words)

                # 1. update parameters
                optim.step(denom=grad_denom)
                optim.zero_grad()

                if training_progress_bar is not None:
                    training_progress_bar.update(grad_denom)
                    training_progress_bar.set_description(' - (Epc {}, Upd {}) '.format(eidx, uidx))

                # 2. learning rate scheduling
                if scheduler is not None and optimizer_configs["schedule_method"] != "loss":
                    scheduler.step(global_step=uidx)

                # 3. update moving average
                if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                    ma.step()

                # 4. update meters
                train_loss_sen_forward_meter.update(train_loss_sen_forward, grad_denom)
                train_loss_sen_backward_meter.update(train_loss_sen_backward, grad_denom)
                train_loss_word_meter.update(train_loss_word, grad_denom)
                train_loss_sen_forward_hter_meter.update(train_loss_sen_forward_hter, grad_denom)
                train_loss_sen_backward_hter_meter.update(train_loss_sen_backward_hter, grad_denom)
                train_loss_word_hter_meter.update(train_loss_word_hter, grad_denom)
                sent_per_sec_meter.update(grad_denom)
                tok_per_sec_meter.update(cum_n_words)

                # 5. reset accumulated variables, update uidx
                update_cycle = training_configs['update_cycle']
                grad_denom = 0
                uidx += 1
                cum_n_words = 0.0
                train_loss_sen_forward = 0.0
                train_loss_sen_backward = 0.0
                train_loss_word = 0.0

                train_loss_sen_forward_hter = 0.0
                train_loss_sen_backward_hter = 0.0
                train_loss_word_hter = 0.0

            else:
                continue
            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):

                lrate = list(optim.get_lrate())[0]

                if summary_writer is not None:
                    summary_writer.add_scalar("Speed(sents/sec)", scalar_value=sent_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("Speed(words/sec)", scalar_value=tok_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_forward", scalar_value=train_loss_sen_forward_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_backward", scalar_value=train_loss_sen_backward_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_word", scalar_value=train_loss_word_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_forward_hter",
                                              scalar_value=train_loss_sen_forward_hter_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_backward_hter",
                                              scalar_value=train_loss_sen_backward_hter_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_word_hter", scalar_value=train_loss_word_hter_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                    summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)

                # Reset Meters
                sent_per_sec_meter.reset()
                tok_per_sec_meter.reset()
                train_loss_sen_forward_meter.reset()
                train_loss_sen_backward_meter.reset()
                train_loss_word_meter.reset()
                train_loss_sen_forward_hter_meter.reset()
                train_loss_sen_backward_hter_meter.reset()
                train_loss_word_hter_meter.reset()

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):
                with cache_parameters(discriminator):

                    if ma is not None:
                        discriminator.load_state_dict(ma.export_ma_params(), strict=False)

                    if not valid_cl:
                        # 使用普通词级别loss验证，训练验证目标不一致
                        valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                            loss_evaluation_qe_v2(discriminator=discriminator,
                                            critic=critic,
                                            critic_mse=critic_mse,
                                            valid_iterator=valid_iterator)
                    else:
                        # 训练验证目标一致，验证也是词级loss+对比loss
                        valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                            loss_evaluation_qe_contrastive_src(discriminator=discriminator,
                                                            critic=critic,
                                                            critic_mse=critic_mse,
                                                            valid_iterator=valid_iterator,
                                                            loss_real_lambda=loss_real_lambda,
                                                            loss_parallel_lambda=loss_parallel_lambda,
                                                            cl_lambda=cl_lambda,)
         
                    # 保存ckpt
                    torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                
                if scheduler is not None and optimizer_configs["schedule_method"] == "loss":
                    scheduler.step(metric=valid_loss)

                model_collections.add_to_collection("history_loss", valid_loss)

                min_valid_loss = np.array(model_collections.get_collection("history_loss")).min()

                if summary_writer is not None:
                    summary_writer.add_scalar("loss_word", valid_loss, global_step=uidx)
                    summary_writer.add_scalar('loss_hter', valid_loss_hter, global_step=uidx)
                    summary_writer.add_scalar('f1good', f1good, global_step=uidx)
                    summary_writer.add_scalar('f1bad', f1bad, global_step=uidx)
                    summary_writer.add_scalar('f1multi', f1multi, global_step=uidx)
                    summary_writer.add_scalar('pearsonr', p, global_step=uidx)

                if eidx >= 0:
                    if valid_loss <= min_valid_loss:
                        bad_count = 0
                        if is_early_stop is False and rank == 0:
                            INFO("save")
                            torch.save(discriminator.state_dict(), best_model_prefix + '.final')
                    else:
                        bad_count += 1
                        if bad_count >= training_configs['early_stop_patience']:
                            is_early_stop = True
                            WARN("Early Stop")

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.6f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.6f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
    
            if is_early_stop is True:
                break
            # ================================================================================== #
        if training_progress_bar is not None:
            training_progress_bar.close()

        if is_early_stop is True:
            break

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def train_discriminator_no_need_2_generate(flags):
    """
    直接用存好的训练数据，而不是一边从平行语料生成一边训
    flags:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # ================================================================================== #
    # Initialization for training on different devices
    # - CPU/GPU
    # - Single/Distributed
    Constants.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if Constants.USE_GPU:
        torch.cuda.set_device(local_rank)
        Constants.CURRENT_DEVICE = "cuda:{0}".format(local_rank)
    else:
        Constants.CURRENT_DEVICE = "cpu"

    # If not root_rank, close logging
    # else write log of training to file.
    if rank == 0:
        write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    else:
        close_logging()

    # ================================================================================== #
    # Parsing configuration files
    # - Load default settings
    # - Load pre-defined settings
    # - Load user-defined settings

    configs = prepare_configs(flags.config_path, flags.predefined_config)

    data_configs = configs['data_configs']
    discriminator_configs = configs['discriminator_configs']
    generator_configs = configs['generator_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    INFO(pretty_configs(configs))
  
    Constants.SEED = training_configs['seed']

    set_seed(Constants.SEED)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    Constants.EOS = vocab_src.eos
    Constants.PAD = vocab_src.pad
    Constants.BOS = vocab_src.bos
    Constants.MASK = vocab_src.mask

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2]),
        QEHterDataset(data_path=data_configs['train_data'][3])
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs["batch_size"],
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=training_configs['buffer_size'],
                                     batching_func=training_configs['batching_key'],
                                     world_size=world_size,
                                     rank=rank,
                                     shuffle=True)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, buffer_size=100000, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 0. Initial

    lrate = optimizer_configs['learning_rate']
    model_collections = Collections()

    best_model_prefix = os.path.join(flags.saveto, flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)

    # 1. Build Model & Criterion
    INFO('Building model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words,
                                padding_idx=vocab_src.pad, **discriminator_configs)
    """
    generator_mlm = build_model(n_src_vocab=vocab_tgt.max_n_words,   # 注意，按照反向改动了
                                n_tgt_vocab=vocab_src.max_n_words,
                                padding_idx=vocab_tgt.pad, **generator_configs)
    """
    generator_mlm = build_model(n_src_vocab=vocab_src.max_n_words, 
                                n_tgt_vocab=vocab_tgt.max_n_words,
                                padding_idx=vocab_tgt.pad, **generator_configs)

    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)
    # 2. Move to GPU
    if Constants.USE_GPU:
        discriminator = discriminator.cuda()
        generator_mlm = generator_mlm.cuda()
        # generator_lm_forward = generator_lm_forward.cuda()
        # generator_lm_backward = generator_lm_backward.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(generator_mlm, flags.pretrain_generator_mlm_path, exclude_prefix=None,
                          device=Constants.CURRENT_DEVICE)

    # 加载MLM的embedding层到判别器中
    INFO('load embedding')
    times_bigger = int(discriminator_configs['d_model'] / generator_configs['d_model'])
    for key in discriminator.encoder.embeddings.embeddings.state_dict().keys():
        discriminator.encoder.embeddings.embeddings.state_dict()[key].copy_(
            generator_mlm.encoder.embeddings.embeddings.state_dict()[key].repeat(1, times_bigger))   # 注意方向
        discriminator.decoder.embeddings.embeddings.state_dict()[key].copy_(
            generator_mlm.decoder.embeddings.embeddings.state_dict()[key].repeat(1, times_bigger))

    INFO('Done. Elapsed time {0}'.format(timer.toc()))
    load_pretrained_model(discriminator, flags.pretrain_discriminator_path, exclude_prefix=None,device=Constants.CURRENT_DEVICE)
    # 4. Build optimizer
    INFO('Building Optimizer...')

    if not flags.multi_gpu:
        optim = Optimizer(name=optimizer_configs['optimizer'],
                          model=discriminator,
                          lr=lrate,
                          grad_clip=optimizer_configs['grad_clip'],
                          optim_args=optimizer_configs['optimizer_params'],
                          update_cycle=training_configs['update_cycle']
                          )
    else:
        optim = dist.DistributedOptimizer(name=optimizer_configs['optimizer'],
                                          model=discriminator,
                                          lr=lrate,
                                          grad_clip=optimizer_configs['grad_clip'],
                                          optim_args=optimizer_configs['optimizer_params'],
                                          device_id=local_rank
                                          )

    # 加载embedding后是否要更新embedding参数。似乎embedding矩阵没有bias。
    # discriminator.encoder.embeddings.embeddings.weight.requires_grad = False
    # discriminator.decoder.embeddings.embeddings.weight.requires_grad = False

    # 5. Build scheduler for optimizer if needed
    scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])

    # 6. build moving average
    if training_configs['moving_average_method'] is not None:
        ma = MovingAverage(moving_average_method=training_configs['moving_average_method'],
                           named_params=discriminator.named_parameters(),
                           alpha=training_configs['moving_average_alpha'])
    else:
        ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # broadcast parameters and optimizer states
    if world_size > 1:
        INFO("Broadcasting model parameters...")
        dist.broadcast_parameters(params=discriminator.state_dict())
        dist.broadcast_parameters(params=generator_mlm.state_dict())
        # dist.broadcast_parameters(params=generator_lm_forward.state_dict())
        # dist.broadcast_parameters(params=generator_lm_backward.state_dict())
        INFO("Broadcasting optimizer states...")
        dist.broadcast_optimizer_state(optimizer=optim.optim)
        INFO('Done.')

    # ================================================================================== #
    # Prepare training

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [1])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]
    is_early_stop = model_collections.get_collection("is_early_stop", [False, ])[-1]

    train_loss_sen_forward_meter = AverageMeter()
    train_loss_sen_backward_meter = AverageMeter()
    train_loss_word_meter = AverageMeter()
    train_loss_sen_forward_hter_meter = AverageMeter()
    train_loss_sen_backward_hter_meter = AverageMeter()
    train_loss_word_hter_meter = AverageMeter()
    sent_per_sec_meter = TimeMeter()
    tok_per_sec_meter = TimeMeter()

    update_cycle = training_configs['update_cycle']
    grad_denom = 0
    train_loss_sen_forward = 0.0
    train_loss_sen_backward = 0.0
    train_loss_word = 0.0
    train_loss_sen_forward_hter = 0.0
    train_loss_sen_backward_hter = 0.0
    train_loss_word_hter = 0.0
    cum_n_words = 0

    if rank == 0:
        summary_writer = SummaryWriter(log_dir=flags.log_path)
    else:
        summary_writer = None

    sent_per_sec_meter.start()
    tok_per_sec_meter.start()

    INFO('Begin training...')

    while True:

        if summary_writer is not None:
            summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()

        if rank == 0:
            training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                         total=len(training_iterator),
                                         unit="sents"
                                         )
        else:
            training_progress_bar = None


        for batch in training_iter:

            seqs_x, seqs_y, xy_label, xy_hter = batch
            

            batch_size = len(seqs_x)
            cum_n_words += sum(len(s) for s in seqs_y)

            
            try:
                x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)

                if Constants.USE_GPU:
                    xy_hter = torch.tensor(xy_hter).squeeze(-1)
                    xy_hter = xy_hter.cuda()

                loss_word, loss_word_hter = compute_forward_discriminator(model=discriminator,
                                                                critic=critic,
                                                                critic_mse=critic_mse,
                                                                seqs_x=x,
                                                                seqs_y=y,
                                                                lm_ids=xy_label,
                                                                hter_label=xy_hter)
                
                # TODO 可以记录每个batch的 loss_word, loss_word_hter

                loss_sen_forward = 0
                loss_sen_forward_hter = 0
                loss_sen_backward = 0
                loss_sen_backward_hter = 0
                if np.isnan(loss_word) or np.isnan(loss_word_hter) or np.isnan(loss_sen_forward) or \
                    np.isnan(loss_sen_forward_hter) or np.isnan(loss_sen_backward) or np.isnan(loss_sen_backward_hter):
                    WARN("NaN detected!")
                update_cycle -= 1
                grad_denom += batch_size

                train_loss_sen_forward += loss_sen_forward
                train_loss_sen_backward += loss_sen_backward
                train_loss_word += loss_word

                train_loss_sen_forward_hter += loss_sen_forward_hter
                train_loss_sen_backward_hter += loss_sen_backward_hter
                train_loss_word_hter += loss_word_hter

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom_count += 1
                else:
                    raise e

            # When update_cycle becomes 0, it means end of one batch. Several things will be done:
            # - update parameters
            # - reset update_cycle and grad_denom, update uidx
            # - learning rate scheduling
            # - update moving average

            if update_cycle == 0:

                # 0. reduce variables
                if world_size > 1:
                    grad_denom = dist.all_reduce_py(grad_denom)
                    train_loss_sen_forward = dist.all_reduce_py(train_loss_sen_forward)
                    train_loss_sen_backward = dist.all_reduce_py(train_loss_sen_backward)
                    train_loss_word = dist.all_reduce_py(train_loss_word)
                    train_loss_sen_forward_hter = dist.all_reduce_py(train_loss_sen_forward_hter)
                    train_loss_sen_backward_hter = dist.all_reduce_py(train_loss_sen_backward_hter)
                    train_loss_word_hter = dist.all_reduce_py(train_loss_word_hter)
                    cum_n_words = dist.all_reduce_py(cum_n_words)

                # 1. update parameters
                optim.step(denom=grad_denom)
                optim.zero_grad()

                if training_progress_bar is not None:
                    training_progress_bar.update(grad_denom)
                    training_progress_bar.set_description(' - (Epc {}, Upd {}) '.format(eidx, uidx))

                # 2. learning rate scheduling
                if scheduler is not None and optimizer_configs["schedule_method"] != "loss":
                    scheduler.step(global_step=uidx)

                # 3. update moving average
                if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                    ma.step()

                # 4. update meters
                train_loss_sen_forward_meter.update(train_loss_sen_forward, grad_denom)
                train_loss_sen_backward_meter.update(train_loss_sen_backward, grad_denom)
                train_loss_word_meter.update(train_loss_word, grad_denom)
                train_loss_sen_forward_hter_meter.update(train_loss_sen_forward_hter, grad_denom)
                train_loss_sen_backward_hter_meter.update(train_loss_sen_backward_hter, grad_denom)
                train_loss_word_hter_meter.update(train_loss_word_hter, grad_denom)
                sent_per_sec_meter.update(grad_denom)
                tok_per_sec_meter.update(cum_n_words)

                # 5. reset accumulated variables, update uidx
                update_cycle = training_configs['update_cycle']
                grad_denom = 0
                uidx += 1
                cum_n_words = 0.0
                train_loss_sen_forward = 0.0
                train_loss_sen_backward = 0.0
                train_loss_word = 0.0

                train_loss_sen_forward_hter = 0.0
                train_loss_sen_backward_hter = 0.0
                train_loss_word_hter = 0.0

            else:
                continue
            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):

                lrate = list(optim.get_lrate())[0]

                if summary_writer is not None:
                    summary_writer.add_scalar("Speed(sents/sec)", scalar_value=sent_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("Speed(words/sec)", scalar_value=tok_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_forward", scalar_value=train_loss_sen_forward_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_backward", scalar_value=train_loss_sen_backward_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_word", scalar_value=train_loss_word_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_forward_hter",
                                              scalar_value=train_loss_sen_forward_hter_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_backward_hter",
                                              scalar_value=train_loss_sen_backward_hter_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_word_hter", scalar_value=train_loss_word_hter_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                    summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)

                # Reset Meters
                sent_per_sec_meter.reset()
                tok_per_sec_meter.reset()
                train_loss_sen_forward_meter.reset()
                train_loss_sen_backward_meter.reset()
                train_loss_word_meter.reset()
                train_loss_sen_forward_hter_meter.reset()
                train_loss_sen_backward_hter_meter.reset()
                train_loss_word_hter_meter.reset()

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):
                with cache_parameters(discriminator):

                    if ma is not None:
                        discriminator.load_state_dict(ma.export_ma_params(), strict=False)

                    valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                        loss_evaluation_qe(discriminator=discriminator,
                                           critic=critic,
                                           critic_mse=critic_mse,
                                           valid_iterator=valid_iterator)

                    # 保存ckpt
                    torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                

                if scheduler is not None and optimizer_configs["schedule_method"] == "loss":
                    scheduler.step(metric=valid_loss)

                model_collections.add_to_collection("history_loss", valid_loss)

                min_valid_loss = np.array(model_collections.get_collection("history_loss")).min()

                if summary_writer is not None:
                    summary_writer.add_scalar("loss_word", valid_loss, global_step=uidx)
                    summary_writer.add_scalar('loss_hter', valid_loss_hter, global_step=uidx)
                    summary_writer.add_scalar('f1good', f1good, global_step=uidx)
                    summary_writer.add_scalar('f1bad', f1bad, global_step=uidx)
                    summary_writer.add_scalar('f1multi', f1multi, global_step=uidx)
                    summary_writer.add_scalar('pearsonr', p, global_step=uidx)

                if eidx >= 0:
                    if valid_loss <= min_valid_loss:
                        bad_count = 0
                        if is_early_stop is False and rank == 0:
                            INFO("save")
                            torch.save(discriminator.state_dict(), best_model_prefix + '.final')
                    else:
                        bad_count += 1
                        if bad_count >= training_configs['early_stop_patience']:
                            is_early_stop = True
                            WARN("Early Stop")

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.6f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.6f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
    
            if is_early_stop is True:
                break
            # ================================================================================== #
        if training_progress_bar is not None:
            training_progress_bar.close()

        if is_early_stop is True:
            break

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def train_discriminator_no_need_2_generate_log(flags):
    """
    直接用存好的训练数据，而不是一边从平行语料生成一边训
    可以存储loss什么的
    flags:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # ================================================================================== #
    # Initialization for training on different devices
    # - CPU/GPU
    # - Single/Distributed
    Constants.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if Constants.USE_GPU:
        torch.cuda.set_device(local_rank)
        Constants.CURRENT_DEVICE = "cuda:{0}".format(local_rank)
    else:
        Constants.CURRENT_DEVICE = "cpu"

    # If not root_rank, close logging
    # else write log of training to file.
    if rank == 0:
        write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    else:
        close_logging()

    # ================================================================================== #
    # Parsing configuration files
    # - Load default settings
    # - Load pre-defined settings
    # - Load user-defined settings

    configs = prepare_configs(flags.config_path, flags.predefined_config)

    data_configs = configs['data_configs']
    discriminator_configs = configs['discriminator_configs']
    generator_configs = configs['generator_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    INFO(pretty_configs(configs))
  
    Constants.SEED = training_configs['seed']

    set_seed(Constants.SEED)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    Constants.EOS = vocab_src.eos
    Constants.PAD = vocab_src.pad
    Constants.BOS = vocab_src.bos
    Constants.MASK = vocab_src.mask

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2]),
        QEHterDataset(data_path=data_configs['train_data'][3])
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    toy_train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['toy_train_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['toy_train_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['toy_train_data'][2]),
        QEHterDataset(data_path=data_configs['toy_train_data'][3])
    )

    finetune_valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['finetune_valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['finetune_valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['finetune_valid_data'][2]),
        QEHterDataset(data_path=data_configs['finetune_valid_data'][3])
    )

    finetune_toy_train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['finetune_toy_train_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['finetune_toy_train_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['finetune_toy_train_data'][2]),
        QEHterDataset(data_path=data_configs['finetune_toy_train_data'][3])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs["batch_size"],
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=training_configs['buffer_size'],
                                     batching_func=training_configs['batching_key'],
                                     world_size=world_size,
                                     rank=rank,
                                     shuffle=True)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, buffer_size=100000, numbering=False)

    toy_train_iterator = DataIterator(dataset=toy_train_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, buffer_size=100000, numbering=False)

    finetune_valid_iterator = DataIterator(dataset=finetune_valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, buffer_size=100000, numbering=False)

    finetune_toy_train_iterator = DataIterator(dataset=finetune_toy_train_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, buffer_size=100000, numbering=False)
                                  
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 0. Initial

    lrate = optimizer_configs['learning_rate']
    model_collections = Collections()

    best_model_prefix = os.path.join(flags.saveto, flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)

    # 1. Build Model & Criterion
    INFO('Building model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words,
                                padding_idx=vocab_src.pad, **discriminator_configs)

    generator_mlm = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words,
                                padding_idx=vocab_src.pad, **generator_configs)

    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)
    # 2. Move to GPU
    if Constants.USE_GPU:
        discriminator = discriminator.cuda()
        generator_mlm = generator_mlm.cuda()
        # generator_lm_forward = generator_lm_forward.cuda()
        # generator_lm_backward = generator_lm_backward.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(generator_mlm, flags.pretrain_generator_mlm_path, exclude_prefix=None,
                          device=Constants.CURRENT_DEVICE)

    # 加载MLM的embedding层到判别器中
    INFO('load embedding')
    times_bigger = int(discriminator_configs['d_model'] / generator_configs['d_model'])
    for key in discriminator.encoder.embeddings.embeddings.state_dict().keys():
        discriminator.encoder.embeddings.embeddings.state_dict()[key].copy_(
            generator_mlm.encoder.embeddings.embeddings.state_dict()[key].repeat(1, times_bigger))
        discriminator.decoder.embeddings.embeddings.state_dict()[key].copy_(
            generator_mlm.decoder.embeddings.embeddings.state_dict()[key].repeat(1, times_bigger))

    INFO('Done. Elapsed time {0}'.format(timer.toc()))
    load_pretrained_model(discriminator, flags.pretrain_discriminator_path, exclude_prefix=None,device=Constants.CURRENT_DEVICE)
    # 4. Build optimizer
    INFO('Building Optimizer...')

    if not flags.multi_gpu:
        optim = Optimizer(name=optimizer_configs['optimizer'],
                          model=discriminator,
                          lr=lrate,
                          grad_clip=optimizer_configs['grad_clip'],
                          optim_args=optimizer_configs['optimizer_params'],
                          update_cycle=training_configs['update_cycle']
                          )
    else:
        optim = dist.DistributedOptimizer(name=optimizer_configs['optimizer'],
                                          model=discriminator,
                                          lr=lrate,
                                          grad_clip=optimizer_configs['grad_clip'],
                                          optim_args=optimizer_configs['optimizer_params'],
                                          device_id=local_rank
                                          )

    # 加载embedding后是否要更新embedding参数。似乎embedding矩阵没有bias。
    # discriminator.encoder.embeddings.embeddings.weight.requires_grad = False
    # discriminator.decoder.embeddings.embeddings.weight.requires_grad = False

    # 5. Build scheduler for optimizer if needed
    scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])

    # 6. build moving average
    if training_configs['moving_average_method'] is not None:
        ma = MovingAverage(moving_average_method=training_configs['moving_average_method'],
                           named_params=discriminator.named_parameters(),
                           alpha=training_configs['moving_average_alpha'])
    else:
        ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # broadcast parameters and optimizer states
    if world_size > 1:
        INFO("Broadcasting model parameters...")
        dist.broadcast_parameters(params=discriminator.state_dict())
        dist.broadcast_parameters(params=generator_mlm.state_dict())
        # dist.broadcast_parameters(params=generator_lm_forward.state_dict())
        # dist.broadcast_parameters(params=generator_lm_backward.state_dict())
        INFO("Broadcasting optimizer states...")
        dist.broadcast_optimizer_state(optimizer=optim.optim)
        INFO('Done.')

    # ================================================================================== #
    # Prepare training

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [1])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]
    is_early_stop = model_collections.get_collection("is_early_stop", [False, ])[-1]

    train_loss_sen_forward_meter = AverageMeter()
    train_loss_sen_backward_meter = AverageMeter()
    train_loss_word_meter = AverageMeter()
    train_loss_sen_forward_hter_meter = AverageMeter()
    train_loss_sen_backward_hter_meter = AverageMeter()
    train_loss_word_hter_meter = AverageMeter()
    sent_per_sec_meter = TimeMeter()
    tok_per_sec_meter = TimeMeter()

    update_cycle = training_configs['update_cycle']
    grad_denom = 0
    train_loss_sen_forward = 0.0
    train_loss_sen_backward = 0.0
    train_loss_word = 0.0
    train_loss_sen_forward_hter = 0.0
    train_loss_sen_backward_hter = 0.0
    train_loss_word_hter = 0.0
    cum_n_words = 0

    if rank == 0:
        summary_writer = SummaryWriter(log_dir=flags.log_path)
    else:
        summary_writer = None

    sent_per_sec_meter.start()
    tok_per_sec_meter.start()

    INFO('Begin training...')

    loss_dict = {
        'pretrain_train_loss':{'loss_word':[], 'loss_hter':[]},
        'pretrain_train_eval':{'loss_word':[], 'loss_hter':[], 'f1_ok':[], 'f1_bad':[], 'f1_multi':[], 'pearsonr':[]},
        'pretrain_val_eval':  {'loss_word':[], 'loss_hter':[], 'f1_ok':[], 'f1_bad':[], 'f1_multi':[], 'pearsonr':[]},
        'finetune_train_eval':{'loss_word':[], 'loss_hter':[], 'f1_ok':[], 'f1_bad':[], 'f1_multi':[], 'pearsonr':[]},
        'finetune_val_eval':  {'loss_word':[], 'loss_hter':[], 'f1_ok':[], 'f1_bad':[], 'f1_multi':[], 'pearsonr':[]},
    }

    while True:

        if summary_writer is not None:
            summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()

        if rank == 0:
            training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                         total=len(training_iterator),
                                         unit="sents"
                                         )
        else:
            training_progress_bar = None


        for batch in training_iter:

            seqs_x, seqs_y, xy_label, xy_hter = batch
            

            batch_size = len(seqs_x)
            cum_n_words += sum(len(s) for s in seqs_y)

            
            try:

                x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)

                if Constants.USE_GPU:
                    xy_hter = torch.tensor(xy_hter).squeeze(-1)
                    xy_hter = xy_hter.cuda()

                loss_word, loss_word_hter = compute_forward_discriminator(model=discriminator,
                                                                critic=critic,
                                                                critic_mse=critic_mse,
                                                                seqs_x=x,
                                                                seqs_y=y,
                                                                lm_ids=xy_label,
                                                                hter_label=xy_hter)
                
                # 记录每个batch的 loss_word, loss_word_hter
                loss_dict['pretrain_train_loss']['loss_word'].append(loss_word)
                loss_dict['pretrain_train_loss']['loss_hter'].append(loss_word_hter)

                loss_sen_forward = 0
                loss_sen_forward_hter = 0
                loss_sen_backward = 0
                loss_sen_backward_hter = 0
                if np.isnan(loss_word) or np.isnan(loss_word_hter) or np.isnan(loss_sen_forward) or \
                    np.isnan(loss_sen_forward_hter) or np.isnan(loss_sen_backward) or np.isnan(loss_sen_backward_hter):
                    WARN("NaN detected!")
                update_cycle -= 1
                grad_denom += batch_size

                train_loss_sen_forward += loss_sen_forward
                train_loss_sen_backward += loss_sen_backward
                train_loss_word += loss_word

                train_loss_sen_forward_hter += loss_sen_forward_hter
                train_loss_sen_backward_hter += loss_sen_backward_hter
                train_loss_word_hter += loss_word_hter

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom_count += 1
                else:
                    raise e

            # When update_cycle becomes 0, it means end of one batch. Several things will be done:
            # - update parameters
            # - reset update_cycle and grad_denom, update uidx
            # - learning rate scheduling
            # - update moving average

            if update_cycle == 0:

                # 0. reduce variables
                if world_size > 1:
                    grad_denom = dist.all_reduce_py(grad_denom)
                    train_loss_sen_forward = dist.all_reduce_py(train_loss_sen_forward)
                    train_loss_sen_backward = dist.all_reduce_py(train_loss_sen_backward)
                    train_loss_word = dist.all_reduce_py(train_loss_word)
                    train_loss_sen_forward_hter = dist.all_reduce_py(train_loss_sen_forward_hter)
                    train_loss_sen_backward_hter = dist.all_reduce_py(train_loss_sen_backward_hter)
                    train_loss_word_hter = dist.all_reduce_py(train_loss_word_hter)
                    cum_n_words = dist.all_reduce_py(cum_n_words)

                # 1. update parameters
                optim.step(denom=grad_denom)
                optim.zero_grad()

                if training_progress_bar is not None:
                    training_progress_bar.update(grad_denom)
                    training_progress_bar.set_description(' - (Epc {}, Upd {}) '.format(eidx, uidx))

                # 2. learning rate scheduling
                if scheduler is not None and optimizer_configs["schedule_method"] != "loss":
                    scheduler.step(global_step=uidx)

                # 3. update moving average
                if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                    ma.step()

                # 4. update meters
                train_loss_sen_forward_meter.update(train_loss_sen_forward, grad_denom)
                train_loss_sen_backward_meter.update(train_loss_sen_backward, grad_denom)
                train_loss_word_meter.update(train_loss_word, grad_denom)
                train_loss_sen_forward_hter_meter.update(train_loss_sen_forward_hter, grad_denom)
                train_loss_sen_backward_hter_meter.update(train_loss_sen_backward_hter, grad_denom)
                train_loss_word_hter_meter.update(train_loss_word_hter, grad_denom)
                sent_per_sec_meter.update(grad_denom)
                tok_per_sec_meter.update(cum_n_words)

                # 5. reset accumulated variables, update uidx
                update_cycle = training_configs['update_cycle']
                grad_denom = 0
                uidx += 1
                cum_n_words = 0.0
                train_loss_sen_forward = 0.0
                train_loss_sen_backward = 0.0
                train_loss_word = 0.0

                train_loss_sen_forward_hter = 0.0
                train_loss_sen_backward_hter = 0.0
                train_loss_word_hter = 0.0

            else:
                continue
            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):

                lrate = list(optim.get_lrate())[0]

                if summary_writer is not None:
                    summary_writer.add_scalar("Speed(sents/sec)", scalar_value=sent_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("Speed(words/sec)", scalar_value=tok_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_forward", scalar_value=train_loss_sen_forward_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_backward", scalar_value=train_loss_sen_backward_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_word", scalar_value=train_loss_word_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_forward_hter",
                                              scalar_value=train_loss_sen_forward_hter_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_backward_hter",
                                              scalar_value=train_loss_sen_backward_hter_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_word_hter", scalar_value=train_loss_word_hter_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                    summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)

                # Reset Meters
                sent_per_sec_meter.reset()
                tok_per_sec_meter.reset()
                train_loss_sen_forward_meter.reset()
                train_loss_sen_backward_meter.reset()
                train_loss_word_meter.reset()
                train_loss_sen_forward_hter_meter.reset()
                train_loss_sen_backward_hter_meter.reset()
                train_loss_word_hter_meter.reset()

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):
                with cache_parameters(discriminator):

                    if ma is not None:
                        discriminator.load_state_dict(ma.export_ma_params(), strict=False)

                    valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                        loss_evaluation_qe(discriminator=discriminator,
                                           critic=critic,
                                           critic_mse=critic_mse,
                                           valid_iterator=valid_iterator)

                    toy_train_loss, toy_train_loss_hter, toy_train_f1good, toy_train_f1bad, toy_train_f1multi, toy_train_p = \
                        loss_evaluation_qe(discriminator=discriminator,
                                           critic=critic,
                                           critic_mse=critic_mse,
                                           valid_iterator=toy_train_iterator)
                    
                    finetune_valid_loss, finetune_valid_loss_hter, finetune_f1good, finetune_f1bad, finetune_f1multi, finetune_p = \
                        loss_evaluation_qe(discriminator=discriminator,
                                           critic=critic,
                                           critic_mse=critic_mse,
                                           valid_iterator=finetune_valid_iterator)

                    finetune_toy_train_loss, finetune_toy_train_loss_hter, finetune_toy_train_f1good, \
                        finetune_toy_train_f1bad, finetune_toy_train_f1multi, finetune_toy_train_p = \
                        loss_evaluation_qe(discriminator=discriminator,
                                           critic=critic,
                                           critic_mse=critic_mse,
                                           valid_iterator=finetune_toy_train_iterator)
                    
                    # 保存 loss记录
                    loss_dict['pretrain_train_eval']['loss_word'].append(toy_train_loss)
                    loss_dict['pretrain_train_eval']['loss_hter'].append(toy_train_loss_hter)
                    loss_dict['pretrain_train_eval']['f1_ok'].append(toy_train_f1good)
                    loss_dict['pretrain_train_eval']['f1_bad'].append(toy_train_f1bad)
                    loss_dict['pretrain_train_eval']['f1_multi'].append(toy_train_f1multi)
                    loss_dict['pretrain_train_eval']['pearsonr'].append(toy_train_p)

                    loss_dict['pretrain_val_eval']['loss_word'].append(valid_loss)
                    loss_dict['pretrain_val_eval']['loss_hter'].append(valid_loss_hter)
                    loss_dict['pretrain_val_eval']['f1_ok'].append(f1good)
                    loss_dict['pretrain_val_eval']['f1_bad'].append(f1bad)
                    loss_dict['pretrain_val_eval']['f1_multi'].append(f1multi)
                    loss_dict['pretrain_val_eval']['pearsonr'].append(p)
                    
                    loss_dict['finetune_train_eval']['loss_word'].append(finetune_toy_train_loss)
                    loss_dict['finetune_train_eval']['loss_hter'].append(finetune_toy_train_loss_hter)
                    loss_dict['finetune_train_eval']['f1_ok'].append(finetune_toy_train_f1good)
                    loss_dict['finetune_train_eval']['f1_bad'].append(finetune_toy_train_f1bad)
                    loss_dict['finetune_train_eval']['f1_multi'].append(finetune_toy_train_f1multi)
                    loss_dict['finetune_train_eval']['pearsonr'].append(finetune_toy_train_p)

                    loss_dict['finetune_val_eval']['loss_word'].append(finetune_valid_loss)
                    loss_dict['finetune_val_eval']['loss_hter'].append(finetune_valid_loss_hter)
                    loss_dict['finetune_val_eval']['f1_ok'].append(finetune_f1good)
                    loss_dict['finetune_val_eval']['f1_bad'].append(finetune_f1bad)
                    loss_dict['finetune_val_eval']['f1_multi'].append(finetune_f1multi)
                    loss_dict['finetune_val_eval']['pearsonr'].append(finetune_p)

                    #INFO("save ckpt")
                    #torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                    


                if scheduler is not None and optimizer_configs["schedule_method"] == "loss":
                    scheduler.step(metric=valid_loss)

                model_collections.add_to_collection("history_loss", valid_loss)

                min_valid_loss = np.array(model_collections.get_collection("history_loss")).min()

                if summary_writer is not None:
                    summary_writer.add_scalar("loss_word", valid_loss, global_step=uidx)
                    summary_writer.add_scalar('loss_hter', valid_loss_hter, global_step=uidx)
                    summary_writer.add_scalar('f1good', f1good, global_step=uidx)
                    summary_writer.add_scalar('f1bad', f1bad, global_step=uidx)
                    summary_writer.add_scalar('f1multi', f1multi, global_step=uidx)
                    summary_writer.add_scalar('pearsonr', p, global_step=uidx)

                if eidx >= 0:
                    if valid_loss <= min_valid_loss:
                        bad_count = 0
                        if is_early_stop is False and rank == 0:
                            INFO("save")
                            torch.save(discriminator.state_dict(), best_model_prefix + '.final')
                    else:
                        bad_count += 1
                        if bad_count >= training_configs['early_stop_patience']:
                            is_early_stop = True
                            WARN("Early Stop")

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.6f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.6f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
    
            if is_early_stop is True:
                break
            # ================================================================================== #
        if training_progress_bar is not None:
            training_progress_bar.close()

        if is_early_stop is True:
            break

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break

    with open(flags.saveto + 'loss_dict.json', 'w', encoding='utf-8') as f:
        json.dump(loss_dict, f, indent=1)


def train_discriminator_no_need_2_generate_cl(flags):
    """
    直接用存好的训练数据，而不是一边从平行语料生成一边训；
    加入平行语料和伪数据的对比学习，修改损失函数，加入对比项
    flags:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # ================================================================================== #
    # Initialization for training on different devices
    # - CPU/GPU
    # - Single/Distributed
    Constants.USE_GPU = flags.use_gpu

    if flags.multi_gpu:
        dist.distributed_init(flags.shared_dir)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = dist.get_local_rank()
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if Constants.USE_GPU:
        torch.cuda.set_device(local_rank)
        Constants.CURRENT_DEVICE = "cuda:{0}".format(local_rank)
    else:
        Constants.CURRENT_DEVICE = "cpu"

    # If not root_rank, close logging
    # else write log of training to file.
    if rank == 0:
        write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    else:
        close_logging()

    # ================================================================================== #
    # Parsing configuration files
    # - Load default settings
    # - Load pre-defined settings
    # - Load user-defined settings

    configs = prepare_configs(flags.config_path, flags.predefined_config)

    data_configs = configs['data_configs']
    discriminator_configs = configs['discriminator_configs']
    generator_configs = configs['generator_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    INFO(pretty_configs(configs))
  
    Constants.SEED = training_configs['seed']

    set_seed(Constants.SEED)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    Constants.EOS = vocab_src.eos
    Constants.PAD = vocab_src.pad
    Constants.BOS = vocab_src.bos
    Constants.MASK = vocab_src.mask

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2]),
        QEHterDataset(data_path=data_configs['train_data'][3]),
        TextLineDataset(data_path=data_configs['train_data'][4],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][5],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][6]),
        QEHterDataset(data_path=data_configs['train_data'][7]),
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3]),
        TextLineDataset(data_path=data_configs['valid_data'][4],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][5],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][6]),
        QEHterDataset(data_path=data_configs['valid_data'][7]),
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=training_configs["batch_size"],
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=training_configs['buffer_size'],
                                     batching_func=training_configs['batching_key'],
                                     world_size=world_size,
                                     rank=rank,
                                     shuffle=True)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, buffer_size=100000, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 0. Initial

    lrate = optimizer_configs['learning_rate']
    model_collections = Collections()

    best_model_prefix = os.path.join(flags.saveto, flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)

    # 1. Build Model & Criterion
    INFO('Building model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words,
                                padding_idx=vocab_src.pad, **discriminator_configs)

    generator_mlm = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words,
                                padding_idx=vocab_src.pad, **generator_configs)

    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)
    # 2. Move to GPU
    if Constants.USE_GPU:
        discriminator = discriminator.cuda()
        generator_mlm = generator_mlm.cuda()
        # generator_lm_forward = generator_lm_forward.cuda()
        # generator_lm_backward = generator_lm_backward.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()

    # 3. Load pretrained model if needed
    load_pretrained_model(generator_mlm, flags.pretrain_generator_mlm_path, exclude_prefix=None,
                          device=Constants.CURRENT_DEVICE)

    # 加载MLM的embedding层到判别器中
    INFO('load embedding')
    times_bigger = int(discriminator_configs['d_model'] / generator_configs['d_model'])
    for key in discriminator.encoder.embeddings.embeddings.state_dict().keys():
        discriminator.encoder.embeddings.embeddings.state_dict()[key].copy_(
            generator_mlm.encoder.embeddings.embeddings.state_dict()[key].repeat(1, times_bigger))
        discriminator.decoder.embeddings.embeddings.state_dict()[key].copy_(
            generator_mlm.decoder.embeddings.embeddings.state_dict()[key].repeat(1, times_bigger))

    INFO('Done. Elapsed time {0}'.format(timer.toc()))
    load_pretrained_model(discriminator, flags.pretrain_discriminator_path, exclude_prefix=None,device=Constants.CURRENT_DEVICE)
    # 4. Build optimizer
    INFO('Building Optimizer...')

    if not flags.multi_gpu:
        optim = Optimizer(name=optimizer_configs['optimizer'],
                          model=discriminator,
                          lr=lrate,
                          grad_clip=optimizer_configs['grad_clip'],
                          optim_args=optimizer_configs['optimizer_params'],
                          update_cycle=training_configs['update_cycle']
                          )
    else:
        optim = dist.DistributedOptimizer(name=optimizer_configs['optimizer'],
                                          model=discriminator,
                                          lr=lrate,
                                          grad_clip=optimizer_configs['grad_clip'],
                                          optim_args=optimizer_configs['optimizer_params'],
                                          device_id=local_rank
                                          )

    # 加载embedding后是否要更新embedding参数。似乎embedding矩阵没有bias。
    # discriminator.encoder.embeddings.embeddings.weight.requires_grad = False
    # discriminator.decoder.embeddings.embeddings.weight.requires_grad = False

    # 5. Build scheduler for optimizer if needed
    scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])

    # 6. build moving average
    if training_configs['moving_average_method'] is not None:
        ma = MovingAverage(moving_average_method=training_configs['moving_average_method'],
                           named_params=discriminator.named_parameters(),
                           alpha=training_configs['moving_average_alpha'])
    else:
        ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # broadcast parameters and optimizer states
    if world_size > 1:
        INFO("Broadcasting model parameters...")
        dist.broadcast_parameters(params=discriminator.state_dict())
        dist.broadcast_parameters(params=generator_mlm.state_dict())
        # dist.broadcast_parameters(params=generator_lm_forward.state_dict())
        # dist.broadcast_parameters(params=generator_lm_backward.state_dict())
        INFO("Broadcasting optimizer states...")
        dist.broadcast_optimizer_state(optimizer=optim.optim)
        INFO('Done.')

    # ================================================================================== #
    # Prepare training

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [1])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]
    is_early_stop = model_collections.get_collection("is_early_stop", [False, ])[-1]

    train_loss_sen_forward_meter = AverageMeter()
    train_loss_sen_backward_meter = AverageMeter()
    train_loss_word_meter = AverageMeter()
    train_loss_sen_forward_hter_meter = AverageMeter()
    train_loss_sen_backward_hter_meter = AverageMeter()
    train_loss_word_hter_meter = AverageMeter()
    sent_per_sec_meter = TimeMeter()
    tok_per_sec_meter = TimeMeter()

    update_cycle = training_configs['update_cycle']
    grad_denom = 0
    train_loss_sen_forward = 0.0
    train_loss_sen_backward = 0.0
    train_loss_word = 0.0
    train_loss_sen_forward_hter = 0.0
    train_loss_sen_backward_hter = 0.0
    train_loss_word_hter = 0.0
    cum_n_words = 0

    if rank == 0:
        summary_writer = SummaryWriter(log_dir=flags.log_path)
    else:
        summary_writer = None

    sent_per_sec_meter.start()
    tok_per_sec_meter.start()

    INFO('Begin training...')

    while True:

        if summary_writer is not None:
            summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()

        if rank == 0:
            training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                         total=len(training_iterator),
                                         unit="sents"
                                         )
        else:
            training_progress_bar = None


        for batch in training_iter:

            parallel_seqs_x, parallel_seqs_y, parallel_xy_label, parallel_xy_hter, \
                pseudo_seqs_x, pseudo_seqs_y, pseudo_xy_label, pseudo_xy_hter = batch


            batch_size = len(parallel_seqs_x)
            cum_n_words += sum(len(s) for s in parallel_seqs_y)

            
            try:
                # 平行语料loss
                x, y, xy_label = prepare_data_qe(parallel_seqs_x, parallel_seqs_y, parallel_xy_label, cuda=Constants.USE_GPU)

                if Constants.USE_GPU:
                    parallel_xy_hter = torch.tensor(parallel_xy_hter).squeeze(-1).cuda()

                parallel_loss_word, parallel_loss_word_hter, parallel_hter_pre = compute_forward_discriminator(model=discriminator,
                                                                        critic=critic,
                                                                        critic_mse=critic_mse,
                                                                        seqs_x=x,
                                                                        seqs_y=y,
                                                                        lm_ids=xy_label,
                                                                        hter_label=parallel_xy_hter,
                                                                        return_hter_pre=True)
                        
                # 伪数据loss
                x, y, xy_label = prepare_data_qe(pseudo_seqs_x, pseudo_seqs_y, pseudo_xy_label, cuda=Constants.USE_GPU)

                if Constants.USE_GPU:
                    pseudo_xy_hter = torch.tensor(pseudo_xy_hter).squeeze(-1).cuda()

                pseudo_loss_word, pseudo_loss_word_hter, pseudo_hter_pre = compute_forward_discriminator(model=discriminator,
                                                                        critic=critic,
                                                                        critic_mse=critic_mse,
                                                                        seqs_x=x,
                                                                        seqs_y=y,
                                                                        lm_ids=xy_label,
                                                                        hter_label=pseudo_xy_hter,
                                                                        return_hter_pre=True)

                # cal CL loss
                hter_pre_diff = pseudo_hter_pre - parallel_hter_pre
                hter_label_diff = pseudo_xy_hter - parallel_xy_hter

                cl_loss_hter = critic_mse(hter_pre_diff, hter_label_diff) * flags.cl_aug
                torch.autograd.backward(cl_loss_hter) 

                loss_sen_forward = 0
                loss_sen_forward_hter = 0
                loss_sen_backward = 0
                loss_sen_backward_hter = 0
                #if np.isnan(loss_word) or np.isnan(loss_word_hter) or np.isnan(loss_sen_forward) or \
                #    np.isnan(loss_sen_forward_hter) or np.isnan(loss_sen_backward) or np.isnan(loss_sen_backward_hter):
                #    WARN("NaN detected!")
                update_cycle -= 1
                grad_denom += batch_size

                train_loss_sen_forward += loss_sen_forward
                train_loss_sen_backward += loss_sen_backward
                train_loss_word += (parallel_loss_word + pseudo_loss_word)

                train_loss_sen_forward_hter += loss_sen_forward_hter
                train_loss_sen_backward_hter += loss_sen_backward_hter
                train_loss_word_hter += (parallel_loss_word_hter + pseudo_loss_word_hter + cl_loss_hter.item()) 

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom_count += 1
                else:
                    raise e

            # When update_cycle becomes 0, it means end of one batch. Several things will be done:
            # - update parameters
            # - reset update_cycle and grad_denom, update uidx
            # - learning rate scheduling
            # - update moving average

            if update_cycle == 0:

                # 0. reduce variables
                if world_size > 1:
                    grad_denom = dist.all_reduce_py(grad_denom)
                    train_loss_sen_forward = dist.all_reduce_py(train_loss_sen_forward)
                    train_loss_sen_backward = dist.all_reduce_py(train_loss_sen_backward)
                    train_loss_word = dist.all_reduce_py(train_loss_word)
                    train_loss_sen_forward_hter = dist.all_reduce_py(train_loss_sen_forward_hter)
                    train_loss_sen_backward_hter = dist.all_reduce_py(train_loss_sen_backward_hter)
                    train_loss_word_hter = dist.all_reduce_py(train_loss_word_hter)
                    cum_n_words = dist.all_reduce_py(cum_n_words)

                # 1. update parameters
                optim.step(denom=grad_denom)
                optim.zero_grad()

                if training_progress_bar is not None:
                    training_progress_bar.update(grad_denom)
                    training_progress_bar.set_description(' - (Epc {}, Upd {}) '.format(eidx, uidx))

                # 2. learning rate scheduling
                if scheduler is not None and optimizer_configs["schedule_method"] != "loss":
                    scheduler.step(global_step=uidx)

                # 3. update moving average
                if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                    ma.step()

                # 4. update meters
                train_loss_sen_forward_meter.update(train_loss_sen_forward, grad_denom)
                train_loss_sen_backward_meter.update(train_loss_sen_backward, grad_denom)
                train_loss_word_meter.update(train_loss_word, grad_denom)
                train_loss_sen_forward_hter_meter.update(train_loss_sen_forward_hter, grad_denom)
                train_loss_sen_backward_hter_meter.update(train_loss_sen_backward_hter, grad_denom)
                train_loss_word_hter_meter.update(train_loss_word_hter, grad_denom)
                sent_per_sec_meter.update(grad_denom)
                tok_per_sec_meter.update(cum_n_words)

                # 5. reset accumulated variables, update uidx
                update_cycle = training_configs['update_cycle']
                grad_denom = 0
                uidx += 1
                cum_n_words = 0.0
                train_loss_sen_forward = 0.0
                train_loss_sen_backward = 0.0
                train_loss_word = 0.0

                train_loss_sen_forward_hter = 0.0
                train_loss_sen_backward_hter = 0.0
                train_loss_word_hter = 0.0

            else:
                continue
            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):

                lrate = list(optim.get_lrate())[0]

                if summary_writer is not None:
                    summary_writer.add_scalar("Speed(sents/sec)", scalar_value=sent_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("Speed(words/sec)", scalar_value=tok_per_sec_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_forward", scalar_value=train_loss_sen_forward_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_backward", scalar_value=train_loss_sen_backward_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_word", scalar_value=train_loss_word_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_forward_hter",
                                              scalar_value=train_loss_sen_forward_hter_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_sen_backward_hter",
                                              scalar_value=train_loss_sen_backward_hter_meter.ave, global_step=uidx)
                    summary_writer.add_scalar("train_loss_word_hter", scalar_value=train_loss_word_hter_meter.ave,
                                              global_step=uidx)
                    summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                    summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)

                # Reset Meters
                sent_per_sec_meter.reset()
                tok_per_sec_meter.reset()
                train_loss_sen_forward_meter.reset()
                train_loss_sen_backward_meter.reset()
                train_loss_word_meter.reset()
                train_loss_sen_forward_hter_meter.reset()
                train_loss_sen_backward_hter_meter.reset()
                train_loss_word_hter_meter.reset()

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):
                with cache_parameters(discriminator):

                    if ma is not None:
                        discriminator.load_state_dict(ma.export_ma_params(), strict=False)

                    valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                        loss_evaluation_qe_cl(discriminator=discriminator,
                                           critic=critic,
                                           critic_mse=critic_mse,
                                           valid_iterator=valid_iterator)

                if scheduler is not None and optimizer_configs["schedule_method"] == "loss":
                    scheduler.step(metric=valid_loss)

                model_collections.add_to_collection("history_loss", valid_loss)

                min_valid_loss = np.array(model_collections.get_collection("history_loss")).min()

                if summary_writer is not None:
                    summary_writer.add_scalar("loss_word", valid_loss, global_step=uidx)
                    summary_writer.add_scalar('loss_hter', valid_loss_hter, global_step=uidx)
                    summary_writer.add_scalar('f1good', f1good, global_step=uidx)
                    summary_writer.add_scalar('f1bad', f1bad, global_step=uidx)
                    summary_writer.add_scalar('f1multi', f1multi, global_step=uidx)
                    summary_writer.add_scalar('pearsonr', p, global_step=uidx)

                if eidx >= 0:
                    if valid_loss <= min_valid_loss:
                        bad_count = 0
                        if is_early_stop is False and rank == 0:
                            INFO("save")
                            torch.save(discriminator.state_dict(), best_model_prefix + '.final')
                    else:
                        bad_count += 1
                        if bad_count >= training_configs['early_stop_patience']:
                            is_early_stop = True
                            WARN("Early Stop")

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.6f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.6f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
    
            if is_early_stop is True:
                break
            # ================================================================================== #
        if training_progress_bar is not None:
            training_progress_bar.close()

        if is_early_stop is True:
            break

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def finetune_qe(flags):
    """
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(flags.saveto,
                                     flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3])
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'],
                                     shuffle=True)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False,
                                  shuffle=True)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    # 加载之前训好的模型参数
    #params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    #discriminator.load_state_dict(params)
    # 如果有训好的，才加载
    load_pretrained_model(discriminator, flags.pretrain_path, exclude_prefix=None,
                          device="cpu")
    
    if flags.criterion == "focal_loss":
        critic = NMTCriterionFocalLoss(padding_idx=0, ignore_idx=0)
    elif flags.criterion == "nll_loss":
        critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    elif flags.criterion == "margin_loss":
        critic = NMTCriterionMarginLoss(padding_idx=0, ignore_idx=0)

    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=discriminator,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None

    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=flags.log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    save_id = 0

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        train_loss = 0
        train_loss_hter = 0

        for batch in training_iter:
            discriminator.eval()
            uidx += 1
            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y, xy_label, xy_hter = batch

            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)
            cum_samples += n_samples_t
            cum_words += n_words_t
            training_progress_bar.update(n_samples_t)

            x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)

            if Constants.USE_GPU:
                xy_hter = torch.tensor(xy_hter)
                xy_hter = xy_hter.cuda()

            loss, loss_hter = compute_forward_discriminator(model=discriminator,
                                                            critic=critic,
                                                            critic_mse=critic_mse,
                                                            seqs_x=x,
                                                            seqs_y=y,
                                                            lm_ids=xy_label,
                                                            hter_label=xy_hter)

            train_loss += loss
            train_loss_hter += loss_hter
            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} TrainLoss_hter: {2:.4f} ".format(
                    uidx, train_loss / training_configs['disp_freq'], train_loss_hter / training_configs['disp_freq']
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
                train_loss = 0
                train_loss_hter = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                    loss_evaluation_qe(discriminator=discriminator,
                                       critic=critic,
                                       critic_mse=critic_mse,
                                       valid_iterator=valid_iterator)

                # 保存ckpt
                torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                torch.save(discriminator.state_dict(), best_model_prefix + '.last_ckpt')

                model_collections.add_to_collection("history_losses", valid_loss)   # 用词级别loss保存
                # model_collections.add_to_collection("history_losses", valid_loss_hter)  # 用句子级loss保存
                min_loss = np.array(model_collections.get_collection("history_losses")).min()

                if valid_loss <= min_loss:   # 用词级别loss保存
                # if valid_loss_hter <= min_loss:  # 用句子级loss保存
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(discriminator.state_dict(), best_model_prefix + ".final")
                        #torch.save(discriminator.state_dict(), best_model_prefix + "." + str(save_id))
                        #INFO("save model id: {0}".format(save_id))
                        #save_id += 1
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.4f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.2f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
            if is_early_stop is True:
                break
        if is_early_stop is True:
            break
        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def finetune_qe_margin_loss(flags):
    """
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(flags.saveto,
                                     flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3])
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'],
                                     shuffle=True)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False,
                                  shuffle=True)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    # 加载之前训好的模型参数
    #params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    #discriminator.load_state_dict(params)
    # 如果有训好的，才加载
    load_pretrained_model(discriminator, flags.pretrain_path, exclude_prefix=None,
                          device="cpu")
    
    if flags.criterion == "focal_loss":
        critic = NMTCriterionFocalLoss(padding_idx=0, ignore_idx=0)
    elif flags.criterion == "nll_loss":
        critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    elif flags.criterion == "margin_loss":
        critic = NMTCriterionMarginLoss(padding_idx=0, ignore_idx=0)

    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=discriminator,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None

    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=flags.log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    save_id = 0

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        train_loss = 0
        train_loss_hter = 0

        for batch in training_iter:
            discriminator.eval()
            uidx += 1
            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y, xy_label, xy_hter = batch

            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)
            cum_samples += n_samples_t
            cum_words += n_words_t
            training_progress_bar.update(n_samples_t)

            x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)

            if Constants.USE_GPU:
                xy_hter = torch.tensor(xy_hter)
                xy_hter = xy_hter.cuda()

            loss, loss_hter = compute_forward_discriminator_margin(model=discriminator,
                                                            critic=critic,
                                                            critic_mse=critic_mse,
                                                            seqs_x=x,
                                                            seqs_y=y,
                                                            lm_ids=xy_label,
                                                            hter_label=xy_hter)

            train_loss += loss
            train_loss_hter += loss_hter
            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} TrainLoss_hter: {2:.4f} ".format(
                    uidx, train_loss / training_configs['disp_freq'], train_loss_hter / training_configs['disp_freq']
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
                train_loss = 0
                train_loss_hter = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                    loss_evaluation_qe_margin(discriminator=discriminator,
                                       critic=critic,
                                       critic_mse=critic_mse,
                                       valid_iterator=valid_iterator)

                # 保存ckpt
                torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                torch.save(discriminator.state_dict(), best_model_prefix + '.last_ckpt')

                model_collections.add_to_collection("history_losses", valid_loss)   # 用词级别loss保存
                # model_collections.add_to_collection("history_losses", valid_loss_hter)  # 用句子级loss保存
                min_loss = np.array(model_collections.get_collection("history_losses")).min()

                if valid_loss <= min_loss:   # 用词级别loss保存
                # if valid_loss_hter <= min_loss:  # 用句子级loss保存
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(discriminator.state_dict(), best_model_prefix + ".final")
                        #torch.save(discriminator.state_dict(), best_model_prefix + "." + str(save_id))
                        #INFO("save model id: {0}".format(save_id))
                        #save_id += 1
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.4f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.2f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
            if is_early_stop is True:
                break
        if is_early_stop is True:
            break
        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def finetune_qe_one_class(flags):
    # 词级别从多分类变成单分类，最后的线性层从3维改成1维
    """
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(flags.saveto,
                                     flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3])
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'],
                                     shuffle=True)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False,
                                  shuffle=True)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    # 加载之前训好的模型参数
    #params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    #discriminator.load_state_dict(params)
    # 如果有训好的，才加载
    load_pretrained_model(discriminator, flags.pretrain_path, exclude_prefix=None,
                          device="cpu")
    
    if flags.criterion == "focal_loss":
        critic = NMTCriterionFocalLoss(padding_idx=0, ignore_idx=0)
    elif flags.criterion == "nll_loss":
        critic = NMTCriterion(padding_idx=0, ignore_idx=0, one_class=True)
    elif flags.criterion == "hinge_loss":
        critic = NMTCriterionHingeLoss(padding_idx=0, ignore_idx=0, one_class=True)

    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=discriminator,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None

    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=flags.log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    no_sigmoid = False
    if flags.criterion == "hinge_loss":
        no_sigmoid = True


    INFO('Begin training...')

    save_id = 0

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        train_loss = 0
        train_loss_hter = 0

        for batch in training_iter:
            discriminator.eval()
            uidx += 1
            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y, xy_label, xy_hter = batch

            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)
            cum_samples += n_samples_t
            cum_words += n_words_t
            training_progress_bar.update(n_samples_t)

            x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)

            if Constants.USE_GPU:
                xy_hter = torch.tensor(xy_hter)
                xy_hter = xy_hter.cuda()

            loss, loss_hter = compute_forward_discriminator_one_class(model=discriminator,
                                                            critic=critic,
                                                            critic_mse=critic_mse,
                                                            seqs_x=x,
                                                            seqs_y=y,
                                                            lm_ids=xy_label,
                                                            hter_label=xy_hter,
                                                            no_sigmoid=no_sigmoid)

            train_loss += loss
            train_loss_hter += loss_hter
            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} TrainLoss_hter: {2:.4f} ".format(
                    uidx, train_loss / training_configs['disp_freq'], train_loss_hter / training_configs['disp_freq']
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
                train_loss = 0
                train_loss_hter = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                    loss_evaluation_qe_one_class(discriminator=discriminator,
                                       critic=critic,
                                       critic_mse=critic_mse,
                                       valid_iterator=valid_iterator,
                                       no_sigmoid=no_sigmoid)

                # 保存ckpt
                torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                torch.save(discriminator.state_dict(), best_model_prefix + '.last_ckpt')

                model_collections.add_to_collection("history_losses", valid_loss)   # 用词级别loss保存
                # model_collections.add_to_collection("history_losses", valid_loss_hter)  # 用句子级loss保存
                min_loss = np.array(model_collections.get_collection("history_losses")).min()

                if valid_loss <= min_loss:   # 用词级别loss保存
                # if valid_loss_hter <= min_loss:  # 用句子级loss保存
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(discriminator.state_dict(), best_model_prefix + ".final")
                        #torch.save(discriminator.state_dict(), best_model_prefix + "." + str(save_id))
                        #INFO("save model id: {0}".format(save_id))
                        #save_id += 1
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.4f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.2f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
            if is_early_stop is True:
                break
        if is_early_stop is True:
            break
        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def finetune_qe_token_mask(flags):
    # 训练时只关注没被mask的token来训练，别的不管
    """
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(flags.saveto,
                                     flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    if flags.add_feature:
        with open(flags.align_bpe_path_mt2src_train, 'r', encoding='utf-8') as falign_m2s_train, \
            open(flags.align_bpe_path_mt2src_dev, 'r', encoding='utf-8') as falign_m2s_dev:
            align_bpe_dict_mt2src_train = json.load(falign_m2s_train)
            align_bpe_dict_mt2src_dev = json.load(falign_m2s_dev)

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3]),
        QEHterDataset(data_path=data_configs['train_data'][4]),
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )
    """
    normal_train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['normal_train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['normal_train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['normal_train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['normal_train_data'][3]),
    )
    """

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'],
                                     shuffle=False)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False,
                                  shuffle=False)
    """
    normal_train_iterator = DataIterator(dataset=normal_train_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False,
                                  shuffle=True)
    """
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    # 加载之前训好的模型参数
    #params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    #discriminator.load_state_dict(params)
    # 如果有训好的，才加载
    load_pretrained_model(discriminator, flags.pretrain_path, exclude_prefix=None,
                          device="cpu")
    
    margin = False
    if flags.criterion == "focal_loss":
        critic = NMTCriterionFocalLoss(padding_idx=0, ignore_idx=0)
    elif flags.criterion == "nll_loss":
        critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    elif flags.criterion == "margin_loss":
        critic = NMTCriterionMarginLoss(padding_idx=0, ignore_idx=0)
        margin = True

    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=discriminator,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None


    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=flags.log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    loss_dict = {
        'normal_train_eval':{'loss_word':[], 'loss_hter':[], 'f1_ok':[], 'f1_bad':[], 'f1_multi':[], 'pearsonr':[]},
        'val_eval':  {'loss_word':[], 'loss_hter':[], 'f1_ok':[], 'f1_bad':[], 'f1_multi':[], 'pearsonr':[]},
    }

    save_id = 0

    while True:
        
        # =========================================== 训练前测试 ================================================ #
        """
        valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
            loss_evaluation_qe(discriminator=discriminator,
                                critic=critic,
                                critic_mse=critic_mse,
                                valid_iterator=valid_iterator)

        normal_train_loss, normal_train_loss_hter, normal_train_f1good, normal_train_f1bad, normal_train_f1multi, normal_train_p = \
            loss_evaluation_qe(discriminator=discriminator,
                                critic=critic,
                                critic_mse=critic_mse,
                                valid_iterator=normal_train_iterator)

        # 保存 loss记录
        loss_dict['val_eval']['loss_word'].append(valid_loss)
        loss_dict['val_eval']['loss_hter'].append(valid_loss_hter)
        loss_dict['val_eval']['f1_ok'].append(f1good)
        loss_dict['val_eval']['f1_bad'].append(f1bad)
        loss_dict['val_eval']['f1_multi'].append(f1multi)
        loss_dict['val_eval']['pearsonr'].append(p)

        loss_dict['normal_train_eval']['loss_word'].append(normal_train_loss)
        loss_dict['normal_train_eval']['loss_hter'].append(normal_train_loss_hter)
        loss_dict['normal_train_eval']['f1_ok'].append(normal_train_f1good)
        loss_dict['normal_train_eval']['f1_bad'].append(normal_train_f1bad)
        loss_dict['normal_train_eval']['f1_multi'].append(normal_train_f1multi)
        loss_dict['normal_train_eval']['pearsonr'].append(normal_train_p)
        """
        # =========================================== 训练前测试结束 ================================================ #

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        train_loss = 0
        train_loss_hter = 0

        id = 0
        for batch in training_iter:
            discriminator.eval()
            uidx += 1
            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y, xy_label, xy_hter, train_idx = batch

            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)
            cum_samples += n_samples_t
            cum_words += n_words_t
            training_progress_bar.update(n_samples_t)

            x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)

            if Constants.USE_GPU:
                xy_hter = torch.tensor(xy_hter).cuda()
                train_idx = torch.tensor(train_idx).cuda().int().squeeze(0).squeeze(0)

            token_mask = torch.zeros_like(xy_label)[:, 2:-1]
            for idx in train_idx:
                token_mask[0, idx] = 1

            if flags.add_feature:
                loss, loss_hter = compute_forward_discriminator_token_mask_add_feature(model=discriminator,
                                                            critic=critic,
                                                            critic_mse=critic_mse,
                                                            seqs_x=x,
                                                            seqs_y=y,
                                                            lm_ids=xy_label,
                                                            hter_label=xy_hter,
                                                            token_mask=token_mask,
                                                            margin=margin,
                                                            align_line=align_bpe_dict_mt2src_train[str(id)],
                                                            )
            else:
                loss, loss_hter = compute_forward_discriminator_token_mask(model=discriminator,
                                                            critic=critic,
                                                            critic_mse=critic_mse,
                                                            seqs_x=x,
                                                            seqs_y=y,
                                                            lm_ids=xy_label,
                                                            hter_label=xy_hter,
                                                            token_mask=token_mask,
                                                            margin=margin,
                                                            )

            train_loss += loss
            train_loss_hter += loss_hter
            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} TrainLoss_hter: {2:.4f} ".format(
                    uidx, train_loss / training_configs['disp_freq'], train_loss_hter / training_configs['disp_freq']
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
                train_loss = 0
                train_loss_hter = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                if flags.add_feature:
                    valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                        loss_evaluation_qe_add_feature(discriminator=discriminator,
                                                    critic=critic,
                                                    critic_mse=critic_mse,
                                                    valid_iterator=valid_iterator,
                                                    align_dict=align_bpe_dict_mt2src_dev,)
                else:
                    valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                        loss_evaluation_qe(discriminator=discriminator,
                                        critic=critic,
                                        critic_mse=critic_mse,
                                        valid_iterator=valid_iterator,)
                """
                normal_train_loss, normal_train_loss_hter, normal_train_f1good, normal_train_f1bad, normal_train_f1multi, normal_train_p = \
                    loss_evaluation_qe(discriminator=discriminator,
                                        critic=critic,
                                        critic_mse=critic_mse,
                                        valid_iterator=normal_train_iterator,
                                        requires_adapter=False)

                # 保存 loss记录
                loss_dict['val_eval']['loss_word'].append(valid_loss)
                loss_dict['val_eval']['loss_hter'].append(valid_loss_hter)
                loss_dict['val_eval']['f1_ok'].append(f1good)
                loss_dict['val_eval']['f1_bad'].append(f1bad)
                loss_dict['val_eval']['f1_multi'].append(f1multi)
                loss_dict['val_eval']['pearsonr'].append(p)

                loss_dict['normal_train_eval']['loss_word'].append(normal_train_loss)
                loss_dict['normal_train_eval']['loss_hter'].append(normal_train_loss_hter)
                loss_dict['normal_train_eval']['f1_ok'].append(normal_train_f1good)
                loss_dict['normal_train_eval']['f1_bad'].append(normal_train_f1bad)
                loss_dict['normal_train_eval']['f1_multi'].append(normal_train_f1multi)
                loss_dict['normal_train_eval']['pearsonr'].append(normal_train_p)
                """

                # 保存ckpt
                torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                #torch.save(discriminator.state_dict(), best_model_prefix + '.last_ckpt')
                #"""
                model_collections.add_to_collection("history_losses", valid_loss)   # 用词级别loss保存
                # model_collections.add_to_collection("history_losses", valid_loss_hter)  # 用句子级loss保存
                min_loss = np.array(model_collections.get_collection("history_losses")).min()

                if valid_loss <= min_loss:   # 用词级别loss保存
                # if valid_loss_hter <= min_loss:  # 用句子级loss保存
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(discriminator.state_dict(), best_model_prefix + ".final")
                        #torch.save(discriminator.state_dict(), best_model_prefix + "." + str(save_id))
                        #INFO("save model id: {0}".format(save_id))
                        #save_id += 1
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx >= training_configs["min_epochs"]:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.4f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.2f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
            if is_early_stop is True:
                break
            
            id += 1

        if is_early_stop is True:
            break
        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break
    """
    with open(flags.saveto + '/loss_dict.json', 'w', encoding='utf-8') as f:
        json.dump(loss_dict, f, indent=1)
    """


def finetune_qe_token_mask_every_sample(flags):
    # 训练时只关注没被mask的token来训练，别的不管
    # 一组对抗样本学会了，再进行下一组
    """
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(flags.saveto,
                                     flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    with open(flags.sample_seg_path, 'r', encoding='utf-8') as f:
        robust_sample_seg = json.load(f)

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3]),
        QEHterDataset(data_path=data_configs['train_data'][4]),
        QEHterDataset(data_path=data_configs['train_data'][5]),
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'],
                                     shuffle=False)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False,
                                  shuffle=True)
    
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    # 加载之前训好的模型参数
    #params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    #discriminator.load_state_dict(params)
    # 如果有训好的，才加载
    load_pretrained_model(discriminator, flags.pretrain_path, exclude_prefix=None,
                          device="cpu")
    
    margin = False
    if flags.criterion == "focal_loss":
        critic = NMTCriterionFocalLoss(padding_idx=0, ignore_idx=0)
    elif flags.criterion == "nll_loss":
        critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    elif flags.criterion == "margin_loss":
        critic = NMTCriterionMarginLoss(padding_idx=0, ignore_idx=0)
        margin = True

    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=discriminator,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None


    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=flags.log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    loss_dict = {
        'normal_train_eval':{'loss_word':[], 'loss_hter':[], 'f1_ok':[], 'f1_bad':[], 'f1_multi':[], 'pearsonr':[]},
        'val_eval':  {'loss_word':[], 'loss_hter':[], 'f1_ok':[], 'f1_bad':[], 'f1_multi':[], 'pearsonr':[]},
    }

    save_id = 0

    robust_sample_num = len(train_bitext_dataset)

    while(True):
        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                    total=robust_sample_num,
                                    unit="sents")
        train_loss = 0
        train_loss_hter = 0

        # 每个普通样本，就是每组对抗样本
        for sent_id in range(len(robust_sample_seg)):
            robust_sent_id_begin = robust_sample_seg[str(sent_id)]['begin']
            robust_sent_id_end = robust_sample_seg[str(sent_id)]['end']

            contrast_group_dataset = []
            for robust_sent_id in range(robust_sent_id_begin, robust_sent_id_end + 1):
                contrast_group_dataset.append(train_bitext_dataset[robust_sent_id])
            
            train_time = 0
            while(True):
                #print("=======")
                #print("sent_id %d" % sent_id)
                #print("train_time %d" % train_time)

                # 这组对抗样本中的每一个
                for batch in contrast_group_dataset:
                    uidx += 1
                    if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                        scheduler.step(global_step=uidx)
                    
                    seqs_x, seqs_y, xy_label, xy_hter, train_idx, _ = batch
                    n_samples_t = len(seqs_x)
                    n_words_t = sum(len(s) for s in seqs_y)
                    cum_samples += n_samples_t
                    cum_words += n_words_t
                    if train_time == 0: 
                        training_progress_bar.update(n_samples_t)

                    x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)

                    if Constants.USE_GPU:
                        xy_hter = torch.tensor(xy_hter).cuda()
                        train_idx = torch.tensor(train_idx).cuda().int().squeeze(0).squeeze(0)

                    token_mask = torch.zeros_like(xy_label)[:, 2:-1]
                    for idx in train_idx:
                        token_mask[0, idx] = 1

                    loss, loss_hter = compute_forward_discriminator_token_mask(model=discriminator,
                                                                    critic=critic,
                                                                    critic_mse=critic_mse,
                                                                    seqs_x=x,
                                                                    seqs_y=y,
                                                                    lm_ids=xy_label,
                                                                    hter_label=xy_hter,
                                                                    token_mask=token_mask,
                                                                    margin=margin)

                    train_loss += loss
                    train_loss_hter += loss_hter

                    # --------------------------------------------
                    if (uidx) % training_configs['update_cycle'] == 0:
                        optim.step()
                        optim.zero_grad()

                    if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                        ma.step()

                    # ================================================================================== #
                    # Display some information
                    if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                        # words per second and sents per second
                        words_per_sec = cum_words / (timer.toc(return_seconds=True))
                        sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                        lrate = list(optim.get_lrate())[0]

                        summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                        summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                        summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                        summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                        INFO("{0} TrainLoss: {1:.4f} TrainLoss_hter: {2:.4f} ".format(
                            uidx, train_loss / training_configs['disp_freq'], train_loss_hter / training_configs['disp_freq']
                        ))

                        # Reset timer
                        timer.tic()
                        cum_words = 0
                        cum_samples = 0
                        train_loss = 0
                        train_loss_hter = 0

                    # ================================================================================== #
                    # Saving checkpoints
                    if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                        model_collections.add_to_collection("uidx", uidx)
                        model_collections.add_to_collection("eidx", eidx)
                        model_collections.add_to_collection("bad_count", bad_count)

                    # ================================================================================== #
                    # Loss Validation & Learning rate annealing
                    if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                            debug=flags.debug):

                        valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                            loss_evaluation_qe(discriminator=discriminator,
                                            critic=critic,
                                            critic_mse=critic_mse,
                                            valid_iterator=valid_iterator,)

                        # 保存ckpt
                        torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                        #"""
                        model_collections.add_to_collection("history_losses", valid_loss)   # 用词级别loss保存
                        min_loss = np.array(model_collections.get_collection("history_losses")).min()

                        if valid_loss <= min_loss:   # 用词级别loss保存
                            bad_count = 0
                            if is_early_stop is False:
                                # 1. save the best model
                                torch.save(discriminator.state_dict(), best_model_prefix + ".final")

                        else:
                            bad_count += 1
                            # At least one epoch should be traversed
                            if bad_count >= training_configs['early_stop_patience'] and eidx >= training_configs["min_epochs"]:
                                is_early_stop = True
                                WARN("Early Stop!")

                        summary_writer.add_scalar("bad_count", bad_count, uidx)

                        INFO("{0} Loss: {1:.2f} Loss_hter: {2:.4f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                            "F1_multi: {5:.4f} pearsonr:{6:.2f} patience: {7}".
                            format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
                    if is_early_stop is True:
                        break
                    
                    #print("here")

                if is_early_stop is True:
                    break

                # 检验一组样本的成果，通过就break
                # TODO
                acc = cal_contrast_group(model=discriminator, 
                                        dataset=contrast_group_dataset,
                                        critic=critic,
                                        critic_mse=critic_mse,)
                
                #print(acc)
                if acc >= 0.8:
                    break
                train_time += 1

            if is_early_stop is True:
                break
        if is_early_stop is True:
            break
        training_progress_bar.close()
        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break

     

def finetune_qe_token_mask_align_attn_v1(flags):
    # 训练时只关注没被mask的token来训练，别的不管
    # 隐式+显式 = 显式，既有隐式对比数据增强，又有显式融入词对齐信息到attn矩阵
    """
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(flags.saveto,
                                     flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    with open(flags.align_bpe_path_mt2src_train, 'r', encoding='utf-8') as falign_m2s_train, \
        open(flags.align_bpe_path_mt2src_dev, 'r', encoding='utf-8') as falign_m2s_dev:
        align_bpe_dict_mt2src_train = json.load(falign_m2s_train)
        align_bpe_dict_mt2src_dev = json.load(falign_m2s_dev)

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3]),
        QEHterDataset(data_path=data_configs['train_data'][4]),
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    normal_train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['normal_train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['normal_train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['normal_train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['normal_train_data'][3]),
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'],
                                     shuffle=False)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False,
                                  shuffle=False)

    normal_train_iterator = DataIterator(dataset=normal_train_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False,
                                  shuffle=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    # 加载之前训好的模型参数
    #params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    #discriminator.load_state_dict(params)
    # 如果有训好的，才加载
    load_pretrained_model(discriminator, flags.pretrain_path, exclude_prefix=None,
                          device="cpu")
    
    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    #print("==================")
    # TODO 尝试冻结模型参数
    #print(discriminator)
    #assert 1==2
    """
    for n, param in discriminator.named_parameters():
        #print(n)
        param.requires_grad = False

    for n, param in discriminator.named_parameters():
        if "adapter" in n: param.requires_grad = True
    """
    #assert 1==2
    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=discriminator,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None


    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=flags.log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    loss_dict = {
        'normal_train_eval':{'loss_word':[], 'loss_hter':[], 'f1_ok':[], 'f1_bad':[], 'f1_multi':[], 'pearsonr':[]},
        'val_eval':  {'loss_word':[], 'loss_hter':[], 'f1_ok':[], 'f1_bad':[], 'f1_multi':[], 'pearsonr':[]},
    }

    save_id = 0

    while True:

        # =========================================== 训练前测试 ================================================ #
        #"""
        valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
            loss_evaluation_qe_align(discriminator=discriminator,
                                       critic=critic,
                                       critic_mse=critic_mse,
                                       valid_iterator=valid_iterator,
                                       align_dict=align_bpe_dict_mt2src_dev)

        normal_train_loss, normal_train_loss_hter, normal_train_f1good, normal_train_f1bad, normal_train_f1multi, normal_train_p = \
            loss_evaluation_qe(discriminator=discriminator,
                                critic=critic,
                                critic_mse=critic_mse,
                                valid_iterator=normal_train_iterator)

        # 保存 loss记录
        loss_dict['val_eval']['loss_word'].append(valid_loss)
        loss_dict['val_eval']['loss_hter'].append(valid_loss_hter)
        loss_dict['val_eval']['f1_ok'].append(f1good)
        loss_dict['val_eval']['f1_bad'].append(f1bad)
        loss_dict['val_eval']['f1_multi'].append(f1multi)
        loss_dict['val_eval']['pearsonr'].append(p)

        loss_dict['normal_train_eval']['loss_word'].append(normal_train_loss)
        loss_dict['normal_train_eval']['loss_hter'].append(normal_train_loss_hter)
        loss_dict['normal_train_eval']['f1_ok'].append(normal_train_f1good)
        loss_dict['normal_train_eval']['f1_bad'].append(normal_train_f1bad)
        loss_dict['normal_train_eval']['f1_multi'].append(normal_train_f1multi)
        loss_dict['normal_train_eval']['pearsonr'].append(normal_train_p)
        #"""
        # =========================================== 训练前测试结束 ================================================ #

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        train_loss = 0
        train_loss_hter = 0

        id = 0
        for batch in training_iter:
            discriminator.eval()
            uidx += 1
            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y, xy_label, xy_hter, train_idx = batch

            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)
            cum_samples += n_samples_t
            cum_words += n_words_t
            training_progress_bar.update(n_samples_t)

            x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)

            if Constants.USE_GPU:
                xy_hter = torch.tensor(xy_hter).cuda()
                train_idx = torch.tensor(train_idx).cuda().int().squeeze(0).squeeze(0)

            token_mask = torch.zeros_like(xy_label)[:, 2:-1]
            for idx in train_idx:
                token_mask[0, idx] = 1
            
            loss, loss_hter = compute_forward_discriminator_token_mask_align_v1(model=discriminator,
                                                            critic=critic,
                                                            critic_mse=critic_mse,
                                                            seqs_x=x,
                                                            seqs_y=y,
                                                            lm_ids=xy_label,
                                                            hter_label=xy_hter,
                                                            token_mask=token_mask,
                                                            align_line=align_bpe_dict_mt2src_train[str(id)])

            train_loss += loss
            train_loss_hter += loss_hter
            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} TrainLoss_hter: {2:.4f} ".format(
                    uidx, train_loss / training_configs['disp_freq'], train_loss_hter / training_configs['disp_freq']
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
                train_loss = 0
                train_loss_hter = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                    loss_evaluation_qe_align(discriminator=discriminator,
                                       critic=critic,
                                       critic_mse=critic_mse,
                                       valid_iterator=valid_iterator,
                                       align_dict=align_bpe_dict_mt2src_dev)
                #"""
                normal_train_loss, normal_train_loss_hter, normal_train_f1good, normal_train_f1bad, normal_train_f1multi, normal_train_p = \
                    loss_evaluation_qe(discriminator=discriminator,
                                        critic=critic,
                                        critic_mse=critic_mse,
                                        valid_iterator=normal_train_iterator,
                                        requires_adapter=False)

                # 保存 loss记录
                loss_dict['val_eval']['loss_word'].append(valid_loss)
                loss_dict['val_eval']['loss_hter'].append(valid_loss_hter)
                loss_dict['val_eval']['f1_ok'].append(f1good)
                loss_dict['val_eval']['f1_bad'].append(f1bad)
                loss_dict['val_eval']['f1_multi'].append(f1multi)
                loss_dict['val_eval']['pearsonr'].append(p)

                loss_dict['normal_train_eval']['loss_word'].append(normal_train_loss)
                loss_dict['normal_train_eval']['loss_hter'].append(normal_train_loss_hter)
                loss_dict['normal_train_eval']['f1_ok'].append(normal_train_f1good)
                loss_dict['normal_train_eval']['f1_bad'].append(normal_train_f1bad)
                loss_dict['normal_train_eval']['f1_multi'].append(normal_train_f1multi)
                loss_dict['normal_train_eval']['pearsonr'].append(normal_train_p)
                

                # 保存ckpt
                torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                torch.save(discriminator.state_dict(), best_model_prefix + '.last_ckpt')
                #"""
                model_collections.add_to_collection("history_losses", valid_loss)   # 用词级别loss保存
                # model_collections.add_to_collection("history_losses", valid_loss_hter)  # 用句子级loss保存
                min_loss = np.array(model_collections.get_collection("history_losses")).min()

                if valid_loss <= min_loss:   # 用词级别loss保存
                # if valid_loss_hter <= min_loss:  # 用句子级loss保存
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(discriminator.state_dict(), best_model_prefix + ".final")
                        #torch.save(discriminator.state_dict(), best_model_prefix + "." + str(save_id))
                        #INFO("save model id: {0}".format(save_id))
                        #save_id += 1
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.4f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.2f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
            if is_early_stop is True:
                break

            id += 1

        if is_early_stop is True:
            break
        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break

    with open(flags.saveto + '/loss_dict.json', 'w', encoding='utf-8') as f:
        json.dump(loss_dict, f, indent=1)


def finetune_qe_align_attn_v1(flags):
    """
    给attn矩阵融入外部词对齐信息
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(flags.saveto,
                                     flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    with open(flags.align_bpe_path_mt2src_train, 'r', encoding='utf-8') as falign_m2s_train, \
        open(flags.align_bpe_path_mt2src_dev, 'r', encoding='utf-8') as falign_m2s_dev:
        align_bpe_dict_mt2src_train = json.load(falign_m2s_train)
        align_bpe_dict_mt2src_dev = json.load(falign_m2s_dev)

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3])
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'],
                                     shuffle=False)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False,
                                  shuffle=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    # 加载之前训好的模型参数
    #params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    #discriminator.load_state_dict(params)
    # 如果有训好的，才加载
    load_pretrained_model(discriminator, flags.pretrain_path, exclude_prefix=None,
                          device="cpu")
    
    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=discriminator,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None

    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=flags.log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    save_id = 0

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        train_loss = 0
        train_loss_hter = 0

        id = 0
        for batch in training_iter:
            # print(id)
            discriminator.eval()
            uidx += 1
            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y, xy_label, xy_hter = batch
            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)
            cum_samples += n_samples_t
            cum_words += n_words_t
            training_progress_bar.update(n_samples_t)

            x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)

            if Constants.USE_GPU:
                xy_hter = torch.tensor(xy_hter)
                xy_hter = xy_hter.cuda()

            loss, loss_hter = compute_forward_discriminator_align_v1(model=discriminator,
                                                                    critic=critic,
                                                                    critic_mse=critic_mse,
                                                                    seqs_x=x,
                                                                    seqs_y=y,
                                                                    lm_ids=xy_label,
                                                                    hter_label=xy_hter,
                                                                    align_line=align_bpe_dict_mt2src_train[str(id)],
                                                                    align_ratio=flags.align_ratio)

            train_loss += loss
            train_loss_hter += loss_hter
            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} TrainLoss_hter: {2:.4f} ".format(
                    uidx, train_loss / training_configs['disp_freq'], train_loss_hter / training_configs['disp_freq']
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
                train_loss = 0
                train_loss_hter = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                    loss_evaluation_qe_align_v1(discriminator=discriminator,
                                            critic=critic,
                                            critic_mse=critic_mse,
                                            valid_iterator=valid_iterator,
                                            align_dict=align_bpe_dict_mt2src_dev,
                                            align_ratio=flags.align_ratio)

                # 保存ckpt
                torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                torch.save(discriminator.state_dict(), best_model_prefix + '.last_ckpt')

                model_collections.add_to_collection("history_losses", valid_loss)   # 用词级别loss保存
                # model_collections.add_to_collection("history_losses", valid_loss_hter)  # 用句子级loss保存
                min_loss = np.array(model_collections.get_collection("history_losses")).min()

                if valid_loss <= min_loss:   # 用词级别loss保存
                # if valid_loss_hter <= min_loss:  # 用句子级loss保存
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(discriminator.state_dict(), best_model_prefix + ".final")
                        #torch.save(discriminator.state_dict(), best_model_prefix + "." + str(save_id))
                        #INFO("save model id: {0}".format(save_id))
                        #save_id += 1
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.4f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.2f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
            if is_early_stop is True:
                break
            
            id += 1

        if is_early_stop is True:
            break
        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def finetune_qe_align_attn_v2(flags):
    """
    训练attention矩阵尽可能接近词对齐信息
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(flags.saveto,
                                     flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    with open(flags.align_bpe_path_mt2src_train, 'r', encoding='utf-8') as falign_m2s_train:
        align_bpe_dict_mt2src_train = json.load(falign_m2s_train)

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3])
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'],
                                     shuffle=False)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False,
                                  shuffle=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    # 加载之前训好的模型参数
    #params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    #discriminator.load_state_dict(params)
    # 如果有训好的，才加载
    load_pretrained_model(discriminator, flags.pretrain_path, exclude_prefix=None,
                          device="cpu")
    
    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=discriminator,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None

    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=flags.log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    save_id = 0

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        train_loss = 0
        train_loss_hter = 0

        id = 0
        for batch in training_iter:
            discriminator.eval()
            uidx += 1
            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y, xy_label, xy_hter = batch

            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)
            cum_samples += n_samples_t
            cum_words += n_words_t
            training_progress_bar.update(n_samples_t)

            x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)

            if Constants.USE_GPU:
                xy_hter = torch.tensor(xy_hter)
                xy_hter = xy_hter.cuda()

            loss, loss_hter, attn_align_loss = compute_forward_discriminator_align_v2(model=discriminator,
                                                                    critic=critic,
                                                                    critic_mse=critic_mse,
                                                                    seqs_x=x,
                                                                    seqs_y=y,
                                                                    lm_ids=xy_label,
                                                                    hter_label=xy_hter,
                                                                    align_line=align_bpe_dict_mt2src_train[str(id)],
                                                                    attn_align_lambda=flags.attn_align_lambda)

            train_loss += loss
            train_loss_hter += loss_hter
            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} TrainLoss_hter: {2:.4f} TrainLoss_Attn_align: {3:.4f}".format(
                    uidx, train_loss / training_configs['disp_freq'], train_loss_hter / training_configs['disp_freq'], attn_align_loss / training_configs['disp_freq']
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
                train_loss = 0
                train_loss_hter = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                    loss_evaluation_qe(discriminator=discriminator,
                                       critic=critic,
                                       critic_mse=critic_mse,
                                       valid_iterator=valid_iterator)

                # 保存ckpt
                torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                #torch.save(discriminator.state_dict(), best_model_prefix + '.last_ckpt')

                model_collections.add_to_collection("history_losses", valid_loss)   # 用词级别loss保存
                # model_collections.add_to_collection("history_losses", valid_loss_hter)  # 用句子级loss保存
                min_loss = np.array(model_collections.get_collection("history_losses")).min()

                if valid_loss <= min_loss:   # 用词级别loss保存
                # if valid_loss_hter <= min_loss:  # 用句子级loss保存
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(discriminator.state_dict(), best_model_prefix + ".final")
                        #torch.save(discriminator.state_dict(), best_model_prefix + "." + str(save_id))
                        #INFO("save model id: {0}".format(save_id))
                        #save_id += 1
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.4f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.2f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
            
            id += 1

            if is_early_stop is True:
                break
        if is_early_stop is True:
            break
        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def finetune_qe_align_attn_v3(flags):
    """
    给attn矩阵融入外部词对齐信息，也是显式地加进来，但是不是通过修改attn的形式，
    而是直接把（如果有）对齐词汇的话，就把对齐词汇的词向量和 原本的词向量加在一起
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(flags.saveto,
                                     flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    with open(flags.align_bpe_path_mt2src_train, 'r', encoding='utf-8') as falign_m2s_train, \
        open(flags.align_bpe_path_mt2src_dev, 'r', encoding='utf-8') as falign_m2s_dev:
        align_bpe_dict_mt2src_train = json.load(falign_m2s_train)
        align_bpe_dict_mt2src_dev = json.load(falign_m2s_dev)

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3])
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'],
                                     shuffle=False)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False,
                                  shuffle=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    # 加载之前训好的模型参数
    #params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    #discriminator.load_state_dict(params)
    # 如果有训好的，才加载
    load_pretrained_model(discriminator, flags.pretrain_path, exclude_prefix=None,
                          device="cpu")
    
    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=discriminator,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None

    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=flags.log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    save_id = 0

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        train_loss = 0
        train_loss_hter = 0

        id = 0
        for batch in training_iter:
            # print(id)
            discriminator.eval()
            uidx += 1
            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y, xy_label, xy_hter = batch
            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)
            cum_samples += n_samples_t
            cum_words += n_words_t
            training_progress_bar.update(n_samples_t)

            x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)

            if Constants.USE_GPU:
                xy_hter = torch.tensor(xy_hter)
                xy_hter = xy_hter.cuda()

            loss, loss_hter = compute_forward_discriminator_align_v3(model=discriminator,
                                                                    critic=critic,
                                                                    critic_mse=critic_mse,
                                                                    seqs_x=x,
                                                                    seqs_y=y,
                                                                    lm_ids=xy_label,
                                                                    hter_label=xy_hter,
                                                                    align_line=align_bpe_dict_mt2src_train[str(id)],
                                                                    attn_align_lambda=flags.attn_align_lambda,)

            train_loss += loss
            train_loss_hter += loss_hter
            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} TrainLoss_hter: {2:.4f} ".format(
                    uidx, train_loss / training_configs['disp_freq'], train_loss_hter / training_configs['disp_freq']
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
                train_loss = 0
                train_loss_hter = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                    loss_evaluation_qe_align_v3(discriminator=discriminator,
                                            critic=critic,
                                            critic_mse=critic_mse,
                                            valid_iterator=valid_iterator,
                                            align_dict=align_bpe_dict_mt2src_dev,
                                            attn_align_lambda=flags.attn_align_lambda,)

                # 保存ckpt
                torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                torch.save(discriminator.state_dict(), best_model_prefix + '.last_ckpt')

                model_collections.add_to_collection("history_losses", valid_loss)   # 用词级别loss保存
                # model_collections.add_to_collection("history_losses", valid_loss_hter)  # 用句子级loss保存
                min_loss = np.array(model_collections.get_collection("history_losses")).min()

                if valid_loss <= min_loss:   # 用词级别loss保存
                # if valid_loss_hter <= min_loss:  # 用句子级loss保存
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(discriminator.state_dict(), best_model_prefix + ".final")
                        #torch.save(discriminator.state_dict(), best_model_prefix + "." + str(save_id))
                        #INFO("save model id: {0}".format(save_id))
                        #save_id += 1
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.4f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.2f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
            if is_early_stop is True:
                break
            
            id += 1

        if is_early_stop is True:
            break
        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def finetune_qe_add_feature(flags):
    """
    加特征
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(flags.saveto,
                                     flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    with open(flags.align_bpe_path_mt2src_train, 'r', encoding='utf-8') as falign_m2s_train, \
        open(flags.align_bpe_path_mt2src_dev, 'r', encoding='utf-8') as falign_m2s_dev:
        align_bpe_dict_mt2src_train = json.load(falign_m2s_train)
        align_bpe_dict_mt2src_dev = json.load(falign_m2s_dev)

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3])
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'],
                                     shuffle=False)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False,
                                  shuffle=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    # 加载之前训好的模型参数
    #params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    #discriminator.load_state_dict(params)
    # 如果有训好的，才加载
    load_pretrained_model(discriminator, flags.pretrain_path, exclude_prefix=None,
                          device="cpu")
    
    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=discriminator,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None

    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=flags.log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    save_id = 0

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        train_loss = 0
        train_loss_hter = 0

        id = 0
        for batch in training_iter:
            # print(id)
            discriminator.eval()
            uidx += 1
            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y, xy_label, xy_hter = batch
            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)
            cum_samples += n_samples_t
            cum_words += n_words_t
            training_progress_bar.update(n_samples_t)

            x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)

            if Constants.USE_GPU:
                xy_hter = torch.tensor(xy_hter)
                xy_hter = xy_hter.cuda()

            loss, loss_hter = compute_forward_discriminator_add_feature(model=discriminator,
                                                                    critic=critic,
                                                                    critic_mse=critic_mse,
                                                                    seqs_x=x,
                                                                    seqs_y=y,
                                                                    lm_ids=xy_label,
                                                                    hter_label=xy_hter,
                                                                    align_line=align_bpe_dict_mt2src_train[str(id)],)

            train_loss += loss
            train_loss_hter += loss_hter
            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} TrainLoss_hter: {2:.4f} ".format(
                    uidx, train_loss / training_configs['disp_freq'], train_loss_hter / training_configs['disp_freq']
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
                train_loss = 0
                train_loss_hter = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                    loss_evaluation_qe_add_feature(discriminator=discriminator,
                                            critic=critic,
                                            critic_mse=critic_mse,
                                            valid_iterator=valid_iterator,
                                            align_dict=align_bpe_dict_mt2src_dev,)

                # 保存ckpt
                torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                torch.save(discriminator.state_dict(), best_model_prefix + '.last_ckpt')

                model_collections.add_to_collection("history_losses", valid_loss)   # 用词级别loss保存
                # model_collections.add_to_collection("history_losses", valid_loss_hter)  # 用句子级loss保存
                min_loss = np.array(model_collections.get_collection("history_losses")).min()

                if valid_loss <= min_loss:   # 用词级别loss保存
                # if valid_loss_hter <= min_loss:  # 用句子级loss保存
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(discriminator.state_dict(), best_model_prefix + ".final")
                        #torch.save(discriminator.state_dict(), best_model_prefix + "." + str(save_id))
                        #INFO("save model id: {0}".format(save_id))
                        #save_id += 1
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.4f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.2f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
            if is_early_stop is True:
                break
            
            id += 1

        if is_early_stop is True:
            break
        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def finetune_qe_align_attn_v4(flags):
    """
    参考【emnlp2020】Towards enhancing faithfulness for neural machine translation 的对齐训练目标，
    训练部分attention head尽可能接近词对齐信息，使用交叉熵损失
    代码相当冗余……
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(flags.saveto,
                                     flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    with open(flags.align_bpe_path_mt2src_train, 'r', encoding='utf-8') as falign_m2s_train:
        align_bpe_dict_mt2src_train = json.load(falign_m2s_train)

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3])
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'],
                                     shuffle=False)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False,
                                  shuffle=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    # 加载之前训好的模型参数
    #params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    #discriminator.load_state_dict(params)
    # 如果有训好的，才加载
    load_pretrained_model(discriminator, flags.pretrain_path, exclude_prefix=None,
                          device="cpu")
    
    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=discriminator,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None

    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=flags.log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    save_id = 0

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        train_loss = 0
        train_loss_hter = 0

        id = 0
        for batch in training_iter:
            discriminator.eval()
            uidx += 1
            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y, xy_label, xy_hter = batch

            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)
            cum_samples += n_samples_t
            cum_words += n_words_t
            training_progress_bar.update(n_samples_t)

            x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)

            if Constants.USE_GPU:
                xy_hter = torch.tensor(xy_hter)
                xy_hter = xy_hter.cuda()

            loss, loss_hter, attn_align_loss = compute_forward_discriminator_align_v4(model=discriminator,
                                                                    critic=critic,
                                                                    critic_mse=critic_mse,
                                                                    seqs_x=x,
                                                                    seqs_y=y,
                                                                    lm_ids=xy_label,
                                                                    hter_label=xy_hter,
                                                                    align_line=align_bpe_dict_mt2src_train[str(id)],
                                                                    attn_align_lambda=flags.attn_align_lambda)

            train_loss += loss
            train_loss_hter += loss_hter
            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} TrainLoss_hter: {2:.4f} TrainLoss_Attn_align: {3:.4f}".format(
                    uidx, train_loss / training_configs['disp_freq'], train_loss_hter / training_configs['disp_freq'], attn_align_loss / training_configs['disp_freq']
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
                train_loss = 0
                train_loss_hter = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                    loss_evaluation_qe(discriminator=discriminator,
                                       critic=critic,
                                       critic_mse=critic_mse,
                                       valid_iterator=valid_iterator)

                # 保存ckpt
                torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                #torch.save(discriminator.state_dict(), best_model_prefix + '.last_ckpt')

                model_collections.add_to_collection("history_losses", valid_loss)   # 用词级别loss保存
                # model_collections.add_to_collection("history_losses", valid_loss_hter)  # 用句子级loss保存
                min_loss = np.array(model_collections.get_collection("history_losses")).min()

                if valid_loss <= min_loss:   # 用词级别loss保存
                # if valid_loss_hter <= min_loss:  # 用句子级loss保存
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(discriminator.state_dict(), best_model_prefix + ".final")
                        #torch.save(discriminator.state_dict(), best_model_prefix + "." + str(save_id))
                        #INFO("save model id: {0}".format(save_id))
                        #save_id += 1
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.4f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.2f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
            
            id += 1

            if is_early_stop is True:
                break
        if is_early_stop is True:
            break
        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def find_align_attn(flags):
    """
    寻找最接近词对齐信息的attn矩阵
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(flags.saveto,
                                     flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    with open(flags.align_bpe_path_mt2src_train, 'r', encoding='utf-8') as falign_m2s_train:
        align_bpe_dict_mt2src_train = json.load(falign_m2s_train)

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3])
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'],
                                     shuffle=False)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False,
                                  shuffle=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    # 加载之前训好的模型参数
    #params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    #discriminator.load_state_dict(params)
    # 如果有训好的，才加载
    load_pretrained_model(discriminator, flags.pretrain_path, exclude_prefix=None,
                          device="cpu")
    
    if Constants.USE_GPU:
        discriminator.cuda()

    # Build iterator and progress bar
    training_iter = training_iterator.build_generator()
    training_progress_bar = tqdm(desc='  - (Epoch 0)   ',
                                    total=len(training_iterator),
                                    unit="sents")

    id = 0
    attn_align_cossim = []
    for batch in training_iter:
        discriminator.eval()
        
        seqs_x, seqs_y, xy_label, xy_hter = batch

        n_samples_t = len(seqs_x)
        training_progress_bar.update(n_samples_t)

        x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)

        if Constants.USE_GPU:
            xy_hter = torch.tensor(xy_hter)
            xy_hter = xy_hter.cuda()

        attn_align_cossim_one_sample = compute_attn_align_cossim(model=discriminator,
                                                                seqs_x=x,
                                                                seqs_y=y,
                                                                lm_ids=xy_label,
                                                                hter_label=xy_hter,
                                                                align_line=align_bpe_dict_mt2src_train[str(id)])
        attn_align_cossim.append(attn_align_cossim_one_sample)

        id += 1
        #if id > 3: break

    attn_align_cossim = torch.stack(attn_align_cossim, dim = 0)
    #print(attn_align_cossim)
    attn_align_cossim = torch.mean(attn_align_cossim, dim = 0)  # 训练集平均下来，每个layer每个head的attn矩阵和对齐矩阵有多接近
    #print(attn_align_cossim.size())
    print("训练集平均下来，每个layer每个head的attn矩阵和对齐矩阵有多接近:")
    print(attn_align_cossim)
    attn_align_cossim_argmax = attn_align_cossim.argmax(dim = -1)    # 每层最重要的head是哪个
    print("每层最重要的head是哪个:")
    print(attn_align_cossim_argmax)


def finetune_qe_robust_aug_anti(flags):
    # 反向数据增强，增强的数据mask掉src，并动态随机生成mt标签
    """
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(flags.saveto,
                                     flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3])
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'],
                                     shuffle=True)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False,
                                  shuffle=True)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    # 加载之前训好的模型参数
    #params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    #discriminator.load_state_dict(params)
    # 如果有训好的，才加载
    load_pretrained_model(discriminator, flags.pretrain_path, exclude_prefix=None,
                          device="cpu")
    
    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=discriminator,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None

    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=flags.log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    save_id = 0

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        train_loss = 0
        train_loss_hter = 0

        for batch in training_iter:
            discriminator.eval()
            uidx += 1
            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y, xy_label, xy_hter = batch

            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)
            cum_samples += n_samples_t
            cum_words += n_words_t
            training_progress_bar.update(n_samples_t)

            x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)

            if Constants.USE_GPU:
                xy_hter = torch.tensor(xy_hter)
                xy_hter = xy_hter.cuda()

            loss, loss_hter = compute_forward_discriminator_robust_aug_anti(model=discriminator,
                                                            critic=critic,
                                                            critic_mse=critic_mse,
                                                            seqs_x=x,
                                                            seqs_y=y,
                                                            lm_ids=xy_label,
                                                            hter_label=xy_hter,
                                                            aug_lambda=flags.aug_lambda)

            train_loss += loss
            train_loss_hter += loss_hter
            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} TrainLoss_hter: {2:.4f} ".format(
                    uidx, train_loss / training_configs['disp_freq'], train_loss_hter / training_configs['disp_freq']
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
                train_loss = 0
                train_loss_hter = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                    loss_evaluation_qe(discriminator=discriminator,
                                       critic=critic,
                                       critic_mse=critic_mse,
                                       valid_iterator=valid_iterator)

                # 保存ckpt
                torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                torch.save(discriminator.state_dict(), best_model_prefix + '.last_ckpt')

                model_collections.add_to_collection("history_losses", valid_loss)   # 用词级别loss保存
                # model_collections.add_to_collection("history_losses", valid_loss_hter)  # 用句子级loss保存
                min_loss = np.array(model_collections.get_collection("history_losses")).min()

                if valid_loss <= min_loss:   # 用词级别loss保存
                # if valid_loss_hter <= min_loss:  # 用句子级loss保存
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(discriminator.state_dict(), best_model_prefix + ".final")
                        #torch.save(discriminator.state_dict(), best_model_prefix + "." + str(save_id))
                        #INFO("save model id: {0}".format(save_id))
                        #save_id += 1
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.4f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.2f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
            if is_early_stop is True:
                break
        if is_early_stop is True:
            break
        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def finetune_qe_robust_aug_dynamic(flags):
    """
    数据增强微调，动态生成伪数据，随机mask掉ok对应的src token来生成，修改标签为bad
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    generator_configs = configs['generator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(flags.saveto,
                                     flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    with open(flags.align_bpe_path_mt2src, 'r', encoding='utf-8') as f:
        align_bpe_dict_mt2src = json.load(f)

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3])
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'],
                                     shuffle=False)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False,
                                  shuffle=True)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    generator_mlm = build_model(n_src_vocab=vocab_tgt.max_n_words,
                                n_tgt_vocab=vocab_src.max_n_words,
                                padding_idx=vocab_tgt.pad, **generator_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    # 加载之前训好的模型参数
    #params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    #discriminator.load_state_dict(params)
    # 如果有训好的，才加载
    load_pretrained_model(generator_mlm, flags.pretrain_generator_mlm_path, exclude_prefix=None,
                          device="cpu")
    load_pretrained_model(discriminator, flags.pretrain_path, exclude_prefix=None,
                          device="cpu")
    
    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        generator_mlm = generator_mlm.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=discriminator,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None

    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=flags.log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    save_id = 0

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        train_loss = 0
        train_loss_hter = 0

        id = 0
        for batch in training_iter:
            discriminator.eval()
            uidx += 1
            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y, xy_label, xy_hter,  = batch

            align_bpe_dict_line = align_bpe_dict_mt2src[str(id)]  # 这句话中每个mt token对齐的src token们

            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)
            cum_samples += n_samples_t
            cum_words += n_words_t
            training_progress_bar.update(n_samples_t)

            # 动态伪数据
            mask_ratio = 0.5

            y_len = len(seqs_y[-1])    # mt句的长度
        
            ok_ids = [i for i in range(y_len) if xy_label[0][i] == 1]

            x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)
            
            if Constants.USE_GPU:
                xy_hter = torch.tensor(xy_hter)
                xy_hter = xy_hter.cuda()
            
            loss, loss_hter = compute_forward_discriminator(model=discriminator,
                                                            critic=critic,
                                                            critic_mse=critic_mse,
                                                            seqs_x=x,
                                                            seqs_y=y,
                                                            lm_ids=xy_label,
                                                            hter_label=xy_hter)
            
            train_loss += loss
            train_loss_hter += loss_hter
            

            if(len(ok_ids)):
                sample_ok_ids = random.sample(ok_ids, math.ceil(len(ok_ids) * mask_ratio))
            
                align_position = []
                change_ids = []
                for i in sample_ok_ids:
                    if str(i) in align_bpe_dict_line:
                        align_position.extend(align_bpe_dict_line[str(i)])
                        change_ids.append(i)
                finally_ok_id = list(set(ok_ids) - set(change_ids))
                
                if len(align_position):  # 可以生成伪数据
                    ok_align_position = list(set(align_position))  # 去重，得到全部mt ok token对应的src token
                    robust_x, robust_y, x_mask, lm_ids_x = prepare_data_all_robust(seqs_x, seqs_y, align_position=ok_align_position, cuda=Constants.USE_GPU)

                    # 生成伪数据
                    with torch.no_grad():
                        mlm_logits_x = generator_mlm(robust_y, x_mask, 'MLM', log_probs=False)  # [batch_size, seq_len, hidden_dim]

                    word_noise_data_x = gen_noise_data5_y2x_robust_must(robust_x, mlm_logits_x, lm_ids_x)
                    word_noise_label = [1 if i in finally_ok_id else 2 for i in range(y_len)]
                    word_noise_label = [0, 0] + word_noise_label + [0]
                    word_noise_label = torch.tensor(word_noise_label).view(n_samples_t, -1).cuda()
                    word_noise_hter = ((word_noise_label == 2).sum() / (word_noise_label != 0).sum()).unsqueeze(0).unsqueeze(1).unsqueeze(2)

                    robust_loss, robust_loss_hter = compute_forward_discriminator(model=discriminator,
                                                                    critic=critic,
                                                                    critic_mse=critic_mse,
                                                                    seqs_x=word_noise_data_x,
                                                                    seqs_y=y,
                                                                    lm_ids=word_noise_label,
                                                                    hter_label=word_noise_hter)

                    train_loss += robust_loss
                    train_loss_hter += robust_loss_hter

            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} TrainLoss_hter: {2:.4f} ".format(
                    uidx, train_loss / training_configs['disp_freq'], train_loss_hter / training_configs['disp_freq']
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
                train_loss = 0
                train_loss_hter = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                    loss_evaluation_qe(discriminator=discriminator,
                                       critic=critic,
                                       critic_mse=critic_mse,
                                       valid_iterator=valid_iterator)

                # 保存ckpt
                torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                torch.save(discriminator.state_dict(), best_model_prefix + '.last_ckpt')

                model_collections.add_to_collection("history_losses", valid_loss)   # 用词级别loss保存
                # model_collections.add_to_collection("history_losses", valid_loss_hter)  # 用句子级loss保存
                min_loss = np.array(model_collections.get_collection("history_losses")).min()

                if valid_loss <= min_loss:   # 用词级别loss保存
                # if valid_loss_hter <= min_loss:  # 用句子级loss保存
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(discriminator.state_dict(), best_model_prefix + ".final")
                        #torch.save(discriminator.state_dict(), best_model_prefix + "." + str(save_id))
                        #INFO("save model id: {0}".format(save_id))
                        #save_id += 1
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.4f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.2f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
            
            id += 1

            if is_early_stop is True:
                break

        if is_early_stop is True:
            break
        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def finetune_qe_log(flags):
    """
    finetune过程中，记录loss啊什么的存起来以备画图
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(flags.saveto,
                                     flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3])
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    toy_train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['toy_train_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['toy_train_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['toy_train_data'][2]),
        QEHterDataset(data_path=data_configs['toy_train_data'][3])
    )

    pretrain_valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['pretrain_valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['pretrain_valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['pretrain_valid_data'][2]),
        QEHterDataset(data_path=data_configs['pretrain_valid_data'][3])
    )

    pretrain_toy_train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['pretrain_toy_train_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['pretrain_toy_train_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['pretrain_toy_train_data'][2]),
        QEHterDataset(data_path=data_configs['pretrain_toy_train_data'][3])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'])

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False)

    toy_train_iterator = DataIterator(dataset=toy_train_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False)

    pretrain_valid_iterator = DataIterator(dataset=pretrain_valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False)

    pretrain_toy_train_iterator = DataIterator(dataset=pretrain_toy_train_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    # 加载之前训好的模型参数
    #params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    #discriminator.load_state_dict(params)
    # 如果有训好的，才加载
    load_pretrained_model(discriminator, flags.pretrain_path, exclude_prefix=None,
                          device="cpu")
    
    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=discriminator,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None

    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=flags.log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    loss_dict = {
        'finetune_train_loss':{'loss_word':[], 'loss_hter':[]},
        'finetune_train_eval':{'loss_word':[], 'loss_hter':[], 'f1_ok':[], 'f1_bad':[], 'f1_multi':[], 'pearsonr':[]},
        'finetune_val_eval':  {'loss_word':[], 'loss_hter':[], 'f1_ok':[], 'f1_bad':[], 'f1_multi':[], 'pearsonr':[]},
        'pretrain_train_eval':{'loss_word':[], 'loss_hter':[], 'f1_ok':[], 'f1_bad':[], 'f1_multi':[], 'pearsonr':[]},
        'pretrain_val_eval':  {'loss_word':[], 'loss_hter':[], 'f1_ok':[], 'f1_bad':[], 'f1_multi':[], 'pearsonr':[]},
    }

    save_id = 0

    while True:

        # =========================================== 训练前测试 ================================================ #
        valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
            loss_evaluation_qe(discriminator=discriminator,
                                critic=critic,
                                critic_mse=critic_mse,
                                valid_iterator=valid_iterator)

        toy_train_loss, toy_train_loss_hter, toy_train_f1good, toy_train_f1bad, toy_train_f1multi, toy_train_p = \
                loss_evaluation_qe(discriminator=discriminator,
                                    critic=critic,
                                    critic_mse=critic_mse,
                                    valid_iterator=toy_train_iterator)

        pretrain_valid_loss, pretrain_valid_loss_hter, pretrain_f1good, pretrain_f1bad, pretrain_f1multi, pretrain_p = \
            loss_evaluation_qe(discriminator=discriminator,
                                critic=critic,
                                critic_mse=critic_mse,
                                valid_iterator=pretrain_valid_iterator)

        pretrain_toy_train_loss, pretrain_toy_train_loss_hter, pretrain_toy_train_f1good, pretrain_toy_train_f1bad, pretrain_toy_train_f1multi, pretrain_toy_train_p = \
                loss_evaluation_qe(discriminator=discriminator,
                                    critic=critic,
                                    critic_mse=critic_mse,
                                    valid_iterator=pretrain_toy_train_iterator)

        # 保存 loss记录
        loss_dict['finetune_train_eval']['loss_word'].append(toy_train_loss)
        loss_dict['finetune_train_eval']['loss_hter'].append(toy_train_loss_hter)
        loss_dict['finetune_train_eval']['f1_ok'].append(toy_train_f1good)
        loss_dict['finetune_train_eval']['f1_bad'].append(toy_train_f1bad)
        loss_dict['finetune_train_eval']['f1_multi'].append(toy_train_f1multi)
        loss_dict['finetune_train_eval']['pearsonr'].append(toy_train_p)

        loss_dict['finetune_val_eval']['loss_word'].append(valid_loss)
        loss_dict['finetune_val_eval']['loss_hter'].append(valid_loss_hter)
        loss_dict['finetune_val_eval']['f1_ok'].append(f1good)
        loss_dict['finetune_val_eval']['f1_bad'].append(f1bad)
        loss_dict['finetune_val_eval']['f1_multi'].append(f1multi)
        loss_dict['finetune_val_eval']['pearsonr'].append(p)

        loss_dict['pretrain_train_eval']['loss_word'].append(pretrain_toy_train_loss)
        loss_dict['pretrain_train_eval']['loss_hter'].append(pretrain_toy_train_loss_hter)
        loss_dict['pretrain_train_eval']['f1_ok'].append(pretrain_toy_train_f1good)
        loss_dict['pretrain_train_eval']['f1_bad'].append(pretrain_toy_train_f1bad)
        loss_dict['pretrain_train_eval']['f1_multi'].append(pretrain_toy_train_f1multi)
        loss_dict['pretrain_train_eval']['pearsonr'].append(pretrain_toy_train_p)

        loss_dict['pretrain_val_eval']['loss_word'].append(pretrain_valid_loss)
        loss_dict['pretrain_val_eval']['loss_hter'].append(pretrain_valid_loss_hter)
        loss_dict['pretrain_val_eval']['f1_ok'].append(pretrain_f1good)
        loss_dict['pretrain_val_eval']['f1_bad'].append(pretrain_f1bad)
        loss_dict['pretrain_val_eval']['f1_multi'].append(pretrain_f1multi)
        loss_dict['pretrain_val_eval']['pearsonr'].append(pretrain_p)

        # =========================================== 训练前测试结束 ================================================ #

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        train_loss = 0
        train_loss_hter = 0

        for batch in training_iter:
            discriminator.eval()  # 这个为什么啊？---不管什么模式，训出来没区别，不用管
            """
            valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                    loss_evaluation_qe(discriminator=discriminator,
                                       critic=critic,
                                       critic_mse=critic_mse,
                                       valid_iterator=valid_iterator)
            print('================================')
            print(valid_loss)
            print(valid_loss_hter)
            print(f1good)
            print(f1bad)
            print(f1multi)
            print(p)
            assert 1==2
            """
            uidx += 1
            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y, xy_label, xy_hter = batch

            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)
            cum_samples += n_samples_t
            cum_words += n_words_t
            training_progress_bar.update(n_samples_t)

            x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)
            

            if Constants.USE_GPU:
                xy_hter = torch.tensor(xy_hter)
                xy_hter = xy_hter.cuda()
                
            loss, loss_hter = compute_forward_discriminator(model=discriminator,
                                                            critic=critic,
                                                            critic_mse=critic_mse,
                                                            seqs_x=x,
                                                            seqs_y=y,
                                                            lm_ids=xy_label,
                                                            hter_label=xy_hter)

            # 记录每个batch的 loss_word, loss_word_hter
            loss_dict['finetune_train_loss']['loss_word'].append(loss)
            loss_dict['finetune_train_loss']['loss_hter'].append(loss_hter)

            train_loss += loss
            train_loss_hter += loss_hter
            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} TrainLoss_hter: {2:.4f} ".format(
                    uidx, train_loss / training_configs['disp_freq'], train_loss_hter / training_configs['disp_freq']
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
                train_loss = 0
                train_loss_hter = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                    loss_evaluation_qe(discriminator=discriminator,
                                       critic=critic,
                                       critic_mse=critic_mse,
                                       valid_iterator=valid_iterator)

                toy_train_loss, toy_train_loss_hter, toy_train_f1good, toy_train_f1bad, toy_train_f1multi, toy_train_p = \
                        loss_evaluation_qe(discriminator=discriminator,
                                           critic=critic,
                                           critic_mse=critic_mse,
                                           valid_iterator=toy_train_iterator)

                pretrain_valid_loss, pretrain_valid_loss_hter, pretrain_f1good, pretrain_f1bad, pretrain_f1multi, pretrain_p = \
                    loss_evaluation_qe(discriminator=discriminator,
                                       critic=critic,
                                       critic_mse=critic_mse,
                                       valid_iterator=pretrain_valid_iterator)

                pretrain_toy_train_loss, pretrain_toy_train_loss_hter, pretrain_toy_train_f1good, pretrain_toy_train_f1bad, pretrain_toy_train_f1multi, pretrain_toy_train_p = \
                        loss_evaluation_qe(discriminator=discriminator,
                                           critic=critic,
                                           critic_mse=critic_mse,
                                           valid_iterator=pretrain_toy_train_iterator)

                # 保存 loss记录
                loss_dict['finetune_train_eval']['loss_word'].append(toy_train_loss)
                loss_dict['finetune_train_eval']['loss_hter'].append(toy_train_loss_hter)
                loss_dict['finetune_train_eval']['f1_ok'].append(toy_train_f1good)
                loss_dict['finetune_train_eval']['f1_bad'].append(toy_train_f1bad)
                loss_dict['finetune_train_eval']['f1_multi'].append(toy_train_f1multi)
                loss_dict['finetune_train_eval']['pearsonr'].append(toy_train_p)

                loss_dict['finetune_val_eval']['loss_word'].append(valid_loss)
                loss_dict['finetune_val_eval']['loss_hter'].append(valid_loss_hter)
                loss_dict['finetune_val_eval']['f1_ok'].append(f1good)
                loss_dict['finetune_val_eval']['f1_bad'].append(f1bad)
                loss_dict['finetune_val_eval']['f1_multi'].append(f1multi)
                loss_dict['finetune_val_eval']['pearsonr'].append(p)

                loss_dict['pretrain_train_eval']['loss_word'].append(pretrain_toy_train_loss)
                loss_dict['pretrain_train_eval']['loss_hter'].append(pretrain_toy_train_loss_hter)
                loss_dict['pretrain_train_eval']['f1_ok'].append(pretrain_toy_train_f1good)
                loss_dict['pretrain_train_eval']['f1_bad'].append(pretrain_toy_train_f1bad)
                loss_dict['pretrain_train_eval']['f1_multi'].append(pretrain_toy_train_f1multi)
                loss_dict['pretrain_train_eval']['pearsonr'].append(pretrain_toy_train_p)

                loss_dict['pretrain_val_eval']['loss_word'].append(pretrain_valid_loss)
                loss_dict['pretrain_val_eval']['loss_hter'].append(pretrain_valid_loss_hter)
                loss_dict['pretrain_val_eval']['f1_ok'].append(pretrain_f1good)
                loss_dict['pretrain_val_eval']['f1_bad'].append(pretrain_f1bad)
                loss_dict['pretrain_val_eval']['f1_multi'].append(pretrain_f1multi)
                loss_dict['pretrain_val_eval']['pearsonr'].append(pretrain_p)

                if uidx >= 78000:
                    INFO("save ckpt")
                    #torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                    torch.save(discriminator.state_dict(), best_model_prefix + '.last_ckpt')
                

                #model_collections.add_to_collection("history_losses", valid_loss)
                model_collections.add_to_collection("history_losses", valid_loss_hter)
                min_loss = np.array(model_collections.get_collection("history_losses")).min()

                #if valid_loss <= min_loss:
                if valid_loss_hter <= min_loss:
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(discriminator.state_dict(), best_model_prefix + ".final")
                        #torch.save(discriminator.state_dict(), best_model_prefix + "." + str(save_id))
                        #INFO("save model id: {0}".format(save_id))
                        #save_id += 1
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.4f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.2f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
            if is_early_stop is True:
                break
        if is_early_stop is True:
            break
        training_progress_bar.close()

        # save epoch ckpt
        #torch.save(discriminator.state_dict(), best_model_prefix + ".epoch_" + str(eidx))

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break

    with open(flags.saveto + '/loss_dict.json', 'w', encoding='utf-8') as f:
        json.dump(loss_dict, f, indent=1)


def finetune_qe_contrastive(flags):
    # 对比学习
    """
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s_%s.log" % (os.path.basename(flags.log_path), time.strftime("%Y%m%d-%H%M%S"))))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    loss_real_lambda = training_configs["loss_real_lambda"]
    loss_parallel_lambda = training_configs["loss_parallel_lambda"]
    loss_switch_lambda = training_configs["loss_switch_lambda"]
    cl_lambda_switch = training_configs["cl_lambda_switch"]
    cl_lambda_parallel = training_configs["cl_lambda_parallel"]

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(flags.saveto,
                                     flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3]),
        TextLineDataset(data_path=data_configs['train_data'][4],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QEAlignDataset(data_path=data_configs['train_data'][5]),
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3]),
        TextLineDataset(data_path=data_configs['valid_data'][4],
                        vocabulary=vocab_tgt,
                        ),
        QEAlignDataset(data_path=data_configs['valid_data'][5]),
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'],
                                     shuffle=True)

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['batch_size'],
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    # 加载之前训好的模型参数
    #params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    #discriminator.load_state_dict(params)
    # 如果有训好的，才加载
    load_pretrained_model(discriminator, flags.pretrain_path, exclude_prefix=None,
                          device="cpu")
    
    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=discriminator,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None

    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=flags.log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    save_id = 0

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        train_loss = 0
        train_loss_cl = 0
        train_loss_hter = 0

        for batch in training_iter:
            discriminator.eval()
            uidx += 1
            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y, xy_label, xy_hter, seqs_pe, seqs_align = batch

            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)
            cum_samples += n_samples_t
            cum_words += n_words_t
            training_progress_bar.update(n_samples_t)

            x, y, pe, xy_label = prepare_data_qe_with_pe(seqs_x, seqs_y, seqs_pe, xy_label, cuda=Constants.USE_GPU)

            if Constants.USE_GPU:
                xy_hter = torch.tensor(xy_hter).cuda()

            loss, loss_parallel, loss_switch, cl_loss, cl_loss_parallel, loss_hter = \
                                compute_forward_discriminator_contrastive(model=discriminator,
                                                                        critic=critic,
                                                                        critic_mse=critic_mse,
                                                                        seqs_x=x,
                                                                        seqs_y=y,
                                                                        seqs_pe=pe,
                                                                        seqs_align=seqs_align,
                                                                        lm_ids=xy_label,
                                                                        hter_label=xy_hter,
                                                                        loss_real_lambda=loss_real_lambda,
                                                                        loss_parallel_lambda=loss_parallel_lambda,
                                                                        loss_switch_lambda=loss_switch_lambda,
                                                                        cl_lambda_switch=cl_lambda_switch,
                                                                        cl_lambda_parallel=cl_lambda_parallel,
                                                                        optim = optim)

            train_loss += loss
            train_loss_cl += cl_loss
            train_loss_hter += loss_hter
            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} TrainLoss_hter: {2:.4f} ".format(
                    uidx, train_loss / training_configs['disp_freq'], train_loss_hter / training_configs['disp_freq']
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
                train_loss = 0
                train_loss_hter = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                valid_loss, valid_loss_cl, valid_loss_hter, f1good, f1bad, f1multi, p = \
                    loss_evaluation_qe_contrastive(discriminator=discriminator,
                                                    critic=critic,
                                                    critic_mse=critic_mse,
                                                    valid_iterator=valid_iterator,
                                                    loss_real_lambda=loss_real_lambda,
                                                    loss_parallel_lambda=loss_parallel_lambda,
                                                    loss_switch_lambda=loss_switch_lambda,
                                                    cl_lambda_switch=cl_lambda_switch,
                                                    cl_lambda_parallel=cl_lambda_parallel,)

                # 保存ckpt
                torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                torch.save(discriminator.state_dict(), best_model_prefix + '.last_ckpt')

                model_collections.add_to_collection("history_losses", valid_loss)   # 用词级别loss保存模型
                # model_collections.add_to_collection("history_losses", valid_loss_hter)
                min_loss = np.array(model_collections.get_collection("history_losses")).min()

                if valid_loss <= min_loss:                                          # 用词级别loss保存模型
                # if valid_loss_hter <= min_loss:
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(discriminator.state_dict(), best_model_prefix + ".final")
                        #torch.save(discriminator.state_dict(), best_model_prefix + "." + str(save_id))
                        #INFO("save model id: {0}".format(save_id))
                        #save_id += 1
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.4f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.2f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
            if is_early_stop is True:
                break
        if is_early_stop is True:
            break
        training_progress_bar.close()

        # save epoch ckpt
        #torch.save(discriminator.state_dict(), best_model_prefix + ".epoch_" + str(eidx))

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def finetune_qe_many(flags):
    """
    finetune过程中，只关心那些训练样本数量高于一定阈值的token
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    train_num_threshold = flags.train_num_threshold

    best_model_prefix = os.path.join(flags.saveto,
                                     flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])


    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3]),
        QEHterDataset(data_path=data_configs['train_data'][4])
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3]),
        QEHterDataset(data_path=data_configs['valid_data'][4])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'])

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    # 加载之前训好的模型参数
    #params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    #discriminator.load_state_dict(params)
    # 如果有训好的，才加载
    load_pretrained_model(discriminator, flags.pretrain_path, exclude_prefix=None,
                          device="cpu")
    
    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=discriminator,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None

    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=flags.log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    save_id = 0

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        train_loss = 0
        train_loss_hter = 0

        for batch in training_iter:
            discriminator.eval()  # 这个为什么啊？---不管什么模式，训出来没区别，不用管
            uidx += 1
            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y, xy_label, xy_hter, y_cnt = batch   # y_cnt，表示出现次数的文件
            y_cnt = torch.tensor(y_cnt).cuda().squeeze(0)
            many_mask = y_cnt >= train_num_threshold
            unk_mask = (torch.tensor(seqs_y).cuda()) != 3
            many_mask *= unk_mask

            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)
            cum_samples += n_samples_t
            cum_words += n_words_t
            training_progress_bar.update(n_samples_t)

            x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)

            if Constants.USE_GPU:
                xy_hter = torch.tensor(xy_hter)
                xy_hter = xy_hter.cuda()
            
            # 传进去一个mask，不足的token位置或者unk都是pad，0；其余足够的是1
            loss, loss_hter = compute_forward_discriminator_token_mask(model=discriminator,
                                                            critic=critic,
                                                            critic_mse=critic_mse,
                                                            seqs_x=x,
                                                            seqs_y=y,
                                                            lm_ids=xy_label,
                                                            hter_label=xy_hter,
                                                            token_mask=many_mask)

            train_loss += loss
            train_loss_hter += loss_hter
            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} TrainLoss_hter: {2:.4f} ".format(
                    uidx, train_loss / training_configs['disp_freq'], train_loss_hter / training_configs['disp_freq']
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
                train_loss = 0
                train_loss_hter = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                    loss_evaluation_qe_many(discriminator=discriminator,
                                       critic=critic,
                                       critic_mse=critic_mse,
                                       valid_iterator=valid_iterator,
                                       train_num_threshold=train_num_threshold)
                
                #if uidx >= 10000:
                INFO("save ckpt")
                torch.save(discriminator.state_dict(), best_model_prefix + '.ckpt.' + str(uidx))
                    #torch.save(discriminator.state_dict(), best_model_prefix + '.last_ckpt')
                
                model_collections.add_to_collection("history_losses", valid_loss)
                #model_collections.add_to_collection("history_losses", valid_loss_hter)
                min_loss = np.array(model_collections.get_collection("history_losses")).min()

                if valid_loss <= min_loss:
                #if valid_loss_hter <= min_loss:
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(discriminator.state_dict(), best_model_prefix + ".final")
                        #torch.save(discriminator.state_dict(), best_model_prefix + "." + str(save_id))
                        #INFO("save model id: {0}".format(save_id))
                        #save_id += 1
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.4f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.2f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
            if is_early_stop is True:
                break
        if is_early_stop is True:
            break
        training_progress_bar.close()

        # save epoch ckpt
        #torch.save(discriminator.state_dict(), best_model_prefix + ".epoch_" + str(eidx))

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def finetune_qe_weight(flags):
    """
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(flags.saveto,
                                     flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3]),
        QEHterDataset(data_path=data_configs['train_data'][4]),
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'])

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params)
    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=discriminator,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None

    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=flags.log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    save_id = 0

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        train_loss = 0
        train_loss_hter = 0

        for batch in training_iter:
            discriminator.eval()
            uidx += 1
            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y, xy_label, xy_hter, ok_prob = batch
            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)
            cum_samples += n_samples_t
            cum_words += n_words_t
            training_progress_bar.update(n_samples_t)

            x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)
            if Constants.USE_GPU:
                xy_hter = torch.tensor(xy_hter)
                xy_hter = xy_hter.cuda()
                ok_prob = torch.tensor(ok_prob).cuda().squeeze(0)
            

            # ================================================================================== #
            # 这里换了个函数哦
            loss, loss_hter = compute_forward_discriminator_modify(model=discriminator,
                                                            critic=critic,
                                                            critic_mse=critic_mse,
                                                            seqs_x=x,
                                                            seqs_y=y,
                                                            lm_ids=xy_label,
                                                            hter_label=xy_hter,
                                                            ok_prob=ok_prob)

            train_loss += loss
            train_loss_hter += loss_hter
            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} TrainLoss_hter: {2:.4f} ".format(
                    uidx, train_loss / training_configs['disp_freq'], train_loss_hter / training_configs['disp_freq']
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
                train_loss = 0
                train_loss_hter = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                    loss_evaluation_qe(discriminator=discriminator,
                                       critic=critic,
                                       critic_mse=critic_mse,
                                       valid_iterator=valid_iterator)

                # model_collections.add_to_collection("history_losses", valid_loss)
                model_collections.add_to_collection("history_losses", valid_loss_hter)
                min_loss = np.array(model_collections.get_collection("history_losses")).min()

                # if valid_loss <= min_loss:
                if valid_loss_hter <= min_loss:
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        #torch.save(discriminator.state_dict(), best_model_prefix + ".final")
                        torch.save(discriminator.state_dict(), best_model_prefix + "." + str(save_id))
                        INFO("save model id: {0}".format(save_id))
                        save_id += 1
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.4f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.2f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
            if is_early_stop is True:
                break
        if is_early_stop is True:
            break
        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def train_qe(flags):
    """
    单纯地用qe数据训练discriminator，无预训练
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(flags.saveto,
                                     flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3])
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'])

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    #params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    #discriminator.load_state_dict(params)
    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=discriminator,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None

    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=flags.log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        train_loss = 0
        train_loss_hter = 0

        for batch in training_iter:
            discriminator.eval()
            uidx += 1
            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y, xy_label, xy_hter = batch
            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)
            cum_samples += n_samples_t
            cum_words += n_words_t
            training_progress_bar.update(n_samples_t)

            x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)
            if Constants.USE_GPU:
                xy_hter = torch.tensor(xy_hter)
                xy_hter = xy_hter.cuda()

            loss, loss_hter = compute_forward_discriminator(model=discriminator,
                                                            critic=critic,
                                                            critic_mse=critic_mse,
                                                            seqs_x=x,
                                                            seqs_y=y,
                                                            lm_ids=xy_label,
                                                            hter_label=xy_hter)

            train_loss += loss
            train_loss_hter += loss_hter
            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} TrainLoss_hter: {2:.4f} ".format(
                    uidx, train_loss / training_configs['disp_freq'], train_loss_hter / training_configs['disp_freq']
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
                train_loss = 0
                train_loss_hter = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                    loss_evaluation_qe(discriminator=discriminator,
                                       critic=critic,
                                       critic_mse=critic_mse,
                                       valid_iterator=valid_iterator)

                # model_collections.add_to_collection("history_losses", valid_loss)
                model_collections.add_to_collection("history_losses", valid_loss_hter)
                min_loss = np.array(model_collections.get_collection("history_losses")).min()

                # if valid_loss <= min_loss:
                if valid_loss_hter <= min_loss:
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(discriminator.state_dict(), best_model_prefix + ".final")
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.4f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.2f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
            if is_early_stop is True:
                break
        if is_early_stop is True:
            break
        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def new_finetune(flags):
    """
    FLAGS:
        saveto: str
        reload: store_true
        config_path: str
        pretrain_path: str, default=""
        model_name: str
        log_path: str
    """

    # write log of training to file.
    log_path = flags.log_path_prefix
    saveto = log_path
    write_log_to_file(os.path.join(log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(saveto,
                                     flags.model_name + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_data_path = []
    for s in ['.src.BPE', '.mt', '.tag', '.hter']:
        ss = log_path + 'wo_' + flags.wo + '_test' + str(flags.test_id) + s
        train_data_path.append(ss)
    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=train_data_path[0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=train_data_path[1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=train_data_path[2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=train_data_path[3])
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['valid_data'][2]),
        QEHterDataset(data_path=data_configs['valid_data'][3])
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'])

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params)
    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=discriminator,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None

    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        train_loss = 0
        train_loss_hter = 0

        for batch in training_iter:
            discriminator.eval()
            uidx += 1
            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y, xy_label, xy_hter = batch
            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)
            cum_samples += n_samples_t
            cum_words += n_words_t
            training_progress_bar.update(n_samples_t)

            x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)
            if Constants.USE_GPU:
                xy_hter = torch.tensor(xy_hter)
                xy_hter = xy_hter.cuda()

            loss, loss_hter = compute_forward_discriminator(model=discriminator,
                                                            critic=critic,
                                                            critic_mse=critic_mse,
                                                            seqs_x=x,
                                                            seqs_y=y,
                                                            lm_ids=xy_label,
                                                            hter_label=xy_hter)

            train_loss += loss
            train_loss_hter += loss_hter
            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} TrainLoss_hter: {2:.4f} ".format(
                    uidx, train_loss / training_configs['disp_freq'], train_loss_hter / training_configs['disp_freq']
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
                train_loss = 0
                train_loss_hter = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p = \
                    loss_evaluation_qe(discriminator=discriminator,
                                       critic=critic,
                                       critic_mse=critic_mse,
                                       valid_iterator=valid_iterator)

                # model_collections.add_to_collection("history_losses", valid_loss)
                model_collections.add_to_collection("history_losses", valid_loss_hter)
                min_loss = np.array(model_collections.get_collection("history_losses")).min()

                # if valid_loss <= min_loss:
                if valid_loss_hter <= min_loss:
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(discriminator.state_dict(), best_model_prefix + ".final")
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} Loss_hter: {2:.4f} F1_good: {3:.4f} F1_bad: {4:.4f} "
                     "F1_multi: {5:.4f} pearsonr:{6:.2f} patience: {7}".
                     format(uidx, valid_loss, valid_loss_hter, f1good, f1bad, f1multi, p, bad_count))
            if is_early_stop is True:
                break
        if is_early_stop is True:
            break
        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def test_qe(flags):
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    discriminator_model_configs = configs['discriminator_configs']

    data_configs = configs['data_configs']
    #print(flags.batch_size)

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    if flags.align_bpe_path_mt2src_test:
        with open(flags.align_bpe_path_mt2src_test, 'r', encoding='utf-8') as f:
            align_bpe_dict_mt2src_test = json.load(f)

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=flags.source_path,
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=flags.target_path,
                        vocabulary=vocab_tgt,
                        )
    )

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=flags.batch_size,
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading transformer model parameters...')
    timer.tic()
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params, strict=False)

    if Constants.USE_GPU:
        discriminator.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #

    INFO('Begin ...')
    result_hter = []
    result_tag = []
    #result_okprob = []

    timer.tic()

    valid_iter = valid_iterator.build_generator()

    add_align = False
    if flags.add_align: add_align = True

    id = 0
    for batch_valid in tqdm(valid_iter):
        seqs_x_valid, seqs_y_valid = batch_valid

        x_valid, y_valid = prepare_data_qe(seqs_x_valid, seqs_y_valid, cuda=Constants.USE_GPU)

        if flags.align_bpe_path_mt2src_test:
            align_line = align_bpe_dict_mt2src_test[str(id)]

            x_len = x_valid.size(-1) - 2
            y_len = y_valid.size(-1) - 3

            align_matrix = torch.zeros([y_len, x_len])

            for i in range(y_len):
                if str(i) in align_line:
                    align_matrix[i] = align_matrix[i].index_fill(0, torch.tensor(align_line[str(i)]), True)

            align_matrix_pad = F.pad(align_matrix, pad = (1, 1, 2, 1)).cuda()

            if add_align:
                with torch.no_grad():
                    log_probs, hter_pre = discriminator(x_valid, y_valid, align_matrix_pad = align_matrix_pad, align_ratio=flags.align_ratio, add_align=add_align)
            elif flags.add_feature:
                with torch.no_grad():
                    log_probs, hter_pre = discriminator(x_valid, y_valid, align_matrix_pad = align_matrix_pad, add_feature = True)
        else:
            with torch.no_grad():
                log_probs, hter_pre = discriminator(x_valid, y_valid, requires_adapter=False)

        hter_pre.squeeze(1).cpu().tolist()
        for val in hter_pre:
            result_hter.append(float(val))
        
        log_probs = log_probs[0].cpu().tolist()
        outputs_tags_valid = []
        outputs_okprob = []
        for i in log_probs:
            outputs_okprob.append(math.exp(i[1]))
            if i[1] < i[2]:
                outputs_tags_valid.append('BAD')  # 1 ok    2 bad
            else:
                outputs_tags_valid.append('OK')

        result_tag.append(outputs_tags_valid)
        #result_okprob.append(outputs_okprob)
        id += 1
        

    def auto_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    auto_mkdir('/'.join(flags.saveto_hter.split('/')[0:-1]))
    with open(flags.saveto_hter, 'w') as f:
        for val in result_hter:
            f.write('%.6f\n' % val)
    
    auto_mkdir('/'.join(flags.saveto_tags.split('/')[0:-1]))
    with open(flags.saveto_tags, 'w') as f:
        for vals in result_tag:
            for val in vals:
                f.write(val + ' ')
            f.write('\n')

    if flags.saveto_okprob:
        auto_mkdir('/'.join(flags.saveto_okprob.split('/')[0:-1]))
        with open(flags.saveto_okprob, 'w') as f:
            for vals in result_okprob:
                for val in vals:
                    f.write(str(val) + ' ')
                f.write('\n')


def test_qe_ensemble(flags):
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    discriminator_model_configs = configs['discriminator_configs']

    data_configs = configs['data_configs']
    #print(flags.batch_size)

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])


    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=flags.source_path,
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=flags.target_path,
                        vocabulary=vocab_tgt,
                        )
    )

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=flags.batch_size,
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    pretrain_path_list = discriminator_model_configs["pretrain_path"]

    result_log_ok_list = []
    result_log_bad_list = []
    result_hter_list = []
    result_tag = []
    
    for pretrain_path in pretrain_path_list:

        INFO('Reloading transformer model parameters...')
        timer.tic()
        params = load_model_parameters(pretrain_path, map_location="cpu")
        discriminator.load_state_dict(params, strict=False)

        if Constants.USE_GPU:
            discriminator.cuda()
        INFO('Done. Elapsed time {0}'.format(timer.toc()))

        # ================================================================================== #

        INFO('Begin ...')
        result_hter = []
        result_log_ok = []
        result_log_bad = []

        timer.tic()

        valid_iter = valid_iterator.build_generator()

        
        id = 0
        for batch_valid in tqdm(valid_iter):
            seqs_x_valid, seqs_y_valid = batch_valid

            x_valid, y_valid = prepare_data_qe(seqs_x_valid, seqs_y_valid, cuda=Constants.USE_GPU)

            with torch.no_grad():
                log_probs, hter_pre = discriminator(x_valid, y_valid, requires_adapter=False)

            hter_pre.squeeze(1).cpu().tolist()
            for val in hter_pre:
                result_hter.append(float(val))
            
            log_probs = log_probs[0].cpu().tolist()
            ok_tmp = []
            bad_tmp = []
            for i in log_probs:
                ok_tmp.append(i[1])
                bad_tmp.append(i[2])
            result_log_ok.append(ok_tmp)
            result_log_bad.append(bad_tmp)

            id += 1

        result_hter_list.append(result_hter)
        result_log_ok_list.append(result_log_ok)
        result_log_bad_list.append(result_log_bad)
    
    result_hter = [sum(x) / len(x) for x in zip(*result_hter_list)]   # 几个模型预测的hter列表取平均
    result_log_ok = []
    for item in zip(*result_log_ok_list):
        result_log_ok.append([sum(x) / len(x) for x in zip(*item)])
    result_log_bad = []
    for item in zip(*result_log_bad_list):
        result_log_bad.append([sum(x) / len(x) for x in zip(*item)])

    for sent_ok, sent_bad in zip(result_log_ok, result_log_bad):
        sent_tag = []
        for ok_prob, bad_prob in zip(sent_ok, sent_bad):
            if ok_prob < bad_prob:
                sent_tag.append('BAD')
            else:
                sent_tag.append('OK')
        result_tag.append(sent_tag)
    

    def auto_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    auto_mkdir('/'.join(flags.saveto_hter.split('/')[0:-1]))
    with open(flags.saveto_hter, 'w') as f:
        for val in result_hter:
            f.write('%.6f\n' % val)
    
    auto_mkdir('/'.join(flags.saveto_tags.split('/')[0:-1]))
    with open(flags.saveto_tags, 'w') as f:
        for vals in result_tag:
            for val in vals:
                f.write(val + ' ')
            f.write('\n')



def test_qe_one_class(flags):
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    discriminator_model_configs = configs['discriminator_configs']

    data_configs = configs['data_configs']
    #print(flags.batch_size)

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    if flags.align_bpe_path_mt2src_test:
        with open(flags.align_bpe_path_mt2src_test, 'r', encoding='utf-8') as f:
            align_bpe_dict_mt2src_test = json.load(f)

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=flags.source_path,
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=flags.target_path,
                        vocabulary=vocab_tgt,
                        )
    )

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=flags.batch_size,
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading transformer model parameters...')
    timer.tic()
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params, strict=False)

    if Constants.USE_GPU:
        discriminator.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #

    INFO('Begin ...')
    result_hter = []
    result_tag = []
    #result_okprob = []

    timer.tic()

    valid_iter = valid_iterator.build_generator()

    add_align = False
    if flags.add_align: add_align = True

    id = 0
    for batch_valid in tqdm(valid_iter):
        seqs_x_valid, seqs_y_valid = batch_valid

        x_valid, y_valid = prepare_data_qe(seqs_x_valid, seqs_y_valid, cuda=Constants.USE_GPU)

        with torch.no_grad():
            log_probs, hter_pre = discriminator(x_valid, y_valid, one_class=True)
    
        hter_pre.squeeze(1).cpu().tolist()
        for val in hter_pre:
            result_hter.append(float(val))
        
        log_probs = log_probs[0].cpu().tolist()

        outputs_tags_valid = []
        for i in log_probs:
            if i < 0.5:
                outputs_tags_valid.append('BAD')
            else:
                outputs_tags_valid.append('OK')

        result_tag.append(outputs_tags_valid)
        id += 1
        

    def auto_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    auto_mkdir('/'.join(flags.saveto_hter.split('/')[0:-1]))
    with open(flags.saveto_hter, 'w') as f:
        for val in result_hter:
            f.write('%.6f\n' % val)
    
    auto_mkdir('/'.join(flags.saveto_tags.split('/')[0:-1]))
    with open(flags.saveto_tags, 'w') as f:
        for vals in result_tag:
            for val in vals:
                f.write(val + ' ')
            f.write('\n')


def test_qe_robust_before(flags):
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    discriminator_model_configs = configs['discriminator_configs']

    data_configs = configs['data_configs']
    #print(flags.batch_size)

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=flags.source_path_1,
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=flags.source_path_2,
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=flags.target_path,
                        vocabulary=vocab_tgt,
                        ),
        QEHterDataset(data_path=flags.robust_idx_path)
    )

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=flags.batch_size,
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading transformer model parameters...')
    timer.tic()
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params)

    if Constants.USE_GPU:
        discriminator.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #

    INFO('Begin ...')
    result_tag_robust = []

    timer.tic()

    valid_iter = valid_iterator.build_generator()

    for batch_valid in tqdm(valid_iter):
        seqs_x_valid_1, seqs_x_valid_2, seqs_y_valid, robust_idx = batch_valid
        robust_idx = torch.tensor(robust_idx).squeeze(0).squeeze(0).int().tolist()

        x_valid_1, y_valid, x_valid_2 = prepare_data_qe_with_src_real(seqs_x_valid_1, seqs_y_valid, seqs_x_valid_2, cuda=Constants.USE_GPU)

        with torch.no_grad():
            log_probs_1, hter_pre = discriminator(x_valid_1, y_valid)
            log_probs_2, hter_pre = discriminator(x_valid_2, y_valid)

        log_probs_1 = log_probs_1[0].cpu().tolist()
        log_probs_2 = log_probs_2[0].cpu().tolist()
        outputs_tags_valid_1 = []
        outputs_tags_valid_2 = []
        for i in log_probs_1:
            if i[1] < i[2]:
                outputs_tags_valid_1.append('BAD')  # 1 ok    2 bad
            else:
                outputs_tags_valid_1.append('OK')
        for i in log_probs_2:
            if i[1] < i[2]:
                outputs_tags_valid_2.append('BAD')  # 1 ok    2 bad
            else:
                outputs_tags_valid_2.append('OK')

        robust_outputs_tags_valid_1 = [outputs_tags_valid_1[i] for i in range(len(robust_idx)) if robust_idx[i] == 1]
        robust_outputs_tags_valid_2 = [outputs_tags_valid_2[i] for i in range(len(robust_idx)) if robust_idx[i] == 1]
        
        result_tag_robust.append(robust_outputs_tags_valid_1 + robust_outputs_tags_valid_2)
        

    def auto_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    auto_mkdir('/'.join(flags.saveto_tags.split('/')[0:-1]))
    with open(flags.saveto_tags, 'w') as f:
        for vals in result_tag_robust:
            for val in vals:
                f.write(val + ' ')
            f.write('\n')


def test_qe_robust(flags):
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    discriminator_model_configs = configs['discriminator_configs']

    data_configs = configs['data_configs']
    #print(flags.batch_size)

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    if flags.align_bpe_path_mt2src_test:
        with open(flags.align_bpe_path_mt2src_test, 'r', encoding='utf-8') as f:
            align_bpe_dict_mt2src_test = json.load(f)

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=flags.source_path,
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=flags.target_path,
                        vocabulary=vocab_tgt,
                        ),
        QEHterDataset(data_path=flags.robust_idx_path)
    )

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=flags.batch_size,
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading transformer model parameters...')
    timer.tic()
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params, strict=False)

    if Constants.USE_GPU:
        discriminator.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #

    INFO('Begin ...')
    result_tag_robust = []

    timer.tic()

    valid_iter = valid_iterator.build_generator()

    add_align = False
    if flags.add_align: add_align = True

    id = 0
    for batch_valid in tqdm(valid_iter):
        #if id >= 100: break
        seqs_x_valid, seqs_y_valid, robust_idx = batch_valid
        robust_idx = torch.tensor(robust_idx).squeeze(0).squeeze(0).int().tolist()

        x_valid, y_valid = prepare_data_qe(seqs_x_valid, seqs_y_valid, cuda=Constants.USE_GPU)

        if flags.align_bpe_path_mt2src_test:
            align_line = align_bpe_dict_mt2src_test[str(id)]

            x_len = x_valid.size(-1) - 2
            y_len = y_valid.size(-1) - 3

            align_matrix = torch.zeros([y_len, x_len])

            for i in range(y_len):
                if str(i) in align_line:
                    align_matrix[i] = align_matrix[i].index_fill(0, torch.tensor(align_line[str(i)]), True)

            align_matrix_pad = F.pad(align_matrix, pad = (1, 1, 2, 1)).cuda()

            if add_align:
                with torch.no_grad():
                    log_probs, hter_pre = discriminator(x_valid, y_valid, align_matrix_pad = align_matrix_pad, align_ratio=flags.align_ratio, add_align=add_align)
            elif flags.add_feature:
                with torch.no_grad():
                    log_probs, hter_pre = discriminator(x_valid, y_valid, align_matrix_pad = align_matrix_pad, add_feature = True)
        else:
            with torch.no_grad():
                log_probs, hter_pre = discriminator(x_valid, y_valid, requires_adapter=False)

        log_probs = log_probs[0].cpu().tolist()
        outputs_tags_valid = []
        for i in log_probs:
            if i[1] < i[2]:
                outputs_tags_valid.append('BAD')  # 1 ok    2 bad
            else:
                outputs_tags_valid.append('OK')

        robust_outputs_tags_valid = [outputs_tags_valid[i] for i in range(len(outputs_tags_valid)) if i in robust_idx]
        
        result_tag_robust.append(robust_outputs_tags_valid)
        id += 1
        

    def auto_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    auto_mkdir('/'.join(flags.saveto_tags.split('/')[0:-1]))
    with open(flags.saveto_tags, 'w') as f:
        for vals in result_tag_robust:
            for val in vals:
                f.write(val + ' ')
            f.write('\n')


def test_qe_record_attn(flags):
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    discriminator_model_configs = configs['discriminator_configs']

    data_configs = configs['data_configs']
    #print(flags.batch_size)

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    if flags.align_bpe_path_mt2src_test:
        with open(flags.align_bpe_path_mt2src_test, 'r', encoding='utf-8') as f:
            align_bpe_dict_mt2src_test = json.load(f)

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=flags.source_path,
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=flags.target_path,
                        vocabulary=vocab_tgt,
                        )
    )

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=flags.batch_size,
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading transformer model parameters...')
    timer.tic()
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params, strict=False)

    if Constants.USE_GPU:
        discriminator.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #

    INFO('Begin ...')
    result_hter = []
    result_tag = []
    result_attn = []

    timer.tic()

    valid_iter = valid_iterator.build_generator()

    add_align = False
    if flags.add_align: add_align = True

    id = 0
    for batch_valid in tqdm(valid_iter):
        if id <= 3: 
            id += 1
            continue
        if id > 7: break
        seqs_x_valid, seqs_y_valid = batch_valid

        x_valid, y_valid = prepare_data_qe(seqs_x_valid, seqs_y_valid, cuda=Constants.USE_GPU)

        align_line = align_bpe_dict_mt2src_test[str(id)]

        x_len = x_valid.size(-1) - 2
        y_len = y_valid.size(-1) - 3

        align_matrix = torch.zeros([y_len, x_len])

        for i in range(y_len):
            if str(i) in align_line:
                align_matrix[i] = align_matrix[i].index_fill(0, torch.tensor(align_line[str(i)]), True)

        align_matrix = align_matrix.cuda()

        def auto_mkdir(path):
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

        auto_mkdir(flags.plot_path_prefix)

        # 这里要改，最好让模型返回attn，然后计算比例
        with torch.no_grad():
            log_probs, hter_pre, ctx_attn = discriminator(x_valid, y_valid, return_attn=True)

            # ctx_attn 一个列表，包括所有layer的attn: [batch_size, head_num, query_len, key_len]
            ctx_attn_all = torch.stack(ctx_attn, dim = 1) # [batch_size, layer_num, head_num, query_len, key_len]
            
            # 选择前两个attn head，作为对齐训练attn
            #ctx_attn_align_head = ctx_attn_all[:, :, :, 2:-1, 1:-1].squeeze(0)   # 去掉无关部分，<bos><eos> 这里把hter也去掉了，可能会有影响？
            ctx_attn_align_head = ctx_attn_all.squeeze(0)
            # 如果包括<bos><eos>的话，attn分数之和就是1。某些mt token还是很关注源端的bos eos的
            ctx_attn_align = ctx_attn_align_head.reshape(-1, y_len + 3, x_len + 2)   # [layer_num * choosed_head_num, seq_y, seq_x]
            ctx_attn_mean = ctx_attn_align.mean(0)

            # 画个图吧！
            plot_attribution_src(attr_token_src = ctx_attn_mean.transpose(0, 1), 
                                plot_path_prefix = flags.plot_path_prefix, 
                                mt_tokens = y_valid[0], src_tokens = x_valid[0],
                                mt_dict = vocab_tgt, src_dict = vocab_src,
                                id = id, gradients_type = "attn")
  
            #align_matrix_repeat = align_matrix.repeat(ctx_attn_align.size(0), 1, 1)
            

        hter_pre.squeeze(1).cpu().tolist()
        for val in hter_pre:
            result_hter.append(float(val))
        
        log_probs = log_probs[0].cpu().tolist()
        outputs_tags_valid = []
        outputs_okprob = []
        for i in log_probs:
            outputs_okprob.append(math.exp(i[1]))
            if i[1] < i[2]:
                outputs_tags_valid.append('BAD')  # 1 ok    2 bad
            else:
                outputs_tags_valid.append('OK')

        result_tag.append(outputs_tags_valid)
        id += 1
        

    def auto_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    auto_mkdir('/'.join(flags.saveto_hter.split('/')[0:-1]))
    with open(flags.saveto_hter, 'w') as f:
        for val in result_hter:
            f.write('%.6f\n' % val)
    
    auto_mkdir('/'.join(flags.saveto_tags.split('/')[0:-1]))
    with open(flags.saveto_tags, 'w') as f:
        for vals in result_tag:
            for val in vals:
                f.write(val + ' ')
            f.write('\n')


def test_qe_show(flags):
    """
    上边的test_qe函数要写入文件，想看结果要打开文件很麻烦；这里搞一个直接显示预测结果标签的
    """
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    discriminator_model_configs = configs['discriminator_configs']

    data_configs = configs['data_configs']
    #print(flags.batch_size)

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=flags.source_path,
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=flags.target_path,
                        vocabulary=vocab_tgt,
                        )
    )

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=flags.batch_size,
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading transformer model parameters...')
    timer.tic()
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params)

    if Constants.USE_GPU:
        discriminator.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #

    INFO('Begin ...')
    #result_hter = []
    #result_tag = []
    #result_okprob = []

    timer.tic()

    valid_iter = valid_iterator.build_generator()

    id = 0

    for batch_valid in tqdm(valid_iter):

        seqs_x_valid, seqs_y_valid = batch_valid

        x_valid, y_valid = prepare_data_qe(seqs_x_valid, seqs_y_valid, cuda=Constants.USE_GPU)

        with torch.no_grad():
            log_probs, hter_pre = discriminator(x_valid, y_valid)

        #hter_pre.squeeze(1).cpu().tolist()
        #for val in hter_pre:
        #    result_hter.append(float(val))
        
        log_probs = log_probs[0].cpu().tolist()

        outputs_tags_valid = []
        for i in log_probs:
            if i[1] < i[2]:
                outputs_tags_valid.append('BAD')  # 1 ok    2 bad
            else:
                outputs_tags_valid.append('OK')

        #result_tag.append(outputs_tags_valid)

        if id == flags.show_id:
            print("pred tag of sample %d:" % id)
            print(' '.join(outputs_tags_valid))
            break

        id += 1


def test_qe_show_onecase(flags):
    """
    从sh文件直接读取一条样例，然后直接在终端打印预测标注。
    """
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    discriminator_model_configs = configs['discriminator_configs']

    data_configs = configs['data_configs']
    #print(flags.batch_size)

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading transformer model parameters...')
    timer.tic()
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params, strict=False)

    if Constants.USE_GPU:
        discriminator.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #

    INFO('Begin ...')

    timer.tic()

    seqs_x_valid = [vocab_src.sent2ids(flags.src_sent)]
    seqs_y_valid = [vocab_tgt.sent2ids(flags.mt_sent)]

    print("mt_sent")
    print(flags.mt_sent)
    print("seqs_y_valid")
    print(seqs_y_valid)

    x_valid, y_valid = prepare_data_qe(seqs_x_valid, seqs_y_valid, cuda=Constants.USE_GPU)

    with torch.no_grad():
        log_probs, hter_pre = discriminator(x_valid, y_valid)

    log_probs = log_probs[0].cpu().tolist()

    outputs_tags_valid = []
    for i in log_probs:
        if i[1] < i[2]:
            outputs_tags_valid.append('BAD')  # 1 ok    2 bad
        else:
            outputs_tags_valid.append('OK')
    
    print("pred tag of input sample:")
    print(" ".join(outputs_tags_valid))


def logits_tsne(flags):
    """
    画图，不同情况下logit的高维投影
    """
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    discriminator_model_configs = configs['discriminator_configs']

    data_configs = configs['data_configs']
    #print(flags.batch_size)

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator_1 = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator_1.eval()
    discriminator_2 = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator_2.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading transformer model parameters...')
    timer.tic()
    params_1 = load_model_parameters(flags.pretrain_path_1, map_location="cpu")
    params_2 = load_model_parameters(flags.pretrain_path_2, map_location="cpu")
    discriminator_1.load_state_dict(params_1)
    discriminator_2.load_state_dict(params_2)

    if Constants.USE_GPU:
        discriminator_1.cuda()
        discriminator_2.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #

    INFO('Begin ...')

    timer.tic()

    source_path = data_configs['train_data'][0]
    target_path = data_configs['train_data'][1]
    tag_path = data_configs['train_data'][2]

    with open(source_path, 'r', encoding='utf-8') as fsrc, open(target_path, 'r', encoding='utf-8') as ftgt, \
        open(tag_path, 'r', encoding='utf-8') as ftag:
        src_lines = fsrc.readlines()
        mt_lines = ftgt.readlines()
        tag_lines = ftag.readlines()

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=1024)

    fig = plt.figure(figsize=(20, 20))
    ax = plt.subplot(111)
    type_x = []
    type_y = []
    for i in range(4):
        type_x.append([])
        type_y.append([])

    # ================================================== 所有token版本 ================================================== #

    num_mt_samples = len(mt_lines)

    id = 0

    for seqs_y_line, tag_line in zip(mt_lines, tag_lines):
        print(id)
        if id >= 50: break

        seqs_y_list = [vocab_tgt.sent2ids(seqs_y_line)]
        tag_list = tag_line.strip('\n').split()

        if id != num_mt_samples - 1:
            seqs_x_list = [vocab_src.sent2ids(src_lines[id]), vocab_src.sent2ids(src_lines[id + 1])]
        else:
            seqs_x_list = [vocab_src.sent2ids(src_lines[id]), vocab_src.sent2ids(src_lines[id - 1])]

        representation_list_sent = []
        for seqs_x in seqs_x_list:
            # 每种src mt排列组合
            seqs_x = [seqs_x]
            seqs_y = seqs_y_list
            x_valid, y_valid = prepare_data_qe(seqs_x, seqs_y, cuda=Constants.USE_GPU)
            with torch.no_grad():
                # 提取表示
                representation_1 = discriminator_1(x_valid, y_valid, get_representation=True).squeeze(0)[2:-1]  # [mt_seq_len, hidden_size]
                representation_2 = discriminator_2(x_valid, y_valid, get_representation=True).squeeze(0)[2:-1]  # [mt_seq_len, hidden_size]
                representation_list_sent.extend([representation_1, representation_2])
        
        representation_list_sent = (torch.stack(representation_list_sent, 0)).transpose(0, 1).cpu()  # [mt_seq_len, 4, hidden_size]

        for tsne_token_id in range(len(seqs_y[0])):
            if tag_list[tsne_token_id] == 'BAD': continue
            X_tsne = tsne.fit_transform(representation_list_sent[tsne_token_id])   # [4, 2] 降维后
            x_min, x_max = X_tsne.min(0), X_tsne.max(0)
            #data_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
            data_norm = X_tsne
            for i in range(data_norm.shape[0]):
                type_x[i].append(data_norm[i][0])
                type_y[i].append(data_norm[i][1])
        
        id += 1

    type0 = plt.scatter(type_x[0], type_y[0], s=10, c='gold', label='normal model with real src')
    type1 = plt.scatter(type_x[1], type_y[1], s=10, c='lightgreen', label='contrastive model with real src')
    type2 = plt.scatter(type_x[2], type_y[2], s=10, c='salmon', label='normal model with switch src')
    type3 = plt.scatter(type_x[3], type_y[3], s=10, c='lightskyblue', label='contrastive model with switch src')

    plt.legend()
    plt.savefig('/home/yanym/data/contrastive_check/only_ok/tsne.jpg')

    # ================================================== 单个token版本 ================================================== #
    search_token = "钢琴"

    id = 0

    for seqs_y_line, tag_line in zip(mt_lines, tag_lines):
        if id == 6999: break
        #print(id)
        seqs_y_list = seqs_y_line.strip('\n').split()
        if search_token not in seqs_y_list: continue
        tag_list = tag_line.strip('\n').split()
        seqs_x_list = [vocab_src.sent2ids(src_lines[id]), vocab_src.sent2ids(src_lines[id + 1])]

        representation_list_sent = []
        for seqs_x in seqs_x_list:
            # TODO
            pass


def test_qe_knn_hter(flags):
    # 用knn来预测hter
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1   # 暂且，如果flags.batch_size不是1的话，词标签就不准

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    discriminator_model_configs = configs['discriminator_configs']

    data_configs = configs['data_configs']
    training_configs = configs['training_configs']
    #print(flags.batch_size)

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=flags.source_path,
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=flags.target_path,
                        vocabulary=vocab_tgt,
                        )
    )

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=flags.batch_size,
                                  use_bucket=False, numbering=False)

    qe_train_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3]),
        QEHterDataset(data_path=data_configs['train_data'][4])
    )

    qe_train_iterator = DataIterator(dataset=qe_train_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'])

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading transformer model parameters...')
    timer.tic()
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params)

    if Constants.USE_GPU:
        discriminator.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #
    INFO('load faiss index')
    
    faiss_index_prefix = flags.faiss_index_prefix
    qe_train_faiss_index_path = faiss_index_prefix + "qe_train_hter.faiss"
    faiss_index = faiss_utils.FAISSIndex(256, "Flat")
    faiss_index.load(qe_train_faiss_index_path)

    # ================================================================================== #

    INFO('Begin ...')
    k = flags.k
    temperature = flags.temperature
    lambda_knn = flags.lambda_knn
    # result_hter = []
    result_hter_knn = []   # 所有样本的knn_hter
    result_hter_model = [] # 所有样本的模型预测hter
    result_tag = []
    result_okprob = []

    timer.tic()

    valid_iter = valid_iterator.build_generator()

    for batch_valid in tqdm(valid_iter):   # TODO batch_size目前=1，比较慢
        # knn预测
        test_feature = cal_hter_logits(discriminator,
                                        batch_valid,
                                        gpu=flags.use_gpu)  # test_feature: [batch_size, d_model]
        test_feature = test_feature.cpu().detach().numpy()

        KNN_distances, KNN_indices = faiss_index.search(          # KNN_indices就是那些knn选中的最相似的训练样本们
            k=k, queries=test_feature)                    # KNN_distances or KNN_indices: [batch_size, k]

        # 把这几个最近邻训练样本的hter取出来，用距离softmax加权，得到knn预测hter
        gold_hter_list = []
        for id in KNN_indices[0]:                           # TODO 针对batch=1的慢方法。。。
            knn_sample = qe_train_dataset.showitem(id)
            gold_hter = float(knn_sample[3].strip('\n'))
            gold_hter_list.append(gold_hter)
        gold_hter_list = torch.tensor(gold_hter_list)
        dd = torch.tensor(-KNN_distances.squeeze(0) / temperature)
        dd = torch.nn.functional.softmax(dd, dim=-1)
        knn_hter = torch.dot(dd, gold_hter_list)
        result_hter_knn.append(knn_hter.item())

        # 模型预测
        seqs_x_valid, seqs_y_valid = batch_valid

        x_valid, y_valid = prepare_data_qe(seqs_x_valid, seqs_y_valid, cuda=Constants.USE_GPU)

        with torch.no_grad():
            log_probs, hter_pre = discriminator(x_valid, y_valid)

        hter_pre.squeeze(1).cpu().tolist()
        for val in hter_pre:
            result_hter_model.append(float(val))
        

        """
        log_probs = log_probs[0].cpu().tolist()
        outputs_tags_valid = []
        outputs_okprob = []
        for i in log_probs:
            outputs_okprob.append(math.exp(i[1]))
            if i[1] < i[2]:
                outputs_tags_valid.append('BAD')  # 1 ok    2 bad
            else:
                outputs_tags_valid.append('OK')

        result_tag.append(outputs_tags_valid)
        result_okprob.append(outputs_okprob)
        """
    
    # 融合预测结果
    #print(result_hter_knn[:10])
    #print(result_hter_model[:10])
    result_hter = lambda_knn * np.array(result_hter_knn) + (1 - lambda_knn) * np.array(result_hter_model)
        

    def auto_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    auto_mkdir('/'.join(flags.saveto_hter.split('/')[0:-1]))
    with open(flags.saveto_hter, 'w') as f:
        for val in result_hter:
            f.write('%.6f\n' % val)
    """
    auto_mkdir('/'.join(flags.saveto_tags.split('/')[0:-1]))
    with open(flags.saveto_tags, 'w') as f:
        for vals in result_tag:
            for val in vals:
                f.write(val + ' ')
            f.write('\n')

    if flags.saveto_okprob:
        auto_mkdir('/'.join(flags.saveto_okprob.split('/')[0:-1]))
        with open(flags.saveto_okprob, 'w') as f:
            for vals in result_okprob:
                for val in vals:
                    f.write(str(val) + ' ')
                f.write('\n')
    """


def test_qe_knn_token(flags):
    # 用knn来预测token
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1   # 暂且，如果flags.batch_size不是1的话，词标签就不准

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    discriminator_model_configs = configs['discriminator_configs']

    data_configs = configs['data_configs']
    training_configs = configs['training_configs']
    #print(flags.batch_size)

    timer = Timer()

    # ============================================ 读取数据 ============================================ #

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    # 测试集
    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=flags.source_path,
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=flags.target_path,
                        vocabulary=vocab_tgt,
                        )
    )

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=flags.batch_size,
                                  use_bucket=False, numbering=False)

    # qe训练集
    qe_train_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3]),
        QEHterDataset(data_path=data_configs['train_data'][4])
    )

    qe_train_iterator = DataIterator(dataset=qe_train_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'])

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ============================================ 加载模型 ============================================ #
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading transformer model parameters...')
    timer.tic()
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params)

    if Constants.USE_GPU:
        discriminator.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ============================================ faiss路径整理 ============================================ #
    #INFO('load faiss index')
    
    faiss_index_prefix = flags.faiss_index_prefix
    qe_train_tag_faiss_path = faiss_index_prefix + "qe_train_tag.faiss_10w"
    qe_train_tag_id_sent_path = faiss_index_prefix + "qe_train_tag_id_sent.tensor"
    qe_train_tag_id_token_path = faiss_index_prefix + "qe_train_tag_id_token.tensor"
    #faiss_index = faiss_utils.FAISSIndex(256, "Flat")
    #faiss_index.load(qe_train_faiss_index_path)

    # ============================================ knn预测 词级别 ============================================ #

    INFO('Begin knn prediction')
    k = flags.k                       # knn的k，选多少个近邻
    temperature = flags.temperature   # softmax的温度
    lambda_knn = flags.lambda_knn     # knn预测 和 模型预测 的权重比例

    result_tag_knn = []               # 所有样本的knn_tag
    result_tag_model = []             # 所有样本的模型预测tag
    #result_tag = []                  # TODO 这俩不知道做什么的
    #result_okprob = []

    timer.tic()
    valid_iter = valid_iterator.build_generator()

    # ok bad 标签和数值互转
    def tag2num(tag):
        # ok 1     bad 0
        if tag == 'BAD': return 0
        else: return 1
    def num2tag(num):
        if num >= 0.5: return 'OK'
        else: return 'BAD'

    sents_knn_tag = []                     # 存放所有样本的knn预测标签
    for batch_valid in tqdm(valid_iter):   # TODO batch_size目前=1，比较慢
        # 取出模型生成的词logits
        hter_feature, tags_feature = get_feature(discriminator,
                                                batch_valid,
                                                gpu=flags.use_gpu)
        tags_feature = tags_feature.cpu().detach().numpy() # [seq_len, hidden_dim]

        # qe训练数据搜索
        faiss_index = faiss.read_index(qe_train_tag_faiss_path)
        KNN_distances, KNN_indices = faiss_index.search(tags_feature, k=k)  # [seq_len, k] 每个token搜索到k个近邻
        
        sent_knn_tag = []  # 一个样本的knn预测标签
        for token_dis, token_ids in zip(KNN_distances, KNN_indices):        # 依次关注该样本中的每个token所找到的k近邻
            # 得到一个token的k个最近邻token的gold_tag们
            gold_tag_list = []
            for id in token_ids:  # 该token的每个最近邻
                # 这个id就是在全训练数据集token列表中的id，需要转换成对应训练集里哪条样本的哪个token
                tag_id_sent = torch.load(faiss_index_prefix + qe_train_tag_id_sent_path)
                tag_id_token = torch.load(faiss_index_prefix + qe_train_tag_id_token_path)
                sent_id = int(tags_sent_id_list[id].item())                 # 找到该token对应sent id
                token_id = int(tags_token_id_list[id].item())               # 找到该token对应sent中的token id
                knn_sample = qe_train_dataset.showitem(sent_id)             # 通过sent id找到该样本
                token_chosen_by_knn = knn_sample[1].split()[token_id]       # 检索到的这个近邻token是什么
                gold_tag = tag2num(knn_sample[2].split()[token_id])         # 检索到的这个近邻token的gold_tag数字化，ok1 bad0
                gold_tag_list.append(gold_tag)
            gold_tag_list = torch.tensor(gold_tag_list, dtype=torch.float)  # 存放该token的k个最近邻token的gold_tag

            dd = torch.tensor(-token_dis / temperature)      # 根据温度调节相似度，并做softmax得到相似度归一化向量
            dd = torch.nn.functional.softmax(dd, dim=-1)
            token_knn_tag_num = torch.dot(dd, gold_tag_list) # 用相似度给每个最近邻gold_tag加权，得到该token的knn预测
            token_knn_tag = num2tag(token_knn_tag_num)
            #print(token_knn_tag)
            sent_knn_tag.append(token_knn_tag)
            #print(sent_knn_tag)

        sents_knn_tag.append(sent_knn_tag)

        #assert 1==2


        # 模型预测
        """
        seqs_x_valid, seqs_y_valid = batch_valid

        x_valid, y_valid = prepare_data_qe(seqs_x_valid, seqs_y_valid, cuda=Constants.USE_GPU)

        with torch.no_grad():
            log_probs, hter_pre = discriminator(x_valid, y_valid)
        """
        """
        log_probs = log_probs[0].cpu().tolist()
        outputs_tags_valid = []
        outputs_okprob = []
        for i in log_probs:
            outputs_okprob.append(math.exp(i[1]))
            if i[1] < i[2]:
                outputs_tags_valid.append('BAD')  # 1 ok    2 bad
            else:
                outputs_tags_valid.append('OK')

        result_tag.append(outputs_tags_valid)
        result_okprob.append(outputs_okprob)
        """
    
    # 融合预测结果
    #print(result_hter_knn[:10])
    #print(result_hter_model[:10])
    #result_hter = lambda_knn * np.array(result_hter_knn) + (1 - lambda_knn) * np.array(result_hter_model)
        

    # ============================================ 将预测结果写入文件 ============================================ #

    def auto_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    auto_mkdir('/'.join(flags.saveto_tags.split('/')[0:-1]))
    with open(flags.saveto_tags, 'w') as f:
        for vals in sents_knn_tag:
            for val in vals:
                f.write(val + ' ')
            f.write('\n')

    print('Finale.')


def attribution_qe_origin(flags):
    """
    计算test输入对预测结果影响的梯度，并做分析的函数，可以选择何种梯度(GI,IG)
    这个origin版本包括一些古早分析比如pos
    """
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    discriminator_model_configs = configs['discriminator_configs']

    data_configs = configs['data_configs']

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['test_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['test_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['test_data'][2]),
        QEHterDataset(data_path=data_configs['test_data'][3])
    )

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=flags.batch_size,
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading transformer model parameters...')
    timer.tic()
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params)

    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #

    INFO('Begin ...')

    timer.tic()

    valid_iter = valid_iterator.build_generator()

    # 读入pos文件
    """
    pos_file = "/home/user_data_182b/yanym/qe/data/wmt19_ende/qe_data/test.pos.BPE"
    vocab_pos_file = "/home/user_data_182b/yanym/qe/data/wmt19_ende/vocab.pos.json"

    pos_data = []
    with open(pos_file, 'r', encoding='utf-8') as f, open(vocab_pos_file, 'r', encoding='utf-8') as vocab_pos:
        vocab_pos = json.load(vocab_pos)
        lines = f.readlines()
        for line in lines:
            pos_line = line.strip('\n').split(' ')
            pos_data_i = [vocab_pos[w][0] for w in pos_line]
            pos_data.append(pos_data_i)
    """

    all_attr_sent_mt_ratio = []
    all_attr_token_mt_ratio = []
    top3src_sent_cnt = 0
    top3mt_sent_cnt = 0
    top3src_token_cnt = 0
    top3mt_token_cnt = 0

    id = 0

    # 不想重跑所以打表
    # gradients_table = []
    # tag_pred_table = []

    for batch_valid in valid_iter:
        # 每个样本
        print(id)
        """
        if id < flags.test_id: 
            id += 1
            continue
        if id > flags.test_id: assert 1==2
        """
        if id > 0: break
        
        seqs_x_valid, seqs_y_valid, xy_label, xy_hter = batch_valid

        x_valid, y_valid, xy_label = prepare_data_qe(seqs_x_valid, seqs_y_valid, xy_label, cuda=Constants.USE_GPU)
        if Constants.USE_GPU:
            xy_hter = torch.tensor(xy_hter)
            xy_hter = xy_hter.cuda()

        if flags.gradients_type == 'g':
            cal_gradients_func = only_gradients
            cal_gradients_func_srcmt = only_gradients_srcmt
        elif flags.gradients_type == 'ig':
            cal_gradients_func = integrated_gradients

        # tag_pred [seq_len] 0ok 1bad
        """
        tag_pred = cal_prediction(discriminator, 
                                    critic,
                                    critic_mse,
                                    x_valid, 
                                    y_valid, 
                                    xy_label, 
                                    xy_hter) 
        tag_real = xy_label.squeeze(0)[2:-1]-1
        """

        plot_path_prefix = flags.plot_path_prefix
        if not os.path.exists(plot_path_prefix):
            os.makedirs(plot_path_prefix)

        
        # attr_sent_mt [mt_seq_len]
        # attr_token_mt [mt_seq_len, mt_seq_len] 
        # attr_sent_src [src_seq_len]
        # attr_token_src [src_seq_len, mt_seq_len]
        """
        attr_sent_mt, attr_token_mt, \
            attr_sent_src, attr_token_src = cal_gradients_func(discriminator, x_valid, y_valid) 
        
        # ========================================================= 分析1：画图
        
        plot_attribution4(attr_sent_mt=attr_sent_mt, 
                            attr_token_mt=attr_token_mt, 
                            attr_sent_src=attr_sent_src, 
                            attr_token_src=attr_token_src,
                            plot_path_prefix=plot_path_prefix, 
                            mt_tokens=y_valid[0][2:-1], 
                            src_tokens=x_valid[0][1:-1], 
                            mt_dict=vocab_tgt, 
                            src_dict=vocab_src, 
                            id=id, 
                            gradients_type=flags.gradients_type)
        """
        
        

        # 补充：src+mt一起，对于mt预测的影响
        # attr_sent_srcmt [src_seq_len + mt_seq_len]
        # attr_token_srcmt [src_seq_len + mt_seq_len, mt_seq_len]
        # attr_sent_mt_ratio [1]
        # attr_token_mt_ratio [mt_seq_len]
        attr_sent_srcmt, attr_token_srcmt, attr_sent_mt_ratio, attr_token_mt_ratio, \
            top3src_sent, top3mt_sent, top3mt_token, top3src_token = cal_gradients_func_srcmt(discriminator, x_valid, y_valid) 
        
        plot_attribution_srcmt(attr_sent_srcmt=attr_sent_srcmt, 
                                attr_token_srcmt=attr_token_srcmt,
                                plot_path_prefix=plot_path_prefix, 
                                mt_tokens=y_valid[0][2:-1], 
                                src_tokens=x_valid[0][1:-1], 
                                mt_dict=vocab_tgt, 
                                src_dict=vocab_src, 
                                id=id, 
                                gradients_type=flags.gradients_type)
        #assert 1==2
        
        all_attr_sent_mt_ratio.append(attr_sent_mt_ratio)
        all_attr_token_mt_ratio.append(attr_token_mt_ratio)
        """
        top3src_sent_cnt += top3src_sent
        top3mt_sent_cnt += top3mt_sent
        top3src_token_cnt += top3src_token
        top3mt_token_cnt += top3mt_token
        """

        id += 1
        #if id == 1: break
        #continue
        #assert 1==2
        

        # 按句子长度加权
        """
        attr_sent_mt_weighted = attr_sent_mt * attr_sent_mt.size(0)
        attr_token_mt_weighted = attr_token_mt * attr_token_mt.size(0)  
        attr_sent_src_weighted = attr_sent_src * attr_sent_src.size(0)
        attr_token_src_weighted = attr_token_src * attr_token_src.size(0)  
        """

        # ========================================================= 分析2：测试ok bad集中于当前词的程度
        """
        diag = torch.tensor([attr_token_mt_weighted[i][i] for i in range(attr_token_mt_weighted.size(-1))])
        # diag [mt_len] 当前mt token对当前标签的影响大小
        # 对比diag和okbad标签的相关性：平均值or互信息

        # 平均值统计：ok bad的重要性集中程度均值是否有差异
        if id == 1:
            tag_att_real, tag_count_real = tag_x_att(diag, tag_real, 2)
            tag_att_pred, tag_count_pred = tag_x_att(diag, tag_pred, 2)
        else:
            x, y= tag_x_att(diag, tag_real, 2)
            tag_att_real += x
            tag_count_real += y
            x, y= tag_x_att(diag, tag_pred, 2)
            tag_att_pred += x
            tag_count_pred += y

        # 互信息计算：ok bad标签向量 和 词重要性集中程度向量 的互信息有多大
        
        att_i = diag.cpu().detach().numpy()
        tag_real = tag_real.cpu().detach().numpy()
        tag_pred = tag_pred.cpu().detach().numpy()

        # tag-mi互信息
        x, y, z = cal_info(metrics.mutual_info_score, att_i, tag_real, tag_pred)
        if id == 1:
            tag_mi_real = x
            tag_mi_pred = y
            tag_mi_tag = z
        else:
            tag_mi_real += x
            tag_mi_pred += y
            tag_mi_tag += z

        # tag-ami互信息
        x, y, z = cal_info(metrics.adjusted_mutual_info_score, att_i, tag_real, tag_pred)
        if id == 1:
            tag_ami_real = x
            tag_ami_pred = y
            tag_ami_tag = z
        else:
            tag_ami_real += x
            tag_ami_pred += y
            tag_ami_tag += z
        
        # tag-nmi互信息
        x, y, z = cal_info(metrics.normalized_mutual_info_score, att_i, tag_real, tag_pred)
        if id == 1:
            tag_nmi_real = x
            tag_nmi_pred = y
            tag_nmi_tag = z
        else:
            tag_nmi_real += x
            tag_nmi_pred += y
            tag_nmi_tag += z

        

        # ========================================================= 分析3：【ok bad】和【mt token 对句子分数贡献】的相关性
        
        # a 平均值统计：ok bad 的重要性均值是否有差异
        if id == 1:
            tag_att_real, tag_count_real = tag_x_att(attr_sent_mt_weighted, tag_real, 2)
            tag_att_pred, tag_count_pred = tag_x_att(attr_sent_mt_weighted, tag_pred, 2)
        else:
            x, y= tag_x_att(attr_sent_mt_weighted, tag_real, 2)
            tag_att_real += x
            tag_count_real += y
            x, y= tag_x_att(attr_sent_mt_weighted, tag_pred, 2)
            tag_att_pred += x
            tag_count_pred += y

        # b 互信息计算：ok bad标签向量 和 词重要性向量 的互信息有多大
        
        att_i = attr_sent_mt_weighted.cpu().detach().numpy()
        tag_real = tag_real.cpu().detach().numpy()
        tag_pred = tag_pred.cpu().detach().numpy()

        # tag-mi互信息
        x, y, z = cal_info(metrics.mutual_info_score, att_i, tag_real, tag_pred)
        if id == 1:
            tag_mi_real = x
            tag_mi_pred = y
            tag_mi_tag = z
        else:
            tag_mi_real += x
            tag_mi_pred += y
            tag_mi_tag += z


        # tag-ami互信息
        x, y, z = cal_info(metrics.adjusted_mutual_info_score, att_i, tag_real, tag_pred)
        if id == 1:
            tag_ami_real = x
            tag_ami_pred = y
            tag_ami_tag = z
        else:
            tag_ami_real += x
            tag_ami_pred += y
            tag_ami_tag += z
        
        # tag-nmi互信息
        x, y, z = cal_info(metrics.normalized_mutual_info_score, att_i, tag_real, tag_pred)
        if id == 1:
            tag_nmi_real = x
            tag_nmi_pred = y
            tag_nmi_tag = z
        else:
            tag_nmi_real += x
            tag_nmi_pred += y
            tag_nmi_tag += z


        # ========================================================= 分析4：【pos标签】和【mt token 对句子分数贡献】的相关性
        pos_line = torch.tensor(pos_data[id-1]).cuda()

        # 平均值统计：不同pos tag的重要性均值是否有差异
        if id == 1:
            pos_att, pos_count = tag_x_att(attr_sent_mt_weighted, pos_line, 16)
        else:
            x, y = tag_x_att(attr_sent_mt_weighted, pos_line, 16)
            pos_att += x
            pos_count += y

        # 互信息计算：pos向量 和 词重要性向量 的互信息有多大
        pos_line = pos_line.cpu().detach().numpy()

        # pos-mi互信息
        if id == 1:
            pos_mi = cal_info(metrics.mutual_info_score, att_i, pos_line)
        else:
            pos_mi += cal_info(metrics.mutual_info_score, att_i, pos_line)

        # pos-ami互信息
        if id == 1:
            pos_ami = cal_info(metrics.adjusted_mutual_info_score, att_i, pos_line)
        else:
            pos_ami += cal_info(metrics.adjusted_mutual_info_score, att_i, pos_line)

        # pos-nmi互信息
        if id == 1:
            pos_nmi = cal_info(metrics.normalized_mutual_info_score, att_i, pos_line)
        else:
            pos_nmi += cal_info(metrics.normalized_mutual_info_score, att_i, pos_line)
        

        id += 1
        #if id > 20:
        #    break
        """
    all_attr_sent_mt_ratio = torch.cat(all_attr_sent_mt_ratio, 0)
    all_attr_token_mt_ratio = torch.cat(all_attr_token_mt_ratio, 0)
    avg_attr_sent_mt_ratio = torch.mean(all_attr_sent_mt_ratio, dim=0)
    avg_attr_token_mt_ratio = torch.mean(all_attr_token_mt_ratio, dim=0)
    print('avg_attr_sent_mt_ratio: %.6f' % avg_attr_sent_mt_ratio.item())
    print('avg_attr_token_mt_ratio %.6f' % avg_attr_token_mt_ratio.item())
    """
    sent_top3mt_ratio = top3mt_sent_cnt / (top3src_sent_cnt + top3mt_sent_cnt)
    print('sent_top3mt_ratio %.6f' % sent_top3mt_ratio)
    token_top3mt_ratio = top3mt_token_cnt / (top3src_token_cnt + top3mt_token_cnt)
    print('token_top3mt_ratio %.6f' % token_top3mt_ratio)
    """


    """
    # ok bad 结果
    print("分析3：【ok bad】和【mt token 对句子分数贡献】的相关性")
    tag_avg_real = torch.mul(tag_att_real, 1/tag_count_real)
    tag_avg_pred = torch.mul(tag_att_pred, 1/tag_count_pred)
    print("tag_avg")
    print(tag_avg_real)
    print(tag_avg_pred)

    tag_mi_real_avg = tag_mi_real / id
    tag_mi_pred_avg = tag_mi_pred / id
    tag_mi_tag_avg = tag_mi_tag / id
    print("tag_mi")
    print(tag_mi_real_avg)
    print(tag_mi_pred_avg)
    print(tag_mi_tag_avg)

    tag_ami_real_avg = tag_ami_real / id
    tag_ami_pred_avg = tag_ami_pred / id
    tag_ami_tag_avg = tag_ami_tag / id
    print("tag_ami")
    print(tag_ami_real_avg)
    print(tag_ami_pred_avg)
    print(tag_ami_tag_avg)
    
    tag_nmi_real_avg = tag_nmi_real / id
    tag_nmi_pred_avg = tag_nmi_pred / id
    tag_nmi_tag_avg = tag_nmi_tag / id
    print("tag_nmi")
    print(tag_nmi_real_avg)
    print(tag_nmi_pred_avg)
    print(tag_nmi_tag_avg)
    
    # pos 结果
    print("分析4：【pos标签】和【mt token 对句子分数贡献】的相关性")
    pos_avg = torch.mul(pos_att, 1/pos_count)
    print("pos_avg")
    print(pos_avg)

    pos_mi_avg = pos_mi / id
    pos_ami_avg = pos_ami / id
    pos_nmi_avg = pos_nmi / id
    print("pos_mi / ami / nmi")
    print(pos_mi_avg)
    print(pos_ami_avg)
    print(pos_nmi_avg)
    """


def attribution_qe_whole(flags):
    """
    计算test输入对预测结果影响的梯度，并做分析的函数，可以选择何种梯度(GI,IG)
    和origin相比，这个是light版本，把暂时用不到的注释删掉了
    """
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    discriminator_model_configs = configs['discriminator_configs']

    data_configs = configs['data_configs']

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['test_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['test_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['test_data'][2]),
        QEHterDataset(data_path=data_configs['test_data'][3]),
    )

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=flags.batch_size,
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading transformer model parameters...')
    timer.tic()
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params)
    if Constants.USE_GPU:
        discriminator.cuda()

    # ================================================================================== #

    INFO('Begin ...')

    timer.tic()

    valid_iter = valid_iterator.build_generator()

    if flags.gradients_type == 'g':
        cal_gradients_func = one_gradients
        cal_gradients_func_srcmt = one_gradients_srcmt
    elif flags.gradients_type == 'ig':
        cal_gradients_func = integrated_gradients

    plot_path_prefix = flags.plot_path_prefix
    if not os.path.exists(plot_path_prefix):
        os.makedirs(plot_path_prefix)
    
    with open(flags.align_bpe_path_src2mt, 'r', encoding='utf-8') as falign_s2m, \
        open(flags.align_bpe_path_mt2src, 'r', encoding='utf-8') as falign_m2s:
        align_bpe_dict_src2mt = json.load(falign_s2m)
        align_bpe_dict_mt2src = json.load(falign_m2s)
    
    all_avg_attr_sent_mt_ratio = []
    all_avg_attr_token_mt_ratio = []
    all_max_attr_sent_mt_ratio = []
    all_max_attr_token_mt_ratio = []

    right_src_align_sum = []
    right_src_context_sum = []
    wrong_src_align_sum = []
    wrong_src_context_sum = []
    
    align_pred_lines = []

    top_k = 1
    topk_src_sent_cnt = 0
    topk_mt_sent_cnt = 0
    topk_src_token_cnt = 0
    topk_mt_token_cnt = 0

    all_attr_src_align_mean = []
    all_attr_src_context_mean = []
    all_attr_mt_self_mean = []
    all_attr_mt_context_mean = []
    all_attr_src_align_ratio = []
    all_attr_src_context_ratio = []
    all_attr_mt_self_ratio = []
    all_attr_mt_context_ratio = []
    k_adj = 1
    all_attr_mt_this_adj_mean = []
    all_attr_mt_this_remote_mean = []

    id = 0

    for batch_valid in valid_iter:
        # 每个样本
        """
        if(id <= 1): 
            id += 1
            continue
        if(id > 3): break
        """
        print(id)

        """
        if id < flags.test_id: 
            id += 1
            continue
        if id > flags.test_id: assert 1==2
        """
        
        seqs_x_valid, seqs_y_valid, xy_label, xy_hter = batch_valid

        x_valid, y_valid, xy_label = prepare_data_qe(seqs_x_valid, seqs_y_valid, xy_label, cuda=Constants.USE_GPU)
        if Constants.USE_GPU:
            xy_hter = torch.tensor(xy_hter)
            xy_hter = xy_hter.cuda()
        
        #print(y_valid[0][2:-1])

        
        # ============================== 梯度计算 ============================== #
        # ====== src或mt，对于mt预测的影响
        # attr_sent_mt [mt_seq_len]
        # attr_token_mt [mt_seq_len, mt_seq_len] 
        # attr_sent_src [src_seq_len]
        # attr_token_src [src_seq_len, mt_seq_len]
        attr_sent_mt, attr_token_mt, \
            attr_sent_src, attr_token_src = cal_gradients_func(discriminator, x_valid, y_valid)
        
        # ====== src+mt一起，对于mt预测的影响
        # attr_sent_srcmt [src_seq_len + mt_seq_len]
        # attr_token_srcmt [src_seq_len + mt_seq_len, mt_seq_len]
        # attr_sent_mt_ratio [1]
        # attr_token_mt_ratio [mt_seq_len]
        attr_sent_srcmt, attr_token_srcmt = cal_gradients_func_srcmt(discriminator, x_valid, y_valid)


        # ============================== 画图 ============================== #
        """
        plot_attribution_all(attr_sent_mt=attr_sent_mt, 
                            attr_token_mt=attr_token_mt, 
                            attr_sent_src=attr_sent_src, 
                            attr_token_src=attr_token_src,
                            attr_sent_srcmt=attr_sent_srcmt,
                            attr_token_srcmt=attr_token_srcmt,
                            plot_path_prefix=plot_path_prefix, 
                            mt_tokens=y_valid[0][2:-1], 
                            src_tokens=x_valid[0][1:-1], 
                            mt_dict=vocab_tgt, 
                            src_dict=vocab_src, 
                            id=id, 
                            gradients_type=flags.gradients_type,
                            lang='enzh')
        """

        # ============================== 分析函数计算 ============================== #
        def cal_attr_ratio(attr_sent_srcmt, attr_token_srcmt, x_valid, top_k):
            src_len = x_valid.size(-1) - 2

            # mt对句子分数贡献比例 [1]
            attr_sent_src_part = attr_sent_srcmt[:src_len]
            attr_sent_mt_part = attr_sent_srcmt[src_len:]

            avg_sent_src = torch.mean(attr_sent_src_part)
            avg_sent_mt = torch.mean(attr_sent_mt_part)
            avg_attr_sent_mt_ratio = torch.tensor([avg_sent_mt / (avg_sent_src + avg_sent_mt)])  # [1]

            max_sent_src, _ = torch.max(attr_sent_src_part, dim=0)
            max_sent_mt, _ = torch.max(attr_sent_mt_part, dim=0)
            max_attr_sent_mt_ratio = torch.tensor([max_sent_mt / (max_sent_src + max_sent_mt)])  # [1]

            # mt对词标签贡献比例 [mt_seq_len]
            attr_token_src_part = attr_token_srcmt[:src_len]
            attr_token_mt_part = attr_token_srcmt[src_len:]

            avg_token_src = torch.mean(attr_token_src_part, dim = 0)  # [mt_seq_len] 每个位置是 平均每个src token对该mt token的贡献
            avg_token_mt = torch.mean(attr_token_mt_part, dim = 0)  # [mt_seq_len]
            avg_attr_token_mt_ratio = avg_token_mt / (avg_token_src + avg_token_mt)  # [mt_seq_len]

            max_token_src, _ = torch.max(attr_token_src_part, dim = 0)  # [mt_seq_len] 每个位置是 最大的src token对该mt token的贡献
            max_token_mt, _ = torch.max(attr_token_mt_part, dim = 0)  # [mt_seq_len]
            max_attr_token_mt_ratio = max_token_mt / (max_token_src + max_token_mt)  # [mt_seq_len]

            # 句子级：前k大
            _, pred = attr_sent_srcmt.topk(top_k, 0)
            pred = pred >= src_len
            topk_mt_sent = torch.sum(pred)   # [1] 对句子分数贡献最大的token是来自mt吗
            topk_src_sent = pred.numel() - topk_mt_sent

            # 词级：前k大
            _, pred = attr_token_srcmt.topk(top_k, 0)
            pred = pred >= src_len
            topk_mt_token = torch.sum(pred)   # [1] 对这句话的每个mt token影响最大的token 是来自于mt的数量
            topk_src_token = pred.numel() - topk_mt_token   # [1] 对这句话的每个mt token影响最大的token 是来自于src的数量

            # 总计
            nonlocal all_avg_attr_sent_mt_ratio, all_avg_attr_token_mt_ratio, \
                all_max_attr_sent_mt_ratio, all_max_attr_token_mt_ratio, \
                topk_src_sent_cnt, topk_mt_sent_cnt, topk_src_token_cnt, topk_mt_token_cnt

            all_avg_attr_sent_mt_ratio.append(avg_attr_sent_mt_ratio)
            all_avg_attr_token_mt_ratio.append(avg_attr_token_mt_ratio)
            all_max_attr_sent_mt_ratio.append(max_attr_sent_mt_ratio)
            all_max_attr_token_mt_ratio.append(max_attr_token_mt_ratio)

            topk_src_sent_cnt += topk_src_sent
            topk_mt_sent_cnt += topk_mt_sent
            topk_src_token_cnt += topk_src_token
            topk_mt_token_cnt += topk_mt_token

        def cal_word_alignment_attr_ratio(align_bpe_dict_mt2src, id, y_valid, attr_token_src, acc_sign):
            nonlocal right_src_align_sum, wrong_src_align_sum, \
                right_src_context_sum, wrong_src_context_sum
            align_bpe_dict_line = align_bpe_dict_mt2src[str(id)]  # 这句话中每个mt token对齐的src token们
            mt_len = attr_token_srcmt.size(-1)
            mt_temp = y_valid[0][2:-1]
            for i in range(mt_len):
                if mt_temp[i] == 3: continue
                attr_src_this = attr_token_src[:, i]  # 所有src_token对第i个mt_token的贡献
                if str(i) in align_bpe_dict_line:  # 如果这个mt token有对齐的src token的话
                    # 对齐的src与上下文src贡献
                    src_align_status = torch.zeros_like(attr_src_this).index_fill(0, torch.tensor(align_bpe_dict_line[str(i)]).cuda(), True)  # 长度同src，01向量代表每个src token是否是第i个mt token的对齐token
                    src_idx_align = torch.where(src_align_status == 1)[0]    # 和第i个mt token对齐的src token索引
                    src_idx_context = torch.where(src_align_status == 0)[0]  # 第i个mt token的上下文src token索引
                    attr_src_this_align = torch.index_select(attr_src_this, 0, src_idx_align) # [length of align src of this mt token] 和第i个mt token对齐的src token的贡献
                    attr_src_this_context = torch.index_select(attr_src_this, 0, src_idx_context) # [length of context src of this mt token] 第i个mt token的上下文src token的贡献
                    attr_src_this_align_sum = torch.sum(attr_src_this_align)     # [1] 和第i个mt token对齐的src token的贡献之和
                    attr_src_this_context_sum = torch.sum(attr_src_this_context) # [1] 第i个mt token的上下文src token的贡献之和
                    if acc_sign[0][i] == '1':   # 这个token预测对了
                        right_src_align_sum.append(attr_src_this_align_sum.item())  # 总计
                        right_src_context_sum.append(attr_src_this_context_sum.item())
                    else:                       # 这个token预测错了
                        wrong_src_align_sum.append(attr_src_this_align_sum.item())
                        wrong_src_context_sum.append(attr_src_this_context_sum.item())

        def cal_word_alignment_top(attr_token_src, y_valid):
            nonlocal align_pred_lines
            value, most_attr_src_index_include_unk = torch.max(attr_token_src, dim=0)
            most_attr_src_index_include_unk = most_attr_src_index_include_unk.tolist()
            mt_temp = y_valid[0][2:-1]
            most_attr_src_index = [most_attr_src_index_include_unk[ii] if mt_temp[ii] != 3 else -1 for ii in range(len(mt_temp))] # 对齐-1表示unk
            align_pred_lines.append(most_attr_src_index)
        
        def cal_attr_different_part(align_bpe_dict, id, x_valid, y_valid, attr_token_srcmt, k_adj):
            nonlocal all_attr_src_align_mean, all_attr_src_context_mean, \
                    all_attr_mt_self_mean, all_attr_mt_context_mean, \
                    all_attr_mt_this_adj_mean, all_attr_mt_this_remote_mean, \
                    all_attr_src_align_ratio, all_attr_src_context_ratio, \
                    all_attr_mt_self_ratio, all_attr_mt_context_ratio

            align_bpe_dict_line = align_bpe_dict[str(id)]  # 这句话中每个mt token对齐的src token们

            src_len = x_valid.size(-1) - 2
            mt_len = attr_token_srcmt.size(-1)
            mt_temp = y_valid[0][2:-1]

            attr_token_src_part = attr_token_srcmt[:src_len]  # [src_len, mt_len]
            attr_token_mt_part = attr_token_srcmt[src_len:]  # [mt_len, mt_len]

            for i in range(mt_len):
                if mt_temp[i] == 3: continue
                attr_src_this = attr_token_src_part[:, i]  # 所有src_token对第i个mt_token的贡献
                attr_mt_this = attr_token_mt_part[:, i]  # 所有mt_token对第i个mt_token的贡献
                attr_mt_this_reverse = attr_mt_this.flip(dims = [0])

                if str(i) in align_bpe_dict_line:  # 如果这个mt token有对齐的src token的话
                    # 对齐的src与上下文src贡献
                    src_align_status = torch.zeros_like(attr_src_this).index_fill(0, torch.tensor(align_bpe_dict_line[str(i)]).cuda(), True)  # 长度同src，01向量代表每个src token是否是第i个mt token的对齐token
                    src_idx_align = torch.where(src_align_status == 1)[0]    # 和第i个mt token对齐的src token索引
                    src_idx_context = torch.where(src_align_status == 0)[0]  # 第i个mt token的上下文src token索引
                    attr_src_this_align = torch.index_select(attr_src_this, 0, src_idx_align) # [length of align src of this mt token] 和第i个mt token对齐的src token的贡献
                    attr_src_this_context = torch.index_select(attr_src_this, 0, src_idx_context) # [length of context src of this mt token] 第i个mt token的上下文src token的贡献
                    attr_src_this_align_mean = torch.mean(attr_src_this_align)     # [1] 和第i个mt token对齐的src token的贡献平均值
                    attr_src_this_context_mean = torch.mean(attr_src_this_context) # [1] 第i个mt token的上下文src token的贡献平均值
                    all_attr_src_align_mean.append(attr_src_this_align_mean.item())    # 总计
                    all_attr_src_context_mean.append(attr_src_this_context_mean.item())

                    # 自身mt token
                    attr_mt_this_self_mean = attr_mt_this[i]
                    all_attr_mt_self_mean.append(attr_mt_this_self_mean.item())    # 总计

                    # 除去自身mt token 之外的所有上下文mt token
                    attr_mt_this_context = torch.cat([attr_mt_this_reverse[mt_len - i : ], attr_mt_this[i + 1 : ]], dim = 0)
                    attr_mt_this_context_mean = torch.mean(attr_mt_this_context) # [1] 第i个mt token的上下文mt token的贡献平均值
                    all_attr_mt_context_mean.append(attr_mt_this_context_mean.item())

                    # 上下文拆分1：自身mt token的左右各k个邻近mt token
                    attr_mt_this_adj = torch.cat([attr_mt_this_reverse[mt_len - i : mt_len - i + k_adj], attr_mt_this[i + 1 : i + 1 + k_adj]], dim = 0)
                    attr_mt_this_adj_mean = torch.mean(attr_mt_this_adj)
                    all_attr_mt_this_adj_mean.append(attr_mt_this_adj_mean.item())

                    # 上下文拆分2：除去 [自身mt token + 左右各k个邻近] 之外的非邻近mt token
                    attr_mt_this_remote = torch.cat([attr_mt_this_reverse[mt_len - i + k_adj : ], attr_mt_this[i + 1 + k_adj : ]], dim = 0)
                    attr_mt_this_remote_mean = torch.mean(attr_mt_this_remote)
                    all_attr_mt_this_remote_mean.append(attr_mt_this_remote_mean.item())

                    # 比例计算
                    parts_sum = attr_src_this_align_mean + attr_src_this_context_mean + attr_mt_this_self_mean + attr_mt_this_context_mean
                    all_attr_src_align_ratio.append((attr_src_this_align_mean / parts_sum).item())
                    all_attr_src_context_ratio.append((attr_src_this_context_mean / parts_sum).item())
                    all_attr_mt_self_ratio.append((attr_mt_this_self_mean / parts_sum).item())
                    all_attr_mt_context_ratio.append((attr_mt_this_context_mean / parts_sum).item())


        #cal_attr_ratio(attr_sent_srcmt, attr_token_srcmt, x_valid, top_k=1)                       # src mt 贡献比例分析
        #cal_word_alignment_attr_ratio(align_bpe_dict_mt2src, id, y_valid, attr_token_src, acc_sign)
        #cal_word_alignment_top(align_bpe_dict_src2mt, attr_token_src, y_valid)                               # 词对齐分析
        cal_attr_different_part(align_bpe_dict_mt2src, id, x_valid, y_valid, attr_token_srcmt, k_adj)    # 四个因素贡献排序分析

        if id >= 0: break
        id += 1
        
    # ============================== 分析结果展示 ============================== #
    def show_attr_ratio():
        nonlocal all_avg_attr_sent_mt_ratio, all_avg_attr_token_mt_ratio, \
                    all_max_attr_sent_mt_ratio, all_max_attr_token_mt_ratio, \
                    topk_mt_sent_cnt, topk_src_sent_cnt, \
                    topk_mt_token_cnt, topk_src_token_cnt
        all_avg_attr_sent_mt_ratio = torch.cat(all_avg_attr_sent_mt_ratio, 0)    # 延长了而已 [sample_num]
        all_avg_attr_token_mt_ratio = torch.cat(all_avg_attr_token_mt_ratio, 0)  # 延长了而已 [all_token_num]
        avg_attr_sent_mt_ratio = torch.mean(all_avg_attr_sent_mt_ratio, dim=0)
        avg_attr_token_mt_ratio = torch.mean(all_avg_attr_token_mt_ratio, dim=0)
        print('avg_attr_sent_mt_ratio: %.6f' % avg_attr_sent_mt_ratio.item())
        print('avg_attr_token_mt_ratio %.6f' % avg_attr_token_mt_ratio.item())

        all_max_attr_sent_mt_ratio = torch.cat(all_max_attr_sent_mt_ratio, 0)    # 延长了而已 [sample_num]
        all_max_attr_token_mt_ratio = torch.cat(all_max_attr_token_mt_ratio, 0)  # 延长了而已 [all_token_num]
        max_attr_sent_mt_ratio = torch.mean(all_max_attr_sent_mt_ratio, dim=0)
        max_attr_token_mt_ratio = torch.mean(all_max_attr_token_mt_ratio, dim=0)
        print('max_attr_sent_mt_ratio: %.6f' % max_attr_sent_mt_ratio.item())
        print('max_attr_token_mt_ratio %.6f' % max_attr_token_mt_ratio.item())

        topk_mt_sent_ratio = topk_mt_sent_cnt / (topk_src_sent_cnt + topk_mt_sent_cnt)
        print('top%d_mt_sent_ratio %.6f' % (top_k, topk_mt_sent_ratio))
        topk_mt_token_ratio = topk_mt_token_cnt / (topk_src_token_cnt + topk_mt_token_cnt)
        print('top%d_mt_token_ratio %.6f' % (top_k, topk_mt_token_ratio))
    
    def show_word_alignment_attr_ratio():
        nonlocal right_src_align_sum, wrong_src_align_sum, \
                right_src_context_sum, wrong_src_context_sum
        
        right_src_align_all = np.sum(right_src_align_sum)
        right_src_context_all = np.sum(right_src_context_sum)
        wrong_src_align_all = np.sum(wrong_src_align_sum)
        wrong_src_context_all = np.sum(wrong_src_context_sum)

        print("tag预测正确的token，对齐的src_token贡献比例 = %.6f" % (right_src_align_all / (right_src_align_all + right_src_context_all)))
        print("tag预测错误的token，对齐的src_token贡献比例 = %.6f" % (wrong_src_align_all / (wrong_src_align_all + wrong_src_context_all)))


    def show_word_alignment_top():
        nonlocal align_bpe_dict_mt2src, align_pred_lines
        num = 0
        ans = 0
        for line_id, align_pred_list in enumerate(align_pred_lines):
            gold_dict_line = align_bpe_dict_mt2src[str(line_id)]   # 这句话中每个mt token对齐的src token们
            for mt_idx in range(len(align_pred_list)):
                if str(mt_idx) not in gold_dict_line: continue    # 没有对齐的排除
                if align_pred_list[mt_idx] == -1: continue    # unk排除
                most_att = align_pred_list[mt_idx]     # 预测最对齐的src
                if most_att in gold_dict_line[str(mt_idx)]: ans += 1
                num += 1
        print("align_token_num = %d" % ans)
        print("all_consider_num = %d" % num)
        print("align_ratio = %.6f" % (ans / num))
    
    def show_attr_different_part():
        nonlocal all_attr_mt_self_mean, all_attr_mt_context_mean, \
                k_adj, all_attr_mt_this_adj_mean, all_attr_mt_this_remote_mean, \
                all_attr_src_align_mean, all_attr_src_context_mean, \
                all_attr_src_align_ratio, all_attr_src_context_ratio, \
                all_attr_mt_self_ratio, all_attr_mt_context_ratio

        print("自身mt_token 贡献均值 = %.6f" % np.mean(all_attr_mt_self_mean))
        print("其余mt_token 贡献均值 = %.6f" % np.mean(all_attr_mt_context_mean))
        print(" —— 其中，邻近的%d个mt_token 贡献均值 = %.6f" % (k_adj, np.mean(all_attr_mt_this_adj_mean)))
        print(" —— 非邻近的mt_token 贡献均值 = %.6f" % np.mean(all_attr_mt_this_remote_mean))
        print("对齐的src_token 贡献均值 = %.6f" % np.mean(all_attr_src_align_mean))
        print("其余src_token 贡献均值 = %.6f" % np.mean(all_attr_src_context_mean))
        print("=================== 比例")
        print("自身mt_token 贡献均值比例 = %.6f" % np.mean(all_attr_mt_self_ratio))
        print("其余mt_token 贡献均值比例 = %.6f" % np.mean(all_attr_mt_context_ratio))
        print("对齐的src_token 贡献均值比例 = %.6f" % np.mean(all_attr_src_align_ratio))
        print("其余src_token 贡献均值比例 = %.6f" % np.mean(all_attr_src_context_ratio))
    
    #show_attr_ratio()               # src mt 贡献比例分析
    #show_word_alignment_attr_ratio()
    #show_word_alignment_top()           # 词对齐分析
    show_attr_different_part()      # 四个因素贡献排序分析


def attribution_qe_right_wrong(flags):
    """
    计算test输入对预测结果影响的梯度，并做分析的函数，可以选择何种梯度(GI,IG)
    和origin相比，这个是light版本，把暂时用不到的注释删掉了
    暂时，和模型好坏相联系的版本!
    """
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    discriminator_model_configs = configs['discriminator_configs']

    data_configs = configs['data_configs']

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['test_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['test_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['test_data'][2]),
        QEHterDataset(data_path=data_configs['test_data'][3]),
    )

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=flags.batch_size,
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading transformer model parameters...')
    timer.tic()
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params)
    if Constants.USE_GPU:
        discriminator.cuda()

    # ================================================================================== #

    INFO('Begin ...')

    timer.tic()

    valid_iter = valid_iterator.build_generator()

    if flags.gradients_type == 'g':
        cal_gradients_func = one_gradients
        cal_gradients_func_srcmt = one_gradients_srcmt
    elif flags.gradients_type == 'ig':
        cal_gradients_func = integrated_gradients

    plot_path_prefix = flags.plot_path_prefix
    if not os.path.exists(plot_path_prefix):
        os.makedirs(plot_path_prefix)
    
    with open(flags.acc_sign_path, 'r', encoding='utf-8') as f_acc_sign:
        acc_sign_list = []
        lines = f_acc_sign.readlines()
        for line in lines:
            line = line.strip('\n').split()
            acc_sign_list.append(line)

    with open(flags.align_bpe_path_mt2src, 'r', encoding='utf-8') as falign_m2s:
        align_bpe_dict_mt2src = json.load(falign_m2s)
    
    all_avg_attr_sent_src_ratio = []
    all_avg_attr_token_src_ratio = []
    all_avg_attr_token_src_ratio_right = []
    all_avg_attr_token_src_ratio_wrong = []
    all_avg_attr_token_src_num_right = []
    all_avg_attr_token_src_num_wrong = []
    all_avg_attr_token_mt_num_right = []
    all_avg_attr_token_mt_num_wrong = []

    right_src_align_sum = []
    right_src_context_sum = []
    wrong_src_align_sum = []
    wrong_src_context_sum = []
    right_src_align_ratio = []
    wrong_src_align_ratio = []
    
    all_attr_src_align_ratio_right = []
    all_attr_src_context_ratio_right = []
    all_attr_mt_self_ratio_right = []
    all_attr_mt_context_ratio_right = []
    all_attr_src_align_ratio_wrong = []
    all_attr_src_context_ratio_wrong = []
    all_attr_mt_self_ratio_wrong = []
    all_attr_mt_context_ratio_wrong = []

    all_attr_src_align_num_right = []
    all_attr_src_context_num_right = []
    all_attr_mt_self_num_right = []
    all_attr_mt_context_num_right = []
    all_attr_src_align_num_wrong = []
    all_attr_src_context_num_wrong = []
    all_attr_mt_self_num_wrong = []
    all_attr_mt_context_num_wrong = []

    id = 0

    for batch_valid in valid_iter:
        # 每个样本
        """
        if(id <= 1): 
            id += 1
            continue
        if(id > 3): break
        """
        print(id)

        """
        if id < flags.test_id: 
            id += 1
            continue
        if id > flags.test_id: assert 1==2
        """
        
        seqs_x_valid, seqs_y_valid, xy_label, xy_hter = batch_valid

        x_valid, y_valid, xy_label = prepare_data_qe(seqs_x_valid, seqs_y_valid, xy_label, cuda=Constants.USE_GPU)
        if Constants.USE_GPU:
            xy_hter = torch.tensor(xy_hter)
            xy_hter = xy_hter.cuda()
        
        #print(y_valid[0][2:-1])

        
        # ============================== 梯度计算 ============================== #
        # ====== src或mt，对于mt预测的影响
        # attr_sent_mt [mt_seq_len]
        # attr_token_mt [mt_seq_len, mt_seq_len] 
        # attr_sent_src [src_seq_len]
        # attr_token_src [src_seq_len, mt_seq_len]
        attr_sent_mt, attr_token_mt, \
            attr_sent_src, attr_token_src = cal_gradients_func(discriminator, x_valid, y_valid)
        
        # ====== src+mt一起，对于mt预测的影响
        # attr_sent_srcmt [src_seq_len + mt_seq_len]
        # attr_token_srcmt [src_seq_len + mt_seq_len, mt_seq_len]
        # attr_sent_mt_ratio [1]
        # attr_token_mt_ratio [mt_seq_len]
        attr_sent_srcmt, attr_token_srcmt = cal_gradients_func_srcmt(discriminator, x_valid, y_valid)


        # ============================== 画图 ============================== #
        #"""
        plot_attribution_all(attr_sent_mt=attr_sent_mt, 
                            attr_token_mt=attr_token_mt, 
                            attr_sent_src=attr_sent_src, 
                            attr_token_src=attr_token_src,
                            attr_sent_srcmt=attr_sent_srcmt,
                            attr_token_srcmt=attr_token_srcmt,
                            plot_path_prefix=plot_path_prefix, 
                            mt_tokens=y_valid[0][2:-1], 
                            src_tokens=x_valid[0][1:-1], 
                            mt_dict=vocab_tgt, 
                            src_dict=vocab_src, 
                            id=id, 
                            gradients_type=flags.gradients_type,
                            lang='enzh')
        #"""

        # ============================== 分析函数计算 ============================== #
        def cal_attr_ratio(attr_sent_srcmt, attr_token_srcmt, x_valid, acc_sign):
            src_len = x_valid.size(-1) - 2
            mt_len = attr_token_srcmt.size(-1)

            # mt对句子分数贡献比例 [1]
            attr_sent_src_part = attr_sent_srcmt[:src_len]
            attr_sent_mt_part = attr_sent_srcmt[src_len:]

            avg_sent_src = torch.sum(attr_sent_src_part)
            avg_sent_mt = torch.sum(attr_sent_mt_part)
            avg_attr_sent_mt_ratio = torch.tensor([avg_sent_mt / (avg_sent_src + avg_sent_mt)])  # [1]
            avg_attr_sent_src_ratio = torch.tensor([avg_sent_src / (avg_sent_src + avg_sent_mt)])  # [1]

            # mt对词标签贡献比例 [mt_seq_len]
            attr_token_src_part = attr_token_srcmt[:src_len]
            attr_token_mt_part = attr_token_srcmt[src_len:]

            avg_token_src = torch.sum(attr_token_src_part, dim = 0)  # [mt_seq_len] 每个位置是 src部分对该mt token的贡献总和
            avg_token_mt = torch.sum(attr_token_mt_part, dim = 0)  # [mt_seq_len]
            avg_attr_token_mt_ratio = avg_token_mt / (avg_token_src + avg_token_mt)  # [mt_seq_len]
            avg_attr_token_src_ratio = avg_token_src / (avg_token_src + avg_token_mt)  # [mt_seq_len]

            # 总计
            nonlocal all_avg_attr_sent_src_ratio, all_avg_attr_token_src_ratio, \
                    all_avg_attr_token_src_ratio_right, all_avg_attr_token_src_ratio_wrong, \
                    all_avg_attr_token_src_num_right, all_avg_attr_token_src_num_wrong, \
                    all_avg_attr_token_mt_num_right, all_avg_attr_token_mt_num_wrong

            all_avg_attr_sent_src_ratio.append(avg_attr_sent_src_ratio)
            all_avg_attr_token_src_ratio.append(avg_attr_token_src_ratio)

            for i in range(mt_len):
                if acc_sign[i] == '1':   # 这个token预测对了:
                    all_avg_attr_token_src_ratio_right.append(avg_attr_token_src_ratio[i].item())
                    all_avg_attr_token_src_num_right.append(avg_token_src[i].item())
                    all_avg_attr_token_mt_num_right.append(avg_token_mt[i].item())
                else:                       # 这个token预测错了
                    all_avg_attr_token_src_ratio_wrong.append(avg_attr_token_src_ratio[i].item())
                    all_avg_attr_token_src_num_wrong.append(avg_token_src[i].item())
                    all_avg_attr_token_mt_num_wrong.append(avg_token_mt[i].item())

        def cal_word_alignment_attr_ratio(align_bpe_dict_line, id, y_valid, attr_token_src, acc_sign):
            nonlocal right_src_align_sum, wrong_src_align_sum, \
                right_src_context_sum, wrong_src_context_sum, \
                right_src_align_ratio, wrong_src_align_ratio

            mt_len = attr_token_src.size(-1)
            mt_temp = y_valid[0][2:-1]
            for i in range(mt_len):
                if mt_temp[i] == 3: continue
                attr_src_this = attr_token_src[:, i]  # 所有src_token对第i个mt_token的贡献
                if str(i) in align_bpe_dict_line:  # 如果这个mt token有对齐的src token的话
                    # 对齐的src与上下文src贡献
                    src_align_status = torch.zeros_like(attr_src_this).index_fill(0, torch.tensor(align_bpe_dict_line[str(i)]).cuda(), True)  # 长度同src，01向量代表每个src token是否是第i个mt token的对齐token
                    src_idx_align = torch.where(src_align_status == 1)[0]    # 和第i个mt token对齐的src token索引
                    src_idx_context = torch.where(src_align_status == 0)[0]  # 第i个mt token的上下文src token索引
                    attr_src_this_align = torch.index_select(attr_src_this, 0, src_idx_align) # [length of align src of this mt token] 和第i个mt token对齐的src token的贡献
                    attr_src_this_context = torch.index_select(attr_src_this, 0, src_idx_context) # [length of context src of this mt token] 第i个mt token的上下文src token的贡献
                    attr_src_this_align_sum = torch.sum(attr_src_this_align)     # [1] 和第i个mt token对齐的src token的贡献之和
                    attr_src_this_context_sum = torch.sum(attr_src_this_context) # [1] 第i个mt token的上下文src token的贡献之和
                    if acc_sign[i] == '1':   # 这个token预测对了
                        #right_src_align_sum.append(attr_src_this_align_sum.item())  # 总计
                        #right_src_context_sum.append(attr_src_this_context_sum.item())
                        right_src_align_ratio.append((attr_src_this_align_sum / (attr_src_this_align_sum + attr_src_this_context_sum)).item())
                    else:                       # 这个token预测错了
                        #wrong_src_align_sum.append(attr_src_this_align_sum.item())
                        #wrong_src_context_sum.append(attr_src_this_context_sum.item())
                        wrong_src_align_ratio.append((attr_src_this_align_sum / (attr_src_this_align_sum + attr_src_this_context_sum)).item())

        def cal_attr_different_part(align_bpe_dict_line, id, x_valid, y_valid, attr_token_srcmt, acc_sign):
            nonlocal all_attr_src_align_ratio_right, all_attr_src_context_ratio_right, \
                all_attr_mt_self_ratio_right, all_attr_mt_context_ratio_right, \
                all_attr_src_align_ratio_wrong, all_attr_src_context_ratio_wrong, \
                all_attr_mt_self_ratio_wrong, all_attr_mt_context_ratio_wrong, \
                all_attr_src_align_num_right, all_attr_src_context_num_right, \
                all_attr_mt_self_num_right, all_attr_mt_context_num_right, \
                all_attr_src_align_num_wrong, all_attr_src_context_num_wrong, \
                all_attr_mt_self_num_wrong, all_attr_mt_context_num_wrong

            src_len = x_valid.size(-1) - 2
            mt_len = attr_token_srcmt.size(-1)
            mt_temp = y_valid[0][2:-1]

            attr_token_src_part = attr_token_srcmt[:src_len]  # [src_len, mt_len]
            attr_token_mt_part = attr_token_srcmt[src_len:]  # [mt_len, mt_len]

            for i in range(mt_len):
                if mt_temp[i] == 3: continue
                attr_src_this = attr_token_src_part[:, i]  # 所有src_token对第i个mt_token的贡献
                attr_mt_this = attr_token_mt_part[:, i]  # 所有mt_token对第i个mt_token的贡献
                attr_mt_this_reverse = attr_mt_this.flip(dims = [0])

                if str(i) in align_bpe_dict_line:  # 如果这个mt token有对齐的src token的话
                    # 对齐的src与上下文src贡献
                    src_align_status = torch.zeros_like(attr_src_this).index_fill(0, torch.tensor(align_bpe_dict_line[str(i)]).cuda(), True)  # 长度同src，01向量代表每个src token是否是第i个mt token的对齐token
                    src_idx_align = torch.where(src_align_status == 1)[0]    # 和第i个mt token对齐的src token索引
                    src_idx_context = torch.where(src_align_status == 0)[0]  # 第i个mt token的上下文src token索引
                    attr_src_this_align = torch.index_select(attr_src_this, 0, src_idx_align) # [length of align src of this mt token] 和第i个mt token对齐的src token的贡献
                    attr_src_this_context = torch.index_select(attr_src_this, 0, src_idx_context) # [length of context src of this mt token] 第i个mt token的上下文src token的贡献
                    attr_src_this_align_sum = torch.sum(attr_src_this_align)     # [1] 和第i个mt token对齐的src token的贡献之和
                    attr_src_this_context_sum = torch.sum(attr_src_this_context) # [1] 第i个mt token的上下文src token的贡献之和

                    # 自身mt token
                    attr_mt_this_self_sum = attr_mt_this[i]

                    # 除去自身mt token 之外的所有上下文mt token
                    attr_mt_this_context = torch.cat([attr_mt_this_reverse[mt_len - i : ], attr_mt_this[i + 1 : ]], dim = 0)
                    attr_mt_this_context_sum = torch.sum(attr_mt_this_context) # [1] 第i个mt token的上下文mt token的贡献平均值
                    
                    # 比例计算
                    parts_sum = attr_src_this_align_sum + attr_src_this_context_sum + attr_mt_this_self_sum + attr_mt_this_context_sum
                    if acc_sign[i] == '1':   # 这个token预测对了
                        all_attr_src_align_ratio_right.append((attr_src_this_align_sum / parts_sum).item())
                        all_attr_src_context_ratio_right.append((attr_src_this_context_sum / parts_sum).item())
                        all_attr_mt_self_ratio_right.append((attr_mt_this_self_sum / parts_sum).item())
                        all_attr_mt_context_ratio_right.append((attr_mt_this_context_sum / parts_sum).item())

                        all_attr_src_align_num_right.append((attr_src_this_align_sum).item())
                        all_attr_src_context_num_right.append((attr_src_this_context_sum).item())
                        all_attr_mt_self_num_right.append((attr_mt_this_self_sum).item())
                        all_attr_mt_context_num_right.append((attr_mt_this_context_sum).item())
                    else:                       # 这个token预测错了
                        all_attr_src_align_ratio_wrong.append((attr_src_this_align_sum / parts_sum).item())
                        all_attr_src_context_ratio_wrong.append((attr_src_this_context_sum / parts_sum).item())
                        all_attr_mt_self_ratio_wrong.append((attr_mt_this_self_sum / parts_sum).item())
                        all_attr_mt_context_ratio_wrong.append((attr_mt_this_context_sum / parts_sum).item())

                        all_attr_src_align_num_wrong.append((attr_src_this_align_sum).item())
                        all_attr_src_context_num_wrong.append((attr_src_this_context_sum).item())
                        all_attr_mt_self_num_wrong.append((attr_mt_this_self_sum).item())
                        all_attr_mt_context_num_wrong.append((attr_mt_this_context_sum).item())


        #cal_attr_ratio(attr_sent_srcmt, attr_token_srcmt, x_valid, acc_sign_list[id])                       # src mt 贡献比例分析
        #cal_word_alignment_attr_ratio(align_bpe_dict_mt2src[str(id)], id, y_valid, attr_token_src, acc_sign_list[id])                # 词对齐分析
        #cal_attr_different_part(align_bpe_dict_mt2src[str(id)], id, x_valid, y_valid, attr_token_srcmt, acc_sign_list[id])    # 四个因素贡献排序分析

        if id >= 4: break
        id += 1
        
    # ============================== 分析结果展示 ============================== #
    def show_attr_ratio():
        nonlocal all_avg_attr_sent_src_ratio, all_avg_attr_token_src_ratio, \
                    all_avg_attr_token_src_ratio_right, all_avg_attr_token_src_ratio_wrong, \
                    all_avg_attr_token_src_num_right, all_avg_attr_token_src_num_wrong, \
                    all_avg_attr_token_mt_num_right, all_avg_attr_token_mt_num_wrong
        all_avg_attr_sent_src_ratio = torch.cat(all_avg_attr_sent_src_ratio, 0)    # 延长了而已 [sample_num]
        all_avg_attr_token_src_ratio = torch.cat(all_avg_attr_token_src_ratio, 0)  # 延长了而已 [all_token_num]
        
        avg_attr_sent_src_ratio = torch.mean(all_avg_attr_sent_src_ratio, dim=0)
        avg_attr_token_src_ratio = torch.mean(all_avg_attr_token_src_ratio, dim=0)
        print('avg_attr_sent_src_ratio: %.6f' % avg_attr_sent_src_ratio.item())
        print('avg_attr_token_src_ratio %.6f' % avg_attr_token_src_ratio.item())
        print('avg_attr_token_src_ratio_right %.6f' % np.mean(all_avg_attr_token_src_ratio_right))
        print('avg_attr_token_src_ratio_wrong %.6f' % np.mean(all_avg_attr_token_src_ratio_wrong))
        print('avg_attr_token_src_num_right %.6f' % np.mean(all_avg_attr_token_src_num_right))
        print('avg_attr_token_mt_num_right %.6f' % np.mean(all_avg_attr_token_mt_num_right))
        print('avg_attr_token_src_num_wrong %.6f' % np.mean(all_avg_attr_token_src_num_wrong))
        print('avg_attr_token_mt_num_wrong %.6f' % np.mean(all_avg_attr_token_mt_num_wrong))
    
    def show_word_alignment_attr_ratio():
        nonlocal right_src_align_sum, wrong_src_align_sum, \
                right_src_context_sum, wrong_src_context_sum, \
                right_src_align_ratio, wrong_src_align_ratio
        
        #right_src_align_all = np.sum(right_src_align_sum)
        #right_src_context_all = np.sum(right_src_context_sum)
        #wrong_src_align_all = np.sum(wrong_src_align_sum)
        #wrong_src_context_all = np.sum(wrong_src_context_sum)
        right_src_align_ratio_all = np.mean(right_src_align_ratio)
        wrong_src_align_ratio_all = np.mean(wrong_src_align_ratio)

        #print("tag预测正确的token，对齐的src_token贡献比例 旧 = %.6f" % (right_src_align_all / (right_src_align_all + right_src_context_all)))
        #print("tag预测错误的token，对齐的src_token贡献比例 旧 = %.6f" % (wrong_src_align_all / (wrong_src_align_all + wrong_src_context_all)))
        print("tag预测正确的token，对齐的src_token贡献比例 = %.6f" % (right_src_align_ratio_all))
        print("tag预测错误的token，对齐的src_token贡献比例 = %.6f" % (wrong_src_align_ratio_all))

    def show_attr_different_part():
        nonlocal all_attr_src_align_ratio_right, all_attr_src_context_ratio_right, \
            all_attr_mt_self_ratio_right, all_attr_mt_context_ratio_right, \
            all_attr_src_align_ratio_wrong, all_attr_src_context_ratio_wrong, \
            all_attr_mt_self_ratio_wrong, all_attr_mt_context_ratio_wrong, \
            all_attr_src_align_num_right, all_attr_src_context_num_right, \
            all_attr_mt_self_num_right, all_attr_mt_context_num_right, \
            all_attr_src_align_num_wrong, all_attr_src_context_num_wrong, \
            all_attr_mt_self_num_wrong, all_attr_mt_context_num_wrong
        
        print("=========== 对于预测正确的token")
        print("自身mt_token 贡献平均比例 = %.6f" % np.mean(all_attr_mt_self_ratio_right))
        print("其余mt_token 贡献平均比例 = %.6f" % np.mean(all_attr_mt_context_ratio_right))
        print("对齐的src_token 贡献平均比例 = %.6f" % np.mean(all_attr_src_align_ratio_right))
        print("其余src_token 贡献平均比例 = %.6f" % np.mean(all_attr_src_context_ratio_right))
        print("自身mt_token 贡献平均值 = %.6f" % np.mean(all_attr_mt_self_num_right))
        print("其余mt_token 贡献平均值 = %.6f" % np.mean(all_attr_mt_context_num_right))
        print("对齐的src_token 贡献平均值 = %.6f" % np.mean(all_attr_src_align_num_right))
        print("其余src_token 贡献平均值 = %.6f" % np.mean(all_attr_src_context_num_right))

        print("=========== 对于预测错误的token")
        print("自身mt_token 贡献平均比例 = %.6f" % np.mean(all_attr_mt_self_ratio_wrong))
        print("其余mt_token 贡献平均比例 = %.6f" % np.mean(all_attr_mt_context_ratio_wrong))
        print("对齐的src_token 贡献平均比例 = %.6f" % np.mean(all_attr_src_align_ratio_wrong))
        print("其余src_token 贡献平均比例 = %.6f" % np.mean(all_attr_src_context_ratio_wrong))
        print("自身mt_token 贡献平均值 = %.6f" % np.mean(all_attr_mt_self_num_wrong))
        print("其余mt_token 贡献平均值 = %.6f" % np.mean(all_attr_mt_context_num_wrong))
        print("对齐的src_token 贡献平均值 = %.6f" % np.mean(all_attr_src_align_num_wrong))
        print("其余src_token 贡献平均值 = %.6f" % np.mean(all_attr_src_context_num_wrong))
    
    #show_attr_ratio()               # src mt 贡献比例分析
    #show_word_alignment_attr_ratio()
    #show_attr_different_part()      # 四个因素贡献排序分析


def attribution_qe(flags):
    """
    计算test输入对预测结果影响的梯度，并做分析的函数，可以选择何种梯度(GI,IG)
    和origin相比，这个是light版本，把暂时用不到的注释删掉了
    每个部分的贡献和，而不是平均值
    """
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    discriminator_model_configs = configs['discriminator_configs']

    data_configs = configs['data_configs']

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['test_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['test_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['test_data'][2]),
        QEHterDataset(data_path=data_configs['test_data'][3]),
    )

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=flags.batch_size,
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading transformer model parameters...')
    timer.tic()
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params)
    if Constants.USE_GPU:
        discriminator.cuda()

    # ================================================================================== #

    INFO('Begin ...')

    timer.tic()

    valid_iter = valid_iterator.build_generator()

    if flags.gradients_type == 'g':
        cal_gradients_func = one_gradients
        cal_gradients_func_srcmt = one_gradients_srcmt
    elif flags.gradients_type == 'ig':
        cal_gradients_func = integrated_gradients

    plot_path_prefix = flags.plot_path_prefix
    if not os.path.exists(plot_path_prefix):
        os.makedirs(plot_path_prefix)

    with open(flags.align_bpe_path_mt2src, 'r', encoding='utf-8') as falign_m2s:
        align_bpe_dict_mt2src = json.load(falign_m2s)
    
    all_avg_attr_sent_src_ratio = []
    all_avg_attr_token_src_ratio = []
    all_avg_attr_sent_mt_ratio = []
    all_avg_attr_token_mt_ratio = []

    src_align_ratio = []
    
    all_attr_src_align_ratio = []
    all_attr_src_context_ratio = []
    all_attr_mt_self_ratio = []
    all_attr_mt_context_ratio = []
    all_attr_src_align_num = []
    all_attr_src_context_num = []
    all_attr_mt_self_num = []
    all_attr_mt_context_num = []

    id = 0

    for batch_valid in valid_iter:
        # 每个样本
        print(id)
        
        seqs_x_valid, seqs_y_valid, xy_label, xy_hter = batch_valid

        x_valid, y_valid, xy_label = prepare_data_qe(seqs_x_valid, seqs_y_valid, xy_label, cuda=Constants.USE_GPU)
        if Constants.USE_GPU:
            xy_hter = torch.tensor(xy_hter)
            xy_hter = xy_hter.cuda()
        
        #print(y_valid[0][2:-1])

        
        # ============================== 梯度计算 ============================== #
        # ====== src或mt，对于mt预测的影响
        # attr_sent_mt [mt_seq_len]
        # attr_token_mt [mt_seq_len, mt_seq_len] 
        # attr_sent_src [src_seq_len]
        # attr_token_src [src_seq_len, mt_seq_len]
        attr_sent_mt, attr_token_mt, \
            attr_sent_src, attr_token_src = cal_gradients_func(discriminator, x_valid, y_valid)
        
        # ====== src+mt一起，对于mt预测的影响
        # attr_sent_srcmt [src_seq_len + mt_seq_len]
        # attr_token_srcmt [src_seq_len + mt_seq_len, mt_seq_len]
        # attr_sent_mt_ratio [1]
        # attr_token_mt_ratio [mt_seq_len]
        attr_sent_srcmt, attr_token_srcmt = cal_gradients_func_srcmt(discriminator, x_valid, y_valid)


        # ============================== 画图 ============================== #
        """
        plot_attribution_all(attr_sent_mt=attr_sent_mt, 
                            attr_token_mt=attr_token_mt, 
                            attr_sent_src=attr_sent_src, 
                            attr_token_src=attr_token_src,
                            attr_sent_srcmt=attr_sent_srcmt,
                            attr_token_srcmt=attr_token_srcmt,
                            plot_path_prefix=plot_path_prefix, 
                            mt_tokens=y_valid[0][2:-1], 
                            src_tokens=x_valid[0][1:-1], 
                            mt_dict=vocab_tgt, 
                            src_dict=vocab_src, 
                            id=id, 
                            gradients_type=flags.gradients_type,
                            lang='enzh')
        """

        # ============================== 分析函数计算 ============================== #
        def cal_attr_ratio(attr_sent_srcmt, attr_token_srcmt, x_valid):
            src_len = x_valid.size(-1) - 2
            mt_len = attr_token_srcmt.size(-1)

            # mt对句子分数贡献比例 [1]
            attr_sent_src_part = attr_sent_srcmt[:src_len]
            attr_sent_mt_part = attr_sent_srcmt[src_len:]

            avg_sent_src = torch.sum(attr_sent_src_part)
            avg_sent_mt = torch.sum(attr_sent_mt_part)
            avg_attr_sent_mt_ratio = torch.tensor([avg_sent_mt / (avg_sent_src + avg_sent_mt)])  # [1]
            avg_attr_sent_src_ratio = torch.tensor([avg_sent_src / (avg_sent_src + avg_sent_mt)])  # [1]

            # mt对词标签贡献比例 [mt_seq_len]
            attr_token_src_part = attr_token_srcmt[:src_len]
            attr_token_mt_part = attr_token_srcmt[src_len:]

            avg_token_src = torch.sum(attr_token_src_part, dim = 0)  # [mt_seq_len] 每个位置是 src部分对该mt token的贡献总和
            avg_token_mt = torch.sum(attr_token_mt_part, dim = 0)  # [mt_seq_len]
            avg_attr_token_mt_ratio = avg_token_mt / (avg_token_src + avg_token_mt)  # [mt_seq_len]
            avg_attr_token_src_ratio = avg_token_src / (avg_token_src + avg_token_mt)  # [mt_seq_len]

            # 总计
            nonlocal all_avg_attr_sent_src_ratio, all_avg_attr_token_src_ratio, \
                    all_avg_attr_sent_mt_ratio, all_avg_attr_token_mt_ratio

            all_avg_attr_sent_src_ratio.append(avg_attr_sent_src_ratio)
            all_avg_attr_token_src_ratio.append(avg_attr_token_src_ratio)
            all_avg_attr_sent_mt_ratio.append(avg_attr_sent_mt_ratio)
            all_avg_attr_token_mt_ratio.append(avg_attr_token_mt_ratio)

        def cal_word_alignment_attr_ratio(align_bpe_dict_line, id, y_valid, attr_token_src):
            nonlocal src_align_ratio

            mt_len = attr_token_src.size(-1)
            mt_temp = y_valid[0][2:-1]
            for i in range(mt_len):
                if mt_temp[i] == 3: continue
                attr_src_this = attr_token_src[:, i]  # 所有src_token对第i个mt_token的贡献
                if str(i) in align_bpe_dict_line:  # 如果这个mt token有对齐的src token的话
                    # 对齐的src与上下文src贡献
                    src_align_status = torch.zeros_like(attr_src_this).index_fill(0, torch.tensor(align_bpe_dict_line[str(i)]).cuda(), True)  # 长度同src，01向量代表每个src token是否是第i个mt token的对齐token
                    src_idx_align = torch.where(src_align_status == 1)[0]    # 和第i个mt token对齐的src token索引
                    src_idx_context = torch.where(src_align_status == 0)[0]  # 第i个mt token的上下文src token索引
                    attr_src_this_align = torch.index_select(attr_src_this, 0, src_idx_align) # [length of align src of this mt token] 和第i个mt token对齐的src token的贡献
                    attr_src_this_context = torch.index_select(attr_src_this, 0, src_idx_context) # [length of context src of this mt token] 第i个mt token的上下文src token的贡献
                    attr_src_this_align_sum = torch.sum(attr_src_this_align)     # [1] 和第i个mt token对齐的src token的贡献之和
                    attr_src_this_context_sum = torch.sum(attr_src_this_context) # [1] 第i个mt token的上下文src token的贡献之和
                    src_align_ratio.append((attr_src_this_align_sum / (attr_src_this_align_sum + attr_src_this_context_sum)).item())
        
        def cal_attr_different_part(align_bpe_dict_line, id, x_valid, y_valid, attr_token_srcmt):
            nonlocal all_attr_src_align_ratio, all_attr_src_context_ratio, \
                all_attr_mt_self_ratio, all_attr_mt_context_ratio, \
                all_attr_src_align_num, all_attr_src_context_num, \
                all_attr_mt_self_num, all_attr_mt_context_num

            src_len = x_valid.size(-1) - 2
            mt_len = attr_token_srcmt.size(-1)
            mt_temp = y_valid[0][2:-1]

            attr_token_src_part = attr_token_srcmt[:src_len]  # [src_len, mt_len]
            attr_token_mt_part = attr_token_srcmt[src_len:]  # [mt_len, mt_len]

            for i in range(mt_len):
                if mt_temp[i] == 3: continue
                attr_src_this = attr_token_src_part[:, i]  # 所有src_token对第i个mt_token的贡献
                attr_mt_this = attr_token_mt_part[:, i]  # 所有mt_token对第i个mt_token的贡献
                attr_mt_this_reverse = attr_mt_this.flip(dims = [0])

                if str(i) in align_bpe_dict_line:  # 如果这个mt token有对齐的src token的话
                    # 对齐的src与上下文src贡献
                    src_align_status = torch.zeros_like(attr_src_this).index_fill(0, torch.tensor(align_bpe_dict_line[str(i)]).cuda(), True)  # 长度同src，01向量代表每个src token是否是第i个mt token的对齐token
                    src_idx_align = torch.where(src_align_status == 1)[0]    # 和第i个mt token对齐的src token索引
                    src_idx_context = torch.where(src_align_status == 0)[0]  # 第i个mt token的上下文src token索引
                    attr_src_this_align = torch.index_select(attr_src_this, 0, src_idx_align) # [length of align src of this mt token] 和第i个mt token对齐的src token的贡献
                    attr_src_this_context = torch.index_select(attr_src_this, 0, src_idx_context) # [length of context src of this mt token] 第i个mt token的上下文src token的贡献
                    attr_src_this_align_sum = torch.sum(attr_src_this_align)     # [1] 和第i个mt token对齐的src token的贡献之和
                    attr_src_this_context_sum = torch.sum(attr_src_this_context) # [1] 第i个mt token的上下文src token的贡献之和

                    # 自身mt token
                    attr_mt_this_self_sum = attr_mt_this[i]

                    # 除去自身mt token 之外的所有上下文mt token
                    attr_mt_this_context = torch.cat([attr_mt_this_reverse[mt_len - i : ], attr_mt_this[i + 1 : ]], dim = 0)
                    attr_mt_this_context_sum = torch.sum(attr_mt_this_context) # [1] 第i个mt token的上下文mt token的贡献平均值
                    
                    # 比例计算
                    parts_sum = attr_src_this_align_sum + attr_src_this_context_sum + attr_mt_this_self_sum + attr_mt_this_context_sum
                    
                    all_attr_src_align_ratio.append((attr_src_this_align_sum / parts_sum).item())
                    all_attr_src_context_ratio.append((attr_src_this_context_sum / parts_sum).item())
                    all_attr_mt_self_ratio.append((attr_mt_this_self_sum / parts_sum).item())
                    all_attr_mt_context_ratio.append((attr_mt_this_context_sum / parts_sum).item())

                    all_attr_src_align_num.append((attr_src_this_align_sum).item())
                    all_attr_src_context_num.append((attr_src_this_context_sum).item())
                    all_attr_mt_self_num.append((attr_mt_this_self_sum).item())
                    all_attr_mt_context_num.append((attr_mt_this_context_sum).item())
        

        cal_attr_ratio(attr_sent_srcmt, attr_token_srcmt, x_valid)                       # src mt 贡献比例分析
        cal_word_alignment_attr_ratio(align_bpe_dict_mt2src[str(id)], id, y_valid, attr_token_src)                # 词对齐分析
        cal_attr_different_part(align_bpe_dict_mt2src[str(id)], id, x_valid, y_valid, attr_token_srcmt)    # 四个因素贡献排序分析

        #if id >= 3: break
        id += 1
        
    # ============================== 分析结果展示 ============================== #
    def show_attr_ratio():
        nonlocal all_avg_attr_sent_src_ratio, all_avg_attr_token_src_ratio, \
                    all_avg_attr_sent_mt_ratio, all_avg_attr_token_mt_ratio
        all_avg_attr_sent_src_ratio = torch.cat(all_avg_attr_sent_src_ratio, 0)    # 延长了而已 [sample_num]
        all_avg_attr_token_src_ratio = torch.cat(all_avg_attr_token_src_ratio, 0)  # 延长了而已 [all_token_num]
        all_avg_attr_sent_mt_ratio = torch.cat(all_avg_attr_sent_mt_ratio, 0)    # 延长了而已 [sample_num]
        all_avg_attr_token_mt_ratio = torch.cat(all_avg_attr_token_mt_ratio, 0)  # 延长了而已 [all_token_num]
        
        avg_attr_sent_src_ratio = torch.mean(all_avg_attr_sent_src_ratio, dim=0)
        avg_attr_token_src_ratio = torch.mean(all_avg_attr_token_src_ratio, dim=0)
        avg_attr_sent_mt_ratio = torch.mean(all_avg_attr_sent_mt_ratio, dim=0)
        avg_attr_token_mt_ratio = torch.mean(all_avg_attr_token_mt_ratio, dim=0)
        print('avg_attr_sent_src_ratio: %.6f' % avg_attr_sent_src_ratio.item())
        print('avg_attr_sent_mt_ratio: %.6f' % avg_attr_sent_mt_ratio.item())
        print('avg_attr_token_src_ratio %.6f' % avg_attr_token_src_ratio.item())
        print('avg_attr_token_mt_ratio %.6f' % avg_attr_token_mt_ratio.item())
    
    def show_word_alignment_attr_ratio():
        nonlocal src_align_ratio
        
        src_align_ratio_all = np.mean(src_align_ratio)

        print("在来自src的贡献中，对齐的src_token贡献比例 = %.6f" % (src_align_ratio_all))

    def show_attr_different_part():
        nonlocal all_attr_src_align_ratio, all_attr_src_context_ratio, \
                all_attr_mt_self_ratio, all_attr_mt_context_ratio, \
                all_attr_src_align_num, all_attr_src_context_num, \
                all_attr_mt_self_num, all_attr_mt_context_num
        
        print("=========== 四个部分的贡献比例or贡献值")
        print("自身mt_token 贡献平均比例 = %.6f" % np.mean(all_attr_mt_self_ratio))
        print("其余mt_token 贡献平均比例 = %.6f" % np.mean(all_attr_mt_context_ratio))
        print("对齐的src_token 贡献平均比例 = %.6f" % np.mean(all_attr_src_align_ratio))
        print("其余src_token 贡献平均比例 = %.6f" % np.mean(all_attr_src_context_ratio))
        print("自身mt_token 贡献平均值 = %.6f" % np.mean(all_attr_mt_self_num))
        print("其余mt_token 贡献平均值 = %.6f" % np.mean(all_attr_mt_context_num))
        print("对齐的src_token 贡献平均值 = %.6f" % np.mean(all_attr_src_align_num))
        print("其余src_token 贡献平均值 = %.6f" % np.mean(all_attr_src_context_num))

    
    show_attr_ratio()               # src mt 贡献比例分析
    show_word_alignment_attr_ratio()
    show_attr_different_part()      # 四个因素贡献排序分析


def infl_knn(flags):
    """
    计算 训练集 对 某测试样本hter 的影响
    这里是用knn筛选的，筛选依据是：取出test样本经过模型后的hter_logits [d_model]维；在训练集中同样这么做，看哪些样本的向量最相似
    """
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    discriminator_model_configs = configs['discriminator_configs']

    data_configs = configs['data_configs']

    # ================================================================================== #
    # Load Data


    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])
    
    train_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3])
    )

    valid_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['test_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['test_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['test_data'][2]),
        QEHterDataset(data_path=data_configs['test_data'][3])
    )

    # 0. load transformer
    INFO('Building transformer model...')
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()

    INFO('Reloading transformer model parameters...')
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params)

    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()

    # ================================================================================== #
    INFO('create faiss index')
    
    faiss_index_prefix = flags.faiss_index_prefix 
    faiss_index_path = faiss_index_prefix + "qe_train_256.faiss_index"
    faiss_index = faiss_utils.FAISSIndex(256, "Flat")
    faiss_index.load(faiss_index_path) 

    train_dataset_size = len(train_dataset)
    """
    for train_id in range(train_dataset_size):
        print(train_id)
        train_sample = train_dataset[train_id]
        feature = cal_hter_logits(discriminator,
                                    train_sample,
                                    gpu=flags.use_gpu)  # [batch_size, d_model]
        feature = feature.cpu().detach().numpy()
        faiss_index.add(feature)
    
    faiss_index.save(faiss_index_path)
    
    assert 1==2
    """

    test_id = flags.test_id
    test_sample = valid_dataset[test_id]

    test_feature = cal_hter_logits(discriminator,
                                    test_sample,
                                    gpu=flags.use_gpu)  # [batch_size, d_model]
    test_feature = test_feature.cpu().detach().numpy()
    KNN_distances, KNN_indices = faiss_index.search(          # KNN_indices就是那些knn选中的最相似的训练样本们
            k=10, queries=test_feature)                    # [num_query, k]
    
    print("knn_indices")
    print(KNN_indices)
    for id in KNN_indices[0]:
        print(train_dataset.showitem(id))


def knn_analysis(flags):
    """
    计算 不同的训练集中 与 某测试样本hter表示(或token表示) 相似的样本
    取出test样本经过模型后的hter_logits [d_model]维；在训练集中同样这么做，看哪些样本的向量最相似
    """

    # ============================================ 读取配置 ============================================ #
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    discriminator_model_configs = configs['discriminator_configs']

    data_configs = configs['data_configs']
    training_configs = configs['training_configs']

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])
    
    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    # ============================================ 加载数据集 ============================================ #

    train_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3]),
        QETagsDataset(data_path=data_configs['train_data'][4],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][5])
    )

    qe_train_iterator = DataIterator(dataset=train_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'])

    parallel_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['parallel_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['parallel_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['parallel_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['parallel_data'][3]),
        QETagsDataset(data_path=data_configs['parallel_data'][4],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['parallel_data'][5]),
    )

    parallel_iterator = DataIterator(dataset=parallel_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'])

    parallel_pseudo_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['parallel_pseudo_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['parallel_pseudo_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['parallel_pseudo_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['parallel_pseudo_data'][3]),
        QETagsDataset(data_path=data_configs['parallel_pseudo_data'][4],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['parallel_pseudo_data'][5])
    )

    parallel_pseudo_iterator = DataIterator(dataset=parallel_pseudo_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'])

    test_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['test_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['test_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['test_data'][2]),
        QEHterDataset(data_path=data_configs['test_data'][3]),
        QETagsDataset(data_path=data_configs['test_data'][4]),
        QEHterDataset(data_path=data_configs['test_data'][5])
    )

    # ============================================ 加载模型 ============================================ #
    INFO('Building transformer model...')
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()

    INFO('Reloading transformer model parameters...')
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params)

    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    
    # 初始化bert模型
    #tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    #bert_model = BertModel.from_pretrained('bert-base-chinese').cuda()

    # ============================================ 路径整理 ============================================ #

    # 读取特征的faiss路径
    if flags.feature_model == 'directqe':
        feature_prefix = flags.feature_prefix 
        os.makedirs(feature_prefix, exist_ok=True)
        # 句子级路径
        qe_train_hter_faiss_path = feature_prefix + "qe_train_hter.faiss"
        parallel_hter_faiss_path = feature_prefix + "parallel_hter.faiss"
        parallel_pseudo_hter_faiss_path = feature_prefix + "parallel_pseudo_hter.faiss"
        # 词级路径
        qe_train_tag_faiss_path = feature_prefix + "qe_train_tag.faiss_10w"
        parallel_tag_faiss_path = feature_prefix + "parallel_tag.faiss_10w"
        parallel_pseudo_tag_faiss_path = feature_prefix + "parallel_pseudo_tag.faiss_10w"
    
    elif flags.feature_model == 'bert':
        feature_prefix = flags.feature_prefix[:-1] + '_bert/'
        os.makedirs(feature_prefix, exist_ok=True)
        # 句子级路径
        qe_train_hter_faiss_path = feature_prefix + "qe_train_hter.faiss"
        parallel_hter_faiss_path = feature_prefix + "parallel_hter.faiss"
        parallel_pseudo_hter_faiss_path = feature_prefix + "parallel_pseudo_hter.faiss"

    else:
        feature_prefix = flags.feature_prefix[:-1] + '_' + flags.feature_model + '/'
        os.makedirs(feature_prefix, exist_ok=True)
        # 句子级路径
        qe_train_hter_faiss_path = feature_prefix + "qe_train_hter.faiss"
        parallel_hter_faiss_path = feature_prefix + "parallel_hter.faiss"
        parallel_pseudo_hter_faiss_path = feature_prefix + "parallel_pseudo_hter.faiss"
    

    # 写入检索结果的路径文件夹
    test_id = flags.test_id
    test_token = flags.test_token
    
    if flags.feature_model == 'directqe':
        knn_log_prefix = flags.knn_log_prefix
        os.makedirs(knn_log_prefix, exist_ok=True)
        # 句子级路径
        #qe_train_hter_knn_log_path = knn_log_prefix + "test" + str(test_id) + "_qe_train_hter.knn_log"
        #parallel_hter_knn_log_path = knn_log_prefix + "test" + str(test_id) + "_parallel_hter.knn_log"
        #parallel_pseudo_hter_knn_log_path = knn_log_prefix + "test" + str(test_id) + "_parallel_pseudo_hter.knn_log"
        # 词级路径
        qe_train_tag_knn_log_path = knn_log_prefix + "test" + str(test_id) + "_token" + str(test_token) + "_qe_train_tag.knn_log"
        parallel_tag_knn_log_path = knn_log_prefix + "test" + str(test_id) + "_token" + str(test_token) + "_parallel_tag.knn_log"
        parallel_pseudo_tag_knn_log_path = knn_log_prefix + "test" + str(test_id) + "_token" + str(test_token) + "_parallel_pseudo_tag.knn_log"
    
    elif flags.feature_model == 'bert':
        knn_log_prefix = flags.knn_log_prefix[:-1] + '_bert/'
        os.makedirs(knn_log_prefix, exist_ok=True)
        # 句子级路径
        qe_train_hter_knn_log_path = knn_log_prefix + "test" + str(test_id) + "_qe_train_hter.knn_log_bert"
        parallel_hter_knn_log_path = knn_log_prefix + "test" + str(test_id) + "_parallel_hter.knn_log_bert"
        parallel_pseudo_hter_knn_log_path = knn_log_prefix + "test" + str(test_id) + "_parallel_pseudo_hter.knn_log_bert"

    else:
        knn_log_prefix = flags.knn_log_prefix[:-1] + '_' + flags.feature_model + '/'
        os.makedirs(knn_log_prefix, exist_ok=True)
        # 句子级路径
        qe_train_hter_knn_log_path = knn_log_prefix + "test" + str(test_id) + "_qe_train_hter.knn_log_" + flags.feature_model
        parallel_hter_knn_log_path = knn_log_prefix + "test" + str(test_id) + "_parallel_hter.knn_log_" + flags.feature_model
        parallel_pseudo_hter_knn_log_path = knn_log_prefix + "test" + str(test_id) + "_parallel_pseudo_hter.knn_log_" + flags.feature_model

    # ============================================ 一次性·保存feature ============================================ #

    INFO('create faiss index')

    """
    nlist = 100 # 划分单元的数量
    quantizer = faiss.IndexFlatL2(256)
    faiss_index = faiss.IndexIVFFLAT(quantizer, 256, nlist, faiss.METRIC_L2)   # 貌似用不了
    assert 1==2
    """
    #faiss_index = faiss.IndexFlatL2(256)
    #faiss_index = faiss_utils.FAISSIndex(256, "Flat")

    # 保存qe训练数据的feature到文件
    """
    if not os.path.exists(qe_train_hter_faiss_path):
        save_feature_hter(discriminator, qe_train_iterator, feature_prefix, key="qe_train", gpu=flags.use_gpu)
        assert 1==2
    """
    """
    if not os.path.exists(qe_train_tag_faiss_path):
        #save_id_list(qe_train_iterator, feature_prefix, key="qe_train", gpu=flags.use_gpu)
        #assert 1==2
        save_feature_tag(discriminator, qe_train_iterator, feature_prefix, key="qe_train", gpu=flags.use_gpu)
    assert 1==2
    """
    # bert模型的
    """
    qe_mt_path = data_configs['train_data'][1]
    save_feature_hter_bert(tokenizer, bert_model, qe_mt_path, feature_prefix_bert, key="qe_train", gpu=flags.use_gpu)
    assert 1==2
    """
    
    # 保存平行语料的feature到文件
    """
    if not os.path.exists(parallel_hter_faiss_path):
        save_feature_hter(discriminator, parallel_iterator, feature_prefix, key="parallel", gpu=flags.use_gpu)
        assert 1==2
    """
    """
    if not os.path.exists(parallel_tag_faiss_path):
        save_id_list(parallel_iterator, feature_prefix, key="parallel", gpu=flags.use_gpu)
        save_feature_tag(discriminator, parallel_iterator, feature_prefix, key = "parallel", gpu=flags.use_gpu)
    assert 1==2
    """
    # bert模型的
    """
    parallel_mt_path = data_configs['parallel_data'][1]
    save_feature_hter_bert(tokenizer, bert_model, parallel_mt_path, feature_prefix_bert, key="parallel", gpu=flags.use_gpu)
    assert 1==2
    """
    
    # 保存平行语料所生成伪数据的feature到文件
    """
    if not os.path.exists(parallel_pseudo_hter_faiss_path):
        save_feature_hter(discriminator, parallel_pseudo_iterator, feature_prefix, key="parallel_pseudo", gpu=flags.use_gpu)
        assert 1==2
    """
    """
    if not os.path.exists(parallel_pseudo_tag_faiss_path):
        #save_id_list(parallel_pseudo_iterator, feature_prefix, key="parallel_pseudo", gpu=flags.use_gpu)
        save_feature_tag(discriminator, parallel_pseudo_iterator, feature_prefix, key = "parallel_pseudo", gpu=flags.use_gpu)
    assert 1==2
    """
    # bert模型的
    """
    parallel_pseudo_mt_path = data_configs['parallel_pseudo_data'][1]
    save_feature_hter_bert(tokenizer, bert_model, parallel_pseudo_mt_path, feature_prefix_bert, key="parallel_pseudo", gpu=flags.use_gpu)
    assert 1==2
    """
    
    # ============================================ 得到测试样本的feature ============================================ #

    # 某测试样本
    test_sample = test_dataset[test_id][:-1]
    #print('test_sample')
    #print(test_dataset.showitem(test_id))

    if flags.feature_model == 'directqe':
        # directqe模型提取的feature
        
        hter_feature, tags_feature = get_feature(discriminator,
                                                test_sample,
                                                gpu=flags.use_gpu)
        hter_feature = hter_feature.cpu().detach().numpy()
        tags_feature = tags_feature[test_token].unsqueeze(0).cpu().detach().numpy() # 把test_token这个词的feature取出来
        
    elif flags.feature_model == 'bert':
        # bert模型提取的feature
        test_mt_line = test_dataset.showitem(test_id)[1]
        test_mt_sent = "".join(test_mt_line.strip('\n').split())
        hter_feature = get_hter_feature_bert(tokenizer,
                                                    bert_model,
                                                    test_mt_sent,
                                                    gpu=flags.use_gpu)
        hter_feature = hter_feature.cpu().detach().numpy()

    else:
        # directqe模型其它版本
        hter_feature, tags_feature = get_feature(discriminator,
                                                test_sample,
                                                gpu=flags.use_gpu)
        hter_feature = hter_feature.cpu().detach().numpy()
        tags_feature = tags_feature[test_token].unsqueeze(0).cpu().detach().numpy() # 把test_token这个词的feature取出来


    # ============================================ 在训练集中检索相似feature ============================================ #
    INFO("begin search...")

    if flags.search_level == 'sent':

        # qe训练数据搜索
        KNN_distances, KNN_indices = faiss_search(qe_train_hter_faiss_path, 
                                                    hter_feature, 
                                                    k=10)
        write_knn_result_sent(KNN_indices,
                            train_dataset,
                            test_id,
                            test_dataset,
                            qe_train_hter_knn_log_path,)

        # 平行语料搜索
        KNN_distances, KNN_indices = faiss_search(parallel_hter_faiss_path, 
                                                    hter_feature, 
                                                    k=10)
        write_knn_result_sent(KNN_indices,
                            parallel_dataset,
                            test_id,
                            test_dataset,
                            parallel_hter_knn_log_path,)

        # 伪数据搜索
        KNN_distances, KNN_indices = faiss_search(parallel_pseudo_hter_faiss_path, 
                                                    hter_feature, 
                                                    k=10)
        write_knn_result_sent(KNN_indices,
                            parallel_pseudo_dataset,
                            test_id,
                            test_dataset,
                            parallel_pseudo_hter_knn_log_path,)


    elif flags.search_level == 'token':

        # qe训练数据搜索
        
        faiss_index = faiss.read_index(qe_train_tag_faiss_path)
        KNN_distances, KNN_indices = faiss_index.search(tags_feature, k=10)  
        
        print("knn_indices==================")
        print(KNN_indices)
        with open(qe_train_tag_knn_log_path, 'w', encoding='utf-8') as f:
            f.write('---------------------------test_sent_id: ' + str(test_id) + ', test_token_id: ' + str(test_token) + '---------------------------\n')
            for ii, test_sample_i in enumerate(test_dataset.showitem(test_id)):
                if ii == 0:
                    f.write("test_token: " + test_dataset.showitem(test_id)[1].split()[test_token])
                    f.write("       model pred: " + test_dataset.showitem(test_id)[4].split()[test_token])
                    f.write("       gold: " + test_dataset.showitem(test_id)[2].split()[test_token] + "\n")
                if ii == 4: break
                f.write(test_sample_i)
            
            f.write("\n=================================================\n\n")
            for id in KNN_indices[0]:
                # 这个id就是在全数据集token列表中的id，需要转换成对应哪条样本的哪个token
                tags_sent_id_list = torch.load(feature_prefix + "qe_train_tag_id_sent.tensor")
                tags_token_id_list = torch.load(feature_prefix + "qe_train_tag_id_token.tensor")
                sent_id = int(tags_sent_id_list[id].item())
                token_id = int(tags_token_id_list[id].item())
                #print(sent_id)
                #print(token_id)
                knn_sample = train_dataset.showitem(sent_id)
                f.write('---------------------------train_sent_id: ' + str(sent_id) + ', train_token_id: ' + str(token_id) + '---------------------------\n')
                for ii, sample_i in enumerate(knn_sample):
                    if ii == 0:
                        f.write("token_chosen_by_knn: " + knn_sample[1].split()[token_id])
                        f.write("       model pred: " + knn_sample[4].split()[token_id])
                        f.write("       gold: " + knn_sample[2].split()[token_id] + "\n")
                    if ii == 4: break
                    f.write(sample_i)
        assert 1==2
        

        # 平行语料搜索
        """
        faiss_index = faiss.read_index(parallel_tag_faiss_path)
        KNN_distances, KNN_indices = faiss_index.search(tags_feature, k=10)  
        
        print("knn_indices==================")
        print(KNN_indices)
        with open(parallel_tag_knn_log_path, 'w', encoding='utf-8') as f:
            f.write('---------------------------test_sent_id: ' + str(test_id) + ', test_token_id: ' + str(test_token) + '---------------------------\n')
            for ii, test_sample_i in enumerate(test_dataset.showitem(test_id)):
                if ii == 0:
                    f.write("test_token: " + test_dataset.showitem(test_id)[1].split()[test_token])
                    f.write("       model pred: " + test_dataset.showitem(test_id)[4].split()[test_token])
                    f.write("       gold: " + test_dataset.showitem(test_id)[2].split()[test_token] + "\n")
                if ii == 4: break
                f.write(test_sample_i)
            
            f.write("\n=================================================\n\n")
            for id in KNN_indices[0]:
                # 这个id就是在全数据集token列表中的id，需要转换成对应哪条样本的哪个token
                tags_sent_id_list = torch.load(feature_prefix + "parallel_tag_id_sent.tensor")
                tags_token_id_list = torch.load(feature_prefix + "parallel_tag_id_token.tensor")
                sent_id = int(tags_sent_id_list[id].item())
                token_id = int(tags_token_id_list[id].item())
                #print(sent_id)
                #print(token_id)
                knn_sample = parallel_dataset.showitem(sent_id)
                f.write('---------------------------train_sent_id: ' + str(sent_id) + ', train_token_id: ' + str(token_id) + '---------------------------\n')
                for ii, sample_i in enumerate(knn_sample):
                    if ii == 0:
                        f.write("token_chosen_by_knn: " + knn_sample[1].split()[token_id])
                        f.write("       model pred: " + knn_sample[4].split()[token_id])
                        f.write("       gold: OK \n")
                    if ii == 2 or ii == 3: continue  # 平行语料多了这么一句
                    if ii == 4: break
                    f.write(sample_i)
        
        assert 1==2
        """
        # 伪数据搜索
        """
        faiss_index = faiss.read_index(parallel_pseudo_tag_faiss_path)
        KNN_distances, KNN_indices = faiss_index.search(tags_feature, k=10)  
        
        print("knn_indices==================")
        print(KNN_indices)
        with open(parallel_pseudo_tag_knn_log_path, 'w', encoding='utf-8') as f:
            f.write('---------------------------test_sent_id: ' + str(test_id) + ', test_token_id: ' + str(test_token) + '---------------------------\n')
            for ii, test_sample_i in enumerate(test_dataset.showitem(test_id)):
                if ii == 0:
                    f.write("test_token: " + test_dataset.showitem(test_id)[1].split()[test_token])
                    f.write("       model pred: " + test_dataset.showitem(test_id)[4].split()[test_token])
                    f.write("       gold: " + test_dataset.showitem(test_id)[2].split()[test_token] + "\n")
                if ii == 4: break
                f.write(test_sample_i)
            
            f.write("\n=================================================\n\n")
            for id in KNN_indices[0]:
                # 这个id就是在全数据集token列表中的id，需要转换成对应哪条样本的哪个token
                tags_sent_id_list = torch.load(feature_prefix + "parallel_pseudo_tag_id_sent.tensor")
                tags_token_id_list = torch.load(feature_prefix + "parallel_pseudo_tag_id_token.tensor")
                sent_id = int(tags_sent_id_list[id].item())
                token_id = int(tags_token_id_list[id].item())
                knn_sample = parallel_pseudo_dataset.showitem(sent_id)
                f.write('---------------------------train_sent_id: ' + str(sent_id) + ', train_token_id: ' + str(token_id) + '---------------------------\n')
                for ii, sample_i in enumerate(knn_sample):
                    if ii == 0:
                        f.write("token_chosen_by_knn: " + knn_sample[1].split()[token_id])
                        f.write("       model pred: " + knn_sample[4].split()[token_id])
                        f.write("       gold: OK" + knn_sample[2].split()[token_id] + "\n")
                    if ii == 4: break
                    f.write(sample_i)

        assert 1==2
        """


    print("Finale.")


def attribution_qe_train(flags):
    """
    计算【训练集】对【测试集显著性分布】的影响
    """
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    discriminator_model_configs = configs['discriminator_configs']

    data_configs = configs['data_configs']

    # ================================================================================== #
    # Load Data


    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])
    
    train_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['toy_train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['toy_train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['toy_train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['toy_train_data'][3])
    )

    valid_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['toy_train_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['toy_train_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['toy_train_data'][2]),
        QEHterDataset(data_path=data_configs['toy_train_data'][3])
    )

    # 0. load transformer
    INFO('Building transformer model...')
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()

    INFO('Reloading transformer model parameters...')
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params)

    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()

    # ================================================================================== #

    INFO('Begin ...')

    test_id = flags.test_id
    test_token_id = flags.test_token
    influence_path_prefix=flags.influence_path_prefix
    if not os.path.exists(influence_path_prefix):
        os.makedirs(influence_path_prefix)
    influence_path = influence_path_prefix + 'test_' + str(test_id) + '_' + str(test_token_id) + '.train_infl'
    log_show_path = influence_path_prefix + 'log_show_' + str(test_id) + '_' + str(test_token_id) + '.train_infl'
    log_token_path = influence_path_prefix + 'log_token_' + str(test_id) + '_' + str(test_token_id) + '.train_infl'
    
    test_sample = valid_dataset[test_id]
    #print('test_sample')
    #print(valid_dataset.showitem(test_id))

    if flags.saliency_level == 'hter':
        # 得到【所有train样本】 对 【该test_token对 句子级分数 (的显著性)】 的影响
        train_influences, infl_sort_id = cal_influence_single_hter(discriminator,
                                                                    critic_mse,
                                                                    test_sample,
                                                                    train_dataset,
                                                                    test_token_id,
                                                                    influence_path,
                                                                    recursion_depth=flags.recursion_depth,
                                                                    gpu=flags.use_gpu,
                                                                    damp=flags.damp,
                                                                    scale=flags.scale,
                                                                    result_instead_of_loss=False,
                                                                    not_saliency=True) # 先不设flag参数，直接在这里改一下
    elif flags.saliency_level == 'token':
        # 得到【所有train样本】 对 【该test_token对 某token预测 (的显著性)】 的影响
        train_influences, infl_sort_id = cal_influence_single_token(discriminator,
                                                                    critic,
                                                                    test_sample,
                                                                    train_dataset,
                                                                    test_token_id,
                                                                    influence_path,
                                                                    recursion_depth=flags.recursion_depth,
                                                                    gpu=flags.use_gpu,
                                                                    damp=flags.damp,
                                                                    scale=flags.scale,
                                                                    not_saliency=True,
                                                                    log_path=log_token_path)
    elif flags.saliency_level == 'both':
        # 得到【所有train样本】 对 【某test样本预测】 的影响，不管是词标记还是句子分数，loss使用hter+token
        train_influences, infl_sort_id = cal_influence_single_both(discriminator,
                                                                    critic,
                                                                    critic_mse,
                                                                    test_sample,
                                                                    train_dataset,
                                                                    influence_path,
                                                                    recursion_depth=flags.recursion_depth,
                                                                    gpu=flags.use_gpu,
                                                                    damp=flags.damp,
                                                                    scale=flags.scale)
    else:
        raise ValueError('check saliency_level please')

    print(train_influences)
    print(infl_sort_id)
    #train_influences = torch.load(influence_path)
    #print(train_influences)
    #print("=============================== 20 most influential train samples")
    
    with open(log_show_path, 'w', encoding='utf-8') as logf:
        logf.write(' '.join([str(id) for id in infl_sort_id]) + '\n')
        for id in infl_sort_id:
            infl_sample = train_dataset.showitem(id)
            #print(str(id) + '---------------------------')
            #print(infl_sample)
            logf.write('---------------------------train id: ' + str(id) + '---------------------------\n')
            for sample_i in infl_sample:
                logf.write(sample_i)

    print("all done")


def attribution_qe_train_detailed(flags):
    """
    在对【测试集显著性分布】的影响最大的几个【训练集样本】中，
    分别计算样本内token的影响分布，并画图
    """
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    discriminator_model_configs = configs['discriminator_configs']

    data_configs = configs['data_configs']

    # ================================================================================== #
    # Load Data


    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])
    
    train_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3])
    )

    valid_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['test_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['test_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['test_data'][2]),
        QEHterDataset(data_path=data_configs['test_data'][3])
    )
    
    # 0. load transformer
    INFO('Building transformer model...')
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()

    INFO('Reloading transformer model parameters...')
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params)

    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()
    
    # ================================================================================== #

    INFO('Begin ...')

    # ================================================================================== #
    # tsne
    train_id_list = [3922,1114,1180]
    token_id_list = [2,0,7]
    representation_list = []

    for train_id,token_id in zip(train_id_list,token_id_list):
        train_sample = train_dataset[train_id]
        seqs_x, seqs_y, xy_label, xy_hter = train_sample
        src_input, mt_input, tag = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=flags.use_gpu)
        y_inp = mt_input[:, :].contiguous()
        discriminator.eval()

        with torch.no_grad():
            # 提取表示
            representation = discriminator(src_input, y_inp, get_representation=True).squeeze(0)[2:-1]
            representation_list.append(representation[token_id].unsqueeze(0))
            
    # print(representation_list)
    representation_list = torch.cat(representation_list, 0).cpu()
    print(representation_list.size())

    from sklearn import manifold,datasets
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(representation_list)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], color=plt.cm.Set1(y[i]), 
                fontdict={'weight': 'bold', 'size': 9})
    # 这几个表示tsne一下
    plt.savefig('/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/occupy/pplt.jpg')

    assert 1==2


    # ================================================================================== #
    # 借用一下下，摘出部分完全见过的测试样本
    import string
    import re

    # enzh
    data_prefix = "/home/user_data_182b/yanym/qe/data/wmt20_enzh/task2_post_edit/need_now/"
    train_mt_file = "train.mt"
    test_mt_file = "test.mt"
    # ende
    data_prefix = "/home/user_data_182b/yanym/qe/data/wmt19_ende/qe_data_bpe/"
    train_mt_file = "train.mt.BPE"
    test_mt_file = "test.mt.BPE"

    
    # 得到训练词汇集合
    train_wordset = set()
    with open(data_prefix+train_mt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.lower()
            tokens = re.split(r"[ ]+", line.strip('\n'))
            if tokens == []: continue
            for w in tokens:
                #w = w.strip(string.punctuation)
                if w == '': continue
                train_wordset.add(w)

    #print(train_wordset)


    # 逐条分析测试样本中的词是否都在训练时出现过
    #all_in_num = 0
    seen_ids = []
    unseen_ids = []
    with open(data_prefix+test_mt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for index,line in enumerate(lines):
            line = line.lower()
            tokens = re.split(r"[ ]+", line.strip('\n'))
            all_in_flag = True
            for token in tokens:
                if token not in train_wordset:
                    all_in_flag = False
                    break
            if all_in_flag == True:
                #all_in_num += 1
                seen_ids.append(index)
            else:
                unseen_ids.append(index)
    
    # 随机seen_ids
    """
    with open(data_prefix+test_mt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        len_testset = len(lines)
        print(len_testset)
    set_seed(10086)
    seen_ids = np.random.choice(range(len_testset), size=89, replace=False)
    """
    # enzh
    seen_test_prefix = '/home/user_data_182b/yanym/qe/data/wmt20_enzh/task2_post_edit/need_now/seen_test/seen_test'
    unseen_test_prefix = '/home/user_data_182b/yanym/qe/data/wmt20_enzh/task2_post_edit/need_now/seen_test/unseen_test'
    # ende
    seen_test_prefix = '/home/user_data_182b/yanym/qe/data/wmt19_ende/qe_data_bpe/seen_test/seen_test'
    unseen_test_prefix = '/home/user_data_182b/yanym/qe/data/wmt19_ende/qe_data_bpe/seen_test/unseen_test'
    
    for which, part in enumerate([".src.BPE", ".mt", ".tag", ".hter"]):
        with open(seen_test_prefix + part, 'w', encoding='utf-8') as logf:
            for id in seen_ids:
                infl_sample = valid_dataset.showitem(id)
                logf.write(infl_sample[which])
        with open(unseen_test_prefix + part, 'w', encoding='utf-8') as logf:
            for id in unseen_ids:
                infl_sample = valid_dataset.showitem(id)
                logf.write(infl_sample[which])

    assert 1==2

    # ================================================================================== #

    # 借用一下下 infl_tensor_2_log
    """
    influence_path_prefix = "/home/user_data_182b/yanym/qe/save/qe_analysis_influence/directqe_ende_gi_weight100_5000/"

    for f_str in ['0_1', '1_0', '2_3', '3_4']:
        infl_path = influence_path_prefix + 'test_' + f_str + '.train_infl'
        train_influences = torch.load(infl_path)
        infl_sort_id = sorted(enumerate(train_influences), key=lambda x:x[1], reverse=True)
        infl_sort_id = [x[0] for x in infl_sort_id]

        log_path = influence_path_prefix + 'log_' + f_str + '.train_infl'
        with open(log_path, 'w', encoding='utf-8') as logf:
            logf.write(' '.join([str(id) for id in infl_sort_id]) + '\n\n')
            for id in infl_sort_id:
                infl_sample = train_dataset.showitem(id)
                logf.write('---------------------------train id: ' + str(id) + '---------------------------\n')
                for sample_i in infl_sample:
                    logf.write(sample_i)

    assert 1==2
    """

    # 借用一下下 对比平行语料和qe语料
    """
    influence_path_prefix = "/home/user_data_182b/yanym/qe/save/qe_analysis_influence/directqe_ende_gi_cat_no_weight_5000_scale100/"

    for f_str in ['0_1', '1_0', '2_3', '3_4']:
        infl_path = influence_path_prefix + 'test_' + f_str + '.cat_train_infl'
        train_influences = torch.load(infl_path)
        infl_sort_id = sorted(enumerate(train_influences), key=lambda x:x[1], reverse=True)
        infl_sort_id = [x[0] for x in infl_sort_id]

        num_parallel = 30000
        # 文件形式
        
        #cmp_path = influence_path_prefix + 'cmp_' + f_str + '.cat_train_infl'
        #with open(cmp_path, 'w', encoding='utf-8') as fcmp:
        #    for id in infl_sort_id:
        #        if id < num_parallel:
        #            fcmp.write('parallel\n')
        #        else:
        #            fcmp.write('qe\n')
        

        # 列表形式
        from_qe_list = [0 if id < num_parallel else 1 for id in infl_sort_id]
        n_first = 20
        qe_prop_first = sum(from_qe_list[:n_first]) / n_first   # 影响力最大的前n个样本 来自qe数据的比例
        print(f_str)
        print('qe_prop_first 20')
        print(qe_prop_first)
        n_first = 100
        qe_prop_first = sum(from_qe_list[:n_first]) / n_first   # 影响力最大的前n个样本 来自qe数据的比例
        print('qe_prop_first 100')
        print(qe_prop_first)

        import math

        qe_prop = []
        parallel_prop = []
        tick_label = []
        every = 5000
        x = np.arange(math.ceil(len(from_qe_list)/every))
        for i in range(math.ceil(len(from_qe_list)/every)):
            qe_num = sum(from_qe_list[i*every:(i+1)*every])
            qe_prop.append(qe_num / every)
            parallel_prop.append(1 - qe_num / every)
            tick_label.append(str(i*every) + '-' + str((i+1)*every))
        bar_width = 0.3

        #绘制并列柱状图
        f, ax = plt.subplots(figsize = (10,10))
        plt.bar(x,qe_prop,bar_width,color='salmon',label='qe')
        plt.bar(x+bar_width,parallel_prop,bar_width,color='orchid',label='parallels')

        plt.legend()#显示图例，即label
        plt.xticks(x+bar_width/2,tick_label, rotation=90)#显示x坐标轴的标签,即tick_label,调整位置，使其落在两个直方图中间位置
        plt.savefig(influence_path_prefix + 'bar_' + f_str + '.jpg')

    assert 1==2
    """

    # ================================================================================== #

    num_first = flags.num_first
    test_id = flags.test_id
    test_token_id = flags.test_token
    influence_path_prefix=flags.influence_path_prefix
    influence_path_prefix = "/home/user_data_182b/yanym/qe/save/qe_analysis_influence/directqe_ende/"

    influence_path = influence_path_prefix + 'test_' + str(test_id) + '_' + str(test_token_id) + '.train_infl'
    
    test_sample = valid_dataset[test_id]

    # ================================================================================== #
    # 借用一下，计算某test loss
    
    for id in [1]:
        loss_score = cal_loss_z_hter(discriminator,
                                                critic_mse,
                                                valid_dataset[id],
                                                gpu=flags.use_gpu)
        
        print('___________________test_id: '+str(id))
        print(loss_score)
        loss_score *= 100
        print(loss_score)
    
    assert 1==2

    # 加载训练样本影响力id排名
    influence_path = "/home/user_data_182b/yanym/qe/save/qe_analysis_influence_enzh/directqe_enzh_sample5000/test_2_1.train_infl"
    infl = torch.load(influence_path)
    infl_sort_id = sorted(enumerate(infl), key=lambda x:x[1], reverse=True)
    infl_sort_id = [x[0] for x in infl_sort_id]

    infl_sort_id = infl_sort_id[:6300]  # 去掉最negative的10%
    #infl_sort_id = infl_sort_id[700:]   # 去掉最positive的10%
    
    # 根据影响力，选择部分train
    new_train_path_prefix = "/home/user_data_182b/yanym/qe/data/wmt20_enzh/task2_post_edit/need_now/new_train/"
    for which, part in enumerate([".src.BPE", ".mt", ".tag", ".hter"]):
        with open(new_train_path_prefix + "wo_n_2" + part, 'w', encoding='utf-8') as logf:
            for id in infl_sort_id:
                infl_sample = train_dataset.showitem(id)
                logf.write(infl_sample[which])
    
    

    
    
    assert 1==2

    # ================================================================================== #
    # 借用一下，计算ner相关性
    en_ner_path = ""
    with open(en_ner_path, 'r', encoding=utf-8) as f_ner:
        ner_dict = json.load(f_ner)
    
    # 或许分段可以每1000个一分？
    which_ner_stat = []  # 7项，每项代表这1000个样本中 包含某ner的样本的数量
    for id in infl_sort_id:
        if which_ner in ner_dict[id]:
            which_ner_stat[id//1000] += 1

    # 画图
    """
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(x, x, label='linear')  # Plot some data on the axes.
    ax.plot(x, x**2, label='quadratic')  # Plot more data on the axes...
    ax.plot(x, x**3, label='cubic')  # ... and some more.
    ax.set_xlabel('x label')  # Add an x-label to the axes.
    ax.set_ylabel('y label')  # Add a y-label to the axes.
    ax.set_title("Simple Plot")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    """
    f, ax = plt.subplots(figsize=(10,10))
    ax.plot(range(7), which_ner_stat, label=str(which_ner))
    ax.set_title("ner distribution stat")
    ax.legend()



    assert 1==2

    # ================================================================================== #

    recursion_depth=flags.recursion_depth
    #plot_path_prefix=flags.plot_path_prefix
    plot_path_prefix = influence_path_prefix + 'plot/'
    if not os.path.exists(plot_path_prefix):
        os.makedirs(plot_path_prefix)

    # s_test_vec只算一次即可
    if flags.saliency_level == 'hter':
        s_test_vec = cal_s_test_hter_not_saliency(discriminator, # 直接在这里改一下
                                        critic_mse,
                                        test_sample,
                                        train_dataset,
                                        test_token_id,
                                        recursion_depth,
                                        gpu=flags.use_gpu)
    elif flags.saliency_level == 'token':
        s_test_vec = cal_s_test_token(discriminator,
                                        critic,
                                        test_sample,
                                        train_dataset,
                                        test_token_id,
                                        recursion_depth,
                                        gpu=flags.use_gpu)
    else:
        raise ValueError('check saliency_level please')

    # 加载训练样本影响力id排名
    infl = torch.load(influence_path)
    infl_sort_id = sorted(enumerate(infl), key=lambda x:x[1], reverse=True)
    infl_sort_id = [x[0] for x in infl_sort_id]

    # 计算【对该test_token的显著性影响力最大 的前n个train样本】的影响分布
    for rank, train_id in enumerate(infl_sort_id[:num_first]):
        print("======================" + str(train_id))
        train_sample = train_dataset[train_id]
        mt_tokens = train_sample[1][0]
        # 这一个训练样本的影响分布 train_influence_saliency [mt_len]
        if flags.saliency_level == 'hter':
            train_influence_saliency = cal_influence_single_detailed_hter(discriminator,
                                                                        critic,
                                                                        critic_mse,
                                                                        test_sample,
                                                                        test_token_id,
                                                                        train_dataset,
                                                                        s_test_vec=s_test_vec,
                                                                        train_sample_id=train_id,
                                                                        gpu=flags.use_gpu)
        elif flags.saliency_level == 'token':
            # 此分支已作废
            train_influence_saliency = cal_influence_single_detailed_token(discriminator,
                                                                        critic,
                                                                        critic_mse,
                                                                        test_sample,
                                                                        test_token_id,
                                                                        train_dataset,
                                                                        s_test_vec=s_test_vec,
                                                                        train_sample_id=train_id,
                                                                        gpu=flags.use_gpu)
        
        train_influence_saliency = F.softmax(train_influence_saliency, dim=-1)
        
        plot_attribution1(train_influence_saliency, plot_path_prefix, \
                        mt_tokens, vocab_tgt, test_id, test_token_id, rank, train_id)
    print("all done")


def part_trainset(flags):
    """
    根据对某test样本的 训练集影响力排名，选择部分训练集，保存下来
    """
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    data_configs = configs['data_configs']

    # ================================================================================== #
    # Load Data


    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])
    
    train_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3])
    )

    # ================================================================================== #

    test_id = flags.test_id
    test_token_id = flags.test_token
    influence_path_prefix=flags.influence_path_prefix
    influence_path = influence_path_prefix + 'test_' + str(test_id) + '_' + str(test_token_id) + '.train_infl'
    
    # ================================================================================== #
    # 加载训练样本影响力id排名
    infl = torch.load(influence_path)
    infl_sort_id = sorted(enumerate(infl), key=lambda x:x[1], reverse=True)
    infl_sort_id = [x[0] for x in infl_sort_id]

    
    infl_sort_id_wo_n = infl_sort_id[:6300]  # 去掉最negative的10%
    infl_sort_id_wo_p = infl_sort_id[700:]   # 去掉最positive的10%
    
    # 根据影响力，选择部分train
    part_train_path_prefix = influence_path_prefix + "part_train_test_" + str(test_id) + '/'
    if not os.path.exists(part_train_path_prefix):
        os.makedirs(part_train_path_prefix)
    for which, part in enumerate([".src.BPE", ".mt", ".tag", ".hter"]):
        with open(part_train_path_prefix + "wo_n_test" + str(test_id) + part, 'w', encoding='utf-8') as logf:
            for id in infl_sort_id_wo_n:
                infl_sample = train_dataset.showitem(id)
                logf.write(infl_sample[which])
        with open(part_train_path_prefix + "wo_p_test" + str(test_id) + part, 'w', encoding='utf-8') as logf:
            for id in infl_sort_id_wo_p:
                infl_sample = train_dataset.showitem(id)
                logf.write(infl_sample[which])

    print("all done")


def infl_tools(flags):
    """
    工具在这里啦
    """
    Constants.USE_GPU = flags.use_gpu

    flags.batch_size = 1

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)

    discriminator_model_configs = configs['discriminator_configs']

    data_configs = configs['data_configs']

    # ================================================================================== #
    # Load Data


    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])
    
    train_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3])
    )

    valid_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['test_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['test_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        QETagsDataset(data_path=data_configs['test_data'][2]),
        QEHterDataset(data_path=data_configs['test_data'][3])
    )

    # 0. load transformer
    INFO('Building transformer model...')
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()

    INFO('Reloading transformer model parameters...')
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params)

    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic)
    INFO(critic_mse)

    if Constants.USE_GPU:
        discriminator.cuda()
        critic = critic.cuda()
        critic_mse = critic_mse.cuda()

    # ================================================================================== #

    INFO('Begin ...')


    # ================================================================================== #
    # 计算某test loss
    if flags.task == 'cal_test_loss':
        INFO('cal test loss')
        test_id = flags.test_id
        #test_token_id = flags.test_token
        #influence_path_prefix=flags.influence_path_prefix
        #influence_path = influence_path_prefix + 'test_' + str(test_id) + '_' + str(test_token_id) + '.train_infl'
        #test_sample = valid_dataset[test_id]
        
        loss_score = cal_loss_z_hter(discriminator,
                                                critic_mse,
                                                valid_dataset[test_id],
                                                gpu=flags.use_gpu)
            
        print('___________________test_id: '+str(test_id))
        print(loss_score)
        loss_score *= 100
        print(loss_score)
    
    # ================================================================================== #

    


    print("all done")


def attribution_ner_analysis(flags):
    """
    计算对某test样本影响力大的训练样本 是否在ner上有相似性
    比如是否能验证 带有同类型词汇的样本 训练更有帮助 这样的猜想
    """
    Constants.USE_GPU = flags.use_gpu

    config_path = os.path.abspath(flags.config_path)

    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)


    data_configs = configs['data_configs']

    # ================================================================================== #
    # Load Data


    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])
    
    train_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        QETagsDataset(data_path=data_configs['train_data'][2],
                      max_len=data_configs['max_len'][1]
                      ),
        QEHterDataset(data_path=data_configs['train_data'][3])
    )

    # ================================================================================== #

    INFO('Begin ...')

    test_id = flags.test_id
    test_token_id = flags.test_token
    # test_sample = valid_dataset[test_id]

    # 加载测试样本和训练集的ner字典
    ner_path_prefix = flags.ner_path_prefix
    test_ner_path = ner_path_prefix + 'test.' + flags.ner_lang + '.ner.json'
    train_ner_path = ner_path_prefix + 'train.' + flags.ner_lang + '.ner.json'
    with open(test_ner_path, 'r', encoding='utf-8') as test_ner:
        test_ner_dict = json.load(test_ner)
        test_ners = set(test_ner_dict[str(test_id)])
        if len(test_ners) == 0:
            print('this test sample has no ner')
            return
    with open(train_ner_path, 'r', encoding='utf-8') as train_ner:
        train_ner_dict = json.load(train_ner)
        num_train = len(train_ner_dict)


    # 加载训练样本影响力id排名
    influence_path_prefix = flags.influence_path_prefix
    influence_path = influence_path_prefix + 'test_' + str(test_id) + '_' + str(test_token_id) + '.train_infl'
    infl = torch.load(influence_path)
    infl_sort_id = sorted(enumerate(infl), key=lambda x:x[1], reverse=True)  # x[1]是影响力数值
    infl_sort_id = [x[0] for x in infl_sort_id]                              # x[0]是样本id
    #print("aaaaaaaaaaaaaaaaaa")
    #print(infl_sort_id[:10])  # 这里没问题
    
    print("================================  test_id: %d" % test_id)
    all_test_ner_stat = {}
    for test_ner in test_ners:  # 这个test样本中包含的ner种类
        # 或许分段可以每1000个一分？
        test_ner_stat = np.zeros((math.ceil(num_train/1000)))  # 7项，每项代表这1000个样本中 包含该ner的样本的数量
        for index,id in enumerate(infl_sort_id):
            if test_ner in train_ner_dict[str(id)]:
                test_ner_stat[index//1000] += 1
        all_test_ner_stat[test_ner] = test_ner_stat
        print(test_ner)
        print(test_ner_stat)
    #print(all_test_ner_stat)

    #assert 1==2
    # 画图
    """
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(x, x, label='linear')  # Plot some data on the axes.
    ax.plot(x, x**2, label='quadratic')  # Plot more data on the axes...
    ax.plot(x, x**3, label='cubic')  # ... and some more.
    ax.set_xlabel('x label')  # Add an x-label to the axes.
    ax.set_ylabel('y label')  # Add a y-label to the axes.
    ax.set_title("Simple Plot")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    """
    
    f, ax = plt.subplots(figsize=(10,10))
    for test_ner,test_ner_stat in all_test_ner_stat.items():
        ax.plot(range(7), test_ner_stat, label=test_ner)
    ax.set_title("NER distribution statistics - test_{}".format(test_id))
    ax.legend()

    tick_label = ['0']          # 为了偏移
    every = 1000
    for i in range(math.ceil(num_train/every)):
        tick_label.append(str(i*every) + '-' + str((i+1)*every))
    print(tick_label)
    ax.set_xlabel('influence rank')
    ax.set_xticklabels(tick_label, rotation=90)
    ax.set_ylabel('num of ner')

    plt.savefig(influence_path_prefix + "infl_ner_test_{}.jpg".format(test_id))



    print("all done")


def train_probe(flags):
    # 从finetune上改

    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(flags.saveto,
                                     "probe" + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])
    vocab_pos = Vocabulary.build_from_file(**data_configs['vocabularies'][2])

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    train_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['train_data'][0],
                        vocabulary=vocab_src,
                        max_len=data_configs['max_len'][0],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][1],
                        vocabulary=vocab_tgt,
                        max_len=data_configs['max_len'][1],
                        ),
        TextLineDataset(data_path=data_configs['train_data'][2],
                        vocabulary=vocab_pos,
                        max_len=data_configs['max_len'][1],
                        )
    )

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['valid_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        TextLineDataset(data_path=data_configs['valid_data'][2],
                        vocabulary=vocab_pos,
                        max_len=data_configs['max_len'][1],
                        )
    )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'])

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params)

    # 0.5. build probe model
    input_dim = 512
    num_pos_class = 16
    probe = LinearProbe(input_dim, num_pos_class)

    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    INFO(critic)

    if Constants.USE_GPU:
        discriminator.cuda()
        probe.cuda()
        critic = critic.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=probe,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None

    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=flags.log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        train_loss = 0
        id = 1
        for batch in training_iter:
            #print("******************* {}".format(id))
            id += 1
            discriminator.eval()
            uidx += 1
            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y, xy_label = batch
            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)
            cum_samples += n_samples_t
            cum_words += n_words_t
            training_progress_bar.update(n_samples_t)

            x, y, xy_label = prepare_data_qe(seqs_x, seqs_y, xy_label, cuda=Constants.USE_GPU)

            loss = compute_forward_probe(qe_model=discriminator,
                                            probe_model=probe,
                                            critic=critic,
                                            seqs_x=x,
                                            seqs_y=y,
                                            pos_line=xy_label,
                                            representation_type=flags.representation)                                                

            train_loss += loss
            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss: {1:.4f} ".format(
                    uidx, train_loss / training_configs['disp_freq']
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
                train_loss = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                valid_loss, valid_acc = \
                    loss_evaluation_probe(discriminator=discriminator,
                                       probe=probe,
                                       critic=critic,
                                       valid_iterator=valid_iterator,
                                       representation=flags.representation)

                # model_collections.add_to_collection("history_losses", valid_loss)
                model_collections.add_to_collection("history_losses", valid_loss)
                min_loss = np.array(model_collections.get_collection("history_losses")).min()

                if valid_loss <= min_loss:
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(probe.state_dict(), best_model_prefix + ".final")
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} Acc: {2:.4f} patience: {3}".
                     format(uidx, valid_loss, valid_acc, bad_count))
            if is_early_stop is True:
                break
        if is_early_stop is True:
            break
        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def test_probe(flags):
    # 测试探针效果
    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    training_configs = configs['training_configs']

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])
    vocab_pos = Vocabulary.build_from_file(**data_configs['vocabularies'][2])

    valid_bitext_dataset = ZipDataset(
        TextLineDataset(data_path=data_configs['test_data'][0],
                        vocabulary=vocab_src,
                        ),
        TextLineDataset(data_path=data_configs['test_data'][1],
                        vocabulary=vocab_tgt,
                        ),
        TextLineDataset(data_path=data_configs['test_data'][2],
                        vocabulary=vocab_pos,
                        max_len=data_configs['max_len'][1],
                        )
    )

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params)

    # 0.5. load probe model
    input_dim = 512
    num_pos_class = 16
    probe = LinearProbe(input_dim, num_pos_class)
    params = load_model_parameters(flags.pretrain_path_probe, map_location="cpu")
    probe.load_state_dict(params)

    critic = NMTCriterion(padding_idx=0, ignore_idx=0)
    INFO(critic)

    if Constants.USE_GPU:
        discriminator.cuda()
        probe.cuda()
        critic = critic.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #

    INFO('Begin testing...')

    timer.tic()

    valid_loss, valid_acc = \
                    loss_evaluation_probe(discriminator=discriminator,
                                       probe=probe,
                                       critic=critic,
                                       valid_iterator=valid_iterator,
                                       representation=flags.representation)

    INFO("Test Loss: {0:.2f} Test Acc: {1:.4f}".
                     format(valid_loss, valid_acc))
    

def train_probe_len(flags):

    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    optimizer_configs = configs['optimizer_configs']
    training_configs = configs['training_configs']

    Constants.SEED = training_configs['seed']
    # Constants.SEED = flags.seed

    set_seed(Constants.SEED)

    best_model_prefix = os.path.join(flags.saveto,
                                     "probe_len" + Constants.MY_BEST_MODEL_SUFFIX)
    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    train_batch_size = training_configs["batch_size"]
    train_buffer_size = training_configs["buffer_size"]

    if flags.len_to_pred == 'mt':
        train_bitext_dataset = ZipDataset(
            TextLineDataset(data_path=data_configs['train_data'][0],
                            vocabulary=vocab_src,
                            max_len=data_configs['max_len'][0],
                            ),
            TextLineDataset(data_path=data_configs['train_data'][1],
                            vocabulary=vocab_tgt,
                            max_len=data_configs['max_len'][1],
                            ),
            QEHterDataset(data_path=data_configs['train_data'][3])
        )

        valid_bitext_dataset = ZipDataset(
            TextLineDataset(data_path=data_configs['valid_data'][0],
                            vocabulary=vocab_src,
                            ),
            TextLineDataset(data_path=data_configs['valid_data'][1],
                            vocabulary=vocab_tgt,
                            ),
            QEHterDataset(data_path=data_configs['valid_data'][3])
        )
    elif flags.len_to_pred == 'pe':
        train_bitext_dataset = ZipDataset(
            TextLineDataset(data_path=data_configs['train_data'][0],
                            vocabulary=vocab_src,
                            max_len=data_configs['max_len'][0],
                            ),
            TextLineDataset(data_path=data_configs['train_data'][1],
                            vocabulary=vocab_tgt,
                            max_len=data_configs['max_len'][1],
                            ),
            QEHterDataset(data_path=data_configs['train_data'][4])
        )

        valid_bitext_dataset = ZipDataset(
            TextLineDataset(data_path=data_configs['valid_data'][0],
                            vocabulary=vocab_src,
                            ),
            TextLineDataset(data_path=data_configs['valid_data'][1],
                            vocabulary=vocab_tgt,
                            ),
            QEHterDataset(data_path=data_configs['valid_data'][4])
        )

    training_iterator = DataIterator(dataset=train_bitext_dataset,
                                     batch_size=train_batch_size,
                                     use_bucket=training_configs['use_bucket'],
                                     buffer_size=train_buffer_size,
                                     batching_func=training_configs['batching_key'])

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================ Begin ======================================== #
    # Build Model & Optimizer
    # We would do steps below on after another
    #     0. load transformer
    #     1. build models & criterion
    #     2. move models & criterion to gpu if needed
    #     3. load pre-trained model if needed
    #     4. build optimizer
    #     5. build learning rate scheduler if needed
    #     6. load checkpoints if needed

    # 00. Initial
    model_collections = Collections()

    # 0. load transformer
    INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Reloading model parameters...')
    timer.tic()
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params)
    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)
    INFO(critic_mse)

    # 0.5. build probe model
    input_dim = 512
    output_dim = 1
    probe = LinearProbe(input_dim, output_dim)

    if Constants.USE_GPU:
        discriminator.cuda()
        probe.cuda()
        critic_mse = critic_mse.cuda()
    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 4. Build optimizer
    INFO('Building Optimizer...')
    optim = Optimizer(name=optimizer_configs['optimizer'],
                      model=probe,
                      lr=lrate,
                      grad_clip=optimizer_configs['grad_clip'],
                      weight_decay=optimizer_configs['weight_decay'],
                      optim_args=optimizer_configs['optimizer_params']
                      )

    # 5. Build scheduler for optimizer if needed
    if optimizer_configs['schedule_method'] is not None:
        scheduler = build_scheduler(schedule_method=optimizer_configs['schedule_method'],
                                    optimizer=optim, scheduler_configs=optimizer_configs['scheduler_configs'])
    else:
        scheduler = None

    # 6. build moving average
    ma = None

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    eidx = model_collections.get_collection("eidx", [0])[-1]
    uidx = model_collections.get_collection("uidx", [0])[-1]
    bad_count = model_collections.get_collection("bad_count", [0])[-1]
    oom_count = model_collections.get_collection("oom_count", [0])[-1]

    summary_writer = SummaryWriter(log_dir=flags.log_path)

    cum_samples = 0
    cum_words = 0

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    while True:

        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)
        # Build iterator and progress bar
        training_iter = training_iterator.build_generator()
        training_progress_bar = tqdm(desc='  - (Epoch %d)   ' % eidx,
                                     total=len(training_iterator),
                                     unit="sents")
        train_loss = 0
        id = 1
        for batch in training_iter:
            #print("******************* {}".format(id))
            id += 1
            discriminator.eval()
            uidx += 1
            if optimizer_configs["schedule_method"] is not None and optimizer_configs["schedule_method"] != "loss":
                scheduler.step(global_step=uidx)

            seqs_x, seqs_y, seqs_len = batch
            n_samples_t = len(seqs_x)
            n_words_t = sum(len(s) for s in seqs_y)
            cum_samples += n_samples_t
            cum_words += n_words_t
            training_progress_bar.update(n_samples_t)

            x, y = prepare_data_qe(seqs_x, seqs_y, cuda=Constants.USE_GPU)
            if Constants.USE_GPU:
                seqs_len = torch.tensor(seqs_len).cuda()
            seqs_len = seqs_len.squeeze(0)
            loss_seqlen = compute_forward_probe_seqlen(qe_model=discriminator,
                                            probe_model=probe,
                                            critic_mse=critic_mse,
                                            seqs_x=x,
                                            seqs_y=y,
                                            len_label=seqs_len,
                                            representation_type=flags.representation)                                                

            train_loss += loss_seqlen
            # --------------------------------------------
            if (uidx) % training_configs['update_cycle'] == 0:
                optim.step()
                optim.zero_grad()

            if ma is not None and eidx >= training_configs['moving_average_start_epoch']:
                ma.step()

            # ================================================================================== #
            # Display some information
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['disp_freq']):
                # words per second and sents per second
                words_per_sec = cum_words / (timer.toc(return_seconds=True))
                sents_per_sec = cum_samples / (timer.toc(return_seconds=True))
                lrate = list(optim.get_lrate())[0]

                summary_writer.add_scalar("Speed(words/sec)", scalar_value=words_per_sec, global_step=uidx)
                summary_writer.add_scalar("Speed(sents/sen)", scalar_value=sents_per_sec, global_step=uidx)
                summary_writer.add_scalar("lrate", scalar_value=lrate, global_step=uidx)
                summary_writer.add_scalar("oom_count", scalar_value=oom_count, global_step=uidx)
                INFO("{0} TrainLoss_seqlen: {1:.4f} ".format(
                    uidx, train_loss / training_configs['disp_freq']
                ))

                # Reset timer
                timer.tic()
                cum_words = 0
                cum_samples = 0
                train_loss = 0

            # ================================================================================== #
            # Saving checkpoints
            if should_trigger_by_steps(uidx, eidx, every_n_step=training_configs['save_freq'], debug=flags.debug):
                model_collections.add_to_collection("uidx", uidx)
                model_collections.add_to_collection("eidx", eidx)
                model_collections.add_to_collection("bad_count", bad_count)

            # ================================================================================== #
            # Loss Validation & Learning rate annealing
            if should_trigger_by_steps(global_step=uidx, n_epoch=eidx, every_n_step=training_configs['loss_valid_freq'],
                                       debug=flags.debug):

                valid_loss, p = \
                    loss_evaluation_probe_seqlen(discriminator=discriminator,
                                       probe=probe,
                                       critic_mse=critic_mse,
                                       valid_iterator=valid_iterator,
                                       representation=flags.representation)

                # model_collections.add_to_collection("history_losses", valid_loss)
                model_collections.add_to_collection("history_losses", valid_loss)
                min_loss = np.array(model_collections.get_collection("history_losses")).min()

                if valid_loss <= min_loss:
                    bad_count = 0
                    if is_early_stop is False:
                        # 1. save the best model
                        torch.save(probe.state_dict(), best_model_prefix + ".final")
                else:
                    bad_count += 1
                    # At least one epoch should be traversed
                    if bad_count >= training_configs['early_stop_patience'] and eidx > 0:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                INFO("{0} Loss: {1:.2f} pearsonr: {2:.2f} patience: {3}".
                     format(uidx, valid_loss, p, bad_count))
            if is_early_stop is True:
                break
        if is_early_stop is True:
            break
        training_progress_bar.close()

        eidx += 1
        if eidx > training_configs["max_epochs"]:
            break


def test_probe_len(flags):
    # 测试探针效果
    # write log of training to file.
    write_log_to_file(os.path.join(flags.log_path, "%s.log" % time.strftime("%Y%m%d-%H%M%S")))
    Constants.USE_GPU = flags.use_gpu
    config_path = os.path.abspath(flags.config_path)
    with open(config_path.strip()) as f:
        configs = yaml.safe_load(f)
    # INFO(pretty_configs(configs))
    # Add default configs
    configs = prepare_configs(flags.config_path, flags.predefined_config)
    discriminator_model_configs = configs['discriminator_configs']
    data_configs = configs['data_configs']
    training_configs = configs['training_configs']

    timer = Timer()

    # ================================================================================== #
    # Load Data

    # INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
    vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

    if flags.len_to_pred == 'mt':
        valid_bitext_dataset = ZipDataset(
            TextLineDataset(data_path=data_configs['test_data'][0],
                            vocabulary=vocab_src,
                            ),
            TextLineDataset(data_path=data_configs['test_data'][1],
                            vocabulary=vocab_tgt,
                            ),
            QEHterDataset(data_path=data_configs['test_data'][3])
        )
    elif flags.len_to_pred == 'pe':
        valid_bitext_dataset = ZipDataset(
            TextLineDataset(data_path=data_configs['test_data'][0],
                            vocabulary=vocab_src,
                            ),
            TextLineDataset(data_path=data_configs['test_data'][1],
                            vocabulary=vocab_tgt,
                            ),
            QEHterDataset(data_path=data_configs['test_data'][4])
        )

    valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                  batch_size=training_configs['valid_batch_size'],
                                  use_bucket=False, numbering=False)

    # INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # 0. load transformer
    # INFO('Building transformer model...')
    timer.tic()
    discriminator = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, **discriminator_model_configs)
    discriminator.eval()
    # INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # INFO('Reloading model parameters...')
    timer.tic()
    params = load_model_parameters(flags.pretrain_path, map_location="cpu")
    discriminator.load_state_dict(params)

    critic_mse = torch.nn.MSELoss(reduce=True, size_average=True)

    # 0.5. load probe model
    input_dim = 512
    output_dim = 1
    probe = LinearProbe(input_dim, output_dim)
    params = load_model_parameters(flags.pretrain_path_probe, map_location="cpu")
    probe.load_state_dict(params)

    if Constants.USE_GPU:
        discriminator.cuda()
        probe.cuda()
        critic_mse = critic_mse.cuda()

    # ================================================================================== #

    # INFO('Begin testing...')

    timer.tic()

    valid_loss, p = \
            loss_evaluation_probe_seqlen(discriminator=discriminator,
                                       probe=probe,
                                       critic_mse=critic_mse,
                                       valid_iterator=valid_iterator,
                                       representation=flags.representation)

    INFO("Test Loss: {0:.2f} pearsonr: {1:.4f}".
                     format(valid_loss, p))
    