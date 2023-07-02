from __future__ import absolute_import, division, print_function

import glob
import logging
import math
import os
import random
import shutil
import tempfile
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn
import torch.nn.functional as F
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm.auto import tqdm, trange
from transformers import (
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
)
from transformers.convert_graph_to_onnx import convert, quantize
from transformers.optimization import AdamW, Adafactor
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from transquest.algo.word_level.microtransquest.format import post_process, prepare_data, format_to_test
from transquest.algo.word_level.microtransquest.model_args import MicroTransQuestArgs
from transquest.algo.word_level.microtransquest.utils import sweep_config_to_sweep_values, InputExample, \
    read_examples_from_file, get_examples_from_df, convert_examples_to_features, LazyQEDataset

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)


class MicroTransQuestModel:
    def __init__(
            self,
            model_type,
            model_name,
            labels=None,
            args=None,
            use_cuda=True,
            cuda_device=-1,
            onnx_execution_provider=None,
            **kwargs,
    ):
        """
        Initializes a NERModel

        Args:
            model_type: The type of model (bert, roberta)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_model.bin).
            labels (optional): A list of all Named Entity labels.  If not given, ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"] will be used.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {
            "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
            "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
            "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
            "xlmroberta": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),  # 这个
        }

        self.args = self._load_model_args(model_name)
        self.grads = {}

        self.attr_ratio = {
            "all_avg_attr_src2src_ratio": [],
            "all_avg_attr_mt2src_ratio": [],
            "all_avg_attr_gap2src_ratio": [],
            "all_max_attr_src2src_ratio": [],
            "all_max_attr_mt2src_ratio": [],
            "all_max_attr_gap2src_ratio": [],

            "all_avg_attr_src2mt_ratio": [],
            "all_avg_attr_mt2mt_ratio": [],
            "all_avg_attr_gap2mt_ratio": [],
            "all_max_attr_src2mt_ratio": [],
            "all_max_attr_mt2mt_ratio": [],
            "all_max_attr_gap2mt_ratio": [],

            "all_avg_attr_src2gap_ratio": [],
            "all_avg_attr_mt2gap_ratio": [],
            "all_avg_attr_gap2gap_ratio": [],
            "all_max_attr_src2gap_ratio": [],
            "all_max_attr_mt2gap_ratio": [],
            "all_max_attr_gap2gap_ratio": [],
            
            "all_avg_attr_src2src_ratio_right": [],
            "all_avg_attr_mt2src_ratio_right": [],
            "all_avg_attr_gap2src_ratio_right": [],
            "all_avg_attr_src2src_ratio_wrong": [],
            "all_avg_attr_mt2src_ratio_wrong": [],
            "all_avg_attr_gap2src_ratio_wrong": [],
            
            "all_avg_attr_src2mt_ratio_right": [],
            "all_avg_attr_mt2mt_ratio_right": [],
            "all_avg_attr_gap2mt_ratio_right": [],
            "all_avg_attr_src2mt_ratio_wrong": [],
            "all_avg_attr_mt2mt_ratio_wrong": [],
            "all_avg_attr_gap2mt_ratio_wrong": [],
            
            "all_avg_attr_src2gap_ratio_right": [],
            "all_avg_attr_mt2gap_ratio_right": [],
            "all_avg_attr_gap2gap_ratio_right": [],
            "all_avg_attr_src2gap_ratio_wrong": [],
            "all_avg_attr_mt2gap_ratio_wrong": [],
            "all_avg_attr_gap2gap_ratio_wrong": [],
        }

        self.word_alignment_pred = {
            "align_pred_lines_src2mt": [],
            "align_pred_lines_mt2src": [],
            "all_src_self_attr_mean": [],
        }

        self.align_dict_src2mt = dict()  # 官方对齐字典，记录每个mt词汇对齐的所有src词汇
        self.align_dict_mt2src = dict()  # 官方对齐字典，记录每个src词汇对齐的所有mt词汇
        
        self.src_acc_sign = dict()
        self.mt_acc_sign = dict()
        self.gap_acc_sign = dict()

        self.attr_different_part = {
            "all_src2mt_align_attr_mean": [],
            "all_src2mt_context_attr_mean": [],
            "all_src_self_attr_mean": [],
            "all_src_context_attr_mean": [],
            "all_src_adj_attr_mean": [],
            "all_src_remote_attr_mean": [],
            "all_src2gap_attr_mean": [],

            "all_mt2src_align_attr_mean": [],
            "all_mt2src_context_attr_mean": [],
            "all_mt_self_attr_mean": [],
            "all_mt_context_attr_mean": [],
            "all_mt_adj_attr_mean": [],
            "all_mt_remote_attr_mean": [],
            "all_mt2gap_attr_mean": [],

            "all_gap_self_attr_mean": [],
            "all_gap_context_attr_mean": [],
            "all_gap2mt_adj_attr_mean": [],
            "all_gap2mt_remote_attr_mean": [],
            "all_gap2src_attr_mean": [],

            "all_src2mt_align_attr_ratio_right": [],
            "all_src2mt_context_attr_ratio_right": [],
            "all_src_self_attr_ratio_right": [],
            "all_src_context_attr_ratio_right": [],
            "all_src2gap_attr_ratio_right": [],
            "all_src2mt_align_attr_ratio_wrong": [],
            "all_src2mt_context_attr_ratio_wrong": [],
            "all_src_self_attr_ratio_wrong": [],
            "all_src_context_attr_ratio_wrong": [],
            "all_src2gap_attr_ratio_wrong": [],

            "all_mt2src_align_attr_ratio_right": [],
            "all_mt2src_context_attr_ratio_right": [],
            "all_mt_self_attr_ratio_right": [],
            "all_mt_context_attr_ratio_right": [],
            "all_mt2gap_attr_ratio_right": [],
            "all_mt2src_align_attr_ratio_wrong": [],
            "all_mt2src_context_attr_ratio_wrong": [],
            "all_mt_self_attr_ratio_wrong": [],
            "all_mt_context_attr_ratio_wrong": [],
            "all_mt2gap_attr_ratio_wrong": [],

        }

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, MicroTransQuestArgs):
            self.args = args

        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if not use_cuda:
            self.args.fp16 = False

        if labels and self.args.labels_list:
            assert labels == self.args.labels_list
            self.args.labels_list = labels
        elif labels:
            self.args.labels_list = labels
        elif self.args.labels_list:
            pass
        else:
            self.args.labels_list = [
                "OK",
                "BAD",
                "SEP",
            ]
        self.num_labels = len(self.args.labels_list)

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if self.num_labels:
            self.config = config_class.from_pretrained(model_name, num_labels=self.num_labels, **self.args.config)
            self.num_labels = self.num_labels
        else:
            self.config = config_class.from_pretrained(model_name, **self.args.config)
            self.num_labels = self.config.num_labels

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        if self.args.onnx:
            from onnxruntime import InferenceSession, SessionOptions

            if not onnx_execution_provider:
                onnx_execution_provider = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"

            options = SessionOptions()
            options.intra_op_num_threads = 1

            if self.args.dynamic_quantize:
                model_path = quantize(Path(os.path.join(model_name, "onnx_model.onnx")))
                self.model = InferenceSession(model_path.as_posix(), options, providers=[onnx_execution_provider])
            else:
                model_path = os.path.join(model_name, "onnx_model.onnx")
                self.model = InferenceSession(model_path, options, providers=[onnx_execution_provider])
        else:
            if not self.args.quantized_model:
                #print("aaaaaaaaaaaaaaaa") 经过了
                """
                print(model_class)  # <class 'transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaForTokenClassification'>
                print(model_name)  # /home/yanym/save/transquest/outputs/best_model

                print(self.config)
                XLMRobertaConfig {
                    "_name_or_path": "xlm-roberta-large",
                    "architectures": [
                        "XLMRobertaForTokenClassification"
                    ],
                    "attention_probs_dropout_prob": 0.1,
                    "bos_token_id": 0,
                    "eos_token_id": 2,
                    "gradient_checkpointing": false,
                    "hidden_act": "gelu",
                    "hidden_dropout_prob": 0.1,
                    "hidden_size": 1024,
                    "initializer_range": 0.02,
                    "intermediate_size": 4096,
                    "layer_norm_eps": 1e-05,
                    "max_position_embeddings": 514,
                    "model_type": "xlm-roberta",
                    "num_attention_heads": 16,
                    "num_hidden_layers": 24,
                    "output_past": true,
                    "pad_token_id": 1,
                    "position_embedding_type": "absolute",
                    "transformers_version": "4.4.0.dev0",
                    "type_vocab_size": 1,
                    "use_cache": true,
                    "vocab_size": 250002
                    }

                """
                #print("model_name")
                #print(model_name)
                #"/home/yanym/transformers/xlm-roberta-large"
                self.model = model_class.from_pretrained(model_name, config=self.config, **kwargs) # 要重载这个model函数的forward
            else:
                #print("bbbbbbbbbbbbbbbb") 没经过
                quantized_weights = torch.load(os.path.join(model_name, "pytorch_model.bin"))
                self.model = model_class.from_pretrained(None, config=self.config, state_dict=quantized_weights)

            if self.args.dynamic_quantize:
                #print("cccccccccccccccc") 没经过
                self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            if self.args.quantized_model:
                #print("dddddddddddddddd") 没经过
                self.model.load_state_dict(quantized_weights)
            if self.args.dynamic_quantize:
                #print("eeeeeeeeeeeeeeee") 没经过
                self.args.quantized_model = True

        self.results = {}

        if self.args.fp16:
            try:
                from torch.cuda import amp
            except AttributeError:
                raise AttributeError("fp16 requires Pytorch >= 1.6. Please update Pytorch or turn off fp16.")

        if model_name in [
            "vinai/bertweet-base",
            "vinai/bertweet-covid19-base-cased",
            "vinai/bertweet-covid19-base-uncased",
        ]:
            self.tokenizer = tokenizer_class.from_pretrained(
                model_name, do_lower_case=self.args.do_lower_case, normalization=True, **kwargs
            )
        else:
            self.tokenizer = tokenizer_class.from_pretrained(
                model_name, do_lower_case=self.args.do_lower_case, **kwargs
            )

        if self.args.special_tokens_list:
            self.tokenizer.add_tokens(self.args.special_tokens_list, special_tokens=True)
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.args.model_name = model_name
        self.args.model_type = model_type

        self.pad_token_label_id = CrossEntropyLoss().ignore_index

        if model_type == "camembert":
            warnings.warn(
                "use_multiprocessing automatically disabled as CamemBERT"
                " fails when using multiprocessing for feature conversion."
            )
            self.args.use_multiprocessing = False

        if self.args.add_tag:
            self.tokenizer.add_tokens([self.args.tag], special_tokens=True)
            self.model.resize_token_embeddings(len(self.tokenizer))

        if self.args.wandb_project and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args.wandb_project = None

    def train_model(
            self, train_data, output_dir=None, show_running_loss=True, args=None, eval_data=None, verbose=True, **kwargs
    ):
        """
        Trains the model using 'train_data'

        Args:
            train_data: train_data should be the path to a .txt file containing the training data OR a pandas DataFrame with 3 columns.
                        If a text file is given the data should be in the CoNLL format. i.e. One word per line, with sentences seperated by an empty line.
                        The first word of the line should be a word, and the last should be a Name Entity Tag.
                        If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.
            eval_data: Evaluation data (same format as train_data) against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"

        if args:
            self.args.update_from_dict(args)

        if self.args.silent:
            show_running_loss = False

        if self.args.evaluate_during_training and eval_data is None:
            if "eval_df" in kwargs:
                warnings.warn(
                    "The eval_df parameter has been renamed to eval_data."
                    " Using eval_df will raise an error in a future version."
                )
                eval_data = kwargs.pop("eval_df")
            else:
                raise ValueError(
                    "evaluate_during_training is enabled but eval_data is not specified."
                    " Pass eval_data to model.train_model() if using evaluate_during_training."
                )

        if not output_dir:
            output_dir = self.args.output_dir

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Use --overwrite_output_dir to overcome.".format(output_dir)
            )

        train_data = prepare_data(train_data, self.args)
        eval_data = prepare_data(eval_data, self.args)

        #eval_data.to_csv("/home/yanym/save/transquest/eval_data.csv")
        #assert 1==2

        self._move_model_to_device()

        train_dataset = self.load_and_cache_examples(train_data)

        os.makedirs(output_dir, exist_ok=True)

        global_step, training_details = self.train(
            train_dataset, output_dir, show_running_loss=show_running_loss, eval_data=eval_data, **kwargs
        )

        self.save_model(model=self.model)

        logger.info(" Training of {} model complete. Saved to {}.".format(self.args.model_type, output_dir))

        return global_step, training_details

    def train(self, train_dataset, output_dir, show_running_loss=True, eval_data=None, verbose=True, **kwargs):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        device = self.device
        model = self.model
        args = self.args

        tb_writer = SummaryWriter(logdir=args.tensorboard_dir)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [p for n, p in model.named_parameters() if n in params]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = warmup_steps if args.warmup_steps == 0 else args.warmup_steps

        if args.optimizer == "AdamW":
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        elif args.optimizer == "Adafactor":
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adafactor_eps,
                clip_threshold=args.adafactor_clip_threshold,
                decay_rate=args.adafactor_decay_rate,
                beta1=args.adafactor_beta1,
                weight_decay=args.weight_decay,
                scale_parameter=args.adafactor_scale_parameter,
                relative_step=args.adafactor_relative_step,
                warmup_init=args.adafactor_warmup_init,
            )
            print("Using Adafactor for T5")
        else:
            raise ValueError(
                "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
                    args.optimizer
                )
            )

        if args.scheduler == "constant_schedule":
            scheduler = get_constant_schedule(optimizer)

        elif args.scheduler == "constant_schedule_with_warmup":
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

        elif args.scheduler == "linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
            )

        elif args.scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "polynomial_decay_schedule_with_warmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                lr_end=args.polynomial_decay_schedule_lr_end,
                power=args.polynomial_decay_schedule_lr_end,
            )

        else:
            raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0)
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if args.model_name and os.path.exists(args.model_name):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                        len(train_dataloader) // args.gradient_accumulation_steps
                )

                logger.info("   Continuing training from checkpoint, will skip to saved global_step")
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info("   Continuing training from global step %d", global_step)
                logger.info("   Will skip the first %d steps in the current epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(**kwargs)
        if args.wandb_project:
            wandb.init(project=args.wandb_project, config={**asdict(args)}, **args.wandb_kwargs)
            wandb.watch(self.model)

        if self.args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        for _ in train_iterator:
            model.train()
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(f"Epoch {epoch_number + 1} of {args.num_train_epochs}")
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )
            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                batch = tuple(t.to(device) for t in batch)

                print("================= batch")
                print(batch)

                inputs = self._get_inputs_dict(batch)
                print("================= inputs")
                print(inputs)

                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        # model outputs are always tuple in pytorch-transformers (see doc)
                        loss = outputs[0]
                else:
                    outputs = model(**inputs)
                    # model outputs are always tuple in pytorch-transformers (see doc)
                    loss = outputs[0]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()

                if show_running_loss:
                    batch_iterator.set_description(
                        f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if self.args.fp16:
                    scaler.scale(loss).backward()
                else:
                    #print(loss)   # tensor(0.6538, grad_fn=<NllLossBackward>)
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        scaler.unscale_(optimizer)
                    if args.optimizer == "AdamW":
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if self.args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                        tb_writer.add_scalar(
                            "loss", (tr_loss - logging_loss) / args.logging_steps, global_step,
                        )
                        logging_loss = tr_loss
                        if args.wandb_project or self.is_sweeping:
                            wandb.log(
                                {
                                    "Training loss": current_loss,
                                    "lr": scheduler.get_last_lr()[0],
                                    "global_step": global_step,
                                }
                            )

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        if args.save_recent_only:
                            del_paths = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
                            for del_path in del_paths:
                                shutil.rmtree(del_path)
                        # Save model checkpoint
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        self.save_model(output_dir_current, optimizer, scheduler, model=model)

                    if args.evaluate_during_training and (
                            args.evaluate_during_training_steps > 0
                            and global_step % args.evaluate_during_training_steps == 0
                    ):

                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        os.makedirs(output_dir_current, exist_ok=True)

                        # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _ = self.eval_model(
                            eval_data,
                            verbose=verbose and args.evaluate_during_training_verbose,
                            wandb_log=False,
                            output_dir=output_dir_current,
                            **kwargs,
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        if args.save_eval_checkpoints:
                            if args.save_recent_only:
                                del_paths = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
                                for del_path in del_paths:
                                    shutil.rmtree(del_path)
                            # Save model checkpoint
                            self.save_model(output_dir_current, optimizer, scheduler, model=model, results=results)

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        for key in results:
                            training_progress_scores[key].append(results[key])
                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(args.output_dir, "training_progress_scores.csv"), index=False,
                        )

                        if args.wandb_project or self.is_sweeping:
                            wandb.log(self._get_last_metrics(training_progress_scores))

                        if not best_eval_metric:
                            best_eval_metric = results[args.early_stopping_metric]
                            self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                        if best_eval_metric and args.early_stopping_metric_minimize:
                            if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir, optimizer, scheduler, model=model, results=results
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if early_stopping_counter < args.early_stopping_patience:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args.early_stopping_metric}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        else:
                            if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir, optimizer, scheduler, model=model, results=results
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if early_stopping_counter < args.early_stopping_patience:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args.early_stopping_metric}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )

            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))

            if args.save_model_every_epoch or args.evaluate_during_training:
                if args.save_recent_only:
                    del_paths = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
                    for del_path in del_paths:
                        shutil.rmtree(del_path)
                # Save model checkpoint
                os.makedirs(output_dir_current, exist_ok=True)

            if args.save_model_every_epoch:
                self.save_model(output_dir_current, optimizer, scheduler, model=model)

            if args.evaluate_during_training and args.evaluate_each_epoch:
                results, _, _ = self.eval_model(
                    eval_data, verbose=verbose and args.evaluate_during_training_verbose, wandb_log=False, **kwargs
                )

                if args.save_recent_only:
                    del_paths = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
                    for del_path in del_paths:
                        shutil.rmtree(del_path)
                # Save model checkpoint

                self.save_model(output_dir_current, optimizer, scheduler, results=results)

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(os.path.join(args.output_dir, "training_progress_scores.csv"), index=False)

                if args.wandb_project or self.is_sweeping:
                    wandb.log(self._get_last_metrics(training_progress_scores))

                if not best_eval_metric:
                    best_eval_metric = results[args.early_stopping_metric]
                    self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                if best_eval_metric and args.early_stopping_metric_minimize:
                    if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )
                else:
                    if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                        early_stopping_counter = 0
                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )

        return (
            global_step,
            tr_loss / global_step if not self.args.evaluate_during_training else training_progress_scores,
        )

    def eval_model(self, eval_data, output_dir=None, verbose=True, silent=False, wandb_log=True, **kwargs):
        """
        Evaluates the model on eval_data. Saves results to output_dir.

        Args:
            eval_data: eval_data should be the path to a .txt file containing the evaluation data or a pandas DataFrame.
                        If a text file is used the data should be in the CoNLL format. I.e. One word per line, with sentences seperated by an empty line.
                        The first word of the line should be a word, and the last should be a Name Entity Tag.
                        If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.

            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            silent: If silent, tqdm progress bars will be hidden.
            wandb_log: If True, evaluation results will be logged to wandb.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results. (eval_loss, precision, recall, f1_score)
            model_outputs: List of raw model outputs
            preds_list: List of predicted tags
        """  # noqa: ignore flake8"
        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()

        eval_dataset = self.load_and_cache_examples(eval_data, evaluate=True)

        result, model_outputs, preds_list = self.evaluate(
            eval_dataset, output_dir, verbose=verbose, silent=silent, wandb_log=wandb_log, **kwargs
        )
        self.results.update(result)

        if verbose:
            logger.info(self.results)

        return result, model_outputs, preds_list

    def evaluate(self, eval_dataset, output_dir, verbose=True, silent=False, wandb_log=True, **kwargs):
        """
        Evaluates the model on eval_dataset.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        device = self.device
        model = self.model
        args = self.args
        pad_token_label_id = self.pad_token_label_id
        eval_output_dir = output_dir

        results = {}

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        model.eval()

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if self.args.fp16:
            from torch.cuda import amp

        for batch in tqdm(eval_dataloader, disable=args.silent or silent, desc="Running Evaluation"):
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)

                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        tmp_eval_loss, logits = outputs[:2]
                else:
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
                out_input_ids = inputs["input_ids"].detach().cpu().numpy()
                out_attention_mask = inputs["attention_mask"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                out_input_ids = np.append(out_input_ids, inputs["input_ids"].detach().cpu().numpy(), axis=0)
                out_attention_mask = np.append(
                    out_attention_mask, inputs["attention_mask"].detach().cpu().numpy(), axis=0,
                )

        eval_loss = eval_loss / nb_eval_steps
        token_logits = preds
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(self.args.labels_list)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        word_tokens = []
        for i in range(len(preds_list)):
            w_log = self._convert_tokens_to_word_logits(
                out_input_ids[i], out_label_ids[i], out_attention_mask[i], token_logits[i],
            )
            word_tokens.append(w_log)

        model_outputs = [[word_tokens[i][j] for j in range(len(preds_list[i]))] for i in range(len(preds_list))]

        extra_metrics = {}
        for metric, func in kwargs.items():
            extra_metrics[metric] = func(out_label_list, preds_list)

        result = {
            "eval_loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1_score": f1_score(out_label_list, preds_list),
            **extra_metrics,
        }

        results.update(result)

        os.makedirs(eval_output_dir, exist_ok=True)
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            if args.classification_report:
                cls_report = classification_report(out_label_list, preds_list, digits=4)
                writer.write("{}\n".format(cls_report))
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        if self.args.wandb_project and wandb_log:
            wandb.init(project=args.wandb_project, config={**asdict(args)}, **args.wandb_kwargs)

            labels_list = sorted(self.args.labels_list)

            truth = [tag for out in out_label_list for tag in out]
            preds = [tag for pred_out in preds_list for tag in pred_out]
            outputs = [np.mean(logits, axis=0) for output in model_outputs for logits in output]

            # ROC
            wandb.log({"roc": wandb.plots.ROC(truth, outputs, labels_list)})

            # Precision Recall
            wandb.log({"pr": wandb.plots.precision_recall(truth, outputs, labels_list)})

            # Confusion Matrix
            wandb.sklearn.plot_confusion_matrix(
                truth, preds, labels=labels_list,
            )

        return results, model_outputs, preds_list

    def predict(self, to_predict, split_on_space=True):
        """
        Performs predictions on a list of text.

        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction.
            split_on_space: If True, each sequence will be split by spaces for assigning labels.
                            If False, to_predict must be a a list of lists, with the inner list being a
                            list of strings consisting of the split sequences. The outer list is the list of sequences to
                            predict on.

        Returns:
            preds: A Python list of lists with dicts containing each word mapped to its NER tag.
            model_outputs: A Python list of lists with dicts containing each word mapped to its list with raw model output.
        """  # noqa: ignore flake8"

        device = self.device
        model = self.model
        args = self.args
        pad_token_label_id = self.pad_token_label_id
        preds = None

        to_predict = format_to_test(to_predict, self.args)

        if split_on_space:
            if self.args.model_type == "layoutlm":
                predict_examples = [
                    InputExample(
                        i, sentence.split(), [self.args.labels_list[0] for word in sentence.split()], x0, y0, x1, y1
                    )
                    for i, (sentence, x0, y0, x1, y1) in enumerate(to_predict)
                ]
                to_predict = [sentence for sentence, *_ in to_predict]
            else:
                predict_examples = [
                    InputExample(i, sentence.split(), [self.args.labels_list[0] for word in sentence.split()])
                    for i, sentence in enumerate(to_predict)
                ]
        else:
            if self.args.model_type == "layoutlm":
                predict_examples = [
                    InputExample(i, sentence, [self.args.labels_list[0] for word in sentence], x0, y0, x1, y1)
                    for i, (sentence, x0, y0, x1, y1) in enumerate(to_predict)
                ]
                to_predict = [sentence for sentence, *_ in to_predict]
            else:
                predict_examples = [
                    InputExample(i, sentence, [self.args.labels_list[0] for word in sentence])
                    for i, sentence in enumerate(to_predict)
                ]

        eval_dataset = self.load_and_cache_examples(None, to_predict=predict_examples)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        #for batch in eval_dataloader:
        #    print(batch)  # 每条数据，4个tensor
        #assert 1==2

        if self.args.onnx:
            model_inputs = self.tokenizer.batch_encode_plus(
                to_predict, return_tensors="pt", padding=True, truncation=True
            )

            for input_ids, attention_mask in zip(model_inputs["input_ids"], model_inputs["attention_mask"]):
                input_ids = input_ids.unsqueeze(0).detach().cpu().numpy()
                attention_mask = attention_mask.unsqueeze(0).detach().cpu().numpy()
                inputs_onnx = {"input_ids": input_ids, "attention_mask": attention_mask}

                # Run the model (None = get all the outputs)
                output = self.model.run(None, inputs_onnx)

                if preds is None:
                    preds = output[0]
                    out_input_ids = inputs_onnx["input_ids"]
                    out_attention_mask = inputs_onnx["attention_mask"]
                else:
                    preds = np.append(preds, output[0], axis=0)
                    out_input_ids = np.append(out_input_ids, inputs_onnx["input_ids"], axis=0)
                    out_attention_mask = np.append(out_attention_mask, inputs_onnx["attention_mask"], axis=0, )
            out_label_ids = np.zeros_like(out_input_ids)
            for index in range(len(out_label_ids)):
                out_label_ids[index][0] = -100
                out_label_ids[index][-1] = -100
        else:
            self._move_model_to_device()

            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None
            model.eval()

            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

            if self.args.fp16:
                from torch.cuda import amp

            for batch in tqdm(eval_dataloader, disable=args.silent, desc="Running Prediction"):
                batch = tuple(t.to(device) for t in batch)   # 每条数据，4个tensor

                with torch.no_grad():
                    inputs = self._get_inputs_dict(batch)  # input_ids, attention_mask, labels

                    if self.args.fp16:
                        with amp.autocast():
                            outputs = model(**inputs)
                            tmp_eval_loss, logits = outputs[:2]
                    else:
                        outputs = model(**inputs)
                        # 我要修改的话，这里就不能只取outputs出来
                        tmp_eval_loss, logits = outputs[:2]
                        #print(logits.size())   # [1, 200, 2]

                    if self.args.n_gpu > 1:
                        tmp_eval_loss = tmp_eval_loss.mean()
                    eval_loss += tmp_eval_loss.item()

                nb_eval_steps += 1

                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                    out_input_ids = inputs["input_ids"].detach().cpu().numpy()
                    out_attention_mask = inputs["attention_mask"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                    out_input_ids = np.append(out_input_ids, inputs["input_ids"].detach().cpu().numpy(), axis=0)
                    out_attention_mask = np.append(
                        out_attention_mask, inputs["attention_mask"].detach().cpu().numpy(), axis=0,
                    )

            eval_loss = eval_loss / nb_eval_steps
        token_logits = preds
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(self.args.labels_list)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        if split_on_space:
            preds = [
                [{word: preds_list[i][j]} for j, word in enumerate(sentence.split()[: len(preds_list[i])])]
                for i, sentence in enumerate(to_predict)
            ]
        else:
            preds = [
                [{word: preds_list[i][j]} for j, word in enumerate(sentence[: len(preds_list[i])])]
                for i, sentence in enumerate(to_predict)
            ]

        word_tokens = []
        for n, sentence in enumerate(to_predict):
            w_log = self._convert_tokens_to_word_logits(
                out_input_ids[n], out_label_ids[n], out_attention_mask[n], token_logits[n],
            )
            word_tokens.append(w_log)

        if split_on_space:
            model_outputs = [
                [{word: word_tokens[i][j]} for j, word in enumerate(sentence.split()[: len(preds_list[i])])]
                for i, sentence in enumerate(to_predict)
            ]
        else:
            model_outputs = [
                [{word: word_tokens[i][j]} for j, word in enumerate(sentence[: len(preds_list[i])])]
                for i, sentence in enumerate(to_predict)
            ]

        sources_tags, targets_tags = post_process(preds, to_predict, args=self.args)

        return sources_tags, targets_tags

    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad
        return hook

    def auto_mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def plot_attribution(self, attr_token, sent, plot_path_prefix, id):
        real_seq_len = attr_token.size(-1)
        font_size = 240 // real_seq_len
        attr_token = attr_token.cpu().detach().numpy()     # [real_seq_len, real_seq_len]
        f, ax = plt.subplots(figsize = (30,30))
        sns.heatmap(attr_token, ax=ax, cmap="YlGnBu", square=True, annot=attr_token, cbar=False, \
            annot_kws={'size':font_size}, fmt='.4f')
        ax.set_xlabel('attribution', fontsize=font_size)
        ax.set_xticklabels(sent,rotation=0)
        ax.set_yticklabels(sent,rotation=0)
        plt.tick_params(labelsize=font_size)
        self.auto_mkdir(plot_path_prefix)
        plt.savefig(plot_path_prefix + "attr_g_token_{}.jpg".format(id))

    def cal_attr_ratio_origin(self, attr_token, srcmt_sent):
        src_sent_list = srcmt_sent.split("[SEP]")[0].split()
        src_len = len(src_sent_list)
        attr_token_all2src = attr_token[:, :src_len]        # 所有token对src的贡献 [seq_len, src_len]
        attr_token_all2mt = attr_token[:, src_len + 2::2]   # 所有token对mt的贡献 [seq_len, mt_len]
        attr_token_all2gap = attr_token[:, src_len + 1::2]  # 所有token对gap的贡献 [seq_len, gap_len]

        # 对src而言，src、mt、gap分别的贡献比例 ============================================= #
        attr_token_src2src = attr_token_all2src[:src_len]           # src对src贡献  [src_len, src_len]
        attr_token_mt2src = attr_token_all2src[src_len + 2 :: 2]    # mt对src贡献   [mt_len, src_len]
        attr_token_gap2src = attr_token_all2src[src_len + 1 :: 2]   # gap对src贡献  [gap_len, src_len]
        # 计算不同部分的贡献
        avg_attr_src2src = torch.sum(attr_token_src2src, dim = 0)  # [src_len]  每个位置是，src部分对该src token的贡献之和
        avg_attr_mt2src = torch.sum(attr_token_mt2src, dim = 0)    # [src_len]
        avg_attr_gap2src = torch.sum(attr_token_gap2src, dim = 0)  # [src_len]

        # 收集贡献比例
        sum_avg_attr_2src = avg_attr_src2src + avg_attr_mt2src + avg_attr_gap2src  # [src_len]
        avg_attr_src2src_ratio = avg_attr_src2src / sum_avg_attr_2src # [src_len]
        avg_attr_mt2src_ratio = avg_attr_mt2src / sum_avg_attr_2src   # [src_len]
        avg_attr_gap2src_ratio = avg_attr_gap2src / sum_avg_attr_2src # [src_len]
        self.attr_ratio["all_avg_attr_src2src_ratio"].append(avg_attr_src2src_ratio)
        self.attr_ratio["all_avg_attr_mt2src_ratio"].append(avg_attr_mt2src_ratio)
        self.attr_ratio["all_avg_attr_gap2src_ratio"].append(avg_attr_gap2src_ratio)

        # 计算贡献最大值
        max_attr_src2src, _ = torch.max(attr_token_src2src, dim = 0)  # [src_len]  每个位置是，src token对该src token的贡献最大值
        max_attr_mt2src, _ = torch.max(attr_token_mt2src, dim = 0)    # [src_len]
        max_attr_gap2src, _ = torch.max(attr_token_gap2src, dim = 0)  # [src_len]

        # 收集贡献最大值比例
        sum_max_attr_2src = max_attr_src2src + max_attr_mt2src + max_attr_gap2src  # [src_len]
        max_attr_src2src_ratio = max_attr_src2src / sum_max_attr_2src # [src_len]
        max_attr_mt2src_ratio = max_attr_mt2src / sum_max_attr_2src   # [src_len]
        max_attr_gap2src_ratio = max_attr_gap2src / sum_max_attr_2src # [src_len]
        self.attr_ratio["all_max_attr_src2src_ratio"].append(max_attr_src2src_ratio)
        self.attr_ratio["all_max_attr_mt2src_ratio"].append(max_attr_mt2src_ratio)
        self.attr_ratio["all_max_attr_gap2src_ratio"].append(max_attr_gap2src_ratio)

        # 对mt而言，src、mt、gap分别的贡献比例 ============================================= #
        attr_token_src2mt = attr_token_all2mt[:src_len]           # src对mt贡献  [src_len, mt_len]
        attr_token_mt2mt = attr_token_all2mt[src_len + 2 :: 2]    # mt对mt贡献   [mt_len, mt_len]
        attr_token_gap2mt = attr_token_all2mt[src_len + 1 :: 2]   # gap对mt贡献  [gap_len, mt_len]

        # 计算贡献平均值
        avg_attr_src2mt = torch.sum(attr_token_src2mt, dim = 0)  # [mt_len]  每个位置是，src部分对该mt token的贡献
        avg_attr_mt2mt = torch.sum(attr_token_mt2mt, dim = 0)    # [mt_len]
        avg_attr_gap2mt = torch.sum(attr_token_gap2mt, dim = 0)  # [mt_len]

        # 收集贡献平均值比例
        sum_avg_attr_2mt = avg_attr_src2mt + avg_attr_mt2mt + avg_attr_gap2mt  # [mt_len]
        avg_attr_src2mt_ratio = avg_attr_src2mt / sum_avg_attr_2mt # [mt_len]
        avg_attr_mt2mt_ratio = avg_attr_mt2mt / sum_avg_attr_2mt   # [mt_len]
        avg_attr_gap2mt_ratio = avg_attr_gap2mt / sum_avg_attr_2mt # [mt_len]
        self.attr_ratio["all_avg_attr_src2mt_ratio"].append(avg_attr_src2mt_ratio)
        self.attr_ratio["all_avg_attr_mt2mt_ratio"].append(avg_attr_mt2mt_ratio)
        self.attr_ratio["all_avg_attr_gap2mt_ratio"].append(avg_attr_gap2mt_ratio)

        # 计算贡献最大值
        max_attr_src2mt, _ = torch.max(attr_token_src2mt, dim = 0)  # [mt_len]  每个位置是，src token对该mt token的贡献最大值
        max_attr_mt2mt, _ = torch.max(attr_token_mt2mt, dim = 0)    # [mt_len]
        max_attr_gap2mt, _ = torch.max(attr_token_gap2mt, dim = 0)  # [mt_len]

        # 收集贡献最大值比例
        sum_max_attr_2mt = max_attr_src2mt + max_attr_mt2mt + max_attr_gap2mt  # [mt_len]
        max_attr_src2mt_ratio = max_attr_src2mt / sum_max_attr_2mt # [mt_len]
        max_attr_mt2mt_ratio = max_attr_mt2mt / sum_max_attr_2mt   # [mt_len]
        max_attr_gap2mt_ratio = max_attr_gap2mt / sum_max_attr_2mt # [mt_len]
        self.attr_ratio["all_max_attr_src2mt_ratio"].append(max_attr_src2mt_ratio)
        self.attr_ratio["all_max_attr_mt2mt_ratio"].append(max_attr_mt2mt_ratio)
        self.attr_ratio["all_max_attr_gap2mt_ratio"].append(max_attr_gap2mt_ratio)

        # 对gap而言，src、mt、gap分别的贡献比例 ============================================= #
        attr_token_src2gap = attr_token_all2gap[:src_len]           # src对gap贡献  [src_len, gap_len]
        attr_token_mt2gap = attr_token_all2gap[src_len + 2 :: 2]    # mt对gap贡献   [mt_len, gap_len]
        attr_token_gap2gap = attr_token_all2gap[src_len + 1 :: 2]   # gap对gap贡献  [gap_len, gap_len]

        # 计算贡献平均值
        avg_attr_src2gap = torch.sum(attr_token_src2gap, dim = 0)  # [gap_len]  每个位置是，src部分对该gap token的贡献
        avg_attr_mt2gap = torch.sum(attr_token_mt2gap, dim = 0)    # [gap_len]
        avg_attr_gap2gap = torch.sum(attr_token_gap2gap, dim = 0)  # [gap_len]

        # 收集贡献平均值比例
        sum_avg_attr_2gap = avg_attr_src2gap + avg_attr_mt2gap + avg_attr_gap2gap  # [gap_len]
        avg_attr_src2gap_ratio = avg_attr_src2gap / sum_avg_attr_2gap # [gap_len]
        avg_attr_mt2gap_ratio = avg_attr_mt2gap / sum_avg_attr_2gap   # [gap_len]
        avg_attr_gap2gap_ratio = avg_attr_gap2gap / sum_avg_attr_2gap # [gap_len]
        self.attr_ratio["all_avg_attr_src2gap_ratio"].append(avg_attr_src2gap_ratio)
        self.attr_ratio["all_avg_attr_mt2gap_ratio"].append(avg_attr_mt2gap_ratio)
        self.attr_ratio["all_avg_attr_gap2gap_ratio"].append(avg_attr_gap2gap_ratio)

        # 计算贡献最大值
        max_attr_src2gap, _ = torch.max(attr_token_src2gap, dim = 0)  # [gap_len]  每个位置是，src token对该gap token的贡献最大值
        max_attr_mt2gap, _ = torch.max(attr_token_mt2gap, dim = 0)    # [gap_len]
        max_attr_gap2gap, _ = torch.max(attr_token_gap2gap, dim = 0)  # [gap_len]

        # 收集贡献最大值比例
        sum_max_attr_2gap = max_attr_src2gap + max_attr_mt2gap + max_attr_gap2gap  # [gap_len]
        max_attr_src2gap_ratio = max_attr_src2gap / sum_max_attr_2gap # [gap_len]
        max_attr_mt2gap_ratio = max_attr_mt2gap / sum_max_attr_2gap   # [gap_len]
        max_attr_gap2gap_ratio = max_attr_gap2gap / sum_max_attr_2gap # [gap_len]
        self.attr_ratio["all_max_attr_src2gap_ratio"].append(max_attr_src2gap_ratio)
        self.attr_ratio["all_max_attr_mt2gap_ratio"].append(max_attr_mt2gap_ratio)
        self.attr_ratio["all_max_attr_gap2gap_ratio"].append(max_attr_gap2gap_ratio)

    def cal_attr_ratio(self, attr_token, srcmt_sent, id):
        src_sent_list = srcmt_sent.split("[SEP]")[0].split()
        src_len = len(src_sent_list)
        attr_token_all2src = attr_token[:, :src_len]        # 所有token对src的贡献 [seq_len, src_len]
        attr_token_all2mt = attr_token[:, src_len + 2::2]   # 所有token对mt的贡献 [seq_len, mt_len]
        attr_token_all2gap = attr_token[:, src_len + 1::2]  # 所有token对gap的贡献 [seq_len, gap_len]

        # 对src而言，src、mt、gap分别的贡献比例 ============================================= #
        attr_token_src2src = attr_token_all2src[:src_len]           # src对src贡献  [src_len, src_len]
        attr_token_mt2src = attr_token_all2src[src_len + 2 :: 2]    # mt对src贡献   [mt_len, src_len]
        attr_token_gap2src = attr_token_all2src[src_len + 1 :: 2]   # gap对src贡献  [gap_len, src_len]
        # 计算不同部分的贡献
        avg_attr_src2src = torch.sum(attr_token_src2src, dim = 0)  # [src_len]  每个位置是，src部分对该src token的贡献之和
        avg_attr_mt2src = torch.sum(attr_token_mt2src, dim = 0)    # [src_len]
        avg_attr_gap2src = torch.sum(attr_token_gap2src, dim = 0)  # [src_len]

        # 收集贡献比例
        sum_avg_attr_2src = avg_attr_src2src + avg_attr_mt2src + avg_attr_gap2src  # [src_len]
        avg_attr_src2src_ratio = avg_attr_src2src / sum_avg_attr_2src # [src_len]
        avg_attr_mt2src_ratio = avg_attr_mt2src / sum_avg_attr_2src   # [src_len]
        avg_attr_gap2src_ratio = avg_attr_gap2src / sum_avg_attr_2src # [src_len]
        self.attr_ratio["all_avg_attr_src2src_ratio"].append(avg_attr_src2src_ratio)
        self.attr_ratio["all_avg_attr_mt2src_ratio"].append(avg_attr_mt2src_ratio)
        self.attr_ratio["all_avg_attr_gap2src_ratio"].append(avg_attr_gap2src_ratio)
        src_acc_line = self.src_acc_sign[id]
        for i in range(sum_avg_attr_2src.size(-1)):
            if src_acc_line[i] == '1':
                self.attr_ratio["all_avg_attr_src2src_ratio_right"].append(avg_attr_src2src_ratio[i].item())
                self.attr_ratio["all_avg_attr_mt2src_ratio_right"].append(avg_attr_mt2src_ratio[i].item())
                self.attr_ratio["all_avg_attr_gap2src_ratio_right"].append(avg_attr_gap2src_ratio[i].item())
            else:
                self.attr_ratio["all_avg_attr_src2src_ratio_wrong"].append(avg_attr_src2src_ratio[i].item())
                self.attr_ratio["all_avg_attr_mt2src_ratio_wrong"].append(avg_attr_mt2src_ratio[i].item())
                self.attr_ratio["all_avg_attr_gap2src_ratio_wrong"].append(avg_attr_gap2src_ratio[i].item())

        # 对mt而言，src、mt、gap分别的贡献比例 ============================================= #
        attr_token_src2mt = attr_token_all2mt[:src_len]           # src对mt贡献  [src_len, mt_len]
        attr_token_mt2mt = attr_token_all2mt[src_len + 2 :: 2]    # mt对mt贡献   [mt_len, mt_len]
        attr_token_gap2mt = attr_token_all2mt[src_len + 1 :: 2]   # gap对mt贡献  [gap_len, mt_len]

        # 计算贡献平均值
        avg_attr_src2mt = torch.sum(attr_token_src2mt, dim = 0)  # [mt_len]  每个位置是，src部分对该mt token的贡献
        avg_attr_mt2mt = torch.sum(attr_token_mt2mt, dim = 0)    # [mt_len]
        avg_attr_gap2mt = torch.sum(attr_token_gap2mt, dim = 0)  # [mt_len]

        # 收集贡献平均值比例
        sum_avg_attr_2mt = avg_attr_src2mt + avg_attr_mt2mt + avg_attr_gap2mt  # [mt_len]
        avg_attr_src2mt_ratio = avg_attr_src2mt / sum_avg_attr_2mt # [mt_len]
        avg_attr_mt2mt_ratio = avg_attr_mt2mt / sum_avg_attr_2mt   # [mt_len]
        avg_attr_gap2mt_ratio = avg_attr_gap2mt / sum_avg_attr_2mt # [mt_len]
        self.attr_ratio["all_avg_attr_src2mt_ratio"].append(avg_attr_src2mt_ratio)
        self.attr_ratio["all_avg_attr_mt2mt_ratio"].append(avg_attr_mt2mt_ratio)
        self.attr_ratio["all_avg_attr_gap2mt_ratio"].append(avg_attr_gap2mt_ratio)
        mt_acc_line = self.mt_acc_sign[id]
        for i in range(sum_avg_attr_2mt.size(-1)):
            if mt_acc_line[i] == '1':
                self.attr_ratio["all_avg_attr_src2mt_ratio_right"].append(avg_attr_src2mt_ratio[i].item())
                self.attr_ratio["all_avg_attr_mt2mt_ratio_right"].append(avg_attr_mt2mt_ratio[i].item())
                self.attr_ratio["all_avg_attr_gap2mt_ratio_right"].append(avg_attr_gap2mt_ratio[i].item())
            else:
                self.attr_ratio["all_avg_attr_src2mt_ratio_wrong"].append(avg_attr_src2mt_ratio[i].item())
                self.attr_ratio["all_avg_attr_mt2mt_ratio_wrong"].append(avg_attr_mt2mt_ratio[i].item())
                self.attr_ratio["all_avg_attr_gap2mt_ratio_wrong"].append(avg_attr_gap2mt_ratio[i].item())

        # 对gap而言，src、mt、gap分别的贡献比例 ============================================= #
        attr_token_src2gap = attr_token_all2gap[:src_len]           # src对gap贡献  [src_len, gap_len]
        attr_token_mt2gap = attr_token_all2gap[src_len + 2 :: 2]    # mt对gap贡献   [mt_len, gap_len]
        attr_token_gap2gap = attr_token_all2gap[src_len + 1 :: 2]   # gap对gap贡献  [gap_len, gap_len]

        # 计算贡献平均值
        avg_attr_src2gap = torch.sum(attr_token_src2gap, dim = 0)  # [gap_len]  每个位置是，src部分对该gap token的贡献
        avg_attr_mt2gap = torch.sum(attr_token_mt2gap, dim = 0)    # [gap_len]
        avg_attr_gap2gap = torch.sum(attr_token_gap2gap, dim = 0)  # [gap_len]

        # 收集贡献平均值比例
        sum_avg_attr_2gap = avg_attr_src2gap + avg_attr_mt2gap + avg_attr_gap2gap  # [gap_len]
        avg_attr_src2gap_ratio = avg_attr_src2gap / sum_avg_attr_2gap # [gap_len]
        avg_attr_mt2gap_ratio = avg_attr_mt2gap / sum_avg_attr_2gap   # [gap_len]
        avg_attr_gap2gap_ratio = avg_attr_gap2gap / sum_avg_attr_2gap # [gap_len]
        self.attr_ratio["all_avg_attr_src2gap_ratio"].append(avg_attr_src2gap_ratio)
        self.attr_ratio["all_avg_attr_mt2gap_ratio"].append(avg_attr_mt2gap_ratio)
        self.attr_ratio["all_avg_attr_gap2gap_ratio"].append(avg_attr_gap2gap_ratio)
        gap_acc_line = self.gap_acc_sign[id]
        for i in range(sum_avg_attr_2gap.size(-1)):
            if gap_acc_line[i] == '1':
                self.attr_ratio["all_avg_attr_src2gap_ratio_right"].append(avg_attr_src2gap_ratio[i].item())
                self.attr_ratio["all_avg_attr_mt2gap_ratio_right"].append(avg_attr_mt2gap_ratio[i].item())
                self.attr_ratio["all_avg_attr_gap2gap_ratio_right"].append(avg_attr_gap2gap_ratio[i].item())
            else:
                self.attr_ratio["all_avg_attr_src2gap_ratio_wrong"].append(avg_attr_src2gap_ratio[i].item())
                self.attr_ratio["all_avg_attr_mt2gap_ratio_wrong"].append(avg_attr_mt2gap_ratio[i].item())
                self.attr_ratio["all_avg_attr_gap2gap_ratio_wrong"].append(avg_attr_gap2gap_ratio[i].item())

    def show_attr_ratio_origin(self):
        # 对src而言，src、mt、gap分别的贡献比例 结果输出 ============================================= #
        all_avg_attr_src2src_ratio = torch.cat(self.attr_ratio["all_avg_attr_src2src_ratio"], 0)  # 延长拼接到一起 [all_src_token_num]
        all_avg_attr_mt2src_ratio = torch.cat(self.attr_ratio["all_avg_attr_mt2src_ratio"], 0)  # 延长拼接到一起 [all_src_token_num]
        all_avg_attr_gap2src_ratio = torch.cat(self.attr_ratio["all_avg_attr_gap2src_ratio"], 0)  # 延长拼接到一起 [all_src_token_num]
        avg_attr_src2src_ratio = torch.mean(all_avg_attr_src2src_ratio, dim = 0)
        avg_attr_mt2src_ratio = torch.mean(all_avg_attr_mt2src_ratio, dim = 0)
        avg_attr_gap2src_ratio = torch.mean(all_avg_attr_gap2src_ratio, dim = 0)
        print("=============== 先算比例：src mt gap 对 src 贡献比例平均值")
        print(avg_attr_src2src_ratio)
        print(avg_attr_mt2src_ratio)
        print(avg_attr_gap2src_ratio)
                
        all_max_attr_src2src_ratio = torch.cat(self.attr_ratio["all_max_attr_src2src_ratio"], 0)  # 延长拼接到一起 [all_src_token_num]
        all_max_attr_mt2src_ratio = torch.cat(self.attr_ratio["all_max_attr_mt2src_ratio"], 0)  # 延长拼接到一起 [all_src_token_num]
        all_max_attr_gap2src_ratio = torch.cat(self.attr_ratio["all_max_attr_gap2src_ratio"], 0)  # 延长拼接到一起 [all_src_token_num]
        max_attr_src2src_ratio = torch.mean(all_max_attr_src2src_ratio, dim = 0)
        max_attr_mt2src_ratio = torch.mean(all_max_attr_mt2src_ratio, dim = 0)
        max_attr_gap2src_ratio = torch.mean(all_max_attr_gap2src_ratio, dim = 0)
        print("=============== 先算比例：src mt gap 对 src 贡献比例最大值")
        print(max_attr_src2src_ratio)
        print(max_attr_mt2src_ratio)
        print(max_attr_gap2src_ratio)

        # 对mt而言，src、mt、gap分别的贡献比例 结果输出 ============================================= #
        all_avg_attr_src2mt_ratio = torch.cat(self.attr_ratio["all_avg_attr_src2mt_ratio"], 0)  # 延长拼接到一起 [all_mt_token_num]
        all_avg_attr_mt2mt_ratio = torch.cat(self.attr_ratio["all_avg_attr_mt2mt_ratio"], 0)  # 延长拼接到一起 [all_mt_token_num]
        all_avg_attr_gap2mt_ratio = torch.cat(self.attr_ratio["all_avg_attr_gap2mt_ratio"], 0)  # 延长拼接到一起 [all_mt_token_num]
        avg_attr_src2mt_ratio = torch.mean(all_avg_attr_src2mt_ratio, dim = 0)
        avg_attr_mt2mt_ratio = torch.mean(all_avg_attr_mt2mt_ratio, dim = 0)
        avg_attr_gap2mt_ratio = torch.mean(all_avg_attr_gap2mt_ratio, dim = 0)
        print("=============== 先算比例：src mt gap 对 mt 贡献比例平均值")
        print(avg_attr_src2mt_ratio)
        print(avg_attr_mt2mt_ratio)
        print(avg_attr_gap2mt_ratio)
        
        all_max_attr_src2mt_ratio = torch.cat(self.attr_ratio["all_max_attr_src2mt_ratio"], 0)  # 延长拼接到一起 [all_mt_token_num]
        all_max_attr_mt2mt_ratio = torch.cat(self.attr_ratio["all_max_attr_mt2mt_ratio"], 0)  # 延长拼接到一起 [all_mt_token_num]
        all_max_attr_gap2mt_ratio = torch.cat(self.attr_ratio["all_max_attr_gap2mt_ratio"], 0)  # 延长拼接到一起 [all_mt_token_num]
        max_attr_src2mt_ratio = torch.mean(all_max_attr_src2mt_ratio, dim = 0)
        max_attr_mt2mt_ratio = torch.mean(all_max_attr_mt2mt_ratio, dim = 0)
        max_attr_gap2mt_ratio = torch.mean(all_max_attr_gap2mt_ratio, dim = 0)
        print("=============== 先算比例：src mt gap 对 mt 贡献比例最大值")
        print(max_attr_src2mt_ratio)
        print(max_attr_mt2mt_ratio)
        print(max_attr_gap2mt_ratio)

        # 对gap而言，src、mt、gap分别的贡献比例 结果输出 ============================================= #
        all_avg_attr_src2gap_ratio = torch.cat(self.attr_ratio["all_avg_attr_src2gap_ratio"], 0)  # 延长拼接到一起 [all_gap_token_num]
        all_avg_attr_mt2gap_ratio = torch.cat(self.attr_ratio["all_avg_attr_mt2gap_ratio"], 0)  # 延长拼接到一起 [all_gap_token_num]
        all_avg_attr_gap2gap_ratio = torch.cat(self.attr_ratio["all_avg_attr_gap2gap_ratio"], 0)  # 延长拼接到一起 [all_gap_token_num]
        avg_attr_src2gap_ratio = torch.mean(all_avg_attr_src2gap_ratio, dim = 0)
        avg_attr_mt2gap_ratio = torch.mean(all_avg_attr_mt2gap_ratio, dim = 0)
        avg_attr_gap2gap_ratio = torch.mean(all_avg_attr_gap2gap_ratio, dim = 0)
        print("=============== 先算比例：src mt gap 对 gap 贡献比例平均值")
        print(avg_attr_src2gap_ratio)
        print(avg_attr_mt2gap_ratio)
        print(avg_attr_gap2gap_ratio)
        
        all_max_attr_src2gap_ratio = torch.cat(self.attr_ratio["all_max_attr_src2gap_ratio"], 0)  # 延长拼接到一起 [all_gap_token_num]
        all_max_attr_mt2gap_ratio = torch.cat(self.attr_ratio["all_max_attr_mt2gap_ratio"], 0)  # 延长拼接到一起 [all_gap_token_num]
        all_max_attr_gap2gap_ratio = torch.cat(self.attr_ratio["all_max_attr_gap2gap_ratio"], 0)  # 延长拼接到一起 [all_gap_token_num]
        max_attr_src2gap_ratio = torch.mean(all_max_attr_src2gap_ratio, dim = 0)
        max_attr_mt2gap_ratio = torch.mean(all_max_attr_mt2gap_ratio, dim = 0)
        max_attr_gap2gap_ratio = torch.mean(all_max_attr_gap2gap_ratio, dim = 0)
        print("=============== 先算比例：src mt gap 对 gap 贡献比例最大值")
        print(max_attr_src2gap_ratio)
        print(max_attr_mt2gap_ratio)
        print(max_attr_gap2gap_ratio)

        print("======================= 标签预测正确错误，分开来算 ============================")

    def show_attr_ratio(self):
        # 对src而言，src、mt、gap分别的贡献比例 结果输出 ============================================= #
        all_avg_attr_src2src_ratio = torch.cat(self.attr_ratio["all_avg_attr_src2src_ratio"], 0)  # 延长拼接到一起 [all_src_token_num]
        all_avg_attr_mt2src_ratio = torch.cat(self.attr_ratio["all_avg_attr_mt2src_ratio"], 0)  # 延长拼接到一起 [all_src_token_num]
        all_avg_attr_gap2src_ratio = torch.cat(self.attr_ratio["all_avg_attr_gap2src_ratio"], 0)  # 延长拼接到一起 [all_src_token_num]
        avg_attr_src2src_ratio = torch.mean(all_avg_attr_src2src_ratio, dim = 0)
        avg_attr_mt2src_ratio = torch.mean(all_avg_attr_mt2src_ratio, dim = 0)
        avg_attr_gap2src_ratio = torch.mean(all_avg_attr_gap2src_ratio, dim = 0)
        print("=============== src mt gap 对 src 贡献平均比例")
        print(avg_attr_src2src_ratio)
        print(avg_attr_mt2src_ratio)
        print(avg_attr_gap2src_ratio)
        print("=============== 对正确预测token")
        print(np.mean(self.attr_ratio["all_avg_attr_src2src_ratio_right"]))
        print(np.mean(self.attr_ratio["all_avg_attr_mt2src_ratio_right"]))
        print(np.mean(self.attr_ratio["all_avg_attr_gap2src_ratio_right"]))
        print("=============== 对错误预测token")
        print(np.mean(self.attr_ratio["all_avg_attr_src2src_ratio_wrong"]))
        print(np.mean(self.attr_ratio["all_avg_attr_mt2src_ratio_wrong"]))
        print(np.mean(self.attr_ratio["all_avg_attr_gap2src_ratio_wrong"]))

        # 对mt而言，src、mt、gap分别的贡献比例 结果输出 ============================================= #
        all_avg_attr_src2mt_ratio = torch.cat(self.attr_ratio["all_avg_attr_src2mt_ratio"], 0)  # 延长拼接到一起 [all_mt_token_num]
        all_avg_attr_mt2mt_ratio = torch.cat(self.attr_ratio["all_avg_attr_mt2mt_ratio"], 0)  # 延长拼接到一起 [all_mt_token_num]
        all_avg_attr_gap2mt_ratio = torch.cat(self.attr_ratio["all_avg_attr_gap2mt_ratio"], 0)  # 延长拼接到一起 [all_mt_token_num]
        avg_attr_src2mt_ratio = torch.mean(all_avg_attr_src2mt_ratio, dim = 0)
        avg_attr_mt2mt_ratio = torch.mean(all_avg_attr_mt2mt_ratio, dim = 0)
        avg_attr_gap2mt_ratio = torch.mean(all_avg_attr_gap2mt_ratio, dim = 0)
        print("=============== src mt gap 对 mt 贡献平均比例")
        print(avg_attr_src2mt_ratio)
        print(avg_attr_mt2mt_ratio)
        print(avg_attr_gap2mt_ratio)
        print("=============== 对正确预测token")
        print(np.mean(self.attr_ratio["all_avg_attr_src2mt_ratio_right"]))
        print(np.mean(self.attr_ratio["all_avg_attr_mt2mt_ratio_right"]))
        print(np.mean(self.attr_ratio["all_avg_attr_gap2mt_ratio_right"]))
        print("=============== 对错误预测token")
        print(np.mean(self.attr_ratio["all_avg_attr_src2mt_ratio_wrong"]))
        print(np.mean(self.attr_ratio["all_avg_attr_mt2mt_ratio_wrong"]))
        print(np.mean(self.attr_ratio["all_avg_attr_gap2mt_ratio_wrong"]))

        # 对gap而言，src、mt、gap分别的贡献比例 结果输出 ============================================= #
        all_avg_attr_src2gap_ratio = torch.cat(self.attr_ratio["all_avg_attr_src2gap_ratio"], 0)  # 延长拼接到一起 [all_gap_token_num]
        all_avg_attr_mt2gap_ratio = torch.cat(self.attr_ratio["all_avg_attr_mt2gap_ratio"], 0)  # 延长拼接到一起 [all_gap_token_num]
        all_avg_attr_gap2gap_ratio = torch.cat(self.attr_ratio["all_avg_attr_gap2gap_ratio"], 0)  # 延长拼接到一起 [all_gap_token_num]
        avg_attr_src2gap_ratio = torch.mean(all_avg_attr_src2gap_ratio, dim = 0)
        avg_attr_mt2gap_ratio = torch.mean(all_avg_attr_mt2gap_ratio, dim = 0)
        avg_attr_gap2gap_ratio = torch.mean(all_avg_attr_gap2gap_ratio, dim = 0)
        print("=============== src mt gap 对 gap 贡献平均比例")
        print(avg_attr_src2gap_ratio)
        print(avg_attr_mt2gap_ratio)
        print(avg_attr_gap2gap_ratio)
        print("=============== 对正确预测token")
        print(np.mean(self.attr_ratio["all_avg_attr_src2gap_ratio_right"]))
        print(np.mean(self.attr_ratio["all_avg_attr_mt2gap_ratio_right"]))
        print(np.mean(self.attr_ratio["all_avg_attr_gap2gap_ratio_right"]))
        print("=============== 对错误预测token")
        print(np.mean(self.attr_ratio["all_avg_attr_src2gap_ratio_wrong"]))
        print(np.mean(self.attr_ratio["all_avg_attr_mt2gap_ratio_wrong"]))
        print(np.mean(self.attr_ratio["all_avg_attr_gap2gap_ratio_wrong"]))

    def cal_word_alignment(self, attr_token, srcmt_sent):
        src_sent_list = srcmt_sent.split("[SEP]")[0].split()
        src_len = len(src_sent_list)

        # src2mt 对一个src token 影响力最大的mt token
        attr_token_all2src = attr_token[:, :src_len]        # 所有token对src的贡献 [seq_len, src_len]
        attr_token_mt2src = attr_token_all2src[src_len + 2 :: 2]    # mt对src贡献   [mt_len, src_len]

        value, most_attr_mt_idx = torch.max(attr_token_mt2src, dim = 0)  # 对src最有贡献的mt idx，认为是模型预测从src到mt的对齐
        most_attr_mt_idx = most_attr_mt_idx.tolist()      # [src_len]
        self.word_alignment_pred["align_pred_lines_src2mt"].append(most_attr_mt_idx)

        # mt2src 对一个mt token 影响力最大的src token
        attr_token_all2mt = attr_token[:, src_len + 2::2]   # 所有token对mt的贡献 [seq_len, mt_len]
        attr_token_src2mt = attr_token_all2mt[:src_len]           # src对mt贡献  [src_len, mt_len]

        value, most_attr_src_idx = torch.max(attr_token_src2mt, dim = 0)  # 对mt最有贡献的src idx，认为是模型预测从mt到src的对齐
        most_attr_src_idx = most_attr_src_idx.tolist()      # [mt_len]
        self.word_alignment_pred["align_pred_lines_mt2src"].append(most_attr_src_idx)

    def show_word_alignment(self):
        # 词对齐结果打印
        # src2mt
        num_src2mt = 0
        ans_src2mt = 0
        for line_id, align_pred_line_src2mt in enumerate(self.word_alignment_pred["align_pred_lines_src2mt"]):
            gold_dict_line_src2mt = self.align_dict_src2mt[line_id]
            for src_idx in range(len(align_pred_line_src2mt)):
                if src_idx not in gold_dict_line_src2mt: continue       # 没有对齐的排除
                most_attr_mt = align_pred_line_src2mt[src_idx]
                if most_attr_mt in gold_dict_line_src2mt[src_idx]: ans_src2mt += 1
                num_src2mt += 1

        print("对一个src token而言，【影响力最大的mt token】有多大比例属于【和它对齐的mt token】")
        print("align_token_num_src2mt = %d" % ans_src2mt)
        print("all_consider_num_src2mt = %d" % num_src2mt)
        print("align_ratio_src2mt = %.6f" % (ans_src2mt / num_src2mt))

        # mt2src
        num_mt2src = 0
        ans_mt2src = 0
        for line_id, align_pred_line_mt2src in enumerate(self.word_alignment_pred["align_pred_lines_mt2src"]):
            gold_dict_line_mt2src = self.align_dict_mt2src[line_id]
            for src_idx in range(len(align_pred_line_mt2src)):
                if src_idx not in gold_dict_line_mt2src: continue       # 没有对齐的排除
                most_attr_mt = align_pred_line_mt2src[src_idx]
                if most_attr_mt in gold_dict_line_mt2src[src_idx]: ans_mt2src += 1
                num_mt2src += 1

        print("对一个mt token而言，【影响力最大的src token】有多大比例属于【和它对齐的src token】")
        print("align_token_num_mt2src = %d" % ans_mt2src)
        print("all_consider_num_mt2src = %d" % num_mt2src)
        print("align_ratio_mt2src = %.6f" % (ans_mt2src / num_mt2src))

    def cal_attr_different_part_origin(self, attr_token, srcmt_sent, id, k_adj):
        src_sent_list = srcmt_sent.split("[SEP]")[0].split()
        src_len = len(src_sent_list)
        mt_sent_list = srcmt_sent.split("[SEP]")[1].split()[1::2]
        mt_len = len(mt_sent_list)
        gap_sent_list = srcmt_sent.split("[SEP]")[1].split()[::2]
        gap_len = len(gap_sent_list)

        # 对每个src token而言，自身src token、上下文src token、对齐mt token、上下文mt token、gap token 分别的贡献计算排序
        gold_dict_line_src2mt = self.align_dict_src2mt[id]            # 这句话中每个src token应该对齐的mt token们

        attr_token_all2src = attr_token[:, :src_len]                  # 所有token对src的贡献 [seq_len, src_len]

        attr_token_src2src = attr_token_all2src[:src_len]             # src对src贡献  [src_len, src_len]
        attr_token_mt2src = attr_token_all2src[src_len + 2 :: 2]      # mt对src贡献   [mt_len, src_len]
        attr_token_gap2src = attr_token_all2src[src_len + 1 :: 2]     # gap对src贡献  [gap_len, src_len]
        #print(attr_token_src2src)

        for i in range(src_len):
            attr_src2this_src = attr_token_src2src[:, i]              # 所有src token对第i个src token的贡献  [src_len]
            attr_mt2this_src = attr_token_mt2src[:, i]                # 所有mt token对第i个src token的贡献   [mt_len]
            attr_gap2this_src = attr_token_gap2src[:, i]              # 所有gap token对第i个src token的贡献  [gap_len]
            attr_src2this_src_reverse = attr_src2this_src.flip(dims = [0])
            
            if i in gold_dict_line_src2mt:                            # 如果这个src token有对齐的mt token的话
                # 【这个src token】的【对齐mt token】和【上下文mt token】的贡献
                align_status_this_src2mt = torch.zeros_like(attr_mt2this_src).index_fill(0, torch.tensor(gold_dict_line_src2mt[i]).cuda(), True) # 长度同mt，01向量代表每个mt token是否是第i个src token的对齐token
                idx_this_src2mt_align = torch.where(align_status_this_src2mt == 1)[0]                                      # 和第i个src token对齐的mt token索引
                idx_this_src2mt_context = torch.where(align_status_this_src2mt == 0)[0]                                    # 第i个src token对齐的mt token的上下文mt token的索引，就是第i个src token不对齐的mt token的索引
                this_src2mt_align_attr = torch.index_select(attr_mt2this_src, 0, idx_this_src2mt_align)                    # [length of aligned mt of this src token] 和第i个src token对齐的mt token的贡献
                this_src2mt_context_attr = torch.index_select(attr_mt2this_src, 0, idx_this_src2mt_context)                # [length of context mt of this src token] 和第i个src token不对齐的mt token的贡献
                this_src2mt_align_attr_mean = torch.mean(this_src2mt_align_attr)                                           # [1] 和第i个src token对齐的mt token的贡献平均值
                this_src2mt_context_attr_mean = torch.mean(this_src2mt_context_attr)                                       # [1] 和第i个src token不对齐的mt token的贡献平均值
                self.attr_different_part["all_src2mt_align_attr_mean"].append(this_src2mt_align_attr_mean.item())          # 总计
                self.attr_different_part["all_src2mt_context_attr_mean"].append(this_src2mt_context_attr_mean.item())      # 总计

                # 【这个src token自身】的贡献
                this_src_self_attr_mean = attr_src2this_src[i]
                self.attr_different_part["all_src_self_attr_mean"].append(this_src_self_attr_mean.item())                  # 总计

                # 【除去自身src token之外的所有上下文src token】的贡献
                this_src_context_attr = torch.cat([attr_src2this_src_reverse[src_len - i : ], attr_src2this_src[i + 1 : ]], dim = 0)
                this_src_context_attr_mean = torch.mean(this_src_context_attr)                                             # [1] 第i个src token的上下文src token的贡献平均值
                self.attr_different_part["all_src_context_attr_mean"].append(this_src_context_attr_mean.item())            # 总计

                # 上下文拆分1：自身src token的【左右各k个邻近src token】的贡献
                this_src_adj_attr = torch.cat([attr_src2this_src_reverse[src_len - i : src_len - i + k_adj], attr_src2this_src[i + 1 : i + 1 + k_adj]], dim = 0)
                this_src_adj_attr_mean = torch.mean(this_src_adj_attr)
                self.attr_different_part["all_src_adj_attr_mean"].append(this_src_adj_attr_mean.item())                    # 总计

                # 上下文拆分2：除去 自身src token + 左右各k个邻近 之外的【非邻近 src token】的贡献
                this_src_remote_attr = torch.cat([attr_src2this_src_reverse[src_len - i + k_adj : ], attr_src2this_src[i + 1 + k_adj : ]], dim = 0)
                this_src_remote_attr_mean = torch.mean(this_src_remote_attr)
                self.attr_different_part["all_src_remote_attr_mean"].append(this_src_remote_attr_mean.item())              # 总计

                # 【全部gap token】的贡献
                this_src2gap_attr_mean = torch.mean(attr_gap2this_src)
                self.attr_different_part["all_src2gap_attr_mean"].append(this_src2gap_attr_mean.item())                    # 总计

        # 对每个mt token而言，自身mt token、上下文mt token、对齐src token、上下文src token、gap token 分别的贡献计算排序
        gold_dict_line_mt2src = self.align_dict_mt2src[id]            # 这句话中每个mt token应该对齐的src token们

        attr_token_all2mt = attr_token[:, src_len + 2::2]             # 所有token对mt的贡献 [seq_len, mt_len]

        attr_token_src2mt = attr_token_all2mt[:src_len]               # src对mt贡献  [src_len, mt_len]
        attr_token_mt2mt = attr_token_all2mt[src_len + 2 :: 2]        # mt对mt贡献   [mt_len, mt_len]
        attr_token_gap2mt = attr_token_all2mt[src_len + 1 :: 2]       # gap对mt贡献  [gap_len, mt_len]

        for i in range(mt_len):
            attr_src2this_mt = attr_token_src2mt[:, i]                # 所有src token对第i个mt token的贡献  [src_len]
            attr_mt2this_mt = attr_token_mt2mt[:, i]                  # 所有mt token对第i个mt token的贡献   [mt_len]
            attr_gap2this_mt = attr_token_gap2mt[:, i]                # 所有gap token对第i个mt token的贡献  [gap_len]
            attr_mt2this_mt_reverse = attr_mt2this_mt.flip(dims = [0])
            
            if i in gold_dict_line_mt2src:                            # 如果这个mt token有对齐的src token的话
                # 【这个mt token】的【对齐src token】和【上下文src token】的贡献
                align_status_this_mt2src = torch.zeros_like(attr_src2this_mt).index_fill(0, torch.tensor(gold_dict_line_mt2src[i]).cuda(), True) # 长度同src，01向量代表每个src token是否是第i个mt token的对齐token
                idx_this_mt2src_align = torch.where(align_status_this_mt2src == 1)[0]                                      # 和第i个mt token对齐的src token索引
                idx_this_mt2src_context = torch.where(align_status_this_mt2src == 0)[0]                                    # 第i个mt token对齐的src token的上下文src token的索引，就是第i个mt token不对齐的src token的索引
                this_mt2src_align_attr = torch.index_select(attr_src2this_mt, 0, idx_this_mt2src_align)                    # [length of aligned src of this mt token] 和第i个mt token对齐的src token的贡献
                this_mt2src_context_attr = torch.index_select(attr_src2this_mt, 0, idx_this_mt2src_context)                # [length of context src of this mt token] 和第i个mt token不对齐的src token的贡献
                this_mt2src_align_attr_mean = torch.mean(this_mt2src_align_attr)                                           # [1] 和第i个mt token对齐的src token的贡献平均值
                this_mt2src_context_attr_mean = torch.mean(this_mt2src_context_attr)                                       # [1] 和第i个mt token不对齐的src token的贡献平均值
                self.attr_different_part["all_mt2src_align_attr_mean"].append(this_mt2src_align_attr_mean.item())          # 总计
                self.attr_different_part["all_mt2src_context_attr_mean"].append(this_mt2src_context_attr_mean.item())      # 总计

                # 【这个mt token自身】的贡献
                this_mt_self_attr_mean = attr_mt2this_mt[i]
                self.attr_different_part["all_mt_self_attr_mean"].append(this_mt_self_attr_mean.item())                    # 总计

                # 【除去自身mt token之外的所有上下文mt token】的贡献
                this_mt_context_attr = torch.cat([attr_mt2this_mt_reverse[mt_len - i : ], attr_mt2this_mt[i + 1 : ]], dim = 0)
                this_mt_context_attr_mean = torch.mean(this_mt_context_attr)                                               # [1] 第i个mt token的上下文mt token的贡献平均值
                self.attr_different_part["all_mt_context_attr_mean"].append(this_mt_context_attr_mean.item())              # 总计

                # 上下文拆分1：自身mt token的【左右各k个邻近mt token】的贡献
                this_mt_adj_attr = torch.cat([attr_mt2this_mt_reverse[mt_len - i : mt_len - i + k_adj], attr_mt2this_mt[i + 1 : i + 1 + k_adj]], dim = 0)
                this_mt_adj_attr_mean = torch.mean(this_mt_adj_attr)
                self.attr_different_part["all_mt_adj_attr_mean"].append(this_mt_adj_attr_mean.item())                      # 总计

                # 上下文拆分2：除去 自身mt token + 左右各k个邻近 之外的【非邻近 mt token】的贡献
                this_mt_remote_attr = torch.cat([attr_mt2this_mt_reverse[mt_len - i + k_adj : ], attr_mt2this_mt[i + 1 + k_adj : ]], dim = 0)
                this_mt_remote_attr_mean = torch.mean(this_mt_remote_attr)
                self.attr_different_part["all_mt_remote_attr_mean"].append(this_mt_remote_attr_mean.item())                # 总计

                # 【全部gap token】的贡献
                this_mt2gap_attr_mean = torch.mean(attr_gap2this_mt)
                self.attr_different_part["all_mt2gap_attr_mean"].append(this_mt2gap_attr_mean.item())                      # 总计

        # 对每个gap token而言，自身gap token、其余gap token、邻近mt token、其余mt token、src token 分别的贡献计算排序
        attr_token_all2gap = attr_token[:, src_len + 1::2]  # 所有token对gap的贡献 [seq_len, gap_len]

        attr_token_src2gap = attr_token_all2gap[:src_len]           # src对gap贡献  [src_len, gap_len]
        attr_token_mt2gap = attr_token_all2gap[src_len + 2 :: 2]    # mt对gap贡献   [mt_len, gap_len]
        attr_token_gap2gap = attr_token_all2gap[src_len + 1 :: 2]   # gap对gap贡献  [gap_len, gap_len]
        attr_token_mtgap2gap = attr_token_all2gap[src_len + 1 : ]   # mt+gap对gap贡献  [mt_len + gap_len, gap_len]

        for i in range(gap_len):
            attr_src2this_gap = attr_token_src2gap[:, i]                # 所有src token对第i个gap token的贡献  [src_len]
            attr_mt2this_gap = attr_token_mt2gap[:, i]                  # 所有mt token对第i个gap token的贡献   [mt_len]
            attr_gap2this_gap = attr_token_gap2gap[:, i]                # 所有gap token对第i个gap token的贡献  [gap_len]
            attr_mtgap2this_gap = attr_token_mtgap2gap[:, i]            # 所有mt+gap token对第i个gap token的贡献  [mt_len + gap_len]
            attr_gap2this_gap_reverse = attr_gap2this_gap.flip(dims = [0])
            attr_mtgap2this_gap_reverse = attr_mtgap2this_gap.flip(dims = [0])

            # 【这个gap token自身】的贡献
            this_gap_self_attr_mean = attr_gap2this_gap[i]
            self.attr_different_part["all_gap_self_attr_mean"].append(this_gap_self_attr_mean.item())                    # 总计

            # 【除去自身gap token之外的所有上下文gap token】的贡献
            this_gap_context_attr = torch.cat([attr_gap2this_gap_reverse[gap_len - i : ], attr_gap2this_gap[i + 1 : ]], dim = 0)
            this_gap_context_attr_mean = torch.mean(this_gap_context_attr)                                               # [1] 第i个gap token的上下文gap token的贡献平均值
            self.attr_different_part["all_gap_context_attr_mean"].append(this_gap_context_attr_mean.item())              # 总计

            # 这个gap token的【左右各k个邻近mt token】的贡献
            this_gap2mt_adj_attr = torch.cat([attr_mtgap2this_gap_reverse[mt_len + gap_len - 2 * i : mt_len + gap_len - 2 * i + k_adj], attr_mtgap2this_gap[2 * i + 1 : 2 * i + 1 + k_adj]], dim = 0)
            this_gap2mt_adj_attr_mean = torch.mean(this_gap2mt_adj_attr)                                                # [1] 第i个gap token的邻近mt token的贡献平均值
            self.attr_different_part["all_gap2mt_adj_attr_mean"].append(this_gap2mt_adj_attr_mean.item())                  # 总计

            # 这个gap token的【非k邻近mt token】的贡献
            this_gap2mt_remote_attr = torch.cat([attr_mtgap2this_gap_reverse[mt_len + gap_len - 2 * i + k_adj + 1 :: 2], attr_mtgap2this_gap[2 * i + 2 + k_adj :: 2]], dim = 0)
            this_gap2mt_remote_attr_mean = torch.mean(this_gap2mt_remote_attr)                                                # [1] 第i个gap token的非邻近mt token的贡献平均值
            self.attr_different_part["all_gap2mt_remote_attr_mean"].append(this_gap2mt_remote_attr_mean.item())                  # 总计

            # 【全部src token】的贡献
            this_gap2src_attr_mean = torch.mean(attr_src2this_gap)
            self.attr_different_part["all_gap2src_attr_mean"].append(this_gap2src_attr_mean.item())                   # 总计
    
    def cal_attr_different_part(self, attr_token, srcmt_sent, id):
        src_sent_list = srcmt_sent.split("[SEP]")[0].split()
        src_len = len(src_sent_list)
        mt_sent_list = srcmt_sent.split("[SEP]")[1].split()[1::2]
        mt_len = len(mt_sent_list)
        gap_sent_list = srcmt_sent.split("[SEP]")[1].split()[::2]
        gap_len = len(gap_sent_list)

        # 对每个src token而言，自身src token、上下文src token、对齐mt token、上下文mt token、gap token 分别的贡献计算排序
        gold_dict_line_src2mt = self.align_dict_src2mt[id]            # 这句话中每个src token应该对齐的mt token们

        attr_token_all2src = attr_token[:, :src_len]                  # 所有token对src的贡献 [seq_len, src_len]

        attr_token_src2src = attr_token_all2src[:src_len]             # src对src贡献  [src_len, src_len]
        attr_token_mt2src = attr_token_all2src[src_len + 2 :: 2]      # mt对src贡献   [mt_len, src_len]
        attr_token_gap2src = attr_token_all2src[src_len + 1 :: 2]     # gap对src贡献  [gap_len, src_len]
        #print(attr_token_src2src)

        src_acc_line = self.src_acc_sign[id]

        for i in range(src_len):
            attr_src2this_src = attr_token_src2src[:, i]              # 所有src token对第i个src token的贡献  [src_len]
            attr_mt2this_src = attr_token_mt2src[:, i]                # 所有mt token对第i个src token的贡献   [mt_len]
            attr_gap2this_src = attr_token_gap2src[:, i]              # 所有gap token对第i个src token的贡献  [gap_len]
            attr_src2this_src_reverse = attr_src2this_src.flip(dims = [0])

            if i in gold_dict_line_src2mt:                            # 如果这个src token有对齐的mt token的话
                # 【这个src token】的【对齐mt token】和【上下文mt token】的贡献
                align_status_this_src2mt = torch.zeros_like(attr_mt2this_src).index_fill(0, torch.tensor(gold_dict_line_src2mt[i]).cuda(), True) # 长度同mt，01向量代表每个mt token是否是第i个src token的对齐token
                idx_this_src2mt_align = torch.where(align_status_this_src2mt == 1)[0]                                      # 和第i个src token对齐的mt token索引
                idx_this_src2mt_context = torch.where(align_status_this_src2mt == 0)[0]                                    # 第i个src token对齐的mt token的上下文mt token的索引，就是第i个src token不对齐的mt token的索引
                this_src2mt_align_attr = torch.index_select(attr_mt2this_src, 0, idx_this_src2mt_align)                    # [length of aligned mt of this src token] 和第i个src token对齐的mt token的贡献
                this_src2mt_context_attr = torch.index_select(attr_mt2this_src, 0, idx_this_src2mt_context)                # [length of context mt of this src token] 和第i个src token不对齐的mt token的贡献
                this_src2mt_align_attr_sum = torch.sum(this_src2mt_align_attr)                                           # [1] 和第i个src token对齐的mt token的贡献之和
                this_src2mt_context_attr_sum = torch.sum(this_src2mt_context_attr)                                       # [1] 和第i个src token不对齐的mt token的贡献之和
                #self.attr_different_part["all_src2mt_align_attr_sum"].append(this_src2mt_align_attr_sum.item())          # 总计
                #self.attr_different_part["all_src2mt_context_attr_sum"].append(this_src2mt_context_attr_sum.item())      # 总计

                # 【这个src token自身】的贡献
                this_src_self_attr_sum = attr_src2this_src[i]
                #self.attr_different_part["all_src_self_attr_sum"].append(this_src_self_attr_sum.item())                  # 总计

                # 【除去自身src token之外的所有上下文src token】的贡献
                this_src_context_attr = torch.cat([attr_src2this_src_reverse[src_len - i : ], attr_src2this_src[i + 1 : ]], dim = 0)
                this_src_context_attr_sum = torch.sum(this_src_context_attr)                                             # [1] 第i个src token的上下文src token的贡献之和
                #self.attr_different_part["all_src_context_attr_sum"].append(this_src_context_attr_sum.item())            # 总计

                # 【全部gap token】的贡献
                this_src2gap_attr_sum = torch.sum(attr_gap2this_src)
                #self.attr_different_part["all_src2gap_attr_sum"].append(this_src2gap_attr_sum.item())                    # 总计

                # 比例计算
                parts_sum = this_src2mt_align_attr_sum + this_src2mt_context_attr_sum + this_src_self_attr_sum + this_src_context_attr_sum + this_src2gap_attr_sum
                if src_acc_line[i] == '1':
                    self.attr_different_part["all_src2mt_align_attr_ratio_right"].append((this_src2mt_align_attr_sum / parts_sum).item())
                    self.attr_different_part["all_src2mt_context_attr_ratio_right"].append((this_src2mt_context_attr_sum / parts_sum).item())
                    self.attr_different_part["all_src_self_attr_ratio_right"].append((this_src_self_attr_sum / parts_sum).item())
                    self.attr_different_part["all_src_context_attr_ratio_right"].append((this_src_context_attr_sum / parts_sum).item())
                    self.attr_different_part["all_src2gap_attr_ratio_right"].append((this_src2gap_attr_sum / parts_sum).item())
                else:
                    self.attr_different_part["all_src2mt_align_attr_ratio_wrong"].append((this_src2mt_align_attr_sum / parts_sum).item())
                    self.attr_different_part["all_src2mt_context_attr_ratio_wrong"].append((this_src2mt_context_attr_sum / parts_sum).item())
                    self.attr_different_part["all_src_self_attr_ratio_wrong"].append((this_src_self_attr_sum / parts_sum).item())
                    self.attr_different_part["all_src_context_attr_ratio_wrong"].append((this_src_context_attr_sum / parts_sum).item())
                    self.attr_different_part["all_src2gap_attr_ratio_wrong"].append((this_src2gap_attr_sum / parts_sum).item())

        # 对每个mt token而言，自身mt token、上下文mt token、对齐src token、上下文src token、gap token 分别的贡献计算排序
        gold_dict_line_mt2src = self.align_dict_mt2src[id]            # 这句话中每个mt token应该对齐的src token们

        attr_token_all2mt = attr_token[:, src_len + 2::2]             # 所有token对mt的贡献 [seq_len, mt_len]

        attr_token_src2mt = attr_token_all2mt[:src_len]               # src对mt贡献  [src_len, mt_len]
        attr_token_mt2mt = attr_token_all2mt[src_len + 2 :: 2]        # mt对mt贡献   [mt_len, mt_len]
        attr_token_gap2mt = attr_token_all2mt[src_len + 1 :: 2]       # gap对mt贡献  [gap_len, mt_len]

        mt_acc_line = self.mt_acc_sign[id]

        for i in range(mt_len):
            attr_src2this_mt = attr_token_src2mt[:, i]                # 所有src token对第i个mt token的贡献  [src_len]
            attr_mt2this_mt = attr_token_mt2mt[:, i]                  # 所有mt token对第i个mt token的贡献   [mt_len]
            attr_gap2this_mt = attr_token_gap2mt[:, i]                # 所有gap token对第i个mt token的贡献  [gap_len]
            attr_mt2this_mt_reverse = attr_mt2this_mt.flip(dims = [0])
            
            if i in gold_dict_line_mt2src:                            # 如果这个mt token有对齐的src token的话
                # 【这个mt token】的【对齐src token】和【上下文src token】的贡献
                align_status_this_mt2src = torch.zeros_like(attr_src2this_mt).index_fill(0, torch.tensor(gold_dict_line_mt2src[i]).cuda(), True) # 长度同src，01向量代表每个src token是否是第i个mt token的对齐token
                idx_this_mt2src_align = torch.where(align_status_this_mt2src == 1)[0]                                      # 和第i个mt token对齐的src token索引
                idx_this_mt2src_context = torch.where(align_status_this_mt2src == 0)[0]                                    # 第i个mt token对齐的src token的上下文src token的索引，就是第i个mt token不对齐的src token的索引
                this_mt2src_align_attr = torch.index_select(attr_src2this_mt, 0, idx_this_mt2src_align)                    # [length of aligned src of this mt token] 和第i个mt token对齐的src token的贡献
                this_mt2src_context_attr = torch.index_select(attr_src2this_mt, 0, idx_this_mt2src_context)                # [length of context src of this mt token] 和第i个mt token不对齐的src token的贡献
                this_mt2src_align_attr_sum = torch.sum(this_mt2src_align_attr)                                           # [1] 和第i个mt token对齐的src token的贡献平均值
                this_mt2src_context_attr_sum = torch.sum(this_mt2src_context_attr)                                       # [1] 和第i个mt token不对齐的src token的贡献平均值
                #self.attr_different_part["all_mt2src_align_attr_sum"].append(this_mt2src_align_attr_sum.item())          # 总计
                #self.attr_different_part["all_mt2src_context_attr_sum"].append(this_mt2src_context_attr_sum.item())      # 总计

                # 【这个mt token自身】的贡献
                this_mt_self_attr_sum = attr_mt2this_mt[i]
                #self.attr_different_part["all_mt_self_attr_sum"].append(this_mt_self_attr_sum.item())                    # 总计

                # 【除去自身mt token之外的所有上下文mt token】的贡献
                this_mt_context_attr = torch.cat([attr_mt2this_mt_reverse[mt_len - i : ], attr_mt2this_mt[i + 1 : ]], dim = 0)
                this_mt_context_attr_sum = torch.sum(this_mt_context_attr)                                               # [1] 第i个mt token的上下文mt token的贡献平均值
                #self.attr_different_part["all_mt_context_attr_sum"].append(this_mt_context_attr_sum.item())              # 总计

                # 【全部gap token】的贡献
                this_mt2gap_attr_sum = torch.sum(attr_gap2this_mt)
                #self.attr_different_part["all_mt2gap_attr_sum"].append(this_mt2gap_attr_sum.item())                      # 总计

                # 比例计算
                parts_sum = this_mt2src_align_attr_sum + this_mt2src_context_attr_sum + this_mt_self_attr_sum + this_mt_context_attr_sum + this_mt2gap_attr_sum
                if mt_acc_line[i] == '1':
                    self.attr_different_part["all_mt2src_align_attr_ratio_right"].append((this_mt2src_align_attr_sum / parts_sum).item())
                    self.attr_different_part["all_mt2src_context_attr_ratio_right"].append((this_mt2src_context_attr_sum / parts_sum).item())
                    self.attr_different_part["all_mt_self_attr_ratio_right"].append((this_mt_self_attr_sum / parts_sum).item())
                    self.attr_different_part["all_mt_context_attr_ratio_right"].append((this_mt_context_attr_sum / parts_sum).item())
                    self.attr_different_part["all_mt2gap_attr_ratio_right"].append((this_mt2gap_attr_sum / parts_sum).item())
                else:
                    self.attr_different_part["all_mt2src_align_attr_ratio_wrong"].append((this_mt2src_align_attr_sum / parts_sum).item())
                    self.attr_different_part["all_mt2src_context_attr_ratio_wrong"].append((this_mt2src_context_attr_sum / parts_sum).item())
                    self.attr_different_part["all_mt_self_attr_ratio_wrong"].append((this_mt_self_attr_sum / parts_sum).item())
                    self.attr_different_part["all_mt_context_attr_ratio_wrong"].append((this_mt_context_attr_sum / parts_sum).item())
                    self.attr_different_part["all_mt2gap_attr_ratio_wrong"].append((this_mt2gap_attr_sum / parts_sum).item())

        """TODO
        # 对每个gap token而言，自身gap token、其余gap token、邻近mt token、其余mt token、src token 分别的贡献计算排序
        attr_token_all2gap = attr_token[:, src_len + 1::2]  # 所有token对gap的贡献 [seq_len, gap_len]

        attr_token_src2gap = attr_token_all2gap[:src_len]           # src对gap贡献  [src_len, gap_len]
        attr_token_mt2gap = attr_token_all2gap[src_len + 2 :: 2]    # mt对gap贡献   [mt_len, gap_len]
        attr_token_gap2gap = attr_token_all2gap[src_len + 1 :: 2]   # gap对gap贡献  [gap_len, gap_len]
        attr_token_mtgap2gap = attr_token_all2gap[src_len + 1 : ]   # mt+gap对gap贡献  [mt_len + gap_len, gap_len]

        for i in range(gap_len):
            attr_src2this_gap = attr_token_src2gap[:, i]                # 所有src token对第i个gap token的贡献  [src_len]
            attr_mt2this_gap = attr_token_mt2gap[:, i]                  # 所有mt token对第i个gap token的贡献   [mt_len]
            attr_gap2this_gap = attr_token_gap2gap[:, i]                # 所有gap token对第i个gap token的贡献  [gap_len]
            attr_mtgap2this_gap = attr_token_mtgap2gap[:, i]            # 所有mt+gap token对第i个gap token的贡献  [mt_len + gap_len]
            attr_gap2this_gap_reverse = attr_gap2this_gap.flip(dims = [0])
            attr_mtgap2this_gap_reverse = attr_mtgap2this_gap.flip(dims = [0])

            # 【这个gap token自身】的贡献
            this_gap_self_attr_sum = attr_gap2this_gap[i]
            self.attr_different_part["all_gap_self_attr_sum"].append(this_gap_self_attr_sum.item())                    # 总计

            # 【除去自身gap token之外的所有上下文gap token】的贡献
            this_gap_context_attr = torch.cat([attr_gap2this_gap_reverse[gap_len - i : ], attr_gap2this_gap[i + 1 : ]], dim = 0)
            this_gap_context_attr_sum = torch.sum(this_gap_context_attr)                                               # [1] 第i个gap token的上下文gap token的贡献之和
            self.attr_different_part["all_gap_context_attr_sum"].append(this_gap_context_attr_sum.item())              # 总计

            # 这个gap token的【左右各k个邻近mt token】的贡献
            this_gap2mt_adj_attr = torch.cat([attr_mtgap2this_gap_reverse[mt_len + gap_len - 2 * i : mt_len + gap_len - 2 * i + k_adj], attr_mtgap2this_gap[2 * i + 1 : 2 * i + 1 + k_adj]], dim = 0)
            this_gap2mt_adj_attr_sum = torch.sum(this_gap2mt_adj_attr)                                                # [1] 第i个gap token的邻近mt token的贡献之和
            self.attr_different_part["all_gap2mt_adj_attr_sum"].append(this_gap2mt_adj_attr_sum.item())                  # 总计

            # 这个gap token的【非k邻近mt token】的贡献
            this_gap2mt_remote_attr = torch.cat([attr_mtgap2this_gap_reverse[mt_len + gap_len - 2 * i + k_adj + 1 :: 2], attr_mtgap2this_gap[2 * i + 2 + k_adj :: 2]], dim = 0)
            this_gap2mt_remote_attr_sum = torch.sum(this_gap2mt_remote_attr)                                                # [1] 第i个gap token的非邻近mt token的贡献之和
            self.attr_different_part["all_gap2mt_remote_attr_sum"].append(this_gap2mt_remote_attr_sum.item())                  # 总计

            # 【全部src token】的贡献
            this_gap2src_attr_sum = torch.sum(attr_src2this_gap)
            self.attr_different_part["all_gap2src_attr_sum"].append(this_gap2src_attr_sum.item())                   # 总计
        """
    
    def show_attr_different_part_origin(self, k_adj):
        print("对每个src token而言 =====================================")
        print("自身src_token 贡献均值 = %.6f" % np.mean(self.attr_different_part["all_src_self_attr_mean"]))
        print("其余src_token 贡献均值 = %.6f" % np.mean(self.attr_different_part["all_src_context_attr_mean"]))
        print(" —— 其中，邻近的%d个src_token 贡献均值 = %.6f" % (k_adj, np.mean(self.attr_different_part["all_src_adj_attr_mean"])))
        print(" —— 非邻近的src_token 贡献均值 = %.6f" % np.mean(self.attr_different_part["all_src_remote_attr_mean"]))
        print("对齐的mt_token 贡献均值 = %.6f" % np.mean(self.attr_different_part["all_src2mt_align_attr_mean"]))
        print("其余mt_token 贡献均值 = %.6f" % np.mean(self.attr_different_part["all_src2mt_context_attr_mean"]))
        print("gap 贡献均值 = %.6f" % np.mean(self.attr_different_part["all_src2gap_attr_mean"]))

        print("对每个mt token而言 =====================================")
        print("自身mt_token 贡献均值 = %.6f" % np.mean(self.attr_different_part["all_mt_self_attr_mean"]))
        print("其余mt_token 贡献均值 = %.6f" % np.mean(self.attr_different_part["all_mt_context_attr_mean"]))
        print(" —— 其中，邻近的%d个mt_token 贡献均值 = %.6f" % (k_adj, np.mean(self.attr_different_part["all_mt_adj_attr_mean"])))
        print(" —— 非邻近的mt_token 贡献均值 = %.6f" % np.mean(self.attr_different_part["all_mt_remote_attr_mean"]))
        print("对齐的src_token 贡献均值 = %.6f" % np.mean(self.attr_different_part["all_mt2src_align_attr_mean"]))
        print("其余src_token 贡献均值 = %.6f" % np.mean(self.attr_different_part["all_mt2src_context_attr_mean"]))
        print("gap 贡献均值 = %.6f" % np.mean(self.attr_different_part["all_mt2gap_attr_mean"]))

        print("对每个gap token而言 =====================================")
        print("自身gap_token 贡献均值 = %.6f" % np.mean(self.attr_different_part["all_gap_self_attr_mean"]))
        print("其余gap_token 贡献均值 = %.6f" % np.mean(self.attr_different_part["all_gap_context_attr_mean"]))
        print("邻近的mt_token 贡献均值 = %.6f" % np.mean(self.attr_different_part["all_gap2mt_adj_attr_mean"]))
        print("非邻近mt_token 贡献均值 = %.6f" % np.mean(self.attr_different_part["all_gap2mt_remote_attr_mean"]))
        print("src 贡献均值 = %.6f" % np.mean(self.attr_different_part["all_gap2src_attr_mean"]))

    def show_attr_different_part(self):
        print("对每个src token而言 =====================================")
        print("对预测正确的token而言，")
        print("自身src_token 贡献平均比例 = %.6f" % np.mean(self.attr_different_part["all_src_self_attr_ratio_right"]))
        print("其余src_token 贡献平均比例 = %.6f" % np.mean(self.attr_different_part["all_src_context_attr_ratio_right"]))
        print("对齐的mt_token 贡献平均比例 = %.6f" % np.mean(self.attr_different_part["all_src2mt_align_attr_ratio_right"]))
        print("其余mt_token 贡献平均比例 = %.6f" % np.mean(self.attr_different_part["all_src2mt_context_attr_ratio_right"]))
        print("gap 贡献平均比例 = %.6f" % np.mean(self.attr_different_part["all_src2gap_attr_ratio_right"]))
        print("对预测错误的token而言，")
        print("自身src_token 贡献平均比例 = %.6f" % np.mean(self.attr_different_part["all_src_self_attr_ratio_wrong"]))
        print("其余src_token 贡献平均比例 = %.6f" % np.mean(self.attr_different_part["all_src_context_attr_ratio_wrong"]))
        print("对齐的mt_token 贡献平均比例 = %.6f" % np.mean(self.attr_different_part["all_src2mt_align_attr_ratio_wrong"]))
        print("其余mt_token 贡献平均比例 = %.6f" % np.mean(self.attr_different_part["all_src2mt_context_attr_ratio_wrong"]))
        print("gap 贡献平均比例 = %.6f" % np.mean(self.attr_different_part["all_src2gap_attr_ratio_wrong"]))
        
        print("对每个mt token而言 =====================================")
        print("对预测正确的token而言，")
        print("自身mt_token 贡献平均比例 = %.6f" % np.mean(self.attr_different_part["all_mt_self_attr_ratio_right"]))
        print("其余mt_token 贡献平均比例 = %.6f" % np.mean(self.attr_different_part["all_mt_context_attr_ratio_right"]))
        print("对齐的src_token 贡献平均比例 = %.6f" % np.mean(self.attr_different_part["all_mt2src_align_attr_ratio_right"]))
        print("其余src_token 贡献平均比例 = %.6f" % np.mean(self.attr_different_part["all_mt2src_context_attr_ratio_right"]))
        print("gap 贡献平均比例 = %.6f" % np.mean(self.attr_different_part["all_mt2gap_attr_ratio_right"]))
        print("对预测错误的token而言，")
        print("自身mt_token 贡献平均比例 = %.6f" % np.mean(self.attr_different_part["all_mt_self_attr_ratio_wrong"]))
        print("其余mt_token 贡献平均比例 = %.6f" % np.mean(self.attr_different_part["all_mt_context_attr_ratio_wrong"]))
        print("对齐的src_token 贡献平均比例 = %.6f" % np.mean(self.attr_different_part["all_mt2src_align_attr_ratio_wrong"]))
        print("其余src_token 贡献平均比例 = %.6f" % np.mean(self.attr_different_part["all_mt2src_context_attr_ratio_wrong"]))
        print("gap 贡献平均比例 = %.6f" % np.mean(self.attr_different_part["all_mt2gap_attr_ratio_wrong"]))
        

    def attribution(self, to_predict, plot_path_prefix, align_file_path, \
            src_tag_acc_sign_path = None, mt_tag_acc_sign_path = None, \
            gap_tag_acc_sign_path = None, k_adj = 1, split_on_space = True):
        """
        梯度分析
        """

        device = self.device
        model = self.model
        args = self.args
        pad_token_label_id = self.pad_token_label_id
        preds = None

        to_predict = format_to_test(to_predict, self.args)
        # ['rowboats and pedalboats can be rented at the boathouse . [SEP] _ 游艇 _ 和 _ 脚踏船 _ 可以 _ 在 _ 船坞 _ 租用 _ . _', 'he then punches out Wolverine and carries him off to the laboratory . [SEP] _ 然后 _ 他 _ 猛击 _ 狼 _ , _ 把 _ 它 _ 带到 _ 实验室 _ . _']

        if split_on_space:
            if self.args.model_type == "layoutlm":
                predict_examples = [
                    InputExample(
                        i, sentence.split(), [self.args.labels_list[0] for word in sentence.split()], x0, y0, x1, y1
                    )
                    for i, (sentence, x0, y0, x1, y1) in enumerate(to_predict)
                ]
                to_predict = [sentence for sentence, *_ in to_predict]
            else:
                predict_examples = [
                    InputExample(i, sentence.split(), [self.args.labels_list[0] for word in sentence.split()])
                    for i, sentence in enumerate(to_predict)
                ]
        else:
            if self.args.model_type == "layoutlm":
                predict_examples = [
                    InputExample(i, sentence, [self.args.labels_list[0] for word in sentence], x0, y0, x1, y1)
                    for i, (sentence, x0, y0, x1, y1) in enumerate(to_predict)
                ]
                to_predict = [sentence for sentence, *_ in to_predict]
            else:
                predict_examples = [
                    InputExample(i, sentence, [self.args.labels_list[0] for word in sentence])
                    for i, sentence in enumerate(to_predict)
                ]

        eval_dataset = self.load_and_cache_examples(None, to_predict=predict_examples)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        self._move_model_to_device()

        model.eval()

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        
        # 学习对齐字典
        with open(align_file_path, 'r', encoding='utf-8') as f:
            for line_id, align_line in enumerate(f.readlines()):
                align_line_dict_mt2src = dict()
                align_line_dict_src2mt = dict()
                align_line_list = align_line.strip('\n').split()
                for align_item in align_line_list:
                    src_idx, mt_idx = align_item.split('-')
                    src_idx = int(src_idx)
                    mt_idx = int(mt_idx)
                    if mt_idx not in align_line_dict_mt2src: align_line_dict_mt2src[mt_idx] = []
                    align_line_dict_mt2src[mt_idx].append(src_idx)
                    if src_idx not in align_line_dict_src2mt: align_line_dict_src2mt[src_idx] = []
                    align_line_dict_src2mt[src_idx].append(mt_idx)
                self.align_dict_mt2src[line_id] = align_line_dict_mt2src
                self.align_dict_src2mt[line_id] = align_line_dict_src2mt

        # 学习预测正确与否标签
        with open(src_tag_acc_sign_path, 'r', encoding='utf-8') as fs, \
            open(mt_tag_acc_sign_path, 'r', encoding='utf-8') as fm, \
            open(gap_tag_acc_sign_path, 'r', encoding='utf-8') as fg:
            src_acc_lines = fs.readlines()
            mt_acc_lines = fm.readlines()
            gap_acc_lines = fg.readlines()
            line_id = 0
            for src_acc_line, mt_acc_line, gap_acc_line in zip(src_acc_lines, mt_acc_lines, gap_acc_lines):
                src_acc_line = src_acc_line.strip('\n').split()
                mt_acc_line = mt_acc_line.strip('\n').split()
                gap_acc_line = gap_acc_line.strip('\n').split()
                self.src_acc_sign[line_id] = src_acc_line
                self.mt_acc_sign[line_id] = mt_acc_line
                self.gap_acc_sign[line_id] = gap_acc_line
                line_id += 1


        id = 0

        for batch in tqdm(eval_dataloader, disable=args.silent, desc="Running Attribution"):
            print(id)
            srcmt_sent = to_predict[id]  # 这个样本，文字版
            srcmt_sent_list = to_predict[id].split()  # 这个样本，文字列表版

            batch = tuple(t.to(device) for t in batch)   # 每条数据，4个tensor
            inputs = self._get_inputs_dict(batch)  # input_ids, attention_mask, labels  各[1, 200]维tensor [batch_size, seq_len]
            #print(model) # XLMRobertaForTokenClassification
            outputs = model(**inputs, output_hidden_states=True)
            tmp_eval_loss, logits, hidden_states = outputs[:3]   # logits [1, 200, 2]   hidden_states 25 * [1, 200, 1024]  1层embedding + 24层模型隐层，(batch_size, sequence_length, hidden_size)

            embeddings_output = hidden_states[0]
            embeddings_output.register_hook(self.save_grad("embeddings_output"))   # embeddings输出 需要保存梯度

            logits_matter, _ = logits.squeeze(0).max(dim = -1)   # [200] 重要的预测结果
            logits_matter = -logits_matter

            sign = torch.where(inputs["labels"][0] == 0)[0]  # 真正每个词一个位置的索引

            # ============================================= 梯度计算，所有token src+mt+gap 对所有token src+mt+gap 的梯度 ============================================= #
            embed_grads_tokens = []
            for i in range(logits_matter.size(-1)):    # 直接是所有 src + mt拼一起的
                if i not in sign: continue
                model.zero_grad()
                logits_matter[i].backward(retain_graph=True)   # 第i个位置回传梯度
                embed_grads_token = torch.abs(self.grads['embeddings_output'].squeeze(0)).sum(-1)   # [200]
                embed_grads_token_real = torch.index_select(embed_grads_token, 0, sign)  # [28] [real_seq_len]
                embed_grads_tokens.append(embed_grads_token_real)
            # 矩阵：每个token对每个token标签预测的贡献
            attr_token = torch.stack(embed_grads_tokens,1)    # [real_seq_len, real_seq_len]
            attr_token = F.normalize(attr_token, dim = 0)

            # ============================================= 画图 ============================================= #
            self.plot_attribution(attr_token, srcmt_sent_list, plot_path_prefix, id)

            # ============================================= 贡献比例分析 ============================================= #
            #self.cal_attr_ratio(attr_token, srcmt_sent, id)
            
            # ============================================= 词对齐分析 ============================================= #
            #self.cal_word_alignment(attr_token, srcmt_sent)

            # ============================================= 不同因素贡献排序分析 ============================================= #
            #self.cal_attr_different_part(attr_token, srcmt_sent, id)
            

            id += 1
            if id >= 0: break

        # ============================================= 分析结果展示 ============================================= #
        #self.show_attr_ratio()                  # 贡献比例分析
        #self.show_word_alignment()              # 词对齐分析
        #self.show_attr_different_part()     # 不同因素贡献排序分析
        

        print("Finale.")
        return

    def _convert_tokens_to_word_logits(self, input_ids, label_ids, attention_mask, logits):

        ignore_ids = [
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token),
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token),
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token),
        ]

        # Remove unuseful positions
        masked_ids = input_ids[(1 == attention_mask)]
        masked_labels = label_ids[(1 == attention_mask)]
        masked_logits = logits[(1 == attention_mask)]
        for id in ignore_ids:
            masked_labels = masked_labels[(id != masked_ids)]
            masked_logits = masked_logits[(id != masked_ids)]
            masked_ids = masked_ids[(id != masked_ids)]

        # Map to word logits
        word_logits = []
        tmp = []
        for n, lab in enumerate(masked_labels):
            if lab != self.pad_token_label_id:
                if n != 0:
                    word_logits.append(tmp)
                tmp = [list(masked_logits[n])]
            else:
                tmp.append(list(masked_logits[n]))
        word_logits.append(tmp)

        return word_logits

    def load_and_cache_examples(self, data, evaluate=False, no_cache=False, to_predict=None):
        """
        Reads data_file and generates a TensorDataset containing InputFeatures. Caches the InputFeatures.
        Utility function for train() and eval() methods. Not intended to be used directly.

        Args:
            data: Path to a .txt file containing training or evaluation data OR a pandas DataFrame containing 3 columns - sentence_id, words, labels.
                    If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.
            evaluate (optional): Indicates whether the examples are for evaluation or for training.
            no_cache (optional): Force feature conversion and prevent caching. I.e. Ignore cached features even if present.

        """  # noqa: ignore flake8"

        process_count = self.args.process_count

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        mode = "dev" if evaluate else "train"
        if not to_predict and isinstance(data, str) and self.args.lazy_loading:
            dataset = LazyQEDataset(data, tokenizer, self.args)
        else:
            if to_predict:
                examples = to_predict
                no_cache = True
            else:
                if isinstance(data, str):
                    examples = read_examples_from_file(
                        data, mode, bbox=True if self.args.model_type == "layoutlm" else False
                    )
                else:
                    if self.args.lazy_loading:
                        raise ValueError("Input must be given as a path to a file when using lazy loading")
                    examples = get_examples_from_df(data, bbox=True if self.args.model_type == "layoutlm" else False)

            cached_features_file = os.path.join(
                args.cache_dir,
                "cached_{}_{}_{}_{}_{}".format(
                    mode, args.model_type, args.max_seq_length, self.num_labels, len(examples),
                ),
            )
            if not no_cache:
                os.makedirs(self.args.cache_dir, exist_ok=True)

            if os.path.exists(cached_features_file) and (
                    (not args.reprocess_input_data and not no_cache)
                    or (mode == "dev" and args.use_cached_eval_features and not no_cache)
            ):
                features = torch.load(cached_features_file)
                logger.info(f" Features loaded from cache at {cached_features_file}")
            else:
                logger.info(" Converting to features started.")
                features = convert_examples_to_features(
                    examples,
                    self.args.labels_list,
                    self.args.max_seq_length,  # 200
                    self.tokenizer,
                    # XLNet has a CLS token at the end
                    cls_token_at_end=bool(args.model_type in ["xlnet"]),
                    cls_token=tokenizer.cls_token,
                    cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                    sep_token=tokenizer.sep_token,
                    # RoBERTa uses an extra separator b/w pairs of sentences,
                    # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                    sep_token_extra=bool(args.model_type in ["roberta"]),
                    # PAD on the left for XLNet
                    pad_on_left=bool(args.model_type in ["xlnet"]),
                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                    pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                    pad_token_label_id=self.pad_token_label_id,
                    process_count=process_count,
                    silent=args.silent,
                    use_multiprocessing=args.use_multiprocessing,
                    chunksize=args.multiprocessing_chunksize,
                )

                if not no_cache:
                    torch.save(features, cached_features_file)

            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
            """
            print('aaaaaaaaaa')
            print(all_input_ids)
            print(all_input_mask)
            print(all_segment_ids)
            print(all_label_ids)
            assert 1==2
            """

            if self.args.model_type == "layoutlm":
                all_bboxes = torch.tensor([f.bboxes for f in features], dtype=torch.long)

            if self.args.onnx:
                return all_label_ids

            if self.args.model_type == "layoutlm":
                dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_bboxes)
            else:
                dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)   # dataset的4个组成部分

        return dataset

    def convert_to_onnx(self, output_dir=None, set_onnx_arg=True):
        """Convert the model to ONNX format and save to output_dir

        Args:
            output_dir (str, optional): If specified, ONNX model will be saved to output_dir (else args.output_dir will be used). Defaults to None.
            set_onnx_arg (bool, optional): Updates the model args to set onnx=True. Defaults to True.
        """  # noqa
        if not output_dir:
            output_dir = os.path.join(self.args.output_dir, "onnx")
        os.makedirs(output_dir, exist_ok=True)

        if os.listdir(output_dir):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Output directory for onnx conversion must be empty.".format(output_dir)
            )

        onnx_model_name = os.path.join(output_dir, "onnx_model.onnx")

        with tempfile.TemporaryDirectory() as temp_dir:
            self.save_model(output_dir=temp_dir, model=self.model)

            convert(
                framework="pt",
                model=temp_dir,
                tokenizer=self.tokenizer,
                output=Path(onnx_model_name),
                pipeline_name="ner",
                opset=11,
            )

        self.args.onnx = True
        self.tokenizer.save_pretrained(output_dir)
        self.config.save_pretrained(output_dir)
        self._save_model_args(output_dir)

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def _get_inputs_dict(self, batch):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[3],
        }
        # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
        if self.args.model_type in ["bert", "xlnet", "albert", "layoutlm"]:
            inputs["token_type_ids"] = batch[2]

        if self.args.model_type == "layoutlm":
            inputs["bbox"] = batch[4]

        return inputs

    def _create_training_progress_scores(self, **kwargs):
        extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
            "global_step": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "train_loss": [],
            "eval_loss": [],
            **extra_metrics,
        }

        return training_progress_scores

    def save_model(self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None):
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            self._save_model_args(output_dir)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def _save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = MicroTransQuestArgs()
        args.load(input_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]
