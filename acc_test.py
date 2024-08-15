# !/usr/bin/env python3

import os
import shutil
import time
import json
import random
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, default_data_collator, get_scheduler

from rich.table import Table
from rich.align import Align
from rich.console import Console
from rich import print

from Augmenter import Augmenter
from inference import inference
from metrics import SpanEvaluator
from modelv20 import UIE, convert_example
# from utils import download_pretrained_model
from iTrainingLogger import iSummaryWriter
from torch.nn.parallel import DataParallel
import time
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


parser = argparse.ArgumentParser()
# parser.add_argument("--pretrained_model", default='uie-base-zh', type=str, choices=['uie-base-zh'], help="backbone of encoder.")
parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--save_dir", default="./checkpoint", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_len", default=300, type=int,help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--auto_neg_rate", default=0.5, type=float, help="Auto negative samples generated ratio.")
parser.add_argument("--auto_pos_rate", default=0.5, type=float, help="Auto positive samples generated ratio.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--valid_steps", default=200, type=int, required=False, help="evaluate frequecny.")
parser.add_argument("--auto_da_epoch", default=0, type=int, required=False, help="auto add positive/negative samples policy frequency.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument('--device', default="cuda:0", help="Select which device to train model, defaults to gpu.")
parser.add_argument('--device_list', default=[1,2], type=list, help="Select which device to train model, defaults to gpu.")
parser.add_argument("--img_log_dir", default='logs', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='Model Performance', type=str, help="Logging image file name.")
parser.add_argument("--txt_log_name", default='tr_log.txt', type=str, help="Logging txt file name.")
args = parser.parse_args()

writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)


def evaluate(model, metric, data_loader, global_step):
    """
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        metric: 评估指标类(metric)
        data_loader: 测试集的dataloader
        global_step: 当前训练步数
    """
    model.eval()
    metric.reset()
    with torch.no_grad():
        for batch in data_loader:
            model.to(args.device)
            start_prob, end_prob = model(input_ids=batch['input_ids'].to(args.device),
                                            token_type_ids=batch['token_type_ids'].to(args.device),
                                            attention_mask=batch['attention_mask'].to(args.device))
            start_ids = batch['start_ids'].to(torch.float32).detach().numpy()
            end_ids = batch['end_ids'].to(torch.float32).detach().numpy()
            num_correct, num_infer, num_label = metric.compute(start_prob.cpu().detach().numpy(), 
                                                                end_prob.cpu().detach().numpy(),  
                                                                start_ids, 
                                                                end_ids)
            metric.update(num_correct, num_infer, num_label)
        
        precision, recall, f1 = metric.accumulate()
        writer.add_scalar('eval-precision', precision, global_step)
        writer.add_scalar('eval-recall', recall, global_step)
        writer.add_scalar('eval-f1', f1, global_step)
        writer.record()
    model.train()
    return precision, recall, f1

if __name__ == '__main__':
    model = UIE(args.device, args.batch_size)
    weights = torch.load('checkpoints/DuIE2/model_best_v20/model.pt', map_location=args.device)
    model.load_state_dict(weights.module.state_dict())
    tokenizer = AutoTokenizer.from_pretrained("checkpoints/DuIE2/model_best_v20")

    model = DataParallel(model, device_ids=args.device_list)
    dataset = load_dataset('text', data_files={'train': args.train_path,
                                                'dev': args.dev_path})    
    # print(dataset)
    convert_func = partial(convert_example, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    dataset = dataset.map(convert_func, batched=True)
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["dev"]
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.batch_size)
    metric = SpanEvaluator()
    global_step, best_f1 = 0, 0              
    precision, recall, f1 = evaluate(model, metric, eval_dataloader, global_step)
    print("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" % (precision, recall, f1))
