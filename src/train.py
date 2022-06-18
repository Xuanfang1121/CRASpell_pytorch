# -*- coding: utf-8 -*-
# @Time    : 2022/6/12 20:04
# @Author  : zxf
import os
import json
import traceback

import torch
from torch.optim import AdamW
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from common.common import logger
from utils.util import collate_fn
from utils.util import DataProcessor
from utils.util import model_evaluate
from config.getConfig import get_config
from model.CRASpellModel import CRASpellModel


def train(config_file):
    Config = get_config(config_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = Config["visible_gpus"]
    device = "cpu" if Config["visible_gpus"] == "-1" else "cuda"
    # checkpath
    if not os.path.exists(Config["output_path"]):
        os.mkdir(Config["output_path"])
    # tokenzier
    tokenizer = BertTokenizer.from_pretrained(Config["pretrain_model_path"],
                                              add_special_tokens=False,  # 不添加CLS,SEP
                                              do_lower_case=True
                                              )
    # dataprocessor
    train_processor = DataProcessor(tokenizer, Config["train_data_path"],
                                    Config["max_seq_length"],
                                    Config["same_py_file"],
                                    Config["simi_py_file"],
                                    Config["stroke_file"])
    train_dataset = train_processor.file2features()
    # add noise data
    train_dataset = train_processor.build_data_noise_feature(train_dataset)
    logger.info("data add noise")
    logger.info("train data size:{}".format(len(train_dataset)))

    test_processor_merr = DataProcessor(tokenizer, Config["multierror_data_path"],
                                        Config["max_seq_length"],
                                        Config["same_py_file"],
                                        Config["simi_py_file"],
                                        Config["stroke_file"],
                                        label_list=None,
                                        is_training=False
                                        )
    test_merr_dataset = test_processor_merr.file2features()
    test_merr_dataset = test_processor_merr.build_data_noise_feature(test_merr_dataset)
    logger.info("data add noise")
    logger.info("test merr size:{}".format(len(test_merr_dataset)))

    test_processor = DataProcessor(tokenizer, Config["test_data_path"],
                                   Config["max_seq_length"],
                                   Config["same_py_file"],
                                   Config["simi_py_file"],
                                   Config["stroke_file"],
                                   label_list=None,
                                   is_training=False
                                   )
    test_dataset = test_processor.file2features()
    test_dataset = test_processor.build_data_noise_feature(test_dataset)
    logger.info("test data size:{}".format(len(test_dataset)))

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=Config["batch_size"],
                                  shuffle=False, collate_fn=collate_fn)
    logger.info("train data pre epoch number:{}".format(len(train_dataloader)))
    test_merr_dataloader = DataLoader(test_merr_dataset,
                                      batch_size=Config["dev_batch_size"],
                                      shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=Config["dev_batch_size"],
                                 shuffle=False, collate_fn=collate_fn)
    # get id2label
    label_list = train_processor.label_list
    model = CRASpellModel(Config["pretrain_model_path"],
                          num_class=len(train_processor.get_label_list()),
                          max_sen_len=Config["max_seq_length"], device=device,
                          alpha=0.05, dropout_rate=0.1)
    model.to(device)
    # 原始的optimizer
    # optimizer = AdamW(model.parameters(), lr=Config["learning_rate"])
    # 加入衰减的
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=Config["learning_rate"])
    best_f1 = 0.0

    for epoch in range(Config["epochs"]):
        model.train()
        for step, batch_data in enumerate(train_dataloader):
            input_ids = batch_data["input_ids"].to(device)
            input_mask = batch_data["input_mask"].to(device)
            segment_ids = batch_data["segment_ids"].to(device)
            labels = batch_data["label_ids"].to(device)
            lmask = batch_data["lmask"].to(device)
            masked_sample = batch_data["masked_sample"].to(device)
            loss, preds = model(input_ids, input_mask,
                                segment_ids, lmask,
                                labels, masked_sample)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % Config["pre_epoch_step_print"] == 0:
                logger.info("epoch:{}/{}, step:{}/{}, loss:{}".format(epoch + 1,
                                                                      Config["epochs"],
                                                                      step + 1,
                                                                      len(train_dataloader),
                                                                      loss))
            if (step + 1) % Config["eval_interval"] == 0:
                logger.info('multi-error result')
                model_evaluate(model, test_merr_dataloader, device, label_list, logger)
                logger.info("test data result")
                f1 = model_evaluate(model, test_dataloader, device, label_list, logger)
                if f1 >= best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), os.path.join(Config["output_path"],
                                                                Config["model_name"]))
        logger.info('multi-error result')
        model_evaluate(model, test_merr_dataloader, device, label_list, logger)
        logger.info("test data result")
        f1 = model_evaluate(model, test_dataloader, device, label_list, logger)
        if f1 >= best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(Config["output_path"],
                                                        Config["model_name"]))


if __name__ == "__main__":
    config_file = "./config/config.ini"
    train(config_file)