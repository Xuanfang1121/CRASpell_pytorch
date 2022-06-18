# -*- coding: utf-8 -*-
# @Time    : 2022/6/11 20:59
# @Author  : zxf
import os
import json
import collections
from copy import deepcopy
from collections import namedtuple

import torch
from transformers import BertTokenizerFast

from utils.mask import Mask
from utils.mask import PinyinConfusionSet
from utils.mask import StrokeConfusionSet


InputExample = namedtuple('InputExample', ['tokens', 'labels', 'domain'])
InputFeatures = namedtuple('InputFeature', ['input_ids', 'input_mask', 'segment_ids',
                                            'lmask', 'label_ids'])


class DataProcessor(object):
    """
    模型数据处理类
    数据格式为 sent1\tsent2
    """
    def __init__(self, tokenizer, input_path, max_sen_len,
                 same_py_file, simi_py_file, stroke_file,
                 label_list=None,
                 is_training=True):
        self.input_path = input_path
        self.max_sen_len = max_sen_len
        self.is_training = is_training
        self.tokenizer = tokenizer
        self.label_list = label_list
        if label_list is not None:
            self.label_map = {}
            for (i, label) in enumerate(self.label_list):
                self.label_map[label] = i
        else:
            # label2id
            self.label_map = self.tokenizer.vocab
            # id2label
            self.label_list = {}
            for key in self.tokenizer.vocab:
                self.label_list[self.tokenizer.vocab[key]] = key

        # same_py_file = '../datas/confusions/same_pinyin.txt'
        # simi_py_file = '../datas/confusions/simi_pinyin.txt'
        # stroke_file = '../datas/confusions/same_stroke.txt'
        tokenizer = self.tokenizer
        pinyin = PinyinConfusionSet(tokenizer, same_py_file)
        jinyin = PinyinConfusionSet(tokenizer, simi_py_file)
        stroke = StrokeConfusionSet(tokenizer, stroke_file)
        self.masker = Mask(same_py_confusion=pinyin, simi_py_confusion=jinyin,
                           sk_confusion=stroke)
        # get data feature
        # self.file2features()

    def sample(self, text_unicode1, text_unicode2, domain=None):
        segs1 = text_unicode1.strip().split(' ')
        segs2 = text_unicode2.strip().split(' ')
        tokens, labels = [], []
        if len(segs1) != len(segs2):
            return None
        for x, y in zip(segs1, segs2):
            tokens.append(x)
            labels.append(y)
        if len(tokens) < 2:
            return None
        return InputExample(tokens=tokens, labels=labels, domain=domain)

    def load_data(self):
        '''sent1 \t sent2'''
        # train_data = open(self.input_path, encoding="utf-8")
        with open(self.input_path, "r", encoding="utf-8") as f:
            train_data = f.readlines()
        instances = []
        n_line = 0
        for ins in train_data:
            n_line += 1
            # if (DEBUG is True) and (n_line > 1000):
            #     break
            tmps = ins.strip().split('\t')
            if len(tmps) < 2:
                continue
            ins = self.sample(tmps[0], tmps[1])
            if ins is not None:
                yield ins

    def convert_single_example(self, example):
        label_map = self.label_map
        tokens = example.tokens
        labels = example.labels
        domain = example.domain
        seg_value = 0
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > self.max_sen_len - 2:
            tokens = tokens[0:(self.max_sen_len - 2)]
            labels = labels[0:(self.max_sen_len - 2)]

        _tokens = []
        _labels = []
        _lmask = []
        segment_ids = []
        _tokens.append("[CLS]")
        _lmask.append(0)
        _labels.append("[CLS]")
        segment_ids.append(seg_value)
        for token, label in zip(tokens, labels):
            _tokens.append(token)
            _labels.append(label)
            _lmask.append(1)
            segment_ids.append(seg_value)
        _tokens.append("[SEP]")
        segment_ids.append(seg_value)
        _labels.append("[SEP]")
        _lmask.append(0)

        input_ids = self.tokenizer.convert_tokens_to_ids(_tokens)
        label_ids = self.tokenizer.convert_tokens_to_ids(_labels)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_sen_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            _lmask.append(0)

        assert len(input_ids) == self.max_sen_len, f'input_ids size:{len(input_ids)}'
        assert len(input_mask) == self.max_sen_len, f'input_mask size:{len(input_mask)}'
        assert len(segment_ids) == self.max_sen_len, f'segment_ids size:{len(segment_ids)}'

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            lmask=_lmask,
            label_ids=label_ids
        )
        return feature

    def get_label_list(self):
        return self.label_list

    def file2features(self):
        data = self.load_data()
        data_features = []
        for (ex_index, example) in enumerate(data):
            feature = self.convert_single_example(example)
            features = {}
            features["input_ids"] = feature.input_ids
            features["input_mask"] = feature.input_mask
            features["segment_ids"] = feature.segment_ids
            features["lmask"] = feature.lmask  # 这个lmask是标记句子的，句子tokens标记为1
            features["label_ids"] = feature.label_ids
            data_features.append(features)
        return data_features

    def build_data_noise_feature(self, features):
        noise_feature = []
        for index, feature in enumerate(features):
            input_ids = feature["input_ids"]
            label_ids = feature["label_ids"]
            # add noise char
            if self.is_training is True:
                masked_sample = self.masker.mask_process(input_ids, label_ids).tolist()
            else:
                masked_sample = input_ids
            feature["masked_sample"] = masked_sample
            noise_feature.append(feature)

        return noise_feature

    def get_feature(self, u_input, u_output=None):
        if u_output is None:
            u_output = u_input
        instance = self.sample(u_input, u_output)
        feature = self.convert_single_example(instance)
        input_ids = feature.input_ids
        input_mask = feature.input_mask
        segment_ids = feature.segment_ids
        label_ids = feature.label_ids
        label_mask = feature.lmask
        return input_ids, input_mask, segment_ids, label_ids, label_mask


def collate_fn(batch_data):
    input_ids = [item["input_ids"] for item in batch_data]
    input_mask = [item["input_mask"] for item in batch_data]
    segment_ids = [item["segment_ids"] for item in batch_data]
    label_ids = [item["label_ids"] for item in batch_data]
    lmask = [item["lmask"] for item in batch_data]
    masked_sample = [item["masked_sample"] for item in batch_data]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.long)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    label_ids = torch.tensor(label_ids, dtype=torch.long)
    lmask = torch.tensor(lmask, dtype=torch.float32)
    masked_sample = torch.tensor(masked_sample, dtype=torch.long)
    noise_mask = torch.eq(input_ids, masked_sample).float()
    return{"input_ids": input_ids,
           "input_mask": input_mask,
           "segment_ids": segment_ids,
           "label_ids": label_ids,
           "lmask": lmask,
           "masked_sample": masked_sample}
           # "noise_mask": noise_mask}


def model_evaluate(model, dev_dataloader, device, label_list, logger):
    """
    模型验证
    """
    model.eval()
    all_inputs, all_golds, all_preds = [], [], []
    all_inputs_sent, all_golds_sent, all_preds_sent = [], [], []
    with torch.no_grad():
        for step, batch_data in enumerate(dev_dataloader):
            input_ids = batch_data["input_ids"].to(device)
            input_mask = batch_data["input_mask"].to(device)
            segment_ids = batch_data["segment_ids"].to(device)
            labels = batch_data["label_ids"].to(device)
            lmask = batch_data["lmask"].to(device)
            masked_sample = batch_data["masked_sample"].to(device)
            # dev_loss, preds, _ = model(input_ids, input_mask,
            #                            segment_ids, lmask,
            #                            labels, masked_sample,
            #                            is_training=False)
            _, logits = model(input_ids, input_mask, segment_ids, lmask,
                              labels, masked_sample)
            preds = torch.argmax(logits, dim=-1)
            batch_size = input_ids.size()[0]
            max_sen_len = input_ids.size()[1]
            gmask = lmask.data.cpu().numpy()
            preds = preds.data.cpu().numpy()
            golds = labels.data.cpu().numpy()
            input_ids = input_ids.data.cpu().numpy()
            for k in range(batch_size):
                tmp1, tmp2, tmp3, tmps4, tmps5, tmps6, tmps7 = [], [], [], [], [], [], []
                for j in range(max_sen_len):
                    if gmask[k][j] == 0:
                        continue
                    all_golds.append(golds[k][j])
                    all_preds.append(preds[k][j])
                    all_inputs.append(input_ids[k][j])
                    tmp1.append(label_list[golds[k][j]])
                    tmp2.append(label_list[preds[k][j]])
                    tmp3.append(label_list[input_ids[k][j]])

                all_golds_sent.append(tmp1)
                all_preds_sent.append(tmp2)
                all_inputs_sent.append(tmp3)

        all_golds = [label_list[k] for k in all_golds]
        all_preds = [label_list[k] for k in all_preds]
        all_inputs = [label_list[k] for k in all_inputs]

        p, r, f = score_f((all_inputs, all_golds, all_preds), logger=logger, only_check=False)
        logger.info("precision:{}, recall:{}, f1_score:{}".format(p, r, f))
        return f


def score_f(ans, logger, print_flg=False, only_check=False):
    total_gold_err, total_pred_err, right_pred_err = 0, 0, 0
    check_right_pred_err = 0
    inputs, golds, preds = ans
    assert len(inputs) == len(golds)
    assert len(golds) == len(preds)
    for ori, god, prd in zip(inputs, golds, preds):
        ori_txt = str(ori)
        god_txt = str(god)   # ''.join(list(map(str, god)))
        prd_txt = str(prd)   # ''.join(list(map(str, prd)))
        if print_flg is True:
            print(ori_txt, '\t', god_txt, '\t', prd_txt)
        if 'UNK' in ori_txt:
            continue
        if ori_txt == god_txt and ori_txt == prd_txt:
            continue
        # if prd_txt != ori_txt:
        #     fout.writelines('%s\t%s\t%s\n' % (ori_txt, god_txt, prd_txt))
        if ori != god:
            total_gold_err += 1
        if prd != ori:
            total_pred_err += 1
        if (ori != god) and (prd != ori):
            check_right_pred_err += 1
            if god == prd:
                right_pred_err += 1
    # fout.close()

    # check p, r, f
    p = 1. * check_right_pred_err / (total_pred_err + 0.001)
    r = 1. * check_right_pred_err / (total_gold_err + 0.001)
    f = 2 * p * r / (p + r + 1e-13)
    logger.info('token num: gold_n:%d, pred_n:%d, right_n:%d' % (total_gold_err, total_pred_err, check_right_pred_err))
    logger.info('token check: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
    if only_check is True:
        return p, r, f

    # correction p, r, f
    pc1 = 1. * right_pred_err / (total_pred_err + 0.001)
    pc2 = 1. * right_pred_err / (check_right_pred_err + 0.001)
    rc = 1. * right_pred_err / (total_gold_err + 0.001)
    fc1 = 2 * pc1 * rc / (pc1 + rc + 1e-13)
    fc2 = 2 * pc2 * rc / (pc2 + rc + 1e-13)
    logger.info('token correction-1: p=%.3f, r=%.3f, f=%.3f' % (pc2, rc, fc2))
    logger.info('token correction-2: p=%.3f, r=%.3f, f=%.3f' % (pc1, rc, fc1))
    return p, r, f


def create_data_infer_feature(text, tokenizer, max_len):
    feature = tokenizer.encode_plus(text, add_special_tokens=True,
                                    # padding="max_length",
                                    max_length=max_len,
                                    truncation=True)
    lmask = deepcopy(feature["attention_mask"])
    lmask[0] = 0
    lmask[-1] = 0
    padding_len = max_len - len(feature['input_ids'])
    feature['input_ids'] = feature['input_ids'] + [tokenizer.pad_token_id] * padding_len
    feature['attention_mask'] = feature['attention_mask'] + [0] * padding_len
    feature['token_type_ids'] = feature['token_type_ids'] + [0] * padding_len
    feature['lmask'] = lmask + [0] * padding_len
    feature['masked_sample'] = feature['input_ids']
    feature['labels'] = None # feature['input_ids']
    return feature