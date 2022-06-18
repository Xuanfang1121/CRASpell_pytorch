# -*- coding: utf-8 -*-
# @Time    : 2022/6/18 11:39
# @Author  : zxf
import os
import json
import operator

import torch
from transformers import BertTokenizer

from common.common import logger
from model.CRASpellModel import CRASpellModel
from utils.util import create_data_infer_feature


def predict(text, model_path, pretrain_model_path, max_length):
    device = "cpu"
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    id2label = {}
    for key in tokenizer.vocab:
        id2label[tokenizer.vocab[key]] = key

    model = CRASpellModel(pretrain_model_path,
                          num_class=len(id2label),
                          max_sen_len=max_length, device=device,
                          alpha=0.05, dropout_rate=0.1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)),
                          strict=True)
    model.to(device)

    # get text feature
    feature = create_data_infer_feature(text, tokenizer, max_length)
    input_ids = torch.tensor([feature["input_ids"]], dtype=torch.long).to(device)
    input_mask = torch.tensor([feature["attention_mask"]], dtype=torch.long).to(device)
    segment_ids = torch.tensor([feature["token_type_ids"]], dtype=torch.long).to(device)
    # labels = torch.tensor([feature["labels"]], dtype=torch.long).to(device)
    labels = None
    lmask = torch.tensor([feature["lmask"]], dtype=torch.float32).to(device)
    masked_sample = torch.tensor([feature["masked_sample"]], dtype=torch.long).to(device)

    with torch.no_grad():
        _, logits = model(input_ids, input_mask,
                           segment_ids, lmask,
                           labels, masked_sample
                           )
    pred = torch.argmax(logits, dim=-1)
    gmask = lmask.data.cpu().numpy().tolist()[0]
    preds = pred.data.cpu().numpy().tolist()[0]
    print(preds)
    preds_result = []
    for j in range(max_length):
        if gmask[j] == 0:
            continue
        preds_result.append(id2label[preds[j]])

    result = {"ori_sent": text, "pred_sent": ''.join(preds_result)}
    return result


def predictv2(test_file, model_path, pretrain_model_path, max_length, output_file):
    device = torch.device("cpu")
    # read data
    ori_sents = []
    true_sents = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            ori_text, true_sent = line.strip().split('\t')
            ori_sents.append(''.join(ori_text.split()))
            true_sents.append(''.join(true_sent.split()))

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    id2label = {}
    for key in tokenizer.vocab:
        id2label[tokenizer.vocab[key]] = key

    model = CRASpellModel(pretrain_model_path,
                          num_class=len(id2label),
                          max_sen_len=max_length, device=device,
                          alpha=0.05, dropout_rate=0.1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)),
                          strict=True)
    model.to(device)

    result = []
    for i in range(len(ori_sents)):
        text = ori_sents[i]
        # get text feature
        feature = create_data_infer_feature(text, tokenizer, max_length)
        input_ids = torch.tensor([feature["input_ids"]], dtype=torch.long).to(device)
        input_mask = torch.tensor([feature["attention_mask"]], dtype=torch.long).to(device)
        segment_ids = torch.tensor([feature["token_type_ids"]], dtype=torch.long).to(device)
        # labels = torch.tensor([feature["labels"]], dtype=torch.long).to(device)
        labels = None
        lmask = torch.tensor([feature["lmask"]], dtype=torch.float32).to(device)
        masked_sample = torch.tensor([feature["masked_sample"]], dtype=torch.long).to(device)

        with torch.no_grad():
            _, logits = model(input_ids, input_mask,
                              segment_ids, lmask,
                              labels, masked_sample
                              )
        pred = torch.argmax(logits, dim=-1)
        gmask = lmask.data.cpu().numpy().tolist()[0]
        preds = pred.data.cpu().numpy().tolist()[0]
        # print(preds)
        preds_result = []
        for j in range(max_length):
            if gmask[j] == 0:
                continue
            preds_result.append(id2label[preds[j]])
        preds_result = ''.join(preds_result)

        if (i + 1) % 100 == 0:
            print("predict processing:{}/{}".format(i + 1, len(ori_sents)))

        result.append({"ori_sent": text,
                       "pred_sent": preds_result,
                       "true_sent": true_sents[i],
                       "label": operator.eq(true_sents[i], preds_result)})

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # # text = "同肘有无数等待的病人将会死亡。"
    # # text = "下个星期，我跟我朋唷打算去法国玩儿"
    # text_ori = "对 不 气 ， 最 近 我 很 忙 ， 所 以 我 不 会 去 你 的 。"
    # text = ''.join(text_ori.split())
    # model_path = "./output/model.pt"
    # pretrain_model_path = "D:/Spyder/pretrain_model/transformers_torch_tf/chinese-roberta-wwm-ext/"
    # max_length = 128
    # result = predict(text, model_path, pretrain_model_path, max_length)
    # print(result)

    test_file = "./data/sighan/sighan15_test.txt"
    model_path = "./output/model.pt"
    pretrain_model_path = "D:/Spyder/pretrain_model/transformers_torch_tf/chinese-roberta-wwm-ext/"
    max_length = 128
    output_file = "./output/pred_result.json"
    predictv2(test_file, model_path, pretrain_model_path, max_length, output_file)
