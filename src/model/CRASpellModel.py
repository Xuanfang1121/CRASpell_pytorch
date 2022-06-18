# -*- coding: utf-8 -*-
# @Time    : 2022/6/16 23:05
# @Author  : zxf
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import BertConfig


def kl_for_log_probs(log_p, log_q):
    p = torch.exp(log_p)
    neg_ent = torch.sum(p * log_p, dim=-1)
    neg_cross_ent = torch.sum(p * log_q, dim=-1)
    kl = neg_ent - neg_cross_ent
    return kl


def gelu(x):
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*np.power(x, 3))))


class CRASpellModel(nn.Module):
    def __init__(self, pretrain_model_path, num_class, max_sen_len, device,
                 alpha=0.05, dropout_rate=0.1):
        super(CRASpellModel, self).__init__()
        self.num_class = num_class
        self.max_sen_len = max_sen_len
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.bert_config = BertConfig.from_pretrained(pretrain_model_path)
        self.bert = BertModel.from_pretrained(pretrain_model_path)
        self.hidden_size = self.bert.config.hidden_size
        self.linear = nn.Linear(self.hidden_size, self.num_class)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        # copy net
        self.copy_linear = nn.Linear(self.hidden_size, 384)
        self.gelu = nn.GELU()
        self.layernorm = nn.LayerNorm(384)
        self.copy_logits = nn.Linear(384, 1)
        self.helper_tensor = torch.ones(1, self.num_class, dtype=torch.float32).to(device)

    def _KL_loss(self, log_probs, log_probs2, mask):
        log_probs1 = torch.reshape(log_probs, [-1, self.max_sen_len, self.num_class])
        log_probs2 = torch.reshape(log_probs2, [-1, self.max_sen_len, self.num_class])
        kl_loss1 = kl_for_log_probs(log_probs1, log_probs2)
        kl_loss2 = kl_for_log_probs(log_probs2, log_probs1)
        kl_loss = (kl_loss1 + kl_loss2) / 2.0
        kl_loss = torch.squeeze(torch.reshape(kl_loss, [-1, 1]))
        kl_loss = torch.sum(kl_loss * mask) / torch.sum(mask)
        return kl_loss

    def forward(self, input_ids, input_mask, segment_ids, lmask,
                labels, masked_sample):

        mask = torch.squeeze(torch.reshape(lmask, [-1, 1]))
        # correct model
        correct_model = self.bert(input_ids=input_ids,
                                  attention_mask=input_mask,
                                  token_type_ids=segment_ids)
        output_seq = correct_model['last_hidden_state']

        # noise model 这块加一个判断 模型推理的时候不会用到 noise 模型
        noise_model = self.bert(input_ids=masked_sample,
                                attention_mask=input_mask,
                                token_type_ids=segment_ids)
        output_seq2 = noise_model["last_hidden_state"]

        # 改变 output的shape
        output = torch.reshape(output_seq, [-1, self.hidden_size])
        output2 = torch.reshape(output_seq2, [-1, self.hidden_size])

        output = self.dropout(output)
        output2 = self.dropout(output2)
        # 获取logits 和 概率
        logits = self.linear(output)
        probabilities = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        logits2 = self.linear(output2)
        # probabilities2 = F.softmax(logits2, dim=-1)
        log_probs2 = F.log_softmax(logits2, dim=-1)
        # one_hot_labels = F.one_hot(labels, num_classes=self.num_class).float()
        # copy 模块
        input_ids = torch.squeeze(torch.reshape(input_ids, [-1, 1]))
        copy_feed = self.copy_linear(output)
        copy_feed = self.gelu(copy_feed)
        # if is_training:
        copy_feed = self.dropout(copy_feed)
        copy_feed = self.layernorm(copy_feed)   # (-1, 384)
        copy_logits = self.copy_logits(copy_feed)
        copy_prob = nn.Sigmoid()(copy_logits)   # (-1, 1)

        # helper_tensor = torch.ones(1, self.num_class, dtype=torch.float32)
        copy_prob = torch.matmul(copy_prob, self.helper_tensor) # (-1, num_class)
        one_hot_labels_of_input = F.one_hot(input_ids, num_classes=self.num_class
                                            ).float()

        cp_probabilities = copy_prob * one_hot_labels_of_input + (1.0 - copy_prob) * probabilities
        cp_probabilities = torch.clip(cp_probabilities, 1e-10, 1.0 - 1e-7)
        cp_log_probs = torch.log(cp_probabilities)

        if labels is not None:
            labels = torch.squeeze(torch.reshape(labels, [-1, 1]))
            one_hot_labels = F.one_hot(labels, num_classes=self.num_class).float()
            cp_per_example_loss = -torch.sum(one_hot_labels * cp_log_probs, dim=-1) * mask
            cp_per_example_loss = torch.sum(cp_per_example_loss) / torch.sum(mask)
            ns_mask = mask
            kl_loss = self._KL_loss(log_probs, log_probs2, ns_mask)

            # probabilities = cp_probabilities
            loss = (1.0 - self.alpha) * cp_per_example_loss + self.alpha * kl_loss
        else:
            loss = None

        # pred_result = torch.reshape(cp_probabilities, shape=(-1, self.max_sen_len, self.num_class))
        # pred_result = torch.argmax(pred_result, dim=2)
        # golden = torch.reshape(labels, shape=(-1, self.max_sen_len))
        cp_probabilities = torch.reshape(cp_probabilities, shape=(-1, self.max_sen_len, self.num_class))
        return loss, cp_probabilities # , kl_loss