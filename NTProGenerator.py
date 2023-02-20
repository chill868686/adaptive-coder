#! -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
os.environ['TF_KERAS'] = '1'

import glob, re
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model

#最长长度
maxlen = 300

#序列分次预处理
def pre_tokenize(seq):
    tokens = [n for n in seq]
    return tokens

class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉padding部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        if mask[1] is None:
            y_mask = 1.0
        else:
            y_mask = K.cast(mask[1], K.floatx())[:, 1:]
        y_true = y_true[:, 1:]  # 目标token_ids
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

class SeqCompletion(AutoRegressiveDecoder):
    """序列补充"""

    def loadM(self,model,tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids = inputs[0]
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.zeros_like(token_ids)
        return self.last_token(self.model).predict([token_ids, segment_ids])

    def generate(self, text, n=1, topp=0.95):
        token_ids, _ = self.tokenizer.encode(text)
        results = self.nextNTProbs([token_ids[:-1]], n, topp=topp)  # 基于随机采样
        #print(results)
        return [text + self.tokenizer.decode(ids) for ids in results]

    def next(self, text, topp=0.95):
        token_ids, _ = self.tokenizer.encode(text)
        probas = self.nextNTProbs([token_ids[:-1]], 1, topp=topp)  # 基于随机采样
        #print(results)
        return probas

    def nextNTProbs(
        self,
        inputs,
        n,
        topp=None,
        states=None,
        temperature=1,
        min_ends=1
    ):
        """随机采样n个结果
        说明：topp表示每一步只从概率最高的且概率之和刚好达到topp的若干个token中采样。
        返回：n个解码序列组成的list。
        """
        inputs = [np.array([i]) for i in inputs]
        output_ids = self.first_output_ids
        results = []

        probas, states = self.predict(inputs, output_ids, states, temperature, 'probas')  # 计算当前概率
        probas = probas[:,4:]
        probas /= probas.sum(axis=1, keepdims=True)  # 确保归一化
        # if probas[3]>.8:
        #     isEnd = True
        p_indices = probas.argsort(axis=1)[:, ::-1]  # 从高到低排序
        r_indices = p_indices.argsort(axis=1)
        probas = np.take_along_axis(probas, p_indices, axis=1)  # 排序概率
        cumsum_probas = np.cumsum(probas, axis=1)  # 累积概率
        flag = np.roll(cumsum_probas >= topp, 1, axis=1)  # 标记超过topp的部分
        flag[:, 0] = False  # 结合上面的np.roll，实现平移一位的效果
        probas[flag] = 0  # 后面的全部置零

        probas /= probas.sum(axis=1, keepdims=True)  # 重新归一化
        probas = np.take_along_axis(probas, r_indices, axis=1)  # 排序概率

        return probas

def getGen(model_path):

    # bert配置 来源chinese_L-12_H-768_A-12
    config_path = '/mnt/adaptive_coder_path/basemodel/bert_config.json'
    checkpoint_path = '/mnt/adaptive_coder_path/basemodel/bert_model.ckpt'

    dict_path = os.path.join(sys.path[0],'vocab.txt')

    # 加载并精简词表，建立分词器
    token_dict, keep_tokens = load_vocab(
        dict_path=dict_path,
        simplified=True,
        startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
    )

    tokenizer = Tokenizer(token_dict,pre_tokenize=pre_tokenize,do_lower_case=False)

    model = build_transformer_model(
        config_path,
        checkpoint_path,
        application='lm',
        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    )

    output = CrossEntropy(1)([model.inputs[0], model.outputs[0]])
    model = Model(model.inputs, output)
    model.compile(optimizer=Adam(1e-5))
    model.summary()
    model.load_weights(model_path).expect_partial()

    seq_completion = SeqCompletion(
        start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
    )
    seq_completion.loadM(model,tokenizer)
    return seq_completion