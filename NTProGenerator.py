#! -*- coding: utf-8 -*-
#载入模型,推断下一序列碱基的概率
from __future__ import print_function

import os
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

maxlen = 300
batch_size = 10
steps_per_epoch = 1000
epochs = 10000

# bert配置
config_path = '../BERTMODELS/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../BERTMODELS/chinese_L-12_H-768_A-12/bert_model.ckpt'

#序列分次预处理
def pre_tokenize(seq):
    tokens = [n for n in seq]
    return tokens

dict_path = 'vocab.txt'


# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)

tokenizer = Tokenizer(token_dict,pre_tokenize=pre_tokenize,do_lower_case=False)

with open('seq_good_256.txt') as f:
    seqs = [seq.strip() for seq in f.readlines()]

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
model.load_weights("best_model.weights")

class SeqCompletion(AutoRegressiveDecoder):
    """基于随机采样的故事续写
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids = inputs[0]
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.zeros_like(token_ids)
        return self.last_token(model).predict([token_ids, segment_ids])

    def generate(self, text, n=1, topp=0.95):
        token_ids, _ = tokenizer.encode(text)
        results = self.nextNTProbs([token_ids[:-1]], n, topp=topp)  # 基于随机采样
        #print(results)
        return [text + tokenizer.decode(ids) for ids in results]

    def next(self, text, topp=0.95):
        token_ids, _ = tokenizer.encode(text)
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
        #     sample_func = lambda p: np.random.choice(len(p), p=p)  # 按概率采样函数
        #     sample_ids = np.apply_along_axis(sample_func, 1, probas)  # 执行采样
        #     sample_ids = sample_ids.reshape((-1, 1))  # 对齐形状
        #     if topp is not None:
        #         sample_ids = np.take_along_axis(
        #             p_indices, sample_ids, axis=1
        #         )  # 对齐原id
        #     output_ids = np.concatenate([output_ids, sample_ids], 1)  # 更新输出
        #     is_end = output_ids[:, -1] == self.end_id  # 标记是否以end标记结束
        #     end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
        #     if output_ids.shape[1] >= self.minlen:  # 最短长度判断
        #         flag = is_end & (end_counts >= min_ends)  # 标记已完成序列
        #         if flag.any():  # 如果有已完成的
        #             for ids in output_ids[flag]:  # 存好已完成序列
        #                 results.append(ids)
        #             flag = (flag == False)  # 标记未完成序列
        #             inputs = [i[flag] for i in inputs]  # 只保留未完成部分输入
        #             output_ids = output_ids[flag]  # 只保留未完成部分候选集
        #             end_counts = end_counts[flag]  # 只保留未完成部分end计数
        #             if len(output_ids) == 0:
        #                 break
        # # 如果还有未完成序列，直接放入结果
        # for ids in output_ids:
        #     results.append(ids)
        # # 返回结果

        return probas

seq_completion = SeqCompletion(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
)


def just_show():
    s1 = 'AAAAA'
    #s2 = 'ACTCC'
    #s3 = 'GCCAAT'
    for s in [s1]:
        t = seq_completion.next(s)
        print(t)
        # print(u'输入: %s' % s)
        # print(u'结果: %s\n' % (''.join(t)))


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model.weights')
        # 演示效果
        just_show()


if __name__ == '__main__':
    just_show()
    input()