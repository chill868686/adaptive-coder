#! -*- coding: utf-8 -*-
#符合约束的碱基序列生成
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

from absl import app
from absl import flags
import logging

flags.DEFINE_string('log', None, 'Name of the log file.')
flags.DEFINE_string('file_path', None, 'Paths to seq files.')
flags.DEFINE_string('adaptive_coder_path', None, 'Paths to the project.')
flags.DEFINE_enum(
    'coding_type', 'training',
    ['training',],
    '')

FLAGS = flags.FLAGS

def pre_tokenize(seq):
    tokens = [n for n in seq]
    return tokens

dict_path = os.path.join(sys.path[0],'vocab.txt')

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)

tokenizer = Tokenizer(token_dict,pre_tokenize=pre_tokenize,do_lower_case=False)
maxlen = 300
batch_size = 10
steps_per_epoch = 1000
epochs = 10000

config_path = '/mnt/adaptive_coder_path/basemodel/bert_config.json'
checkpoint_path = '/mnt/adaptive_coder_path/basemodel/bert_model.ckpt'

model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='lm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

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

output = CrossEntropy(1)([model.inputs[0], model.outputs[0]])

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, seq in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                seq, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.lowest = 1e10
        self.save_path = os.path.join(FLAGS.adaptive_coder_path, 'models', FLAGS.file_path.split('.')[-2].split('/')[-1]+'.weights')
    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights(self.save_path)
        # 演示效果
        just_show()

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
        results = self.random_sample([token_ids[:-1]], n, topp=topp)  # 基于随机采样
        return [text + tokenizer.decode(ids) for ids in results]


seq_completion = SeqCompletion(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
)

def just_show():
    s1 = 'ATCCGG'
    s2 = 'ACTCC'
    s3 = 'GCCAAT'
    for s in [s1, s2, s3]:
        t = SeqCompletion.generate(s)
        print(u'输入: %s' % s)
        print(u'结果: %s\n' % (''.join(t)))

def main(_argv):
    with open(FLAGS.file_path) as f:
        seqs = [seq.strip() for seq in f.readlines()]

    evaluator = Evaluator()
    train_generator = data_generator(seqs, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass