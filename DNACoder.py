#! -*- coding: utf-8 -*-
#测试多媒体文件的编解码
#补充，对于空序列的序列化处理
#循环内将缩小为
import os
import arithmeticcoding_fast
import NTProGenerator
import numpy as np
from absl import app
from absl import flags
import logging
from tqdm import tqdm

flags.DEFINE_string('log', None, 'Name of the log file.')
flags.DEFINE_string('file_path', None, 'Paths to data files.')
flags.DEFINE_string('adaptive_coder_path', None, 'Paths to the project.')
flags.DEFINE_enum(
    'coding_type', 'en_decoding',
    ['en_decoding', 'encoding', 'decoding'],
    '')

FLAGS = flags.FLAGS

id2nt = ['A','T','C','G']

def main(_argv):
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    fh = logging.FileHandler(os.path.join(FLAGS.adaptive_coder_path,'log',FLAGS.log), encoding="UTF-8")
    #logger.addHandler(sh)
    logger.addHandler(fh)
    alphabet_size = 4
    index = 32
    f = open(FLAGS.file_path, 'rb')
    f2 = open(os.path.join(FLAGS.adaptive_coder_path, 'results/decodes',FLAGS.file_path.split('/')[-1]), 'wb')
    f_log = open(os.path.join(FLAGS.adaptive_coder_path, 'results/encodes', FLAGS.file_path.split('/')[-1] + '.dna'), 'w')
    bitout = arithmeticcoding_fast.BitOutputStream(f2)
    enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
    bitin = arithmeticcoding_fast.BitInputStream(f)
    dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
    curSeq = ""
    i = 0
    #for i in range(1000000):
    while True:
        prob = NTProGenerator.seq_completion.next(curSeq)[0]
        cumul = np.zeros(alphabet_size + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)
        c = dec.read(cumul, alphabet_size)
        nt = id2nt[c]
        logger.info("%d %s %s" % (i,str(prob),nt))
        curSeq += nt
        #node = node[id2nt_dict[str(c)]]
        #print(id2nt_dict[str(c)])
        enc.write(cumul, c)

        if (i+1)%256 == 0:
            f_log.write(curSeq+'\n')
            curSeq = ""
        if dec.isEnd:
            index-=1
        if index==0:
            break
        i += 1
    f_log.write(curSeq + '\n')
    f_log.close()
    enc.finish()
    bitout.close()
    bitin.close()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass