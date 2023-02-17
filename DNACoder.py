#! -*- coding: utf-8 -*-
import os
import arithmeticcoding_fast
import NTProGenerator   #载入神经网络
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
nt2id = {'A':0,'T':1,'C':2,'G':3}
def main(_argv):
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    fh = logging.FileHandler(os.path.join(FLAGS.adaptive_coder_path,'log',FLAGS.log), encoding="UTF-8")
    #logger.addHandler(sh)
    logger.addHandler(fh)
    alphabet_size = 4
    index = 32
    generator = NTProGenerator.getGen()
    if FLAGS.coding_type == 'en_decoding':
        fin = open(FLAGS.file_path, 'rb')   #media file
        fout = open(os.path.join(FLAGS.adaptive_coder_path, 'results/decodes', FLAGS.file_path.split('/')[-1]), 'wb')
        fseq = open(os.path.join(FLAGS.adaptive_coder_path, 'results/encodes', FLAGS.file_path.split('/')[-1] + '.dna'), 'w')
        bitout = arithmeticcoding_fast.BitOutputStream(fout)
        enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
        bitin = arithmeticcoding_fast.BitInputStream(fin)
        dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
        curSeq = ""
        i = 0
        while True:
            prob = generator.next(curSeq)[0]
            cumul = np.zeros(alphabet_size + 1, dtype=np.uint64)
            cumul[1:] = np.cumsum(prob * 10000000 + 1)
            c = dec.read(cumul, alphabet_size)
            nt = id2nt[c]
            logger.info("%d %s select:%s" % (i, str(prob), nt))
            curSeq += nt
            enc.write(cumul, c)

            if (i + 1) % 256 == 0:
                fseq.write(curSeq + '\n')
                curSeq = ""
            if dec.isEnd:
                index -= 1
            if index == 0:
                break
            i += 1
        fseq.write(curSeq + '\n')
        fseq.close()
        enc.finish()
        bitout.close()
        bitin.close()
    if FLAGS.coding_type == 'encoding':
        fin = open(FLAGS.file_path, 'rb')   #media file
        #fout = open(os.path.join(FLAGS.adaptive_coder_path, 'results/decodes', FLAGS.file_path.split('/')[-1]), 'wb')
        fseq = open(os.path.join(FLAGS.adaptive_coder_path, 'results/encodes', FLAGS.file_path.split('/')[-1] + '.dna'),'w')
        #bitout = arithmeticcoding_fast.BitOutputStream(fout)
        #enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
        bitin = arithmeticcoding_fast.BitInputStream(fin)
        dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
        curSeq = ""
        i = 0
        while True:
            prob = generator.next(curSeq)[0]
            cumul = np.zeros(alphabet_size + 1, dtype=np.uint64)
            cumul[1:] = np.cumsum(prob * 10000000 + 1)
            c = dec.read(cumul, alphabet_size)
            nt = id2nt[c]
            logger.info("%d %s select:%s" % (i, str(prob), nt))
            curSeq += nt
            #enc.write(cumul, c)

            if (i + 1) % 256 == 0:
                fseq.write(curSeq + '\n')
                curSeq = ""
            if dec.isEnd:
                index -= 1
            if index == 0:
                break
            i += 1
        fseq.write(curSeq + '\n')
        fseq.close()
        #enc.finish()
        #bitout.close()
        bitin.close()
    if FLAGS.coding_type == 'decoding':
        fin = open(FLAGS.file_path, 'r')   #DNA file
        fout = open(os.path.join(FLAGS.adaptive_coder_path, 'results/decodes', FLAGS.file_path.split('/')[-1][:-4]), 'wb')
        #fseq = open(os.path.join(FLAGS.adaptive_coder_path, 'results/encodes', FLAGS.file_path.split('/')[-1] + '.dna'),'w')
        bitout = arithmeticcoding_fast.BitOutputStream(fout)
        enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
        #bitin = arithmeticcoding_fast.BitInputStream(fin)
        #dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
        curSeq = ""
        i = 0
        end = False
        while True:
            prob = generator.next(curSeq)[0]
            cumul = np.zeros(alphabet_size + 1, dtype=np.uint64)
            cumul[1:] = np.cumsum(prob * 10000000 + 1)
            #c = dec.read(cumul, alphabet_size)
            while True:
                nt = fin.read(1)
                if not nt:
                    end = True
                    break
                if not nt.strip():
                    curSeq = ""
                    break
                else:
                    i += 1
                    curSeq += nt
                    c = nt2id[nt]
                    enc.write(cumul, c)
                    logger.info("%d %s %s" % (i, str(prob), nt))
                    break
            if end:
                break

        #fseq.write(curSeq + '\n')
        #fseq.close()
        enc.finish()
        bitout.close()
        #bitin.close()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass