#! -*- coding: utf-8 -*-
#测试多媒体文件的编解码
#补充，对于空序列的序列化处理
#循环内将缩小为

import arithmeticcoding_fast
import NTProGenerator
import numpy as np

id2nt = ['A','T','C','G']

if __name__ == "__main__":
    alphabet_size = 4
    index = 32
    f = open('./randomstream_4.10.rs', 'rb')
    #f = open('./mutimedias/123.png', 'rb')
    f2 = open('text3.rs', 'wb')
    #f = open('./randomstream_4.10.rs', 'rb')
    #f2 = open('randomstream_decoded.rs', 'wb')
    bitout = arithmeticcoding_fast.BitOutputStream(f2)
    enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
    bitin = arithmeticcoding_fast.BitInputStream(f)
    dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
    curSeq = ""
    i = 0
    #for i in range(1000000):
    while True:
        print(i)

        prob = NTProGenerator.seq_completion.next(curSeq)[0]
        print(prob)
        cumul = np.zeros(alphabet_size + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)
        c = dec.read(cumul, alphabet_size)
        nt = id2nt[c]
        print(nt)
        curSeq += nt
        #node = node[id2nt_dict[str(c)]]
        #print(id2nt_dict[str(c)])
        enc.write(cumul, c)

        if (i+1)%256 == 0:
            f_log = open('seq_log.txt','a')
            f_log.write(curSeq+'\n')
            f_log.close()
            curSeq = ""
        if dec.isEnd:
            index-=1
        if index==0:
            break
        i += 1
    enc.finish()
    bitout.close()
    bitin.close()
