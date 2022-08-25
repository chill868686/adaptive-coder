#1.局部CG平衡优化
#2.引物
#3.ATCG
import random
import re
import Bio.Seq as Seq

ENDONUCLEASE = {
    #0:taSeq, 1:optSeq
    "AccI":["GTCGAC","CCGGTCGACCGG"],
    "AflIII":["ACATGT","CCCACATGTGGG"],
    "AscI":["GGCGCGCC","TTGGCGCGCCAA"],
    "AvaI":["CCCGGG","TCCCCCGGGGGA"],
    "BamHI":["GGATCC","CGCGGATCCGCG"],
    "BglII":["AGATCT","GGAAGATCTTCC"],
    "BssHII":["GCGCGC","TTGGCGCGCCAA"],
    #"BstEII":"GGT(A/T)ACC" 非唯一目标序列
    #"BstXI":"CCAATGCATTGG" 最优序列非回文
    #"ClaI":["ATCGAT","CCCATCGATGGG"],
    "EcoRI":["GAATTC","CCGGAATTCCGG"],
    "HaeIII":["GGCC","TTGCGGCCGCAA"],
    "HindIII":["AAGCTT","CCCAAGCTTGGG"],
    "KpnI":["GGTACC","GGGGGTACCCCC"],
    "MluI":["ACGCGT","CGACGCGTCG"],
    "NcoI":["CCATGG","CATGCCATGGCATG"],
    "NdeI":["CATATG","GGGAATTCCATATGGAATTCCC"],
    "NheI":["GCTAGC","CTAGCTAGCTAG"],
    #"NotI":"GCGGCCGC" 最优序列非回文
    #"NsiI": 最优序列非回文
    "PacI":["TTAATTAA","CCTTAATTAAGG"],
    #"PmeI":"GTTTAAAC" 最优序列非回文
    #"PstI" 最优序列非回文
    #"PvuI":["CGATCG","TCGCGATCGCGA"],
    "SacI":["GAGCTC","CGAGCTCG"],
    "SacII":["CCGCGG","TCCCCGCGGGGA"],
    #"SalI": 最优序列非回文
    "ScaI":["AGTACT","AAAAGTACTTTT"],
    "SmaI":["CCCGGG","TCCCCCGGGGGA"],
    "SpeI":["ACTAGT","CTAGACTAGTCTAG"],
    "SphI":["GCATGC","ACATGCATGCATGT"],
    "StuI":["AGGCCT","AAAAGGCCTTTT"],
    "XbaI":["TCTAGA","CTAGTCTAGACTAG"],
    "XboI":["CTCGAG","CCGCTCGAGCGG"],
    "XmaI":["CCCGGG","TCCCCCCGGGGGGA"]
}

PRIMER = [
    ["AAGGCAAGTTGTTACCAGCA","TGCGACCGTAATCAAACCAA"],
    ["TTCGTTCGTCGTTGATTGGT","AAACGGAGCCATGAGTTTGT"],
    ["GAAGAGTTTAGCCACCTGGT","CAAGTAACCGGCAACAACTG"],
    ["CTGTCCATAGCCTTGTTCGT","GCGGAAACGTAGTGAAGGTA"]
]

lPrimer = [pair[0] for pair in PRIMER] + [str(Seq.Seq(pair[1]).reverse_complement()) for pair in PRIMER]

SCAFFOLD = ["AATTCCGG"]

PLASMID = {}
nts = ['A','T','C','G']
maxlen = 256

dRule = {
    #1.特定序列(酶切位点)
    "blockSeq":[ENDONUCLEASE[e][0] for e in ENDONUCLEASE if e not in PLASMID] + SCAFFOLD,

    #2.正则表达式,改为内嵌
        #2.1 均聚物, 短串联重复[重复6次以上、单元长度在6以下]
        #2.2 反向互补, Oligo设计之间, 尚未考虑完全**********************,当碱基长度目标是4^60
    #"pattern":r'([ATCG]{1,6})\1{5}'
    #3.统计GC比例，改为内嵌
    #需要预定一个目标区域，是区域内的CG平衡
}

def hamming(s1, s2):
    return sum([a != b for a, b in zip(s1, s2)])

#以20个碱基为CG平衡单元


def isGoodSeq(seq):
    # seq input为字符串
    # 列表
    for badSeq in dRule["blockSeq"]:
        if badSeq in seq:
            #print(seq, "endoSeq", badSeq)
            return False,"endoSeq"
    pattern = r'([ATCG]{1,6})\1{5}'
    if re.findall(pattern, seq):
        #print(seq, "repeatSeq")
        return False,"repeatSeq"

    #20个碱基为单元
    for i in range(0,len(seq)-19,20):
        check_unit = seq[i:i+20]
        C = list(check_unit).count("C")
        G = list(check_unit).count("G")
        if C + G > 12 or 20 - C - G > 12:
            #print(C+G,(C+G)/256,maxlen*0.6,len(seq) - C - G)
            return False,"unbalance"
    for i in range(len(seq)-19):
        check_unit = seq[i:i+20]
        for p in lPrimer:
            if hamming(check_unit,p)<5:
                return False, "primerClosed"
    return True, "well"
#print(isGoodSeq("GAGCGTCGGAGAACACTCATAGTGAAGTCCACCCATATATCATGCGTCGGTTTGACCGAAACTGGACTCTGAATGCCCCTGCAACTTTCACACGTAAGTTGGACCAAAAGGTACGAATCCCAACGCAGCTCTGTCGAAATGAGGGATGGTTGGGATTCGCATACTTGTGCTGCGATTCTGACGTAGCGCAACCTTAACTTCTTAGGGCTGTCCTGAATCTGCACATCTCGTAATGGCTATCCGCCTACTTAAATCA"))
#input()
# for s in lPrimer:
#
# input()

# f = open('seq_log.txt','r')
# i=0
# for seq in f.readlines():
#     if not isGoodSeq(seq.strip())[0]:
#         print(isGoodSeq(seq.strip()))
#         i+=1
# print(i)
# input()
while True:

    seq = ''
    for _ in range(maxlen):
        seq += random.choice(nts)
    igs,r = isGoodSeq(seq)
    print(igs,r)
    if igs:
        with open("seq_good_256_0228.txt","a+", encoding="utf-8") as f:
            f.write(seq+'\n')
    #else:
    #    with open("seq_bad_256","a+", encoding="utf-8") as f:
    #        f.write(r+seq+'\n')