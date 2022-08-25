import random
import re

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
    "ClaI":["ATCGAT","CCCATCGATGGG"],
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
    "PvuI":["CGATCG","TCGCGATCGCGA"],
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
PLASMID = {}
nts = ['A','T','C','G']
maxlen = 256

dRule = {
    #1.特定序列(酶切位点)
    "blockSeq":[ENDONUCLEASE[e][0] for e in ENDONUCLEASE if e not in PLASMID],

    #2.正则表达式,改为内嵌
        #2.1 均聚物, 短串联重复[重复6次以上、单元长度在6以下]
        #2.2 反向互补, Oligo设计之间, 尚未考虑完全**********************,当碱基长度目标是4^60
    #"pattern":r'([ATCG]{1,6})\1{5}'
    #3.统计GC比例，改为内嵌
    #需要预定一个目标区域，是区域内的CG平衡
}

def isCGB(seq):
    C = list(seq).count("C")
    G = list(seq).count("G")
    if C + G > maxlen * 0.6 or len(seq) - C - G > maxlen * 0.6:
        print(C+G,(C+G)/256,maxlen*0.6,len(seq) - C - G)
        return False
    return True


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
    # 手动写CG平衡
    if not isCGB(seq):
        #print(seq, "unbalance")
        return False,"unbalance"
    return True, "well"

#print(isGoodSeq("GCAACGCACTGTAACAGGCGTTCGCCGAAGCTAGTGTACAGCTTGCCTTGGACTAAATTAAATCATTTGGAAGCCTCTACGACTATATGTTAGCATGTCTCGCTCGTTGGCCCGGGGTCAGGGATAAGGTTCTGGTCAATGCAGGTAACCCATACAAGCTTCGCAAGACGCTATCCACGGCAGGTTATACCTGGAATCTCCGTTCTATGAGCTGTCTCTGAGTCACTCCGTAGACGGCCGGGCAGGCCTCATTGCG"))
#input()

f = open('seq_log.txt','r')
i=0
for seq in f.readlines():
    if not isGoodSeq(seq.strip())[0]:
        print(isGoodSeq(seq.strip()))
        i+=1
print(i)
input()
while True:

    seq = ''
    for _ in range(maxlen):
        seq += random.choice(nts)
    igs,r = isGoodSeq(seq)
    if igs:
        with open("seq_good_256.txt","a+", encoding="utf-8") as f:
            f.write(seq+'\n')
    #else:
    #    with open("seq_bad_256","a+", encoding="utf-8") as f:
    #        f.write(r+seq+'\n')