#策略模式生成
#读写配置文件 pip install configparser
import collections
import random
import re
import sys
import Bio.Seq as Seq
from configparser import ConfigParser
import collections

class SGFilter():
    def isGoodSeq(self):
        pass

class EndoFilter(SGFilter):
    def __init__(self):
        self.ENDONUCLEASE = {
        # 0:taSeq, 1:optSeq
        "AccI": ["GTCGAC", "CCGGTCGACCGG"],
        "AflIII": ["ACATGT", "CCCACATGTGGG"],
        "AscI": ["GGCGCGCC", "TTGGCGCGCCAA"],
        "AvaI": ["CCCGGG", "TCCCCCGGGGGA"],
        "BamHI": ["GGATCC", "CGCGGATCCGCG"],
        "BglII": ["AGATCT", "GGAAGATCTTCC"],
        "BssHII": ["GCGCGC", "TTGGCGCGCCAA"],
        # "BstEII":"GGT(A/T)ACC" 非唯一目标序列
        # "BstXI":"CCAATGCATTGG" 最优序列非回文
        # "ClaI":["ATCGAT","CCCATCGATGGG"],
        "EcoRI": ["GAATTC", "CCGGAATTCCGG"],
        "HaeIII": ["GGCC", "TTGCGGCCGCAA"],
        "HindIII": ["AAGCTT", "CCCAAGCTTGGG"],
        "KpnI": ["GGTACC", "GGGGGTACCCCC"],
        "MluI": ["ACGCGT", "CGACGCGTCG"],
        "NcoI": ["CCATGG", "CATGCCATGGCATG"],
        "NdeI": ["CATATG", "GGGAATTCCATATGGAATTCCC"],
        "NheI": ["GCTAGC", "CTAGCTAGCTAG"],
        # "NotI":"GCGGCCGC" 最优序列非回文
        # "NsiI": 最优序列非回文
        "PacI": ["TTAATTAA", "CCTTAATTAAGG"],
        # "PmeI":"GTTTAAAC" 最优序列非回文
        # "PstI" 最优序列非回文
        # "PvuI":["CGATCG","TCGCGATCGCGA"],
        "SacI": ["GAGCTC", "CGAGCTCG"],
        "SacII": ["CCGCGG", "TCCCCGCGGGGA"],
        # "SalI": 最优序列非回文
        "ScaI": ["AGTACT", "AAAAGTACTTTT"],
        "SmaI": ["CCCGGG", "TCCCCCGGGGGA"],
        "SpeI": ["ACTAGT", "CTAGACTAGTCTAG"],
        "SphI": ["GCATGC", "ACATGCATGCATGT"],
        "StuI": ["AGGCCT", "AAAAGGCCTTTT"],
        "XbaI": ["TCTAGA", "CTAGTCTAGACTAG"],
        "XboI": ["CTCGAG", "CCGCTCGAGCGG"],
        "XmaI": ["CCCGGG", "TCCCCCCGGGGGGA"]
    }
        self.blockSeq = [self.ENDONUCLEASE[e][0] for e in self.ENDONUCLEASE]
    def isGoodSeq(self,seq):
        for badSeq in self.blockSeq:
            if badSeq in seq:
                return False, self.__class__.__name__
        return True, self.__class__.__name__

class PrimerFilter(SGFilter):
    def __init__(self):
        self.PRIMER = [
            ["AAGGCAAGTTGTTACCAGCA", "TGCGACCGTAATCAAACCAA"],
            ["TTCGTTCGTCGTTGATTGGT", "AAACGGAGCCATGAGTTTGT"],
            ["GAAGAGTTTAGCCACCTGGT", "CAAGTAACCGGCAACAACTG"],
            ["CTGTCCATAGCCTTGTTCGT", "GCGGAAACGTAGTGAAGGTA"]
        ]
        self.lPrimer = [pair[0] for pair in self.PRIMER] + [str(Seq.Seq(pair[1]).reverse_complement()) for pair in self.PRIMER]

    def hamming(self,s1, s2):
        return sum([a != b for a, b in zip(s1, s2)])
    def isGoodSeq(self,seq):
        for i in range(len(seq) - 19):
            check_unit = seq[i:i + 20]
            for p in self.lPrimer:
                if self.hamming(check_unit, p) < 5:
                    return False, self.__class__.__name__
        return True, self.__class__.__name__

class ScaffoldFilter(SGFilter):
    def __init__(self):
        self.SCAFFOLD = ["AATTCCGG"]
        self.blockSeq = self.SCAFFOLD
    def isGoodSeq(self,seq):
        for badSeq in self.blockSeq:
            if badSeq in seq:
                return False, self.__class__.__name__
        return True, self.__class__.__name__

class GCFilter(SGFilter):
    def __init__(self):
        pass
    def isGoodSeq(self, seq):
        for i in range(0, len(seq) - 19, 20):
            check_unit = seq[i:i + 20]
            C = list(check_unit).count("C")
            G = list(check_unit).count("G")
            if C + G > 12 or 20 - C - G > 12:
                # print(C+G,(C+G)/256,maxlen*0.6,len(seq) - C - G)
                return False, self.__class__.__name__
        return True, self.__class__.__name__

class RepeatFilter(SGFilter):
    def __init__(self):
        self.pattern = r'([ATCG]{1,6})\1{5}'
    def isGoodSeq(self,seq):
        if re.findall(self.pattern, seq):
            return False, self.__class__.__name__
        else:
            return True, self.__class__.__name__

class myFilter(SGFilter):
    """
    defined your own filter, then use by add it into 'seqgen.conf' file
    """
    def __init__(self):
        pass
    def isGoodSeq(self,seq):
        pass

class SeqGenerator():
    def __init__(self):
        #read config
        cp = ConfigParser()
        cp.read("seqgen.conf")
        self.nts = ['A', 'T', 'C', 'G']
        self.maxlen = int(cp.get("config", "maxLen"))
        self.gennum = int(cp.get("config", "genNum"))
        self.strategies = cp.get("config", "strategies").split('>')
        print(self.maxlen,self.strategies)

    def isGoodSeq(self, seq):
        outputs = []
        objects = [globals()[s]() for s in self.strategies]
        flag = True
        for object in objects:
            output = object.isGoodSeq(seq)
            outputs.append(output)
            if output[0] == False:
                flag = False
        return flag,outputs

    def gen(self, path="../seq_good_test.txt"):
        i = 0
        while True:
            seq = ''
            for _ in range(self.maxlen):
                seq += random.choice(self.nts)
            flag,outputs = self.isGoodSeq(seq)
            print(flag,outputs)
            if flag:
                i+=1
                with open(path, "a+", encoding="utf-8") as f:
                    f.write(seq + '\n')
                if i>=self.gennum:
                    break
    def check(self, path="../seq_good_test.txt"):
        with open(path,'r') as f:
            seqs = [line.strip() for line in f.readlines()]

        goodList = []
        infoList = []
        for seq in seqs:
            flag, outputs = self.isGoodSeq(seq)
            goodList.append(flag)
            infoList += outputs
        print(collections.Counter(goodList))
        print(collections.Counter(infoList))

if __name__ == "__main__":
    model = sys.argv[1]
    path = sys.argv[2]
    sg = SeqGenerator()
    if model == "gen":
        sg.gen(path)
    elif model == "check":
        sg.check(path)
    else:
        print("""seqgen [gen|check] [path(of gen|check file)]""")

