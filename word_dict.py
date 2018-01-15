# -*- coding: utf-8 -*
import unicodedata,string,re

MAX_DIC = 9000

d = {}
d["SOS"] = 0
d["EOS"] = 1

def nomarlize_string(s):
    s = s.low().strip()
    s = re.sub(r"")

with open("./data/word2idx.txt") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.split(":")
        line = tmp[0].decode("utf8")
        string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[-－＿+——！，。？?、~@#￥%……&*（）]+".decode("utf8"), "".decode("utf8"), line)
        if len(string) >= 1:
            d[string] = len(d)
        if len(d) >= MAX_DIC:
            break
d["UNK"] = len(d)

import json
dic_str = json.dumps(d)
fw = open("./data/word2idx.json","wb")
fw.write(dic_str)
fw.close()