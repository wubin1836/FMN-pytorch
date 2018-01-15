# -*- coding: utf-8 -*
import json
import jieba
import codecs

S = []
T = []
D = []

with open("./data/word2idx.json") as f:
    word_dict = json.load(f)

print word_dict

with codecs.open("./data/sample_data.txt","r","utf-8") as f:
    s_tmp = []

    q = [] #for each session with it's query
    d = []

    q_d_tmp = [] #for each query with it's document
    q_tmp = []

    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line == "============":#session end
            d.append(q_d_tmp)
            q.append(q_tmp)
            S.append(q[:-1])
            T.append(q[-1])
            D.append(d[:-1])
            q = []
            d = []

        elif line == "------------":#query end
            d.append(q_d_tmp)
            q.append(q_tmp)
            q_tmp = []
            q_d_tmp = []

        else:
            tmp = line.strip().split(" ")
            index_tmp = [word_dict["SOS"]]
            seg_list = jieba.cut(tmp[0], cut_all=False)
            for item in seg_list:
                if word_dict.has_key(item):
                    index_tmp.append(word_dict[item])
                else:
                    index_tmp.append(word_dict["UNK"])
            index_tmp.append(word_dict["EOS"])
            if (tmp[-1] == "0" or tmp[-1] == "1") and len(tmp) > 1:  # document
                q_d_tmp.append([index_tmp,int(tmp[-1])])
            else:
                q_tmp = index_tmp

import cPickle as pickle
output = open("./data/S.pkl","wb")
pickle.dump(S,output)
output.close()

output = open("./data/T.pkl","wb")
pickle.dump(T,output)
output.close()

output = open("./data/D.pkl","wb")
pickle.dump(D,output)
output.close()

for i in range(len(S)):
    print S[i]
    print T[i]
    print D[i]

    print "=========="