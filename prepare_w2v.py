import pandas as pd
from sklearn.metrics import f1_score
import codecs
import time
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import gensim
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
# from pyltp import Segmentor
from sklearn.externals import joblib
from sklearn.svm import SVC
import string
import re
#
# def w2c_train():
#     corpus=[]
#     for eachline in codecs.open('embedding/word-of-mouth-car-cut.csv','rU','utf-8'):
#         new_line=eachline.strip().split()
#         corpus.append(new_line)
#     len(corpus)
#     sentences = Word2Vec(corpus, size=300, window=5, min_count=5, workers=4)
#     sentences.save('embedding/car_w2c_model')
# stopwords = (
#     'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
#      'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
#      'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
#      'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the'
#      'and', 'if', 'or', 'because', 'as', 'until', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
#      'between', 'into', 'through', 'during', 'before', 'after',
#      'then', 'once', 'here', 'there', 'when', 'where', 'why',
#      'all', 'any', 'both', 'each', 'few',
#      'own', 'same', 'than', 'too', 'very', 's', 't', 'now')

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def prepare_w2v():
    # w2v_model = KeyedVectors.load_word2vec_format('~/embedding/glove.42B.300d.w2v.txt',
    #                                               binary=False)
    w2v_model = KeyedVectors.load_word2vec_format('E:/embedding/glove.42B.300d.w2v.txt',
                                                  binary=False, limit=10000000)
    # print(w2v_model.maost_similar("happy"))
    print('a' in w2v_model)
    print('and' in w2v_model)
    dim = len(w2v_model['good'])
    print(dim)
    # exit(-1)


    f1 = codecs.open("dataset2/rt-polarity.neg", encoding='utf-8')
    f2 = codecs.open("dataset2/rt-polarity.pos", encoding='utf-8')

    f_list = [f1, f2]
    fw1 = codecs.open("embedding_all_Glove300.txt", 'w', encoding='utf-8')
    # fw2 = codecs.open("embedding_test_GN300.txt", 'w', encoding='utf-8')

    all_set = set()
    test_set = set()

    # train_x_text = []
    length_l = []
    length_l_in = []

    for f in f_list:
        for line in f:
            # print(line)
            line = line.strip('\r\n')
            # print(line)
            line = clean_str(line)
            # print(line)
            words = line.split()
            length_l.append(len(words))
            l_in = 0
            for w in words:
                if w in w2v_model:
                    l_in += 1
                all_set.add(w)
            length_l_in.append(l_in)


    print("%d %d", np.mean(length_l), np.mean(length_l_in))
    print(len(all_set))
    in_set = set()
    miss = 0
    for w in all_set:
        if w in w2v_model:
            in_set.add(w)
        else:
            wi = w.split('-')
            for nw in wi:
                if nw in w2v_model:
                    in_set.add(nw)
                else:
                    miss += 1
                    # print(nw)
    print(len(in_set))
    print("miss:%d"% miss)

    fw1.write(str(len(in_set)) + ' ' + str(dim)+'\n')
    for w in in_set:
        if w in w2v_model:
            # in_set.add(w)
            embeds = w2v_model[w]
            fw1.write(w)
            for i in embeds:
                fw1.write(' ' + str(i))
            fw1.write('\n')
        else:
            print('error')
            exit(-1)


    # for w in test_set:
    #     if w in w2v_model:
    #         embeds = w2v_model[w]
    #         fw2.write(w + '\t')
    #         for i in embeds:
    #             fw2.write(' ' + str(i))
    #         fw2.write('\n')
    #     else:
    #         wi = w.split('-')
    #         for nw in wi:
    #             if nw in w2v_model:
    #                 embeds = w2v_model[nw]
    #                 fw1.write(nw + '\t')
    #                 for i in embeds:
    #                     fw1.write(' ' + str(i))
    #                 fw1.write('\n')

def test():
    w2v_model = KeyedVectors.load_word2vec_format('embedding_all_GN300.txt')
    print(w2v_model.most_similar("unhappy"))


if __name__ == '__main__':
    # main()
    prepare_w2v()
    test()