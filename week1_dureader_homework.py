#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
import jieba

from gensim.models import word2vec
import multiprocessing


path = r'D:/Projects/MRC/week1_机器阅读理解发展及任务解析/MRC_DuReader/'
train_set = 'data/preprocessed/trainset/search.train.json'
stopwords_set = 'stopwords/cn_stopwords.txt'

dureader_seg_res = 'data/dureader_segment.txt'



# 加载停用词
stopwords = []
with open(path+stopwords_set, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stopwords.append(line.strip())
print(len(stopwords))



# 构造问题和答案分词词库
corpus = []
with open(path+train_set,'r',encoding='utf-8') as f:
    for row in f:
        qa = []
        data = json.loads(row)
        question = data['segmented_question']
        answers = data['segmented_answers']
        qa += [q for q in question if q not in stopwords] + [word for answer in answers for word in answer if word not in stopwords]
        corpus.append(qa)

with open(path + dureader_seg_res, "w", encoding='utf-8') as dureader:
    for row in corpus:
        dureader.write(" ".join(row))

print(corpus[:3])



def train_wordVectors(sentences, embedding_size = 128, window = 5, min_count = 5):
    '''
    :param sentences: sentences可以是LineSentence或者PathLineSentences读取的文件对象，也可以是
                    The `sentences` iterable can be simply a list of lists of tokens,如lists=[['我','是','中国','人'],['我','的','家乡','在','广东']]
    :param embedding_size: 词嵌入大小
    :param window: 窗口
    :param min_count:Ignores all words with total frequency lower than this.
    :return: w2vModel
    '''
    w2vModel = word2vec.Word2Vec(sentences, vector_size=embedding_size, window=window, min_count=min_count,workers=multiprocessing.cpu_count())
    return w2vModel

def save_wordVectors(w2vModel,word2vec_path):
    w2vModel.save(word2vec_path)


def load_wordVectors(word2vec_path):
    w2vModel = word2vec.Word2Vec.load(word2vec_path)
    return w2vModel

word2vec_sentence = word2vec.LineSentence(path+dureader_seg_res)

# 简单训练
model = word2vec.Word2Vec(word2vec_sentence, hs=1, min_count=1, window=3, vector_size=100)
print(model.wv.similarity('微信', '好友'))


# 训练：
word2vec_path = path + 'models/word2vec_model.model'
model2 = train_wordVectors(word2vec_sentence, vector_size=128, window=5, min_count=5)

save_wordVectors(model2,word2vec_path)
model2 = load_wordVectors(word2vec_path)

print(model2.wv.similarity('微信', '好友'))