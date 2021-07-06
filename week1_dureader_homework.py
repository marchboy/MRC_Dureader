#!/usr/bin/env python
# -*- coding: utf-8 -*-


from collections import defaultdict
from io import BufferedIOBase
import json
import jieba
import logging

from smart_open import open
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
import multiprocessing


class MRC_Dureader():
    def __init__(self, trainset_path, stopwords_path, dureader_path) -> None:
        self.trainset_path = trainset_path
        self.stopwords_path = stopwords_path
        self.dureader_path = dureader_path
        self.stop_words_list = self.load_stopwords()

    def load_stopwords(self):
        stop_words_list = []
        with open(self.stopwords_path, "r", encoding="utf-8") as stop_words_handler:
            for word in stop_words_handler:
                stop_words_list.append(word.strip())
        return stop_words_list

    def load_trainset(self):
        corpus = []
        # i = 1
        with open(self.trainset_path, "r", encoding="utf-8") as trainset_handler:
            for line in trainset_handler:
                ques_ans = []
                line = json.loads(line)
                question = line.get("segmented_question")
                answers = line.get("segmented_answers")

                ques_ans += [ques for ques in question if ques not in self.stop_words_list] + \
                     [ans for answer in answers for ans in answer if ans not in self.stop_words_list]
                corpus.append(ques_ans)
                # i += 1
                # if i >= 50:
                #     break

        # 保存词汇到本地
        with open(self.dureader_path, "w", encoding="utf-8") as dureader:
            for word in corpus:
                dureader.write(" ".join(word))
        return corpus

    def words_count(self) -> dict:
        # 词汇统计
        word_count_dict = defaultdict(int)
        for qa in self.load_trainset():
            for word in qa:
                word_count_dict[word] += 1
        word_counts = [(k, v) for k, v in sorted(word_count_dict.items(), key=lambda item: item[1], reverse=True)]
        return word_counts


def train_word2vec(sentences, embedding_size = 128, window = 5, min_count = 5):
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


if __name__ == "__main__":
    path = r'F:/0003_projects/KKB_MRC/week1_机器阅读理解发展及任务解析/MRC_Dureader/'
    train_set = 'data/preprocessed/trainset/test.txt'
    stopwords_set = 'stopwords/cn_stopwords.txt'
    dureader_seg_res = 'data/dureader_segment.txt'

    trainset_path = r"E:/dureader_preprocessed/preprocessed/trainset/search.train.json"
    mrc_dureader = MRC_Dureader(trainset_path, path+stopwords_set, path+dureader_seg_res)
    
    # 打印词频, 前100
    word_counts = mrc_dureader.words_count()
    print(word_counts[:100])

    # 读取文本语料
    word2vec_sentence = word2vec.LineSentence(path+dureader_seg_res)

    # 简单训练
    model = word2vec.Word2Vec(word2vec_sentence, hs=1, min_count=1, window=3, vector_size=100)
    print(model.wv.similarity('微信', '聊天记录'))


    # 训练：
    word2vec_path = path + 'models/word2vec_model.model'
    model2 = train_word2vec(word2vec_sentence, embedding_size=128, window=5, min_count=5)
    
    # 保存模型
    save_wordVectors(model2, word2vec_path)
    model2 = load_wordVectors(word2vec_path)

    print(model2.wv.similarity('微信', '聊天记录'))
