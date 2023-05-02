# -*- coding: gbk -*-
import random
import re
import jieba
import gensim
import matplotlib.pyplot as plt
import matplotlib

def process(file_path):
    paragraphs = []
    label = []

    punctuation = u'[A-Za-z0-9_.!+-=——,$%^，。？、~@#￥%……&*《》<>「」{}【】()/]（）'

    with open(file_path + 'inf.txt','r',encoding = 'utf-8') as f:
        file_names = f.readline().split(',')
    f.close()

    for file_name in (file_names):
        with open(file_path + file_name + '.txt','r',encoding='gb18030') as f:
            corpus = f.read()
            corpus = re.sub(punctuation,'',corpus)
            corpus = corpus.replace('\u3000','')
            for paragraph in corpus.split('\n'):
                if(len(paragraph) > 500):
                    paragraphs.append(paragraph)
                    label.append(file_name)

    para_sample = []
    label_sample = []
    serial_number = random.sample(range(len(paragraphs)), 200)
    para_sample.extend([paragraphs[i] for i in serial_number])
    label_sample.extend([label[i] for i in serial_number])

    return para_sample,label_sample

def dividword(paragraphs,labels):
    divid_word = []
    label_word = []
    divid_char = []
    label_char = []
    for i,para in enumerate(paragraphs):
        words = [word for word in jieba.cut(para)]
        divid_word.append(words)
        label_word.append(labels[i])

        chars = [char for char in para]
        divid_char.append(chars)
        label_char.append(labels[i])

    return divid_word,label_word,divid_char,label_char

if __name__ == "__main__":
    file_path = 'D:/语料库/'
    paragraphs, label = process(file_path)
    divid_word,label_word,divid_char,label_char = dividword(paragraphs,label)
    topicnum_range = range(2,20)
    dictionary_word = gensim.corpora.Dictionary(divid_word)
    corpus_word = [dictionary_word.doc2bow(word) for word in divid_word]
    dictionary_char = gensim.corpora.Dictionary(divid_char)
    corpus_char = [dictionary_char.doc2bow(char) for char in divid_char]

    perplexity_word = []
    coherence_word = []
    perplexity_char = []
    coherence_char = []
    for topicnum in topicnum_range:
        ldamodel_word = gensim.models.ldamodel.LdaModel(corpus=corpus_word,num_topics=topicnum,id2word=dictionary_word,passes=20)
        pep_wd = -ldamodel_word.log_perplexity(corpus_word)
        perplexity_word.append(pep_wd)
        cv_wd = gensim.models.CoherenceModel(model=ldamodel_word, texts=divid_word, dictionary=dictionary_word,coherence='c_v')
        coherence_word.append(cv_wd.get_coherence())

        ldamodel_char = gensim.models.ldamodel.LdaModel(corpus=corpus_char,num_topics=topicnum,id2word=dictionary_char,passes=20)
        pep_ch = -ldamodel_char.log_perplexity(corpus_char)
        perplexity_char.append(pep_ch)
        cv_ch = gensim.models.CoherenceModel(model=ldamodel_char, texts=divid_char, dictionary=dictionary_char,coherence='c_v')
        coherence_char.append(cv_ch.get_coherence())


    plt.plot(topicnum_range, perplexity_word)
    plt.legend()
    plt.xlabel('主题数目')
    plt.ylabel('Perplexity')
    plt.rcParams['font.sans-serif']=['SimHei']
    matplotlib.rcParams['axes.unicode_minus']=False
    plt.title('主题数目-困惑性（以词为单位）')
    plt.savefig('1')
    plt.cla()
    plt.clf()

    plt.plot(topicnum_range, perplexity_char,label='以字为单位')
    plt.legend()
    plt.xlabel('主题数目')
    plt.ylabel('Perplexity')
    plt.rcParams['font.sans-serif']=['SimHei']
    matplotlib.rcParams['axes.unicode_minus']=False
    plt.title('主题数目-困惑性（以字为单位）')
    plt.savefig('2')
    plt.cla()
    plt.clf()

    plt.plot(topicnum_range, coherence_word,label='以词为单位')
    plt.plot(topicnum_range, coherence_char,label='以字为单位')
    plt.legend()
    plt.xlabel('主题数目')
    plt.ylabel('Coherence')
    plt.rcParams['font.sans-serif']=['SimHei']
    matplotlib.rcParams['axes.unicode_minus']=False
    plt.title('主题数目-主题一致性比较图')
    plt.savefig('3')
    plt.cla()
    plt.clf()
