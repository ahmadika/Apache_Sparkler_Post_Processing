#This code is for training a Word2Vec model to be used for retreiving close words for Sparkler ML
#Models Training

import word2vec
import configparser
import codecs
import numpy as np
from nltk.corpus import stopwords
#Reading configuration
def read_config(file, section):
    Config = configparser.ConfigParser()
    Config.read(file)
    configs = {}
    options = Config.options(section)
    for option in options:
        configs[option] = Config.get(section, option)
    return configs
#Reading paralle corpus
def read_lines(file_lines):
    print("Hi")
    sentences = []
    for line in file_lines:
        tokens = line.split()
        sentences.append(tokens)
    return np.asarray(sentences)

def removeStopWords(stopwords, list):
    return([word for word in list if word not in stopwords])

def closeWords(model,word, topN):
    indexes, metrics = model.cosine(word)
    list = model.generate_response(indexes, metrics).tolist()
    return list[:topN]

def cleanText(file, outName):

    stop_words = set(stopwords.words('english'))
    file1 = open(file)
    line = file1.read()  # Use this to read file content as a stream:
    words = line.split()
    for r in words:
        if not r in stop_words:
            appendFile = open(str(outName) + '.txt', 'a')
            appendFile.write(" " + r)
            appendFile.close()
            # for char in r:
            #     if char in "?.!-_":
            #         line.replace(char, ' ')
            # for rr in r.split():
            #     appendFile = open(str(outName)+'.txt', 'a')
            #     appendFile.write(" " + rr)
            #     appendFile.close()
    return str(outName)+'.txt'
def main():
    argv1="sparkler.config"
    argv2="Sparkler"
    configs = read_config(argv1,argv2)
    train_file_obj = codecs.open(configs['train_file'],'r')
    train_lines = train_file_obj.readlines()
    train_file_obj.close()
    dimension_input = int(configs['dimension_input'])
    word2vec.word2phrase(cleanText('data/200andOcean.txt','clean200Ocean'), 'ocean-full-phrases', verbose=True)
    word2vec.word2vec('ocean-full-phrases', 'ocean.bin', size=dimension_input, verbose=True, min_count=5)
    model = word2vec.load('ocean.bin')
    word='ocean'
    print(closeWords(model, word, 5))
    print(model.vectors.shape)
if __name__=='__main__':
    main()
