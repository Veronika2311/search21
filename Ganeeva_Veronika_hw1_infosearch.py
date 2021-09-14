# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import stanza

def get_path_list():
    curr_dir = os.getcwd()
    data_path = os.path.join(curr_dir, 'friends-data')
    
    list_of_filepaths = []
    for root, dirs, files in os.walk(data_path):
        for name in files:
            list_of_filepaths.append(os.path.join(root, name))
    
    return list_of_filepaths


def get_text(filepath):
    with open(filepath, 'r', encoding='utf8') as f:
        text = f.read()
    return text


def preprocess(text, nlp):
    doc = nlp(text)
    #что тут происходит -- фильтруются предлоги, союзы, пунктуация и прочие служебные части речи
    postags_stop = ['ADP', 'AUX', 'CCONJ', 'DET', 'INTJ', 'PART', 'PRON', 'PUNCT', 'SCONJ']
    lemms = []
    for sent in doc.sentences:
        for word in sent.words:
            if  word.upos not in postags_stop:
                lemms.append(word.lemma.lower())
    return ' '.join(lemms)


def get_matrix(vectorizer, list_of_filepaths, text_analyzer):
    #работает не сильно быстро, ибо stanza. Какой-нибудь пайморфи быстрее, но там так имена лемматизируются, что лучше не надо
    X = vectorizer.fit_transform(preprocess(get_text(filepath), text_analyzer) for filepath in tqdm(list_of_filepaths))
    
    return X, vectorizer


def get_answers(X, vectorizer):
    vocabulary = vectorizer.get_feature_names()
    matrix_freq = np.asarray(X.sum(axis=0)).ravel()
    min_words = []
    max_words = []
    min_freq = min(matrix_freq)
    print('Значение минимальной частотности')
    print(min_freq)
    max_freq = max(matrix_freq)
    print('Значение максимальной частотности')
    print(max_freq)
    for i, el in enumerate(matrix_freq):
        if el == min_freq:
            min_words.append(vocabulary[i])
        elif el == max_freq:
            max_words.append(vocabulary[i])
    print('Самое частое слово')
    print(max_words[0])
    print('Самое редкое слово (одно, а то их много)')
    print(min_words[0])
    
    print('Слова, которые встречаются во всех документах')
    for i, st in enumerate(X.toarray().transpose()):
        if min(st) > 0:
            print(vocabulary[i])
    
      
    persons = [['моника', 'мон'], ['рэйчел', 'рэйч'], ['чендлер','чэндлер', 'чен'], 
             ['фиби', 'фибс'], ['росс'], ['джоуи', 'джои', 'джо']]

    names_freq = []
    for person in persons:
        person_freq = 0
        for name in person:
            person_freq += sum(X.toarray().transpose()[vectorizer.vocabulary_.get(name)])
        names_freq.append(person_freq)
    #print(names_freq)
    print('Самое частое имя: ' + persons[names_freq.index(max(names_freq))][0])


def main():
    text_analyzer = stanza.Pipeline(lang='ru', processors='tokenize,lemma,pos')
    vectorizer = CountVectorizer(analyzer='word')
    list_of_filepaths = get_path_list()
    X, vectorizer = get_matrix(vectorizer, list_of_filepaths, text_analyzer)
    get_answers(X, vectorizer)
    
if __name__ == "__main__":
    main()
