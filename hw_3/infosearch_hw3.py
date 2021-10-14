import os
import json
import numpy as np
from scipy import sparse
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy.lang.ru import Russian

def get_text():
    '''
    Читаем файл, в этом дз определённый
    '''
    with open('questions_about_love.jsonl', 'r', encoding='utf8') as f:
        corpus = list(f)[:50000]
    
    return corpus

def preprocess(text, nlp):
    '''
    Здесь фильтруются предлоги, союзы и прочие служебные части речи.
    Также происходит фильтрация пунктуации.
    '''
    doc = nlp(text)
    postags_stop = ['ADP', 'AUX', 'CCONJ', 'DET', 'INTJ', 'PART', 'PRON', 'PUNCT', 'SCONJ']
    lemms = []
    for word in doc:
        if word.pos_ not in postags_stop and str(word) not in '.,?!\'\"()[]{}<>1234567890':
            lemms.append(word.lemma_.lower())
    return ' '.join(lemms)


def sort_answers(answers):
    '''
    Сортируем ответы по рейтингу автора
    '''
    raits = []
    for answer in answers:
        if answer['author_rating']['value']:
            raits.append(int(answer['author_rating']['value']))
        else:
            raits.append(0)
    if len(raits) >= 1:
        max_rait_ind = np.argmax(np.array(raits))
        best_answer = answers[max_rait_ind]['text']
    else:
        best_answer = ''
    
    return best_answer


def get_corpus(text_analyzer):
    '''
    Обрабатываем корпус, полчучаем корпус сырых ответов и корпус препроцесшенных
    '''
    print('Пожалуйста, подождите, корпус обрабатывается...')
    corpus_preproc = [] 
    corpus_origin_answers = []
    for question_answers in tqdm(get_text()):
        answers = json.loads(question_answers)['answers']
        text = sort_answers(answers)
        corpus_origin_answers.append(text)
        preprocessed_text = preprocess(text, text_analyzer)
        corpus_preproc.append(preprocessed_text)
        
    
    return corpus_preproc, corpus_origin_answers


def index_corpus(corpus):
    '''
    Векторизуем корпус
    '''
    print('Векторизация корпуса...')
    tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    count_vectorizer = CountVectorizer()
    
    X_tf = tf_vectorizer.fit_transform(corpus)
    X_tfidf = tfidf_vectorizer.fit_transform(corpus)
    X_count = count_vectorizer.fit_transform(corpus)    
    idf_vect = tfidf_vectorizer.idf_
    len_of_doc_vect = X_count.sum(axis=1)
    avgdl = len_of_doc_vect.mean()
    k = 2.0
    b = 0.75
    
    values, rows, cols = [], [], []
    

    for i, j in zip(*X_tf.nonzero()):
        #числитель
        n = idf_vect[j] * X_tf[i, j] * (k + 1)
        #знаменатель
        m = X_tf[i, j] + (k * (1 - b + b * len_of_doc_vect[i, 0] / avgdl))
        result = n / m
        rows.append(i)
        cols.append(j)
        values.append(result)
        
    corpus_matrix = sparse.csr_matrix((values, (rows, cols)))
        
    return corpus_matrix, count_vectorizer



def index_search(vectorizer, query):
    '''
    Векторизуем запрос
    '''
    Y  = vectorizer.transform([query]).toarray()
    
    return Y


def search_docs(X, Y):
    '''
    Считаем близость запроса и документов корпуса
    '''
    
    return X.dot(Y.T)


def range_of_results(corpus, scores):
    '''
    Ранжируем выдачу, выдаём пять самых близких документов
    '''
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    return np.array(corpus)[sorted_scores_indx.ravel()][:5]


def main():
    '''
    Главная функция: собирем всё воедино
    '''
    text_analyzer = Russian()
    corpus_preproc, corpus_origin_answers = get_corpus(text_analyzer)
    corpus_matrix, count_vectorizer = index_corpus(corpus_preproc)
    while True:
        user_string = input('Введите поисковый запрос: ')
        if user_string == '':
            break
        query = preprocess(user_string, text_analyzer)
        Y = index_search(count_vectorizer, query)
        result = range_of_results(corpus_origin_answers, search_docs(corpus_matrix, Y))
        for el in result:
            print(el)

if __name__ == "__main__":
    main()