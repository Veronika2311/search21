#import os
import json
import numpy as np
from scipy import sparse
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
#from spacy.lang.ru import Russian
from gensim.models import KeyedVectors
import torch
from transformers import AutoTokenizer, AutoModel
#! -m spacy download ru_core_news_sm
import ru_core_news_sm


def get_text():
    '''
    Читаем файл, в этом дз определённый
    '''
    with open('questions_about_love.jsonl', 'r', encoding='utf8') as f:
        corpus = list(f)[:10000]

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


    return corpus_preproc, np.array(corpus_origin_answers)


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

    return corpus_matrix, count_vectorizer, tfidf_vectorizer, X_tfidf, X_count

def normalize(vector):
    return vector / np.linalg.norm(vector)


def get_embeddings_fasttext(text, model):
    text = text.split()
    vectors_of_words = np.zeros((len(text), model.vector_size))
    for i, word in enumerate(text):
        vectors_of_words[i] = model[word]
    if vectors_of_words.shape[0] == 0: #is 0:
        vector_of_text = np.zeros((model.vector_size,))
    else:
        mean_of_words = np.mean(vectors_of_words, axis=0)
        vector_of_text = normalize(mean_of_words)
    return vector_of_text


def get_embeddings_bert(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    vector_of_text = embeddings[0].cpu().numpy()
    vector_of_text = normalize(vector_of_text)

    return vector_of_text


def get_matrix_texts_fasttext(corpus_preproc, model):
    vectors_of_texts = np.zeros((len(corpus_preproc), model.vector_size))
    for i, text in enumerate(corpus_preproc):
        vector_of_text = get_embeddings_fasttext(text, model)
        vectors_of_texts[i] = vector_of_text

    return vectors_of_texts


def get_matrix_texts_bert(corpus_preproc, model, tokenizer):
    vectors_of_texts = np.zeros((len(corpus_preproc), 312))
    for i, text in enumerate(corpus_preproc):
        vector_of_text = get_embeddings_bert(text, model, tokenizer)
        vectors_of_texts[i] = vector_of_text

    return vectors_of_texts


def save_vectorizer(filename, vectorizer):
    joblib.dump(vectorizer, filename)

def save_sparse(filename, matrix):
    sparse.save_npz(filename, matrix)

def save_nparray(filename, nparray):
    np.save(filename, nparray)


def main():
    '''
    Главная функция: собирем всё воедино
    '''
    print(0)
    text_analyzer = ru_core_news_sm.load()
    #corpus_filepath = 'questions_about_love.jsonl'
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    bert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    print(1)
    fasttext_model = KeyedVectors.load('araneum_none_fasttextcbow_300_5_2018.model')
    print(2)
    corpus_preproc, corpus_origin_answers = get_corpus(text_analyzer)
    print(3)
    corpus_matrix, count_vectorizer, tfidf_vectorizer, X_tfidf, X_count = index_corpus(corpus_preproc)
    print(4)
    fasttext_corpus_matrix = get_matrix_texts_fasttext(corpus_origin_answers, fasttext_model)
    bert_corpus_matrix = get_matrix_texts_bert(corpus_origin_answers, bert_model, tokenizer)
    print(5)
    save_vectorizer('count_vectorizer', count_vectorizer)
    save_vectorizer('tfidf_vectorizer', tfidf_vectorizer)
    save_sparse('count_matrix', X_count)
    save_sparse('tfidf_matrix', X_tfidf)
    save_sparse('bm25_matrix', corpus_matrix)
    save_nparray('bert_corpus_matrix', bert_corpus_matrix)
    save_nparray('fasttext_corpus_matrix', fasttext_corpus_matrix)
    save_nparray('ans_corpus', corpus_origin_answers)


#if __name__ == "__main__":
#    main()