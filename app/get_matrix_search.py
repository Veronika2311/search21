#import os
import json
import numpy as np
from scipy import sparse
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#from sklearn.metrics.pairwise import cosine_similarity
#import joblib
#from spacy.lang.ru import Russian
#from gensim.models import KeyedVectors
import torch
#from transformers import AutoTokenizer, AutoModel
#! -m spacy download ru_core_news_sm
#import ru_core_news_sm
from get_matrix_preproc import *


def load_vectorizer(filename):
    vectorizer = joblib.load(filename)
    return vectorizer

def load_sparse(filename):
    matrix = sparse.load_npz(filename)
    return matrix

def load_nparray(filename):
    nparray = np.load(filename)
    return nparray


def index_search_vect(vectorizer, query):
    '''
    Векторизуем запрос
    '''
    Y  = vectorizer.transform([query])
    return Y


def dot_nparray_search(X, Y):
    return np.dot(X, Y.T)


def dot_sparse_search(X, Y):
    return X.dot(Y.T)


def cosine_search(X, Y):
    return cosine_similarity(X, Y).transpose()[0]


def range_of_results(corpus, scores):
    '''
    Ранжируем выдачу, выдаём пять самых близких документов
    '''
    if sparse.issparse(scores):
        scores = scores.toarray()
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    #print(sorted_scores_indx.shape)
    #print(sorted_scores_indx)
    return corpus[sorted_scores_indx.ravel()[:5]]


def vectorizers_search(query, text_analyzer, vectorizer, corpus, corpus_matrix):
    query = preprocess(query, text_analyzer)
    query_vect = index_search_vect(vectorizer, query)
    #print(corpus_matrix.shape)
    #print(query_vect)
    similarity_vect = cosine_search(corpus_matrix, query_vect)
    #print(similarity_vect.shape)
    result = range_of_results(corpus, similarity_vect)
    return result


def bm25_search(query, text_analyzer, vectorizer, corpus, corpus_matrix):
    query = preprocess(query, text_analyzer)
    query_vect = index_search_vect(vectorizer, query)
    similarity_vect = dot_sparse_search(corpus_matrix, query_vect)
    result = range_of_results(corpus, similarity_vect)
    return result


def fasttext_search(query, model, count_vectorizer, corpus, corpus_matrix):
    query_vect = normalize(get_embeddings_fasttext(query, model))
    similarity_vect = dot_nparray_search(corpus_matrix, query_vect)
    result = range_of_results(corpus, similarity_vect)
    return result


def bert_search(query, model, tokenizer, corpus, corpus_matrix):
    query_vect = normalize(get_embeddings_bert(query, model, tokenizer))
    similarity_vect = dot_nparray_search(corpus_matrix, query_vect)
    result = range_of_results(corpus, similarity_vect)
    return result
