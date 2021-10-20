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
import base64
import streamlit as st


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


def range_of_results(corpus, scores, count_results):
    '''
    Ранжируем выдачу, выдаём пять самых близких документов
    '''
    if sparse.issparse(scores):
        scores = scores.toarray()
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    result_scores_indx = sorted_scores_indx.ravel()[:count_results]
    return corpus[result_scores_indx], scores[sorted_scores_indx]


def vectorizers_search(query, text_analyzer, vectorizer, \
                       corpus, corpus_matrix, count_results):
    query = preprocess(query, text_analyzer)
    query_vect = index_search_vect(vectorizer, query)
    similarity_vect = cosine_search(corpus_matrix, query_vect)
    result = range_of_results(corpus, similarity_vect, count_results)
    return result


def bm25_search(query, text_analyzer, vectorizer, \
                corpus, corpus_matrix, count_results):
    query = preprocess(query, text_analyzer)
    query_vect = index_search_vect(vectorizer, query)
    similarity_vect = dot_sparse_search(corpus_matrix, query_vect)
    result = range_of_results(corpus, similarity_vect, count_results)
    return result[0], result[1].ravel()


def fasttext_search(query, model, count_vectorizer, \
                    corpus, corpus_matrix, count_results):
    query_vect = normalize(get_embeddings_fasttext(query, model))
    similarity_vect = dot_nparray_search(corpus_matrix, query_vect)
    result = range_of_results(corpus, similarity_vect, count_results)
    return result


def bert_search(query, model, tokenizer, corpus, corpus_matrix, count_results):
    query_vect = normalize(get_embeddings_bert(query, model, tokenizer))
    similarity_vect = dot_nparray_search(corpus_matrix, query_vect)
    result = range_of_results(corpus, similarity_vect, count_results)
    return result

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
      background-image: url("data:image/png;base64,%s");
      background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
