from get_matrix_preproc import *
from get_matrix_search import *
from gensim.models import KeyedVectors
from transformers import AutoTokenizer, AutoModel
import ru_core_news_sm
import time
import streamlit as st


@st.cache(allow_output_mutation=True)
def load_all():
    text_analyzer = ru_core_news_sm.load()
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    bert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    fasttext_model = KeyedVectors.load('araneum_none_fasttextcbow_300_5_2018.model')
    count_vectorizer = load_vectorizer('count_vectorizer')
    tfidf_vectorizer = load_vectorizer('tfidf_vectorizer')
    count_matrix = load_sparse('count_matrix.npz')
    tfidf_matrix = load_sparse('tfidf_matrix.npz')
    bm25_matrix = load_sparse('bm25_matrix.npz')
    bert_corpus_matrix = load_nparray('bert_corpus_matrix.npy')
    fasttext_corpus_matrix = load_nparray('fasttext_corpus_matrix.npy')
    ans_corpus = load_nparray('ans_corpus.npy')

    return text_analyzer, tokenizer, bert_model, fasttext_model, \
        count_vectorizer, tfidf_vectorizer, count_matrix, \
        tfidf_matrix, bm25_matrix, bert_corpus_matrix, \
        fasttext_corpus_matrix, ans_corpus


def main():
    st.title('Поиск')
    text_analyzer, tokenizer, bert_model, fasttext_model, count_vectorizer, \
        tfidf_vectorizer, count_matrix, tfidf_matrix, bm25_matrix, \
        bert_corpus_matrix, fasttext_corpus_matrix, ans_corpus = load_all()

    option = st.sidebar.selectbox('Опция поиска',
        ('count_vectorizer', 'tfidf_vectorizer', 'bm25', 'fasttext', 'bert'))
    query = st.text_input('Поисковый запрос')
    start_query_time = time.time()
    if option == 'count_vectorizer':
        result = vectorizers_search(query, text_analyzer, count_vectorizer, ans_corpus, count_matrix)
    elif option == 'tfidf_vectorizer':
        result = vectorizers_search(query, text_analyzer, tfidf_vectorizer, ans_corpus, tfidf_matrix)
    elif option == 'bm25':
        result = bm25_search(query, text_analyzer, count_vectorizer, ans_corpus, bm25_matrix)
    elif option == 'fasttext':
        result = fasttext_search(query, fasttext_model, count_vectorizer, ans_corpus, fasttext_corpus_matrix)
    elif option == 'bert':
        result = bert_search(query, bert_model, tokenizer, ans_corpus, bert_corpus_matrix)
    end_query_time = time.time()
    st.write(str(result))
    st.write(end_query_time - start_query_time)


if __name__ == '__main__':
    main()
