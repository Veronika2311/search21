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

    text_analyzer, tokenizer, bert_model, fasttext_model, count_vectorizer, \
        tfidf_vectorizer, count_matrix, tfidf_matrix, bm25_matrix, \
        bert_corpus_matrix, fasttext_corpus_matrix, ans_corpus = load_all()

    st.title('Поиск по корпусу ответов mail.ru')
    st.markdown('*Раздел "Отношения"*')
    st.sidebar.image('ruby-heart.png')
    set_background('background.jpg')

    option = st.sidebar.selectbox('Выберите опцию поиска: ',
        ('СountVectorizer', 'TfidfVectorizer', 'Okapi BM25', \
         'Fasttext', 'Bert (rubert-tiny)'))

    count_results = st.sidebar.slider('Выберите количество результатов: ', 5, 100)
    query = st.text_input('Введите запрос: ')
    st.button('Искать в корпусе')

    if query == '':
        st.markdown('*Ваш запрос пока ничего не содержит*')
    else:
        start_query_time = time.time()
        if option == 'СountVectorizer':
            result = vectorizers_search(query, text_analyzer, \
                                        count_vectorizer, ans_corpus, \
                                        count_matrix, count_results)
        elif option == 'TfidfVectorizer':
            result = vectorizers_search(query, text_analyzer, \
                                        tfidf_vectorizer, ans_corpus, \
                                        tfidf_matrix, count_results)
        elif option == 'Okapi BM25':
            result = bm25_search(query, text_analyzer, count_vectorizer, \
                                 ans_corpus, bm25_matrix, count_results)
        elif option == 'Fasttext':
            result = fasttext_search(query, fasttext_model, \
                                     count_vectorizer, ans_corpus, \
                                     fasttext_corpus_matrix, count_results)
        elif option == 'Bert (rubert-tiny)':
            result = bert_search(query, bert_model, tokenizer, ans_corpus, \
                                 bert_corpus_matrix, count_results)
        end_query_time = time.time()

        st.markdown('*Время выполнения поиска: ' + \
                    str(round(end_query_time - start_query_time, 3)) + ' секунды*')

        st.markdown('### Результаты:')
        st.write('***********************************************')
        for i, answer in enumerate(result[0]):
            st.write(answer)
            st.markdown('*Близость ответа к запросу: ' + \
                        str(round(result[1][i], 7)) + '*')
            st.write('***********************************************')
            #res_print = [answer, ' <br\> ', '*Близость ответа к запросу: ',
            #             str(round(result[1][i], 7)), '*']
            #st.markdown(''.join(res_print))

        end_print_time = time.time()
        st.markdown('*Время печати запросов: ' + \
                    str(round(end_print_time - end_query_time, 3)) + ' секунды*')


if __name__ == '__main__':
    main()
