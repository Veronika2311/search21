from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import os
from tqdm import tqdm
import stanza

def get_path_list(folder='friends-data'):
    '''
    Обходим дерево файлов, получаем список путей к ним.
    По умолчанию в папке "friends-data", но для других тоже нужна
    '''
    data_path = os.path.join(os.getcwd(), folder)
    
    list_of_filepaths = []
    for root, dirs, files in os.walk(data_path):
        for name in files:
            list_of_filepaths.append(os.path.join(root, name))
    
    return list_of_filepaths


def get_text(filepath):
    '''
    Читаем файл
    '''
    with open(filepath, 'r', encoding='utf8') as f:
        text = f.read()
    return text


def preprocess(text, nlp):
    '''
    Здесь фильтруются предлоги, союзы и прочие служебные части речи.
    Также происходит фильтрация пунктуации.
    Препроцессинг взят из первой части дз
    '''
    doc = nlp(text)
    postags_stop = ['ADP', 'AUX', 'CCONJ', 'DET', 'INTJ', 'PART', 'PRON', 'PUNCT', 'SCONJ']
    lemms = []
    for sent in doc.sentences:
        for word in sent.words:
            if  word.upos not in postags_stop:
                lemms.append(word.lemma.lower())
    return ' '.join(lemms)


def get_corpus(text_analyzer):
    '''
    Если есть сохранённые предобработанные тексты, то используем их.
    Если нет -- создаём и сохраняем, чтобы не ждать по полчаса каждый раз.
    '''
    
    path_to_preprocessed = os.path.join(os.getcwd(), 'friends-data-preprocessed')
    
    if os.path.isdir(path_to_preprocessed):
        list_of_filepaths = get_path_list('friends-data-preprocessed')
        corpus = [get_text(filepath) for filepath in tqdm(list_of_filepaths)]
    else:
        print('Пожалуйста, подождите, корпус обрабатывается...')
        corpus = []
        shutil.copytree(os.path.join(os.getcwd(), 'friends-data'),  path_to_preprocessed)
        list_of_filepaths = get_path_list('friends-data-preprocessed')
        for filepath in tqdm(list_of_filepaths):
            preprocessed_text = preprocess(get_text(filepath), text_analyzer)
            corpus.append(preprocessed_text)
            with open(filepath, 'w', encoding='utf8') as f:
                f.write(preprocessed_text)
                
    return corpus

def normalize(x):
    return x / np.linalg.norm(x)

def index_corpus(vectorizer, corpus):
    '''
    Векторизуем корпус
    '''
    X = vectorizer.fit_transform(corpus).toarray()
    return X


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
    return cosine_similarity(X, Y).transpose()[0]


def main():
    vectorizer = TfidfVectorizer()
    text_analyzer = stanza.Pipeline(lang='ru', processors='tokenize,lemma,pos')
    
    query = preprocess(input('Введите поисковый запрос: '),  text_analyzer)
    
    list_of_filepaths = get_path_list()
    corpus = get_corpus(text_analyzer)


    corpus = get_corpus(text_analyzer)
    result = search_docs(index_corpus(vectorizer, corpus), index_search(vectorizer, query))
    for i in np.argsort(result)[::-1]:
        
        if result[i] != 0.0:
            print(list_of_filepaths[i])
            print(result[i])


if __name__ == "__main__":
    main()