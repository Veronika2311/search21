import os
import json
import numpy as np
from scipy import sparse
from tqdm import tqdm
from sklearn.preprocessing import normalize
from spacy.lang.ru import Russian
from gensim.models import KeyedVectors
import torch
from transformers import AutoTokenizer, AutoModel

def get_file(filepath):
    '''
    Читаем файл, в этом дз определённый
    '''
    with open(filepath, 'r', encoding='utf8') as f:
        corpus = list(f)[:50]
    
    return corpus


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


def get_corpus(corpus_json):
    '''
    Обрабатываем корпус, полчучаем корпус сырых ответов и корпус препроцесшенных
    '''
    print('Пожалуйста, подождите, корпус обрабатывается...')
    corpus_origin_answers = []
    for question_answers in tqdm(corpus_json):
        answers = json.loads(question_answers)['answers']
        text = sort_answers(answers)
        corpus_origin_answers.append(text)

    return corpus_origin_answers

def normalize(vector):
    return vector / np.linalg.norm(vector)


def get_embeddings_fasttext(text, model):
    text = text.split()
    vectors_of_words = np.zeros((len(text), model.vector_size))
    for i, word in enumerate(text):
        vectors_of_words[i] = model[word]
    if vectors_of_words.shape[0] is 0:
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


def get_similarity(corpus_matrix, query):
    return np.dot(corpus_matrix, query.T)


def range_of_results(corpus, scores):
    '''
    Ранжируем выдачу, выдаём пять самых близких документов
    '''
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    return np.array(corpus)[sorted_scores_indx.ravel()][:5]


def main():
    print('Начинаем обработку')
    corpus_filepath = 'questions_about_love.jsonl'
    print('Загружаем модель...')
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    bert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    fasttext_model = KeyedVectors.load('araneum_none_fasttextcbow_300_5_2018.model')
    print('Получаем корпус...')
    corpus_origin_answers = get_corpus(corpus_json)
    print('Получаем матрицу эмбеддингов...')
    fasttext_corpus_matrix = get_matrix_texts_fasttext(corpus_origin_answers, fasttext_model)  
    bert_corpus_matrix = get_matrix_texts_bert(corpus_origin_answers, bert_model, tokenizer)
    while True:
        query = input('Введите поисковый запрос: ')
        if query == '':
            break
        option = input('Выбепите опцию. 1 для fasttext, 2 для bert. По умолчанию bert')
        if option == '1':
            query_vector = get_embeddings_fasttext(query, fasttext_model)
            similarity = get_similarity(fasttext_corpus_matrix, query_vector)            
        else:
            query_vector = get_embeddings_bert(query, model, tokenizer)
            similarity = get_similarity(bert_corpus_matrix, query_vector)
        
        result = range_of_results(corpus_origin_answers, similarity)
        for el in result:
            print(el)

if __name__ == "__main__":
    main()
