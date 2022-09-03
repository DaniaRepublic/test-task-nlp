import pandas as pn
# для выявления имем с помощью NER 
import spacy
nlp = spacy.load('ru_core_news_md')
from nltk import word_tokenize
from nltk.util import ngrams

data = pn.read_csv('test_data.csv')
data['insight'] = ''

# разделим реплики из разных диалогов
dialogs = list()
for dlg_id in data['dlg_id'].unique():
    dialog = data[data['dlg_id'] == dlg_id]
    manager_part = dialog[dialog['role'] == 'manager']
    client_part = dialog[dialog['role'] == 'client']
    dialogs.append((manager_part, client_part))

'''
предположим, что менеджер представляется, как:
1. меня зовут <ИМЯ>
2. это <ИМЯ>
...
заметим, что порядок слов не так важен

также предположм, что менеджер представляет коммпанию, как:
1. компания <КОМПАНИЯ>

рассмотрев биграмы, триграмы, ngrams реплик можно выделить совпадающие с предположениями
'''

intro_ngrams = {
    'bigrams': [
        [
            'это'
            # <ИМЯ>
        ],
        [
            'я'
            # <ИМЯ>
        ]
        # и т.п.
    ],
    'trigrams': [
        [
            'меня', 
            'зовут'
            # <ИМЯ>
        ],
        [
            'моё',
            'имя'
            # <ИМЯ>
        ]
        # и т.д.
    ]
    # <NGRAMS> : [
    #   [
    #       ...
    #   ],
    #   ...
    # ]
}

# регулярное выражение под которое попадают компании
company_re = '(?i)компания(.*бизнес)'

# так как грамматика в репликах не соблюдена, ngrams с n>1 могут дать неверный NER результат
def isName(word: str) -> bool:
    # name entity recognition
    name_entity = nlp(word).ents
    if name_entity:
        # если entity типа "человек", то это имя
        if name_entity[0].label_ == 'PER':
            return True
    return False
    
# представился ли человек и как его зовут
def ngramRelevance(sentence: list, n: int, intros: list) -> tuple:
    # отберем ngrams в которых слова совпадают с 
    # предположением о том, как менеджер может представиться
    relevant_ngrams = [ngram for ngram in list(ngrams(sentence, n)) 
        if all([intro in ngram for intro in intros])]
    if relevant_ngrams:
        for ngram in relevant_ngrams:
            word_list = list(ngram)
            for intro_word in intros:
                word_list.remove(intro_word)
            is_name = isName(word_list[0])
            if is_name:
                return True, word_list[0]
    return False, None

def findIntroductionsInDF(df: pn.DataFrame) -> list:
    # определим, в каких случаях менеджер представил себя
    introductions_info = list()
    for sentence, idx in zip(df, df.index.to_list()):
        # пройтись по всем ngrams из intro_ngrams
        for i, key in enumerate(intro_ngrams.keys()):
            for ngram in intro_ngrams[key]:
                ngram_relevance, name = ngramRelevance(sentence, i+2, ngram)
                if ngram_relevance:
                    introductions_info.append((idx, name))
                    break # уже представился
    return introductions_info if introductions_info else [(None, None)]

meta_info = list()
for dialog in dialogs:
    # найдем представление и имя
    manager_id = dialog[0]['dlg_id'].unique()[0]
    manager_sentences = dialog[0]['text'].transform(str.lower).transform(word_tokenize)
    manager_introduced = findIntroductionsInDF(manager_sentences)
    # найдем название компании
    matches = dialog[0]['text'].str.extract(company_re).dropna()
    business_name = None
    if len(matches) > 0:
        business_name = matches.iloc[0][0]
    meta_info.append((manager_id, business_name, manager_introduced))

# находит совпадения с регулярными выражениями в датафрейме
def findMatchesInDF(df: pn.DataFrame, regexps: list) -> pn.DataFrame:
    matches_df = pn.DataFrame()
    for regexp in regexps:
        matches = df[df['text'].str.contains(regexp, regex=True)]
        matches_df = pn.concat([matches_df, matches])
    return matches_df

# регулярные выражения приветствия и прощания
greetings_re = [
    '(?i).*здравствуйте.*', 
    '(?i).*добр(?:ый день|ое утро|ый вечер).*'
]
goodbyes_re = [
    '(?i).*до свидания.*', 
    '(?i).*всего доброго.*'
]
# хранит id менеджеров, которые поздоровались и попрощались
polite_managers = list()
# находит приветствия и прощания
for dialog in dialogs:
    manager_sentences = dialog[0]
    greetings_df = findMatchesInDF(manager_sentences, greetings_re)
    goodbyes_df = findMatchesInDF(manager_sentences, goodbyes_re)
    greeting_ids = greetings_df.index.to_list()
    goodbye_ids = goodbyes_df.index.to_list()
    if greeting_ids and goodbye_ids:
        manager_id = manager_sentences['dlg_id'].unique()[0]
        polite_managers.append(manager_id)
    for greeting_id in greeting_ids:
        data.loc[greeting_id, ['insight']] += 'greeting=True;'
    for goodbye_id in goodbye_ids:
        data.loc[goodbye_id, ['insight']] += 'goodbye=True;'

# вспомогательный csv с доп. информацией 
meta_data = pn.DataFrame(columns=['dlg_id', 'manager_name', 'company_name', 'manager_polite'])
for info in meta_info:
    dlg_id = info[0]
    row = pn.DataFrame({
        'dlg_id': [dlg_id], 
        'company_name': [info[1]],
        'manager_name': [info[2][0][1]],
        'manager_polite': [dlg_id in polite_managers]
        })
    meta_data = pn.concat([meta_data, row], sort=False)
    # добавим флаг на представление в реплике
    id_ = info[2][0][0]
    if id_:
        data.loc[id_, 'insight'] += 'saidName=True;'

# запишем результаты в локальные файлы
meta_data.to_csv('./meta-info.csv')
data.to_csv('./test-data-res.csv')