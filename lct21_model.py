import re
import nltk
from bs4 import BeautifulSoup
from pymorphy2 import MorphAnalyzer
from wordcloud import WordCloud

from lightfm import LightFM
from lightfm.evaluation import precision_at_k

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from pathlib import Path
DATA_DIR = Path('data')

def logs_preproc(df_input):
    """
    Очистка и предобработка датасета логов df_input
    """
    df = df_input.copy()

    # Извлечем news_id
    df['news_id'] = df.url_clean.str.extract(r".*/(\d*(?:050|073))/.*")
    # Удалим записи с неидентифицированными ссылками
    df.dropna(inplace=True)

    df.news_id = df.news_id.astype(int)

    # Извлечем дополнительную информацию из временной метки
    df['hour_sid'] = (df.date_time.sub(df.date_time.min()).dt.total_seconds() // 3600).astype(int)  # час от 0 до 23
    df['date'] = df.date_time.dt.date
    df['week_sid'] = df.date_time.dt.isocalendar().week
    df['weekday_sid'] = df.date_time.dt.weekday
    df['hour'] = df.date_time.dt.hour

    # Зададим вес по умолчанию для каждого фактического просмотра
    df['event_weight'] = 1.

    return(df.sort_values(['user_id', 'date_time']))

def news_preproc(df_input):
    """
    Очистка и предобработка датасета новостей df_input
    """
    df_json = df_input.copy()
    df_json['tag_ids'] = df_json.tags.apply(lambda x:[item['id'] for item in x])
    df_json['sphere_id'] = df_json.sphere.apply(lambda x:x['id'])
    df_json['sphere_ids'] = df_json.spheres.apply(lambda x:[item['id'] for item in x])
    df_json['void'] = ''
    df_json.void = df_json.void.str.split()
    df_json.organizations.mask(df_json.organizations.isna(), df_json.void, inplace=True)
    df_json['organization_ids'] = df_json.organizations

    # Заполняем пропуски превью и текста, подставляя замещающие столбцы
    df_json.preview_text.mask(df_json.preview_text.isna(), df_json.preview, inplace=True)
    df_json.full_text.mask(df_json.full_text.isna(), df_json.text, inplace=True)

    df_json.theme_id = df_json.theme_id.fillna(0).astype(int)
    df_json.territory_area_id = df_json.territory_area_id.fillna(0).astype(int)
    df_json.territory_district_id = df_json.territory_district_id.fillna(0).astype(int)
    df_json.oiv_id = df_json.oiv_id.fillna(0).astype(int)
    df_json.label = df_json.label.fillna('')
    cols = ['id', 'title', 'url', 'published_at', 
            'label', 'tag_ids', 
            'theme_id', 'theme_ids', 
            'sphere_id', 'sphere_ids', 
            'territory_area_id', 'territory_district_id', 
            'preview_text', 'full_text', 
            'oiv_id', 'organization_ids']

    return df_json[cols].set_index('id')


def load_organizations():
    """
    Загрузка организаций. Источник - mos.ru/api, выявлен путем изучения сайта mos.ru
    """
    if not (DATA_DIR/'organizations.csv').exists():
        list_organizations = []
        href = 'https://www.mos.ru/api/structure/v1/frontend/json/ru/institutions?per-page=50'
        while True:
            json_page = json.load(urlopen(href))
            list_organizations += json_page['items']
            print(f"Загружено {len(list_organizations)} организаций"+" "*10, end='\r')
            if href != json_page['_links']['last']['href']:
                href = json_page['_links']['next']['href']
            else:
                break
        # Список организаций/департаменов сохраняем в таблицу df_organizations
        df_organizations = (
            pd.DataFrame(list_organizations)
            .drop(columns=['yandex_metrika_id','edo_id','rich_snippet','external_card_title','reception_description','icon'])
            .set_index('id')
        )
        df_organizations.lead_institution_id = df_organizations.lead_institution_id.fillna(0).astype(int)
        df_organizations.to_csv(DATA_DIR/'organizations.csv')
    else:
        df_organizations = pd.read_csv(DATA_DIR/'organizations.csv', index_col='id')

    return df_organizations


def load_areas():
    """
    Загрузка округов и районов. Источник - mos.ru/api, ссылка извлечена из файла 'areas + districts.json', 
    предоставленного организаторами
    """
    if not (DATA_DIR/'areas.csv').exists():
        href = 'https://www.mos.ru/api/directories/v2/frontend/json/territory/districts?expand=areas&per-page=50&page=1'
        json_page = json.load(urlopen(href))
        s_areas_districts = pd.DataFrame(json_page['items']).set_index(['id', 'title']).areas
        df_areas = (
            s_areas_districts[s_areas_districts.str.len()>0]
            .explode()
            .transform({'area_id': lambda x:x['id'], 'area_title': lambda x:x['title']})
            .reset_index()
            .set_index('area_id')
            .rename(columns={'id': 'district_id', 'title': 'district_title'})
        )
        df_areas.to_csv(DATA_DIR/'areas.csv')
    else:
        df_areas = pd.read_csv(DATA_DIR/'areas.csv', index_col='area_id')

    return df_areas


def train_test_split_dt(df_input, date_time_last):
    """
    Формирование тренировочного датасета df_train и тестовой выборки df_test c отсечкой по времени
    """
    news_list = df_input.news_id.unique()
    news_id_to_x = dict(zip(news_list, np.arange(len(news_list))))
    
    user_list = df_input.user_id.unique()
    user_id_to_y = dict(zip(user_list, np.arange(len(user_list))))

    # Разделим датасет лога на тренировочный и тестовый
    df_train = df_input[df_input.date_time<=date_time_last]
    df_test = df_input[df_input.date_time>date_time_last]
    print(f"Размер тестовой выборки {df_test.shape}")
    
    data_train = csr_matrix((np.ones(len(df_train)), (df_train.news_id.map(news_id_to_x), df_train.user_id.map(user_id_to_y))), shape=(len(news_list), len(user_list))).transpose()
    data_test = csr_matrix((np.ones(len(df_test)), (df_test.news_id.map(news_id_to_x), df_test.user_id.map(user_id_to_y))), shape=(len(news_list), len(user_list))).transpose()

    return data_train, data_test, news_list, user_list, news_id_to_x, user_id_to_y

def lightFM_fit(df_input):
    model = LightFM(learning_rate=0.05, loss='bpr')
    model.fit(df_input, epochs=50)
    return model

