from lct21_model import logs_preproc, news_preproc, train_test_split_dt, lightFM_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import streamlit as st

st.set_page_config(
    page_title="ЛЦТ-2021: Рекомендательная система новостей mos.ru",
    layout='wide',
    initial_sidebar_state='auto',
)

sidebar_selection = st.sidebar.radio(
    'Выберите режим работы:',
    ['Выберите режим работы', 'Демонстрация работы модели', 'Ответ на тестовое задание'],
)

# @st.cache(ttl=3*60*60, suppress_st_warning=True)
def get_data():
    FILE_LOGS = 'data/dataset_news_1.xlsx'
    FILE_NEWS =  'data/news.jsn'
    df_logs = logs_preproc(st.cache(pd.read_excel)(FILE_LOGS))
    df_news = news_preproc(st.cache(pd.read_json)(FILE_NEWS))
    return df_logs, df_news

st.title("ЛИДЕРЫ ЦИФРОВОЙ ТРАНСФОРМАЦИИ 2021 // ХАКАТОН")
st.write("Команда DST-OFF")

df_logs, df_news = get_data()

if sidebar_selection == 'Выберите режим работы':
    st.subheader("ДЕМОНСТРАЦИОННЫЙ СТЕНД")
    st.info("Разработка рекомендательной системы новостей для посетителей mos.ru")
    st.text("В меню слева выберите режим работы демонстрационного стенда")
    st.text("\nДополнительные материалы, разметка новостей и описание работы приведены в ноутбуках")

elif sidebar_selection == 'Демонстрация работы модели':

    with st.expander('1. Загрузка данных'):
        st.write(f"Логи загружены. Всего загружены сведения о {len(df_logs)} просмотрах.")
        st.dataframe(df_logs.sample(50))
        st.write(f"Новости загружены. Всего загружено {len(df_news)} новости.")
        st.dataframe(df_news.sample(50).astype(str))

    st.write("Выберите id пользователя для демонстрации:")
    demo_user_id = st.selectbox('id пользователя', df_logs.user_id.unique())

    demo_logs = (
        df_logs[df_logs.user_id==demo_user_id].sort_values('date_time')
        .merge(df_news[['published_at','title','preview_text','full_text']], left_on='news_id', right_index=True)
    )

    demo_views_count =  len(demo_logs)

    st.write(f"У пользователя c id={demo_user_id} всего в августе было {demo_views_count} просмотров:")
    st.write(f"Выделим для демонстрации {demo_views_count//20} интервалов.")

    demo_cursor = demo_views_count%20
    if demo_views_count==0:
        demo_cursor = 20

    prev_demo_cursor =0

    while demo_cursor <= demo_views_count:
        last_date_demo_cursor = demo_logs.iloc[demo_cursor-1].date_time
        with st.expander(f"{demo_cursor - prev_demo_cursor} просмотров с {demo_logs.iloc[prev_demo_cursor].date_time} "
                         f"по {last_date_demo_cursor}"):
            st.dataframe(demo_logs.iloc[prev_demo_cursor:demo_cursor][['date_time','url_clean','published_at','title']])

            # Выделим тренировочный датасет строго по датам до точки отсечки - до конце интервала
            data_train, data_test, news_list, user_list, news_id_to_x, user_id_to_y = train_test_split_dt(
                df_logs, last_date_demo_cursor)

            st.write('')

            # Рассчитаем ранги классического алгоритма рекомендательной системы на основе алгоритма матричной факторизации
            model = lightFM_fit(data_train)
            scores = model.predict(int(user_id_to_y[demo_user_id]), np.arange(len(news_list)))

            # Рассчитаем ранги классического алгоритма рекомендательной системы на основе алгоритма матричной факторизации
            df_submission = pd.DataFrame(scores).reset_index().rename(columns={0:'score'}).sort_values('score', ascending=False)
            df_submission['news_id'] = df_submission['index'].apply(lambda x:news_list[x]).astype(int)
            df_submission = df_submission.merge(df_news[['published_at','title','url']], left_on='news_id', right_index=True)
            df_submission = df_submission[['news_id','url','published_at','title', 'score']]

            # Поощряем новости, с даты публикации которых прошло не более 2 дней, добавляем к скору +3
            df_submission.score = (df_submission.score.mask(df_submission.published_at.between(
                last_date_demo_cursor-dt.timedelta(days=2), last_date_demo_cursor+dt.timedelta(days=7)), df_submission.score+1.5))

            # Дополнительно поощряем самые свежие новости, добавляем к скору еще +3
            df_submission.score = (df_submission.score.mask(df_submission.published_at.between(
                last_date_demo_cursor-dt.timedelta(days=0), last_date_demo_cursor+dt.timedelta(days=3)), df_submission.score+1.5))

            # Устанавливаем запретительный штраф (-100) по ранее просмотренные новости
            df_submission.score = (df_submission.score.mask(df_submission.news_id.isin(
                df_logs[(df_logs.user_id==demo_user_id)&(df_logs.date_time<=last_date_demo_cursor)].news_id), df_submission.score-100))
            df_submission.sort_values('score', ascending=False, inplace=True)

            st.write(f"Предсказание на период после {last_date_demo_cursor}")
            st.dataframe(df_submission.head(20))

        prev_demo_cursor = demo_cursor
        demo_cursor +=20

    st.info("Последнее предсказание может являться ответом на тестовое задание для пользователя с указанным id."
            "Мы рекомендуем 20 новостей на момент последней записи в логах для каждого пользователя."
            "Разница заключается в том, что при домонстрации модель обучается строго по дату последней деятельности пользователя в логе,"
            "а при формировании общего ответа мы обучаем модель на полном объеме данных - на всем логе.")

elif sidebar_selection == 'Ответ на тестовое задание':
    with st.expander('1. Загрузка данных'):
        st.write(f"Логи загружены. Всего загружены сведения о {len(df_logs)} просмотрах.")
        st.dataframe(df_logs.sample(50))
        st.write(f"Новости загружены. Всего загружено {len(df_news)} новости.")
        st.dataframe(df_news.sample(50).astype(str))

    # Выделим тренировочный датасет на последнюю дату лога
    data_train, data_test, news_list, user_list, news_id_to_x, user_id_to_y = train_test_split_dt(
        df_logs, df_logs.date_time.max())

    df_logs.date_time.max()

    # Рассчитаем ранги классического алгоритма рекомендательной системы на основе алгоритма матричной факторизации
    model = lightFM_fit(data_train)

    list_submission = []

    for u_id in user_list:

        scores = model.predict(int(user_id_to_y[u_id]), np.arange(len(news_list)))

        # Рассчитаем ранги классического алгоритма рекомендательной системы на основе алгоритма матричной факторизации
        df_submission = pd.DataFrame(scores).reset_index().rename(columns={0:'score'}).sort_values('score', ascending=False)
        df_submission['news_id'] = df_submission['index'].apply(lambda x:news_list[x]).astype(int)
        df_submission = df_submission.merge(df_news[['published_at','title','url']], left_on='news_id', right_index=True)
        df_submission = df_submission[['news_id','url','published_at','title', 'score']]

        # Последняя дата в логе для пользователя
        last_date_user_log = df_logs[df_logs.user_id==u_id].date_time.max()

        # Поощряем новости, с даты публикации которых прошло не более 2 дней, добавляем к скору +3
        df_submission.score = (df_submission.score.mask(df_submission.published_at.between(
            last_date_user_log-dt.timedelta(days=2), last_date_user_log+dt.timedelta(days=7)), df_submission.score+1.5))

        # Дополнительно поощряем самые свежие новости, добавляем к скору еще +3
        df_submission.score = (df_submission.score.mask(df_submission.published_at.between(
            last_date_user_log-dt.timedelta(days=0), last_date_user_log+dt.timedelta(days=3)), df_submission.score+1.5))

        # Устанавливаем запретительный штраф (-100) по ранее просмотренные новости
        df_submission.score = (df_submission.score.mask(df_submission.news_id.isin(
            df_logs[df_logs.user_id==u_id].news_id), df_submission.score-100))

        df_submission.sort_values('score', ascending=False, inplace=True)

        list_submission.append([u_id]+df_submission.head(20).news_id.to_list())

    df_predict = pd.DataFrame(list_submission, columns=['user_id']+[f"book_id_{i+1}" for i in range(20)])
    st.dataframe(df_predict)

    @st.cache
    def convert_df(df):
        return df.to_csv(sep=';', index=False).encode('utf-8')

    csv = convert_df(df_predict)

    st.download_button(
        label="Скачать CSV",
        data=csv,
        file_name='submissions.csv',
        mime='text/csv',
    )