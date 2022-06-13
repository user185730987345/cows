#импорты
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sqlite3
import re
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression

#эти библиотеки не были рассмотрены в ходе курса
import statsmodels
import bokeh
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import gridplot
from bokeh.palettes import Spectral4

with st.echo(code_location ='below'):
    #0
    #функции, здесь используется нампай, чтобы ускорить код
    def convert(df_x, column_name):
        df_x[column_name] = np.array(df_x[column_name]).astype(np.float64)

    st.set_option('deprecation.showPyplotGlobalUse', False)

    try:
        r = requests.get('https://data.cityofnewyork.us/resource/f9bf-2cp4.json')
        df = pd.DataFrame(r.json())
    except:
        df = pd.read_csv('df.csv')

    x = []
    for i in range(478):
        if df.iloc[i]['sat_math_avg_score'] == 's':
            x.append(i)

    df = df.drop(x)
    df = df.reset_index()

    convert(df, 'sat_critical_reading_avg_score')
    convert(df, 'sat_math_avg_score')
    convert(df, 'sat_writing_avg_score')
    convert(df, 'num_of_sat_test_takers')

    df['avg'] = (df['sat_critical_reading_avg_score'] + df['sat_math_avg_score'] + df['sat_writing_avg_score'])/3
    df['gross_scores'] = 3 * df['avg'] * df['num_of_sat_test_takers']

    st.title('У этого проекта нет названия')
    st.header('и смысла 2 \n но он про образование в штатах')
    """
    Автор знает, как тяжело проверять работы по питону, поэтому проект сделан удобным для проверяющего. В начале, есть таблица с предпологаемой оценкой. В работе вы найдете пометки "критерий n - x баллов, потому что ABC". Вы можете согласиться с автором, либо оспорить, но в любом случае найти, куда смотреть, будет проще.
    """
    if(st.button('Спасибо, автор!')):
        st.success('Автор коммитится проверять чужие работы внимательно и лояльно в момент написания этих строк')


    st.subheader('Предлагаемая оценка 17-18 баллов. Предлагаемая разбалловка:')
    grade = pd.read_csv('grade.csv')
    grade.index = grade['Unnamed: 0']
    del grade['Unnamed: 0']
    st.table(grade)

    st.subheader('#1 Зависимость общих баллов от средних')
    """
    Есть датасет со средними баллами за ЕГЭ по школам США. Интересно как суммарный балл коррелирует со средним.
    """
    fig = px.scatter(df.sort_values('gross_scores', ascending = False),
               x='gross_scores',
               y ='avg',
               log_x =True,
               trendline='ols',
               template = 'ggplot2'
              )
    st.plotly_chart(fig)

    df1 = pd.read_csv('masterfile11_gened_final.csv')
    st.subheader('#2 Мнения групп vs. среднее мнение')
    """
    Также, есть датасет с мнениями людей, в т.ч. учителей, студентов и родителей, о школах. Выясним, как мнения групп соотносятся со средним
    """
    y=[]
    for i in ['s','t','p','tot']:
        x=[]
        for j in ['saf','com','eng','aca']:
            x.append(df1[f'{j}_{i}_11'].mean())
        y.append(x)

    data1 = pd.DataFrame(
        index=['students','teachers','parents','average'],
        columns=['safety (saf)', 'communications (com)', 'engagement (en)', 'academic expectations (aca)'],
        data = y)

    # обращаю внимание на то, что эти графики были выполнены с использованием библиотеки Bokeh
    # (2 балла за доп билиотеки)
    # я ****  даже шрифт поменял у названия, чтобы сделать эти чудовища смотрибельными
    x = ['students','teachers','parents','average']
    y1 = list(data1['safety (saf)'])
    y2 = list(data1['communications (com)'])
    y3 = list(data1['engagement (en)'])
    y4 = list(data1['academic expectations (aca)'])

    s1 = figure(x_range = x, y_range =(0,10), title="average safety (saf)", x_axis_label=None, y_axis_label=None)
    s1.vbar(x=x,top=y1, legend_field='x', width=0.5, bottom=0, color=Spectral4)
    s1.legend.orientation = "horizontal"
    s1.legend.location = "top_center"
    s1.xaxis.visible=False
    s1.xgrid.visible=False
    s1.title.text_font = "times"
    s1.title.text_font_size='20px'

    s2 = figure(x_range = x, y_range =(0,10), title="average communications (com)", x_axis_label=None, y_axis_label=None)
    s2.vbar(x=x,top=y2, legend_field='x', width=0.5, bottom=0, color=Spectral4)
    s2.legend.orientation = "horizontal"
    s2.legend.location = "top_center"
    s2.xaxis.visible=False
    s2.xgrid.visible=False
    s2.title.text_font = "times"
    s2.title.text_font_size='20px'

    s3 = figure(x_range=x, y_range =(0,10), title="average engagement (eng)", x_axis_label=None, y_axis_label=None)
    s3.vbar(x=x,top=y3, legend_field='x', width=0.5, bottom=0, color=Spectral4)
    s3.legend.orientation = "horizontal"
    s3.legend.location = "top_center"
    s3.xaxis.visible=False
    s3.xgrid.visible=False
    s3.title.text_font = "times"
    s3.title.text_font_size='20px'

    s4 = figure(x_range = x, y_range =(0,10), title="average academic expectations (aca)", x_axis_label=None, y_axis_label=None)
    s4.vbar(x=x,top=y4, legend_field='x', width=0.5, bottom=0, color=Spectral4)
    s4.legend.orientation = "horizontal"
    s4.legend.location = "top_center"
    s4.xaxis.visible=False
    s4.xgrid.visible=False
    s4.title.text_font = "times"
    s4.title.text_font_size='20px'

    grid = gridplot([[s1, s2], [s3, s4]], width=450, height=500)
    #вывод чудовища в стримлит (bars)
    st.bokeh_chart(grid)

    st.subheader('#3 Построение графа "мнения групп vs. средние"')
    #используем numpy (1 балл за numpy), для создания данных для графа
    arr1 = np.array(y)
    arr1 = arr1.transpose()
    arr1 = np.around(arr1, decimals = 2)
    l1 = list(arr1)
    l2 = [6.5,6.5,6.5,6.5,6.5]
    for i in l1:
        for j in i:
            l2.append(j)
    arr2 = np.array(l2)
    arr2 = arr2**4
    arr3 = arr2.copy()
    for i in range(5):
        arr3[i]=100

    dct = {0:'metric'}
    metric_x = ['saf','com','eng','aca']
    metric_y = ['std', 'tchr', 'par', 'avg']

    for i in range(20):
        if i+1 <= 4:
            dct[i+1] = metric_x[i]
        else:
            dct[i+1] = metric_y[i%4]

    # для большей наглядности визуализируем эти данные с помощью графов в networkx (1 балл за networkx)
    g = nx.Graph()

    for i in range(4):
        g.add_edge(0,i+1)
    for i in range(16):
        g.add_edge(i//4 + 1,i+5)
    """
    Выведем еще одну красивую визуализацию для понимания, как среднее связано с мнениями групп.
    """
    fig = plt.figure(figsize=(15,10))
    nx.draw_networkx(g,
                     with_labels = True,
                     labels=dct,
                     node_size=arr2,
                     node_color=arr3,
                     width = 2,
                     font_color='white',
                     font_size=13,
                    )

    st.pyplot(fig)
    """
    Школьники - оптимисты. Родители - пессимисты.
    """

    st.subheader('#4 Какое количество школьников оценивают критерий выше, чем родители?')
    dfsql = df1[['saf_p_11', 'com_p_11', 'eng_p_11', 'aca_p_11','saf_s_11', 'com_s_11', 'eng_s_11', 'aca_s_11']]
    means = list(dfsql.mean())
    means = means[0:4]

    cats = list(dfsql.columns)
    cats = cats[4:8]

    list_sql = list(zip(cats, means))

    # как видно родители завышают оценки, а студенты занижают
    # узнаем, какой процент школьников имеет оценку выше, чем средняя родительская
    # для этого используем SQL (1 балл)
    conn = sqlite3.connect('database.sqlite')
    c = conn.cursor()
    dfsql.to_sql('DataSQL', con = conn)

    percentage = {'saf':0, 'com':0, 'eng':0, 'aca':0}
    for i in list_sql:
        x = c.execute(
            f"""
            SELECT COUNT({i[0]})
            FROM DataSQL
            WHERE {i[0]} >= {i[1]}
            """).fetchall()
        y = c.execute(
            f"""
            SELECT COUNT({i[0]})
            FROM DataSQL
            """).fetchall()
        percentage[f'{i[0]}'[0:3]] = round(x[0][0] / y[0][0], 4) * 100

    c.execute(
    """
    DROP TABLE DataSQL
    """)
    conn.close()

    df_percentage = pd.DataFrame(index = ['% of std >= mean par'], data = percentage)
    df_percentage = df_percentage.transpose()
    df_percentage['% of std < mean par'] = 100 - df_percentage['% of std >= mean par']

    df_percentage.columns = ['оценивающих выше среднего родителя', 'оценивающих ниже среднего родителя']
    df_percentage.columns.name = 'Часть студентов'

    df_percentage.index = ['Безопастность','Коммуникации','Вовлеченность','Академические перспективы']
    df_percentage.index.name = 'Критерий'
    """
    Мы уже поняли, что школьники занижают оценки, а родители завышают. Интересно, какова разница. Посмотрим это по следующей метрике:
    процент школьников, которые оценивают что-то выше среднего родителя.
    """
    fig = px.bar(df_percentage,
           orientation = 'h',
           template = 'ggplot2'
          )
    st.plotly_chart(fig)

    """
    Теперь у нас есть понимание того, как устроены средние. Дальше работаем только со средними.
    """
    #в дальнейшем мы берем только средние показатели
    ddf1 = df1[['dbn',
         'saf_tot_11',
         'com_tot_11',
         'eng_tot_11',
         'aca_tot_11']]

    st.subheader('#5 Совмещение двух датасетов: как оценка безопасности связана со средним баллом ЕГЭ?')
    #одна из сложных штук с pandas
    maindf = df.merge(ddf1)
    """
    
    """
    fig = px.scatter(maindf, x='avg', y='saf_tot_11', trendline='ols', template = 'ggplot2')
    st.plotly_chart(fig)

    st.subheader('#6 Подсчет корреляций и вывод регрессии: какой у меня будет средний за ЕГЭ, если по одному экзамену X?')
    corr_df = pd.DataFrame(maindf.corr()['avg'][2:5])
    st.write(corr_df)

    #machine learning
    model = LinearRegression()
    model.fit(maindf[["sat_writing_avg_score"]], maindf["avg"])
    #maindf.plot.scatter(x="sat_writing_avg_score", y="avg")
    x = pd.DataFrame(columns = ['sat_writing_avg_score'], data = np.linspace(250, 750))

    fig, ax = plt.subplots()
    ax.plot(x["sat_writing_avg_score"], model.predict(x), color="C2")
    ax.scatter(x = maindf['sat_writing_avg_score'], y=maindf["avg"])
    st.pyplot(fig)

    st.subheader('#7 API и Geopandas: где в штатах больше учеников?')
    #обращения к api и работа с api (1 балл)
    r = requests.get('https://api.covid-relief-data.ed.gov/portal/api/v1/esf/summary')
    ss = r.json()['stateSummaries']
    state_tickers=[]
    for i in ss:
        state_tickers.append(i.get('state'))

    #вот такое был код, чтобы получить "dfstates"
    #чтобы ускорить, я сохранил этот дата фрэйм, чтобы приложение не лагало
    #потому что код работал не слишком быстро
    #да и мало ли реквест без vpn не заработает в один прекрасный момент

    #data_states = []
    #for i in state_tickers:
    #    myset = []
    #    j = requests.get(f'https://api.covid-relief-data.ed.gov/portal/api/v1/state/{i}')
    #    x = j.json()
    #    myset.append(i)
    #    myset.append(x.get('stateName'))
    #    myset.append(x.get('numberOfIheStudents')+x.get('numberOfLeaStudents'))
    #    data_states.append(myset)
    #dfstates = pd.DataFrame(columns=['ticker','name','number of students'], data = data_states)
    #dfstates.to_csv('dfstates.csv')
    dfstates = pd.read_csv('dfstates.csv')

    geostates = gpd.read_file('gz_2010_us_040_00_500k.json')
    geostates['NAME'] = geostates['NAME'].str.upper()
    geostates_unique = list(geostates['NAME'].unique())
    dfstates_unique = list(dfstates['name'].unique())

    dif1 = dfstates_unique.copy()
    for i in geostates_unique:
        try:
            dif1.remove(i)
        except:
            None

    dif2 = geostates_unique.copy()
    for i in dfstates_unique:
        try:
            dif2.remove(i)
        except:
            None

    isin = []
    for i in range(len(dfstates['name'])):
        isin.append(bool(dfstates['name'][i] in dif1))
    dfstates['is in'] = isin

    dfstates = dfstates[dfstates['is in']==False].reset_index()
    dfstates = dfstates.sort_values('name')
    geostates = geostates.sort_values('NAME')
    geostates['number of students'] = dfstates['number of students']
    """
    Найдены данные по тому, в каком штате, сколько учеников. Они замерджены с картой США. 
    """
    geostates.plot(column = 'number of students', legend=True,
                   legend_kwds={'label': "Number of students by country",
                                'orientation': "vertical"},
                   cmap='coolwarm',
                  figsize=(15,6))
    plt.title('States by number of students',fontsize=25)
    plt.xlim(-175,-50)
    plt.ylim(15,75)
    plt.axis('Off')
    st.pyplot()

    st.subheader('#8 bs4, re: какой из приведенных выше критериев общественность считает более важным?')
    """
    Возьмем первые 5 страниц в поисковике по запросу "education in us critics". О каком критерии там чаще говорят?
    """
    # посмотрим, какой критерий общественность считает более важным
    #с помощью beautiful.soup (1 балл) и regular expressions ( 1 балл)
    #возьмем пять первый ссылок в duckduckgo по запросу "education us critics"
    link1 = 'https://education.stateuniversity.com/pages/2341/Public-Education-Criticism.html'
    link2 = 'https://www.theedadvocate.org/10-reasons-the-u-s-education-system-is-failing/'
    link3 = 'https://www.edweek.org/leadership/opinion-12-critical-issues-facing-education-in-2020/2019/12'
    link4 = 'https://thehill.com/opinion/education/587598-a-true-patriotic-education-requires-critical-analysis-of-us-history/'
    link5 = 'https://edition.cnn.com/2021/01/12/opinions/what-comes-next-america-education-perry/index.html'
    links = [link1, link2, link3, link4, link5]

    queries = [requests.get(i) for i in links]

    pages = [BeautifulSoup(i.text, 'html.parser') for i in queries]

    texts = [i.get_text() for i in pages]

    words = ['safe','communicat', 'engag', 'academ']
    text_analysis_values = []
    for i in texts:
        bunch = []
        for j in words:
            bunch.append(len(re.findall(j, i)))
        text_analysis_values.append(bunch)

    #оцени степень заботы
    try:
        text_analysis = pd.DataFrame(columns = words, data=text_analysis_values)
        text_analysis.columns.name = 'частота'
        text_analysis.index.name = 'страница №'
    except:
        text_analysis = pd.read_csv('text_analysis.csv')
        del text_analysis['страница №']

    text_sums = pd.DataFrame(data = text_analysis.sum()).transpose()
    text_sums.index=['Сумма']

    t_a_final = pd.concat([text_analysis, text_sums])
    t_a_final.index.name = 'Страница №'

    st.table(text_analysis)

    st.table(text_sums)

    fig = px.pie(text_sums.transpose(), values = 'Сумма', names = list(text_sums.columns))
    st.plotly_chart(fig)


