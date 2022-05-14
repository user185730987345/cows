import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm

with st.echo(code_location='below'):

    df = pd.read_csv('mammals.csv')
    df = df.dropna()
    df = df.reset_index(drop=True)

    df.columns = ['Species', 'Body Weight (kg)', 'Brain Weight (kg)', 'Non dreaming sleep (hr)', 'Dreaming sleep (hr)',
           'Total sleep (hr)', 'Life span (yr)', 'Gestation', 'Predation', 'Exposure',
           'Danger']

    df['Brain Weight (kg)'] = df['Brain Weight (kg)']/1000
    df['Brain / Body'] = df['Brain Weight (kg)']/df['Body Weight (kg)']
    df['Dreaming / Total sleep'] = df['Dreaming sleep (hr)']/df['Total sleep (hr)']

    st.title('У этого исследования нет, названия')
    st.header('И смысла')
    st.subheader('Но зато оно веселое и красивое, а автор потренировался питонить')

    status = st.radio("Выбери одно", ('Я бы хотел знать, какую часть времени сна корова мечтает', 'Я бы не хотел это узнать'))
    if (status == 'Я бы хотел знать, какую часть времени сна корова мечтает'):
        st.success("Супер! Жми на кнопку снизу")
    else:
        st.error("Зануда! Ну ладно, жми на кнопку снизу")

    if(st.button("Какая корова? О чем мечтает?")):
        st.text("Внизу кликер, ищи корову.")

    choice = st.selectbox("Животное", df['Species'].iloc[:10])
    if choice=='Cow':
        st.success('Все верно, cow это корова')
    st.write('Ты выбрал', choice, 'давай посмотрим в таблице сколько она мечтает во сне часов в день')
    st.write(df[df['Species']==choice])

    """
    Ого, корова спит только четыре часа. При этом мечтает всего 40 минут в день. Интересно, а как у других животных и есть ли связь с объемом мозга
    """

    fig = px.bar(df,
                 x='Species',
                 y='Brain / Body',
                 color='Dreaming / Total sleep',
                 title='У каких животных сколько в весе занимает мозг и сколько они мечтают от времени сна')
    st.plotly_chart(fig)

    """
    Если ты был внимателен, то узнал, что Еж большой мечтатель, а Ехидна наооборот почти не мечтает.
    """

    df1 = df.sort_values('Brain / Body')

    fig = px.scatter(df1,
                     x='Brain / Body',
                     y='Dreaming / Total sleep',
                     opacity=0.65,
                     trendline='ols',
                     trendline_color_override='yellow',
                     title='Как связаны относительные мозги и относительные мечтания?')
    st.plotly_chart(fig)

    """
    Как видно, чем умнее, тем меньше мечтает. 
    Дисклеймер: не воспринимайте серьезно.
    
    Давай посмотрим, может зависимость пропадет, если взять абсолютное количество мозгов?
    """

    df2 = df.sort_values('Brain Weight (kg)')
    df2 = df2[df2['Species']!='Asianelephant']

    fig = px.scatter(df2,
                     x='Brain Weight (kg)',
                     y='Dreaming / Total sleep',
                     opacity=0.65,
                     trendline='ols',
                     trendline_color_override='purple',
                     title='Как связаны абсолютные мозги и относительные мечтания?',
                     log_x=True)
    st.plotly_chart(fig)

    """
    Как видно, есть положительная зависимость, так что и переживать не о чем.
    
    Бонус: как связаны мозг и сон?
    """

    fig = px.scatter(df2,
                     x='Brain Weight (kg)',
                     y='Total sleep (hr)',
                     size='Brain / Body',
                     opacity=1,
                     trendline='ols',
                     trendline_color_override='blue',
                     title='Может больше спать поможет? Чем больше точка, тем больше весит мозг в теле',
                     log_x=True)
    st.plotly_chart(fig)

    """
    Как видно, автор проекта много спал)
    """

    if st.checkbox('Понравился проект?'):
        st.text('Я старался')
