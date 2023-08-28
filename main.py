import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


df= pd.read_csv('./weatherAUS.csv')                                               #Variable que lee y almacena el archivo como dataframe

st.title("Modelo para pronosticar el clima en Australia 🏄‍♂️🦘🐊")
st.image('./imagen.png')

city = ('Melbourne', 'Sydney','Albury','Adelaide')                                 #opciones de ciudades para el usuario
selected_city = st.selectbox("Selecciona una ciudad de interés", city)             #Se almacena la ciudad elegida con una función de st para seleccionar de Acción

n_years = st.slider("Años a predecir: ", 1, 4)
period = n_years * 365

@st.cache_data                                                                      # usa la cache para no tener que volver a cargar esto al cambiar de acción
def cargar_datos(ciudad):
    """Función para cargar datos
    var data = guarda los datos que se descargan desde yfinance, ticker es lo que entre, star el dia que inicia y today la fecha actual"""
    city = df[df['Location'] == f"{ciudad}"]                                         #creación de dataframe con la ciudad elegida
    city['Date'] = pd.to_datetime(city['Date'])                                      #parsean los datos a tipo datetime de la columna "Date"
    city.reset_index(inplace=True)                                                   #Resetea el indice y pone los dias encambio
    city['Year'] = city['Date'].apply(lambda x: x.year)                              #Se crea una nueva columna al filtrar las fechas menores a 2015
    city = city[city['Year'] < 2015]                                                 # Se crea otro df a partir de las ciudades que cumplan la condicion de años
    return city

data_load_state = st.text("Cargando datos...")
city = cargar_datos(selected_city)
data_load_state.text("Cargando datos... Hecho!")

st.subheader('Histórico')
st.write(city.tail())

def plot_historico():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=city['Date'],y=city['Temp9am'], name= 'Temperatura a las 9am'))
    fig.add_trace(go.Scatter(x=city['Date'], y=city['Temp3pm'], name='Temperatura a las 3pm'))
    fig.layout.update(title_text="Datos serie de tiempo",xaxis_rangeslider_visible=True, xaxis_title = 'Fecha', yaxis_title = "Temperatura")
    st.plotly_chart(fig)
plot_historico()
def forecast():
    """
        Se encarga de entrenar el modelo prophet y ejecutarlo, posteriormente graficará la predicción según los años escogidos
        y también los componente de temporalidad
    """
    #Forecasting
    df_train = city[['Date','Temp3pm']]                                                #se crea el dataframe con las columnas date y temp3pm
    df_train.dropna(inplace= True)                                                     #Limpieza de datos
    df_train.rename(columns = {"Date":"ds","Temp3pm":"y"}, inplace = True)             #Renombrar las columnas porque asi lo requiere el prophet

    m = Prophet(daily_seasonality= True)                                                #Crear instancia
    m.fit(df_train)                                                                     #Entrenamiento del modelo
    future = m.make_future_dataframe(periods=period)                                    #Se crea el dataframe con fechas futuras// periods se refiere a los periodos ingresados en n_years
    forecast = m.predict(future)                                                        #Se hace el forecast

    st.subheader('Forecast Data')
    st.write(forecast.tail())

    """tabs en donde se da una breve explicación de que significa lo que arroja el forecast"""
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["ds", "trend", "yhat","additive","daily/weekly/yearly"])


    tab1.subheader("ds")
    tab1.write("Fecha a la que se refiere la predicción")

    tab2.subheader("trend")
    tab2.write("Tendencia estimada para la fecha")

    tab3.subheader("yhat")
    tab3.write("Valor pronosticado para la fecha")
    tab3.write('yhat_lower/Upper: Valor inferior/superior del intervalo de confianza de la predicción')

    tab4.subheader("additive")
    tab4.write("Componente que representa la estacionalidad semanal y anual si hay")

    tab5.subheader("daily/weekly/yearly")
    tab5.write("Componentes de estacionalidad diaria-semanal-anual")

    fig1 = plot_plotly(m,forecast)                                        #Figura 1 en donde se observa el forecast a través de los años
    st.plotly_chart(fig1)

    st.write('forecast components')
    fig2 = m.plot_components((forecast))                                  #Figura 2 en donde se observa los componentes de temporalidad
    st.write(fig2)

st.title("Pronosticar")
result = st.button("Haz click acá!")                                        #Variable que almacena el estado del botón
if result:
    forecast()