
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly as plt
import plotly.graph_objects as go
import exchange_calendars as xcals
from datetime import date

def download_data(tickers, start_date='2001-01-01', end_date=date.today().strftime('%Y-%m-%d')):
    data = yf.download(tickers, start=start_date, end=end_date)

    return data['Close']

def calcular_fechas(hoy: pd.Timestamp):
    # Obtén el calendario de la bolsa de México
    xmex = xcals.get_calendar("XMEX")

    # Si el día de la semana es lunes (0 en el sistema Python weekday()), retrocede 3 días
    if hoy.weekday() == 0:
        prev_business_day = hoy - pd.Timedelta(days=3)
    # De lo contrario, solo retrocede un día
    else:
        prev_business_day = hoy - pd.Timedelta(days=1)

    # Si el día calculado no es un día hábil, busca el día hábil más reciente
    if not xmex.is_session(prev_business_day):
        prev_business_day = xmex.previous_close(prev_business_day).to_pydatetime()

    ayer = prev_business_day

    # Crear un diccionario para almacenar los resultados
    resultado = {}

    # Mes hasta la fecha
    primer_dia_mes = xmex.date_to_session(hoy.replace(day=1), direction="next")
    if hoy == primer_dia_mes:
        # Si hoy es el primer día hábil del mes, toma el primer día hábil del mes anterior
        mes_anterior = hoy - pd.DateOffset(months=1)
        primer_dia_mes = xmex.date_to_session(mes_anterior.replace(day=1), direction="next")

    # Calcula los días hábiles entre el primer día del mes y hoy
    dias_habiles = len(xmex.sessions_in_range(primer_dia_mes, hoy))+1

    # Usa estos días hábiles para obtener la ventana de sesiones
    mes_hasta_fecha = xmex.sessions_window(hoy, -dias_habiles)

    # Año hasta la fecha
    primer_dia_año = xmex.date_to_session(hoy.replace(month=1, day=1), direction="next")
    if hoy == primer_dia_año:
        # Si hoy es el primer día hábil del año, toma el primer día hábil del año anterior
        año_anterior = hoy - pd.DateOffset(years=1)
        primer_dia_año = xmex.date_to_session(año_anterior.replace(month=1, day=1), direction="next")

    # Calcula los días hábiles entre el primer día del año y hoy
    dias_habiles = len(xmex.sessions_in_range(primer_dia_año, hoy))+1

    # Usa estos días hábiles para obtener la ventana de sesiones
    año_hasta_fecha = xmex.sessions_window(hoy, -dias_habiles)

    # Fecha de hace un mes
    hace_un_mes = hoy - pd.DateOffset(months=1)

    # Encuentra el día hábil más cercano en el pasado a hace_un_mes
    dia_habil_hace_un_mes = xmex.date_to_session(hace_un_mes, direction="previous")

    # Obtén todas las sesiones desde hace_un_mes hasta hoy
    ultimos_30_dias = xmex.sessions_in_range(dia_habil_hace_un_mes, hoy)

    # Fecha de hace tres meses
    hace_tres_meses = hoy - pd.DateOffset(months=3)

    # Encuentra el día hábil más cercano en el pasado a hace_tres_meses
    dia_habil_hace_tres_meses = xmex.date_to_session(hace_tres_meses, direction="previous")

    # Obtén todas las sesiones desde hace_tres_meses hasta hoy
    ultimos_90_dias = xmex.sessions_in_range(dia_habil_hace_tres_meses, hoy)

    # Fecha de hace seis meses
    hace_seis_meses = hoy - pd.DateOffset(months=6)

    # Encuentra el día hábil más cercano en el pasado a hace_seis_meses
    dia_habil_hace_seis_meses = xmex.date_to_session(hace_seis_meses, direction="previous")

    # Obtén todas las sesiones desde hace_seis_meses hasta hoy
    ultimos_180_dias = xmex.sessions_in_range(dia_habil_hace_seis_meses, hoy)

    # Fecha de hace un año
    hace_un_año = hoy - pd.DateOffset(years=1)

    # Encuentra el día hábil más cercano en el pasado a hace_un_año
    dia_habil_hace_un_año = xmex.date_to_session(hace_un_año, direction="previous")

    # Obtén todas las sesiones desde hace_un_año hasta hoy
    ultimos_365_dias = xmex.sessions_in_range(dia_habil_hace_un_año, hoy)

    resultado['mes_hasta_fecha'] = mes_hasta_fecha
    resultado['año_hasta_fecha'] = año_hasta_fecha
    resultado['ultimos_30_dias'] = ultimos_30_dias
    resultado['ultimos_90_dias'] = ultimos_90_dias
    resultado['ultimos_180_dias'] = ultimos_180_dias
    resultado['ultimos_365_dias'] = ultimos_365_dias

    return resultado

def anualizar_rendimiento(rendimiento_bruto, dias):
    rendimiento_anualizado = rendimiento_bruto / dias * 360
    return rendimiento_anualizado

def calcular_rendimiento_bruto(precio_inicio, precio_fin, dias):
    # Calcular el cambio porcentual en el precio
    cambio_pct = (precio_fin / precio_inicio) - 1

    # Calcular el rendimiento bruto
    rendimiento_bruto = cambio_pct
    return rendimiento_bruto

def calcular_rendimiento(precios, ventanas_de_tiempo, nombre_benchmark):
    rendimientos = []

    for periodo, ventana in ventanas_de_tiempo.items():
        # Obtén los precios de inicio y fin para la ventana de tiempo actual
        precio_inicio = precios.loc[ventana[0], nombre_benchmark]
        precio_fin = precios.loc[ventana[-1], nombre_benchmark]

        # Calcula el rendimiento bruto y anualizado
        rendimiento_bruto = calcular_rendimiento_bruto(precio_inicio, precio_fin, (ventana[-1] - ventana[0]).days)
        rendimiento_anualizado = anualizar_rendimiento(rendimiento_bruto, (ventana[-1] - ventana[0]).days)

        # Agrega el rendimiento a la lista de rendimientos
        rendimientos.append({
            'Periodo': periodo,
            'Rendimiento_bruto': rendimiento_bruto*100,
            'Rendimiento_anualizado': rendimiento_anualizado*100
        })

    # Convierte la lista de rendimientos en un dataframe
    df_rendimientos = pd.DataFrame(rendimientos)

    return df_rendimientos

tickers = ['GOVT', 'XLV', "GLD", "MCHI", "IVV"]

activos=download_data(tickers)
activos = activos.dropna()

df_activos = activos.copy()

# Opciones de navegación
st.sidebar.title("Navegación")
option = st.sidebar.radio("Seleccione una página", ["Activos", "Portafolios"])


if option == "Activos":
    st.title("Resumen y Estadisticas del activo")
    activo = st.sidebar.selectbox(
        "Elige un activo",
        ('GOVT', 'XLV', "GLD", "MCHI", "IVV")
    )
    df_activo = df_activos[activo]

    # Crear la figura
    fig = go.Figure()

    # Agregar los datos del activo a la figura
    fig.add_trace(go.Scatter(x=df_activo.index, y=df_activo.values, mode='lines'))

    # Establecer títulos y etiquetas
    fig.update_layout(title='Precio de cierre historico del activo',
                    xaxis_title='Fecha',
                    yaxis_title='Precio de Cierre (en $)')

    st.plotly_chart(fig)

    st.text("Pon la fecha a la que quieres los rendimientos:")

    hoy = st.date_input('Introduce la fecha')
    hoy = pd.Timestamp(hoy)
    ventanas_de_tiempo = calcular_fechas(hoy)

    df_rendimientos = calcular_rendimiento(activos, ventanas_de_tiempo, activo)

    st.dataframe(df_rendimientos)




