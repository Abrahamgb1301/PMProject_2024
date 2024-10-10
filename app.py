import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# Funciones auxiliares
def obtener_datos_acciones(simbolos, start_date, end_date):
    data = yf.download(simbolos, start=start_date, end=end_date)['Close']
    return data

def calcular_metricas(df):
    returns = df.pct_change()
    cumulative_returns = (1 + returns).cumprod() - 1
    normalized_prices = df / df.iloc[0] * 100
    return returns, cumulative_returns, normalized_prices

def calcular_rendimientos_portafolio(returns, weights):
    return (returns * weights).sum(axis=1)

def calcular_rendimiento_ventana(returns, window):
    return (1 + returns).rolling(window=window).apply(np.prod) - 1

def calcular_beta(portfolio_returns, index_returns):
    cov = np.cov(portfolio_returns, index_returns)[0, 1]
    var = np.var(index_returns)
    return cov / var

def calcular_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calcular_sortino_ratio(returns, risk_free_rate=0.02, target_return=0):
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < target_return]
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    return np.sqrt(252) * excess_returns.mean() / downside_deviation

# Configuración de la página
st.set_page_config(page_title="Analizador de Portafolio", layout="wide")
st.title("Analizador de Portafolio de Inversión")

# Entrada de símbolos y pesos
simbolos_input = st.text_input("Ingrese los símbolos de las acciones separados por comas (por ejemplo: AAPL,GOOGL,MSFT):", "AAPL,GOOGL,MSFT,AMZN,FB")
pesos_input = st.text_input("Ingrese los pesos correspondientes separados por comas (deben sumar 1):", "0.2,0.2,0.2,0.2,0.2")

simbolos = [s.strip() for s in simbolos_input.split(',')]
pesos = [float(w.strip()) for w in pesos_input.split(',')]

# Selección de la ventana de tiempo
end_date = datetime.now()
start_date_options = {
    "1 mes": end_date - timedelta(days=30),
    "3 meses": end_date - timedelta(days=90),
    "6 meses": end_date - timedelta(days=180),
    "1 año": end_date - timedelta(days=365),
    "3 años": end_date - timedelta(days=3*365),
    "5 años": end_date - timedelta(days=5*365),
    "10 años": end_date - timedelta(days=10*365)
}
selected_window = st.selectbox("Seleccione la ventana de tiempo para el análisis:", list(start_date_options.keys()))
start_date = start_date_options[selected_window]

if len(simbolos) != len(pesos) or abs(sum(pesos) - 1) > 1e-6:
    st.error("El número de símbolos debe coincidir con el número de pesos, y los pesos deben sumar 1.")
else:
    # Obtener datos
    benchmark = 'SPY'  # Cambiamos a SPY como benchmark
    all_symbols = simbolos + [benchmark]
    df_stocks = obtener_datos_acciones(all_symbols, start_date, end_date)
    returns, cumulative_returns, normalized_prices = calcular_metricas(df_stocks)
    
    # Rendimientos del portafolio
    portfolio_returns = calcular_rendimientos_portafolio(returns[simbolos], pesos)
    portfolio_cumulative_returns = (1 + portfolio_returns).cumprod() - 1

    # Sección 1: Análisis de Activos Individuales
    st.header("Análisis de Activos Individuales")
    
    selected_asset = st.selectbox("Seleccione un activo para analizar:", simbolos)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Rendimiento Total", f"{cumulative_returns[selected_asset].iloc[-1]:.2%}")
    col2.metric("Sharpe Ratio", f"{calcular_sharpe_ratio(returns[selected_asset]):.2f}")
    col3.metric("Sortino Ratio", f"{calcular_sortino_ratio(returns[selected_asset]):.2f}")
    
    # Gráfico de precio normalizado del activo seleccionado
    fig_asset = px.line(normalized_prices[selected_asset], title=f'Precio Normalizado de {selected_asset} (Base 100)')
    st.plotly_chart(fig_asset)
    
    # Beta del activo vs benchmark
    beta_asset = calcular_beta(returns[selected_asset], returns[benchmark])
    st.metric("Beta vs SPY", f"{beta_asset:.2f}")

    # Sección 2: Análisis del Portafolio
    st.header("Análisis del Portafolio")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Rendimiento Total del Portafolio", f"{portfolio_cumulative_returns.iloc[-1]:.2%}")
    col2.metric("Sharpe Ratio del Portafolio", f"{calcular_sharpe_ratio(portfolio_returns):.2f}")
    col3.metric("Sortino Ratio del Portafolio", f"{calcular_sortino_ratio(portfolio_returns):.2f}")

    # Gráfico de rendimientos acumulados del portafolio vs benchmark
    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(go.Scatter(x=portfolio_cumulative_returns.index, y=portfolio_cumulative_returns, name='Portafolio'))
    fig_cumulative.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[benchmark], name='SPY'))
    fig_cumulative.update_layout(title='Rendimientos Acumulados: Portafolio vs SPY', xaxis_title='Fecha', yaxis_title='Rendimiento Acumulado')
    st.plotly_chart(fig_cumulative)

    # Beta del portafolio vs benchmark
    beta_portfolio = calcular_beta(portfolio_returns, returns[benchmark])
    st.metric("Beta del Portafolio vs SPY", f"{beta_portfolio:.2f}")

    # Rendimientos en diferentes ventanas de tiempo
    st.subheader("Rendimientos en Diferentes Ventanas de Tiempo")
    ventanas = [1, 7, 30, 90, 180, 252]
    rendimientos_ventanas = pd.DataFrame(index=['Portafolio'] + simbolos + [benchmark])
    
    for ventana in ventanas:
        rendimientos_ventanas[f'{ventana}d'] = pd.Series({
            'Portafolio': calcular_rendimiento_ventana(portfolio_returns, ventana).iloc[-1],
            **{symbol: calcular_rendimiento_ventana(returns[symbol], ventana).iloc[-1] for symbol in simbolos + [benchmark]}
        })
    
    st.dataframe(rendimientos_ventanas.style.format("{:.2%}"))

    # Gráfico de comparación de rendimientos
    fig_comparison = go.Figure()
    for index, row in rendimientos_ventanas.iterrows():
        fig_comparison.add_trace(go.Bar(x=ventanas, y=row, name=index))
    fig_comparison.update_layout(title='Comparación de Rendimientos', xaxis_title='Días', yaxis_title='Rendimiento', barmode='group')
    st.plotly_chart(fig_comparison)



