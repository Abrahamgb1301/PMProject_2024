import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Funciones auxiliares
def obtener_datos_acciones(simbolos, periodo='5y'):
    data = yf.download(simbolos, period=periodo)['Close']
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

if len(simbolos) != len(pesos) or abs(sum(pesos) - 1) > 1e-6:
    st.error("El número de símbolos debe coincidir con el número de pesos, y los pesos deben sumar 1.")
else:
    # Obtener datos
    df_stocks = obtener_datos_acciones(simbolos)
    returns, cumulative_returns, normalized_prices = calcular_metricas(df_stocks)
    
    # Rendimientos del portafolio
    portfolio_returns = calcular_rendimientos_portafolio(returns, pesos)
    portfolio_cumulative_returns = (1 + portfolio_returns).cumprod() - 1

    # Gráfico de precios normalizados
    fig_normalized = px.line(normalized_prices, title='Precios Normalizados (Base 100)')
    fig_normalized.add_scatter(x=normalized_prices.index, y=(1 + portfolio_cumulative_returns) * 100, name='Portafolio')
    st.plotly_chart(fig_normalized)

    # Gráfico de rendimientos acumulados
    fig_cumulative = px.line(cumulative_returns, title='Rendimientos Acumulados')
    fig_cumulative.add_scatter(x=cumulative_returns.index, y=portfolio_cumulative_returns, name='Portafolio')
    st.plotly_chart(fig_cumulative)

    # Métricas del portafolio
    st.subheader("Métricas del Portafolio")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rendimiento Total", f"{portfolio_cumulative_returns.iloc[-1]:.2%}")
    col2.metric("Sharpe Ratio", f"{calcular_sharpe_ratio(portfolio_returns):.2f}")
    col3.metric("Sortino Ratio", f"{calcular_sortino_ratio(portfolio_returns):.2f}")

    # Comparación con S&P 500
    sp500 = obtener_datos_acciones(['^GSPC'])
    sp500_returns = sp500.pct_change()
    
    # Beta vs S&P 500
    betas = {}
    for symbol in simbolos:
        betas[symbol] = calcular_beta(returns[symbol], sp500_returns['^GSPC'])
    betas['Portafolio'] = calcular_beta(portfolio_returns, sp500_returns['^GSPC'])
    
    st.subheader("Betas vs S&P 500")
    st.bar_chart(pd.Series(betas))

    # Rendimientos en diferentes ventanas de tiempo
    st.subheader("Rendimientos en Diferentes Ventanas de Tiempo")
    ventanas = [1, 7, 30, 90, 180, 252]
    rendimientos_ventanas = pd.DataFrame()
    for ventana in ventanas:
        rendimientos_ventanas[f'{ventana}d'] = calcular_rendimiento_ventana(portfolio_returns, ventana)
    rendimientos_ventanas.index = ['Portafolio']
    
    for symbol in simbolos:
        symbol_returns = calcular_rendimiento_ventana(returns[symbol], ventanas[-1])
        rendimientos_ventanas.loc[symbol] = symbol_returns.iloc[-1]
    
    sp500_returns_windows = calcular_rendimiento_ventana(sp500_returns['^GSPC'], ventanas[-1])
    rendimientos_ventanas.loc['S&P 500'] = sp500_returns_windows.iloc[-1]
    
    st.dataframe(rendimientos_ventanas.style.format("{:.2%}"))

    # Gráfico de comparación de rendimientos
    fig_comparison = go.Figure()
    for index, row in rendimientos_ventanas.iterrows():
        fig_comparison.add_trace(go.Bar(x=ventanas, y=row, name=index))
    fig_comparison.update_layout(title='Comparación de Rendimientos', xaxis_title='Días', yaxis_title='Rendimiento', barmode='group')
    st.plotly_chart(fig_comparison)




