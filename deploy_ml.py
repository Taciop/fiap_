# Importar bibliotecas
import streamlit as st
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


st.title("Previsão de Preços do Petróleo")
st.write("Acompanhe a tendência histórica e faça previsões futuras com base nos dados.")


@st.cache
def load_data():
    
    file_path = "C:\\Fiap\\petroleo_5 - Página1.xlsx"  
    data = pd.read_excel(file_path, parse_dates=['data'])
    data = data.rename(columns={"data": "ds", "preco": "y"})
    return data


data = load_data()


model = Prophet()
model.fit(data)


days = st.slider("Selecione o número de dias para prever:", min_value=7, max_value=90, value=30)


future = model.make_future_dataframe(periods=days)
forecast = model.predict(future)


st.write(f"Previsões para os próximos {days} dias:")
forecast_table = forecast[['ds', 'yhat']].iloc[-days:]  # Selecionar apenas os dias previstos
forecast_table.columns = ['Data', 'Preço Previsto (USD)']  # Renomear colunas para exibição
st.dataframe(forecast_table)

st.write("Gráfico de Previsões:")
fig1 = model.plot(forecast)
st.pyplot(fig1)


st.write("Componentes das previsões:")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)


train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

forecast_test = forecast.set_index('ds').loc[test['ds']]
rmse = mean_squared_error(test['y'], forecast_test['yhat'], squared=False)
mae = mean_absolute_error(test['y'], forecast_test['yhat'])

st.write("**Métricas de Desempenho no Conjunto de Teste:**")
st.write(f"- RMSE (Erro Quadrático Médio): {rmse:.2f}")
st.write(f"- MAE (Erro Médio Absoluto): {mae:.2f}")
