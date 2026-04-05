import streamlit as st
import pandas as pd
from src.data_loader import load_csv
from src.visualization import plot_results

st.title("📈 Time Series Forecast Dashboard")

data = load_csv()
st.subheader("Historical Close Prices")
st.line_chart(data['Close'])

st.subheader("Forecast Simulation (Example)")

example_pred = data['Close'][-60:] * 1.02
plot_results(data['Close'][-60:], example_pred, "Example Forecast")