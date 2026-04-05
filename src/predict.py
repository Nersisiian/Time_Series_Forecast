# src/predict.py
import pandas as pd
from tensorflow.keras.models import load_model
from src.preprocessing import scale_data, create_sequences
import os

data = pd.read_csv('data/aapl.csv', index_col=0, parse_dates=True)

scaled_data, scaler = scale_data(data['Close'])
seq_length = min(60, len(data)-1)
X, y = create_sequences(scaled_data, seq_length)

lstm_model = load_model('models/lstm_model.h5')
transformer_model = load_model('models/transformer_model.keras')

pred_lstm = lstm_model.predict(X)
pred_lstm = scaler.inverse_transform(pred_lstm)

pred_transformer = transformer_model.predict(X)
pred_transformer = scaler.inverse_transform(pred_transformer)

os.makedirs('data', exist_ok=True)
df_pred = pd.DataFrame({
    'Actual': data['Close'][seq_length:].values,
    'LSTM_Predicted': pred_lstm.flatten(),
    'Transformer_Predicted': pred_transformer.flatten()
})
df_pred.to_csv('data/aapl_predicted.csv', index=True)
print("Предсказание сохранено в data/aapl_predicted.csv")