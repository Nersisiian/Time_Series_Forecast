# src/train.py
from src.data_loader import load_csv
from src.preprocessing import scale_data, create_sequences
from src.models.lstm_model import build_lstm
from src.models.transformer_model import build_transformer
import numpy as np
import os

data = load_csv()
num_rows = len(data)
print(f"Количество строк в CSV: {num_rows}")

scaled_data, scaler = scale_data(data['Close'])

seq_length = min(60, num_rows - 1)
print(f"Используем seq_length = {seq_length}")

X_lstm, y = create_sequences(scaled_data, seq_length)
print("X_lstm.shape:", X_lstm.shape)
print("y.shape:", y.shape)

if X_lstm.shape[0] == 0:
    raise ValueError("Массив X_lstm пустой. Нужно больше строк в CSV или уменьшить seq_length.")

lstm_model = build_lstm((X_lstm.shape[1], 1))
lstm_model.fit(X_lstm, y, epochs=5, batch_size=2)

transformer_model = build_transformer((X_lstm.shape[1], 1))
transformer_model.fit(X_lstm, y, epochs=5, batch_size=2)

os.makedirs('models', exist_ok=True)
lstm_model.save('models/lstm_model.h5')
transformer_model.save('models/transformer_model.keras')

print("Обучение завершено. Модели сохранены в папке 'models'.")