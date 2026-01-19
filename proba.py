import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import os
import shutil

CSV_FILE = 'vel.csv'
CSV_OUT_FILE = 'mjerenja.csv'
LOOKBACK = 12        
TRAIN_SENSOR_ROW = 10 
EPOCHS = 10
SAVED_MODEL_DIR = "traffic_saved_model" # mapa gdje se sprema model

# UČITAVANJE
print(f"Učitavam {CSV_FILE}...")
if not os.path.exists(CSV_FILE):
    print("GREŠKA: Nema vel.csv!")
    exit()

df = pd.read_csv(CSV_FILE, header=None)
train_data = df.iloc[TRAIN_SENSOR_ROW]

with open(CSV_OUT_FILE, 'w', encoding='utf-8') as f:
    for i in range(0, len(train_data), 13):
        row = train_data[i:i + 13]
        f.write(','.join(format(x, ".18e") for x in row))
        if i + 13 < len(train_data):
            f.write('\n')

print(f"Mjerenja su upisana u {CSV_OUT_FILE}.")

print(len(train_data))
