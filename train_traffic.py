import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import os
import shutil

CSV_FILE = 'vel.csv'
LOOKBACK = 12        
TRAIN_SENSOR_INDEX = 0 
EPOCHS = 10
SAVED_MODEL_DIR = "traffic_saved_model" # mapa gdje se sprema model

# UČITAVANJE
print(f"Učitavam {CSV_FILE}...")
if not os.path.exists(CSV_FILE):
    print("GREŠKA: Nema vel.csv!")
    exit()

df = pd.read_csv(CSV_FILE, header=None)
train_data = df.iloc[:, TRAIN_SENSOR_INDEX].values.astype('float32') # stupci su senzori, redovi su vremenski koraci

print(f"Učitano {len(train_data)} mjerenja za senzor u stupcu {TRAIN_SENSOR_INDEX}.")

# SLIDING WINDOW PRIPREMA PODATAKA
def create_dataset(dataset, look_back=12):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        X.append(a)
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)

X_train, y_train = create_dataset(train_data, LOOKBACK)
X_train = X_train.reshape(X_train.shape[0], LOOKBACK, 1) # 3D oblik za LSTM

# IZGRADNJA NEURONSKE MREŽE I TRENING
print("Treniram model...")
model = Sequential()
model.add(Input(shape=(LOOKBACK, 1), batch_size=1, name="input")) # točno specificiranje batch_size zbog wasm-a
model.add(LSTM(64, activation='relu', return_sequences=True, unroll=True)) # 1. LSTM sloj
model.add(LSTM(32, activation='relu', unroll=True)) # 2. LSTM sloj
model.add(Dense(1)) # izlazni sloj - predviđena brzina vozila

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, verbose=1) #treniranje

# SPREMANJE (EXPORT) - TensorFlow SavedModel format:
# .pb -> protobuffer datoteka s grafom modela
# variables/ -> mapa s težinama modela
# assets/ -> dodatne datoteke (npr. tokenizer), može biti prazna
# fingerprint -> datoteka s hash vrijednostima za provjeru integriteta
print(f"Spremam model u mapu '{SAVED_MODEL_DIR}'...")

# Brisanje stare mape ako postoji
if os.path.exists(SAVED_MODEL_DIR):
    shutil.rmtree(SAVED_MODEL_DIR)

try:
    model.export(SAVED_MODEL_DIR)
    print("Uspješno spremljeno, model.export()!")
except Exception as e:
    print(f"model.export() nije uspio, model.save(): {e}")
    model.save(SAVED_MODEL_DIR)
    print("Uspješno spremljeno, model.save()!")