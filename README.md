# UPUTE ZA POKRETANJE

Ovaj sustav se može podijeliti na dva podsustava: Python podsustav za treniranje i konverziju AI
modela, te Rust/WebAssembly podsustav za izvršavanje predikcija.

## Preduvjeti su instalirani sljedeći alati:
- Python (3.10 ili novija) – treniranje i konverzija modela
- Rust (i Cargo) – kompajliranje izvornog koda u Wasm kod
- WasmEdge Runtime – izvršavanje WebAssembly koda

## Treniranje i konverzija modela:
- osigurati da se datoteka s mjerenjima senzora (**vel.csv**) nalazi u istoj mapi kao i Python skripta
**train_traffic.py** (u datoteci redci predstavljaju mjerenja u vremenu, a stupci senzore)
- instalacija potrebnih biblioteka (paziti da verzije biblioteka budu međusobno kompatibilne):
```bash
pip install "numpy<2.0" pandas tensorflow "tf2onnx>=1.16.0"
```
- treniranje modela:
  ```bash
  python train_traffic.py
  ```
- konverzija modela:
  ```bash
  python -m tf2onnx.convert --saved-model traffic_saved_model --output traffic_lstm.onnx --opset 13
  ```
  (dobivena datoteka **traffic_lstm.onnx**)
  
## Kompajliranje i pokretanje (Rust i WebAssembly):
- instalacija podrške za WASI (WebAssembly System Interface) unutar Rust-a:
```bash
rustup target add wasm32-wasip1
```
- kopirati generiranu datoteku **traffic_lstm.onnx** i datoteku s podacima **vel.csv** u root direktorij
Rust projekta
- kompajliranje Rust koda u WebAssembly:
```bash
cargo build --target wasm32-wasip1 --release
```
(izvršna Wasm datoteka **traffic_lstm_wasi.wasm**
nalazit će se u mapi **target/wasm32-wasip1/release/**)
- pokretanje simulacije:
```bash
wasmedge --dir .:. target/wasm32-wasip1/release/traffic_lstm_wasi.wasm
```
Nakon izvođenja posljednje naredbe, pokreće se WebAssembly kod u kojem je ugrađen LSTM model.
U terminalu će se u vremenskim koracima prikazivati zadnjih 3 mjerenja od 12 mjerenja koja predstavljaju
ulazni niz, predviđena brzina, stvarna brzina te apsolutno odstupanje (greška modela). Sustav se izvodi u
beskonačnoj petlji te na taj način simulirajući kontinuirani nadzor prometa.
