use tract_onnx::prelude::*;
use std::io::{self, BufRead};

fn main() -> TractResult<()> {
    const LOOKBACK: usize = 12; // model gleda 12 koraka unazad
    const EXPECTED_INPUT_LEN: usize = LOOKBACK + 1; // prethodne vrijednosti + vrijednost za provjeru

    // UČITAVANJE MODELA
    let model_bytes = include_bytes!("../traffic_lstm.onnx"); // onnx model je ugrađen u wasm datoteku

    eprintln!("Inicijaliziranje AI modela...");
    let model = tract_onnx::onnx()
        .model_for_read(&mut (model_bytes as &[u8]))? // učitavanje modela iz bajtova
        .with_input_fact(0, f32::fact([1, LOOKBACK, 1]).into())? // definiranje oblika ulaza zbog WasmEdge ograničenja
        .into_optimized()?
        .into_runnable()?;

    eprintln!("Model spreman. Čitanje podataka sa Stdin (format: 13 brojeva odvojenih zarezom)...");

    // ČITANJE SA STDIN
    let stdin = io::stdin();
    let mut handle = stdin.lock(); // zaključavamo stdin za efikasno čitanje
    let mut buffer = String::new(); // buffer za čitanje linija

    // GLAVNA PETLJA (BESKONAČNA)
    loop {
        buffer.clear();
        match handle.read_line(&mut buffer) {
            Ok(0) => {
                eprintln!("Input zatvoren. Gašenje.");
                break;
            }
            Ok(_) => { // uspješno pročitana linija
                // čišćenje
                let trimmed = buffer.trim();
                if trimmed.is_empty() { continue; }

                // pretvorba linije u vektor f32, greške pri parsiranju se ignoriraju
                let numbers: Vec<f32> = trimmed.split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();

                if numbers.len() != EXPECTED_INPUT_LEN {
                    eprintln!("Primljeno {} brojeva, očekivano {}.", numbers.len(), EXPECTED_INPUT_LEN);
                    println!("ERROR: Bad input length");
                    continue;
                }

                let input_slice = &numbers[0..LOOKBACK];
                let actual_value = numbers[LOOKBACK];

                // pretvaranje ulaza u 3D tenzor
                let input_tensor = tensor1(input_slice).into_shape(&[1, LOOKBACK, 1]);

                // provjera tenzora
                if input_tensor.is_err() {
                    eprintln!("Problem s tenzorom.");
                    continue;
                }

                // izvršavanje modela
                let result = model.run(tvec!(input_tensor.unwrap().into()));

                // obrada rezultata
                match result {
                    Ok(res) => {
                        let output_tensor = res[0].to_array_view::<f32>().unwrap();
                        let predicted_value = output_tensor[[0, 0]];
                        let diff = (predicted_value - actual_value).abs();

                        println!("PREDVIĐENO: {:.2}, STVARNO: {:.2}, ODSTUPANJE: {:.2}", predicted_value, actual_value, diff);
                    },
                    Err(e) => {
                        eprintln!("Model run failed: {:?}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Greška pri čitanju stdina: {}", e);
                break;
            }
        }
    }

    Ok(())
}