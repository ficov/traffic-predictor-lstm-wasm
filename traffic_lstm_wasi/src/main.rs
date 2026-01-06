use tract_onnx::prelude::*;
use std::fs::File;
use std::io::{self, BufRead};
use std::thread;
use std::time::Duration;

fn main() -> TractResult<()> {

    const LOOKBACK: usize = 12;          // model gleda 12 koraka unazad
    const TARGET_SENSOR_ROW: usize = 10; // uzimamo podatke 10. senzora za test kako bi se isprobalo na drugim mjerenjima
    const CSV_FILENAME: &str = "vel.csv";

    println!("LSTM TRAFFIC MONITORING (WasmEdge)");

    // UČITAVANJE MODELA
    let model_bytes = include_bytes!("../traffic_lstm.onnx"); // onnx model je ugrađen u wasm datoteku

    println!("[INFO] Inicijaliziram AI model...");
    let model = tract_onnx::onnx()
        .model_for_read(&mut (model_bytes as &[u8]))? // učitavanje modela iz bajtova
        .with_input_fact(0, f32::fact([1, LOOKBACK, 1]).into())? // definiranje oblika ulaza zbog WasmEdge ograničenja
        .into_optimized()?
        .into_runnable()?;

    // ČITANJE PODATAKA
    println!("[INFO] Čitam podatke iz '{}' (Senzor #{})...", CSV_FILENAME, TARGET_SENSOR_ROW);
    
    let file = File::open(CSV_FILENAME).expect("GREŠKA: Nema vel.csv u folderu!");
    let reader = io::BufReader::new(file);

    // dohvat ciljane linije iz CSV-a
    let line = reader.lines().nth(TARGET_SENSOR_ROW)
        .expect("CSV nema dovoljno redova!")
        .expect("Greška pri čitanju linije");

    // pretvorba linije u vektor f32, greške pri parsiranju se ignoriraju
    let data: Vec<f32> = line.split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("[INFO] Učitano {} zapisa. Pokrećem simulaciju...", data.len());

    // simulacija slanja podataka sa senzora u stvarnom vremenu
    // beskonačna petlja za ponavljanje simulacije
    loop {
        println!("\n[RESTART] Pokrećem simulaciju (početak niza podataka)");
        println!("\n  VRIJEME  |   INPUT (Zadnjih 15 min) | PREDVIĐENO | STVARNO | ODSTUPANJE");
        println!("-----------|--------------------------|------------|---------|-----------");
        
        for i in 0..(data.len() - LOOKBACK) {
            
            // sliding window ulaznih podataka
            let input_slice = &data[i..i + LOOKBACK];
            let actual_value = data[i + LOOKBACK]; // stvarna vrijednost za usporedbu
    
            // pretvaranje ulaza u 3D tenzor
            let input_tensor = tensor1(input_slice).into_shape(&[1, LOOKBACK, 1])?;
    
            // izvršavanje modela
            let result = model.run(tvec!(input_tensor.into()))?;
            
            // čitanje rezultata (predviđene brzine)
            let output_tensor = result[0].to_array_view::<f32>()?;
            let predicted_value = output_tensor[[0, 0]];
    
            // izračun greške (apsolutna vrijednost)
            let diff = (predicted_value - actual_value).abs();
    
            // ispis rezultata, prikazujući zadnjih 15 minuta ulaza, ali se šalje zadnjih 12
            println!(
                " T={:04}    | .....{:5.1}, {:5.1}, {:5.1} |   {:6.2}   | {:6.2}  |   {:4.2}", 
                i, 
                input_slice[9], input_slice[10], input_slice[11], 
                predicted_value, 
                actual_value,
                diff
            );
    
            // simulacija slanja svakih 3 sekunde sa senzora
            thread::sleep(Duration::from_millis(3000)); 
        }

        println!("\n[INFO] Simulacija završena. Ponovno pokrećem...\n");
        thread::sleep(Duration::from_millis(2000));

    }

}