use std::error::Error;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use rustfft::{FftPlanner, num_complex::Complex};
use ndarray::s; // Used for HDF5 slicing
use std::fs::File;
use std::io::ErrorKind;

// Configuration
const CHUNK_SIZE: usize = 892; // Power of 2 is best for FFT
const SAMPLE_RATE: f64 = 1000.0; // Replace with your actual rate

#[derive(PartialEq)]
enum FileType {
    CSV,
    HDF5,
    Unknown,
}

use std::io;
use thiserror::Error;

#[derive(Debug, Error)]
enum DetectError {
    #[error("IO error")]
    Io(#[from] io::Error),

    #[error("Invalid file format")]
    InvalidFormat,
}


fn main() -> Result<(), Box<dyn Error>> {
    // 1. Get filename from args or hardcode
    const ALL_FILES: Vec<String> = env::args().collect();

    Ok(())
}

fn file_looper(file_vector: Vec<String>) {
    for filename in &file_vector {
        println!("Fourier Transforming file: {}", filename);

        let path = Path::new(filename);

        let file_type = match detect_file_type(path) {
            Ok(file_type) => file_type,
            Err(err) => {
                println!("Error processing {}: {}", filename, err);
                continue;
            }
        };

        println!("Detected file type: {:?}", file_type);
        
        match file_type {
            FileType::CSV => process_csv_chunks(path)?,
            // FileType::HDF5 => process_hdf5_chunks(path, "dataset_name")?, // later....
            FileType::Unknown => println!("Unknown file format!"),
        }
    }
}


// --- DETECTION LOGIC ---

fn detect_file_type(path: &Path) -> Result<FileType, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut buffer = [0u8; 8]; // Read first 8 bytes
    file.read_exact(&mut buffer)?;

    // HDF5 Magic Bytes: \x89 H D F \r \n \x1a \n
    let hdf5_magic = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];

    if buffer == hdf5_magic {
        return Ok(FileType::HDF5);
    }

    // Heuristic for CSV: usually starts with readable ASCII
    // Check if the first few bytes are valid UTF-8 text
    if buffer.iter().all(|b| b.is_ascii() && !b.is_ascii_control()) {
        return Ok(FileType::CSV);
    }


    Ok(FileType::Unknown)
}

// --- FFT PROCESSING LOGIC ---

fn process_fft_chunk(buffer: &mut Vec<Complex<f64>>, chunk_index: usize) {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(buffer.len());
    
    fft.process(buffer);

    // Example output: Peak frequency in this chunk
    // Note: In real apps, you might aggregate this into a spectrogram
    let mut max_mag = 0.0;
    let mut max_bin = 0;
    
    for (i, val) in buffer.iter().enumerate().take(buffer.len() / 2) {
        let mag = val.norm();
        if mag > max_mag {
            max_mag = mag;
            max_bin = i;
        }
    }
    
    let peak_freq = max_bin as f64 * SAMPLE_RATE / buffer.len() as f64;
    println!("Chunk {}: Peak Frequency = {:.2} Hz (Mag: {:.2})", chunk_index, peak_freq, max_mag);
}

// --- CSV CHUNKING ---

fn process_csv_chunks(path: &Path) -> Result<(), Box<dyn Error>> {
    println!("Processing CSV in chunks...");
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);
    
    let mut buffer: Vec<Complex<f64>> = Vec::with_capacity(CHUNK_SIZE);
    let mut chunk_count = 0;

    // We assume the CSV has a header "signal" or just raw numbers. 
    // This iterates row by row.
    for result in rdr.deserialize::<f64>() { // Assuming single column of floats
        let val = result?;
        buffer.push(Complex { re: val, im: 0.0 });

        if buffer.len() == CHUNK_SIZE {
            process_fft_chunk(&mut buffer, chunk_count);
            buffer.clear(); // Clear for next chunk
            chunk_count += 1;
        }
    }
    
    // Process remaining samples if needed
    if !buffer.is_empty() {
        // Zero-pad to reach power of 2 if strict about size
        while buffer.len() < CHUNK_SIZE {
            buffer.push(Complex { re: 0.0, im: 0.0 });
        }
        process_fft_chunk(&mut buffer, chunk_count);
    }

    Ok(())
}
