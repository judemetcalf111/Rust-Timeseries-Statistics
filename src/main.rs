use std::error::Error;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::{env, io};
use rustfft::{FftPlanner, num_complex::Complex};
use csv::Writer;

#[derive(Debug, PartialEq)]
enum FileType {
    CSV,
    Unknown,
}

fn main() -> Result<(), Box<dyn Error>> {
    // Collect args, skipping the executable name
    let args: Vec<String> = env::args().skip(1).collect();

    if args.is_empty() {
        println!("Please provide at least one file path.");
        return Ok(());
    }

    file_looper(args)?;

    Ok(())
}

fn file_looper(file_vector: Vec<String>) -> Result<(), Box<dyn Error>> {
    for filename in &file_vector {
        let path = Path::new(filename);

        let file_type = match detect_file_type(path) {
            Ok(file_type) => file_type,
            Err(e) => {
                eprintln!("Error detecting type for {}: {}", filename, e);
                continue;
            }
        };

        match file_type {
            FileType::CSV => {
                if let Err(e) = process_full_signal(path) {
                    eprintln!("Error processing {}: {}", filename, e);
                }
            },
            FileType::Unknown => println!("Unknown file format: {}", filename),
        }
    }
    Ok(())
}

fn detect_file_type(path: &Path) -> Result<FileType, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut buffer = [0u8; 8];
    let n = file.read(&mut buffer)?;

    if n > 0 && buffer[..n].iter().all(|&b| b.is_ascii() && !b.is_ascii_control()) {
        return Ok(FileType::CSV);
    }
    Ok(FileType::Unknown)
}

fn process_full_signal(path: &Path) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    
    let mut signal2: Vec<Complex<f64>> = Vec::new();
    let mut signal3: Vec<Complex<f64>> = Vec::new();
    let mut timestamps: Vec<f64> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        // Assuming: Col 0 = Time, Col 1 = Signal A, Col 2 = Signal B
        let t: f64 = record.get(0).unwrap_or("0").parse()?;
        let s2: f64 = record.get(1).unwrap_or("0").parse()?;
        let s3: f64 = record.get(2).unwrap_or("0").parse()?;

        timestamps.push(t);
        signal2.push(Complex { re: s2, im: 0.0 });
        signal3.push(Complex { re: s3, im: 0.0 });
    }

    let n = signal2.len();
    if n < 2 { return Err("Not enough data".into()); }

    // Calculate Sample Rate: Fs = 1 / average delta T
    let total_time = timestamps.last().unwrap() - timestamps.first().unwrap();
    let sample_rate = (n as f64 - 1.0) / total_time;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    fft.process(&mut signal2);
    fft.process(&mut signal3);

    // Save to CSV
    save_fft_results(path, &signal2, sample_rate)?;

    println!("Processed {} ({} samples) at {:.2} Hz", path.display(), n, sample_rate);
    Ok(())
}

fn save_fft_results(original_path: &Path, data: &[Complex<f64>], sample_rate: f64) -> Result<(), Box<dyn Error>> {
    // Create new filename: "original_fft.csv"
    let mut new_path = PathBuf::from(original_path);
    let stem = original_path.file_stem().unwrap().to_str().unwrap();
    new_path.set_file_name(format!("{}_fft.csv", stem));

    let mut wtr = Writer::from_path(&new_path)?;
    wtr.write_record(&["Frequency", "Magnitude"])?;

    let n = data.len();
    for (i, complex) in data.iter().enumerate().take(n / 2) {
        let freq = i as f64 * sample_rate / n as f64;
        let magnitude = (complex.re.powi(2) + complex.im.powi(2)).sqrt();
        wtr.write_record(&[freq.to_string(), magnitude.to_string()])?;
    }

    wtr.flush()?;
    println!("Results saved to: {:?}", new_path);
    Ok(())
}
