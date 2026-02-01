use std::error::Error;
use std::fs::{File, create_dir_all};
use std::io::Read;
use std::path::Path;
use std::env;
// use std::f64::consts::PI;

use rustfft::{FftPlanner, num_complex::Complex};
use csv::Writer;
use ndarray::{Array1, s}; // ndarray imports

#[derive(Debug, PartialEq)]
enum FileType {
    CSV,
    Unknown,
}

fn main() -> Result<(), Box<dyn Error>> {
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

        match detect_file_type(path) {
            Ok(FileType::CSV) => {
                if let Err(e) = process_full_signal(path) {
                    eprintln!("Error processing {}: {}", filename, e);
                }
            },
            Ok(FileType::Unknown) => println!("Unknown file format: {}", filename),
            Err(e) => eprintln!("Error detecting type for {}: {}", filename, e),
        }
    }
    Ok(())
}

fn detect_file_type(path: &Path) -> Result<FileType, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut buffer = [0u8; 8];
    let n = file.read(&mut buffer)?;

    if n > 0 && buffer[..n].iter().all(|&b| b.is_ascii() && (b == b'\n' || b == b'\r' || !b.is_ascii_control())) {
        return Ok(FileType::CSV);
    }
    Ok(FileType::Unknown)
}

fn process_full_signal(path: &Path) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    
    // Create output directory safely
    let stem = path.file_stem().unwrap_or_default().to_str().unwrap_or("data");
    let timeseries_path = path.parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{}_timeseries_data", stem));
        
    create_dir_all(&timeseries_path)?;

    // Temporary vectors to hold read data
    let mut raw_s2: Vec<Complex<f64>> = Vec::new();
    let mut raw_s3: Vec<Complex<f64>> = Vec::new();
    let mut timestamps: Vec<f64> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        // Assuming: Col 0 = Time, Col 1 = Signal A, Col 2 = Signal B
        let t: f64 = record.get(0).unwrap_or("0").parse()?;
        let s2: f64 = record.get(1).unwrap_or("0").parse()?;
        let s3: f64 = record.get(2).unwrap_or("0").parse()?;

        timestamps.push(t);
        raw_s2.push(Complex { re: s2, im: 0.0 });
        raw_s3.push(Complex { re: s3, im: 0.0 });
    }

    let n = raw_s2.len();
    if n < 2 { return Err("Not enough data".into()); }

    // Convert to ndarray for easier math
    let mut signal2 = Array1::from(raw_s2);
    let mut signal3 = Array1::from(raw_s3);

    // Calculate Sample Rate
    let total_time = timestamps.last().unwrap() - timestamps.first().unwrap();
    let sample_rate = (n as f64 - 1.0) / total_time;
    let dt = 1.0 / sample_rate;

    // Calculate ACF (before windowing or FFT)
    // Takes every tenth length, 
    // since the resolution will be more than good enough at that scale.
    // Ensures speed!!!
    calculate_and_save_acf(&timeseries_path, &signal2, 10, dt, "ACF_x")?;
    calculate_and_save_acf(&timeseries_path, &signal3, 10, dt, "ACF_y")?;

    // // Apply Windowing (Hann) For the Future
    // // We apply the window to the signal in-place to reduce spectral leakage
    // apply_hann_window(&mut signal2);
    // apply_hann_window(&mut signal3);

    // FFT 
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    // RustFFT requires a mutable slice. ndarray provides this via .as_slice_mut().
    // If the array is not contiguous (it usually is here), this would return None, so we unwrap.
    fft.process(signal2.as_slice_mut().unwrap());
    fft.process(signal3.as_slice_mut().unwrap());

    // Save Results
    save_fft_results(&timeseries_path, &signal2, sample_rate, "square_fftx")?;
    save_fft_results(&timeseries_path, &signal3, sample_rate, "square_ffty")?;  
    save_full_fft_results(&timeseries_path, &signal2, &signal3, sample_rate, "full_PSD")?;

    println!("Processed {:?} ({} samples) at {:.2} Hz", path, n, sample_rate);
    Ok(())
}

// /// Applies a Hann window to the signal in-place using ndarray
// fn apply_hann_window(signal: &mut Array1<Complex<f64>>) {
//     let n = signal.len();
//     let n_f = n as f64;
    
//     // Create the window array
//     let window = Array1::from_shape_fn(n, |i| {
//         0.5 * (1.0 - (2.0 * PI * i as f64 / (n_f - 1.0)).cos())
//     });

//     // Element-wise multiplication
//     // Zip the signal with the window and update signal.re (im is 0 anyway for real signals)
//     for (val, &win_val) in signal.iter_mut().zip(window.iter()) {
//         val.re *= win_val;
//         val.im *= win_val; // If signal was complex, we'd scale both parts
//     }
// }

fn calculate_and_save_acf(
    dir: &Path, 
    data: &Array1<Complex<f64>>, 
    sample_size: u64, 
    timestep: f64, 
    suffix: &str
) -> Result<(), Box<dyn Error>> {
    let filename: String = format!("{}.csv", suffix);
    let full_path: std::path::PathBuf = dir.join(filename);

    let plot_name: String = format!("Plot_{}.pdf", suffix);
    let plot_path: std::path::PathBuf = dir.join(plot_name);
    let plot_string: &str = plot_path.to_str().expect("Path contained invalid Unicode characters");
    
    let mut wtr = Writer::from_path(&full_path)?;
    wtr.write_record(&["Time", "ACF"])?;

    let n = data.len();
    let n_f = n as f64;
    
    // Calculate mean using iterator
    let sum_re: f64 = data.iter().map(|c| c.re).sum();
    let mean = sum_re / n_f;

    // Center the data (remove mean)
    let centered: Array1<f64> = data.map(|c| c.re - mean);
    
    // Calculate Variance
    let variance = centered.fold(0.0, |acc, &x| acc + x * x) / n_f;

    // Create max_iterations
    let max_iterations: usize = ((n_f - 1.0) / sample_size as f64).floor() as usize;
    let mut plot_data: Vec<(f64, Vec<f64>)> = Vec::with_capacity(max_iterations);

    for pre_k in 0..=max_iterations {

        let k: usize = pre_k * sample_size as usize;
        // Slice logic for ndarray: data[0..N-k] dot data[k..N]
        let slice_a = centered.slice(s![..n-k]);
        let slice_b = centered.slice(s![k..]);

        let sum_product: f64 = slice_a.dot(&slice_b);

        let acf_val: f64 = if variance.abs() > 1e-9 {
            sum_product / (n_f * variance)
        } else {
            0.0
        };
        
        let time: f64 = k as f64 * timestep;
        wtr.write_record(&[time.to_string(), acf_val.to_string()])?;

        plot_data.push((time, vec![acf_val]));
    }

    wtr.flush()?;

    let _: complot::Plot = (
        plot_data.into_iter(),
        complot::complot!(
            plot_string, 
            xlabel = "Lag", 
            ylabel = "ACF Function",
            title = format!("Plotting: {}", suffix)
        )
    ).into();

    Ok(())
}

fn save_fft_results(
    dir: &Path, 
    data: &Array1<Complex<f64>>, 
    sample_rate: f64, 
    suffix: &str
    ) -> Result<(), Box<dyn Error>> {
        
    let filename: String = format!("{}.csv", suffix);
    let full_path: std::path::PathBuf = dir.join(filename);

    let plot_name: String = format!("Plot_{}.pdf", suffix);
    let plot_path: std::path::PathBuf = dir.join(plot_name);
    let plot_string: &str = plot_path.to_str().expect("Path contained invalid Unicode characters");

    let mut wtr: Writer<File> = Writer::from_path(&full_path)?;
    wtr.write_record(&["Frequency", "Magnitude"])?;

    let max_interations: usize = data.len();

    let mut plot_data: Vec<(f64, Vec<f64>)> = Vec::with_capacity(max_interations);

    let n = max_interations;
    // Only iterate up to Nyquist frequency (N/2)
    for i in 0..n / 2 {
        let freq = i as f64 * sample_rate / n as f64;
        let magnitude = data[i].norm(); // .norm() is sqrt(re^2 + im^2)

        plot_data.push((freq.log10(), vec![magnitude.log10()]));
        wtr.write_record(&[freq.to_string(), magnitude.to_string()])?;
    }
    
    wtr.flush()?;

    let _: complot::Plot = (
        plot_data.into_iter(),
        complot::complot!(
            plot_string, 
            xlabel = "Frequency (log10)", 
            ylabel = "Squared FT Magnitude (log10)",
            title = format!("Plotting: {}", suffix)
        )
    ).into();

    Ok(())
}

fn save_full_fft_results(
    dir: &Path, 
    datax: &Array1<Complex<f64>>, 
    datay: &Array1<Complex<f64>>, 
    sample_rate: f64, 
    suffix: &str
    ) -> Result<(), Box<dyn Error>> {
        
    let filename: String = format!("{}.csv", suffix);
    let full_path: std::path::PathBuf = dir.join(filename);

    let plot_name: String = format!("Plot_{}.pdf", suffix);
    let plot_path: std::path::PathBuf = dir.join(plot_name);
    let plot_string: &str = plot_path.to_str().expect("Path contained invalid Unicode characters");

    let mut wtr: Writer<File> = Writer::from_path(&full_path)?;
    wtr.write_record(&["Frequency", "Combined_Magnitude"])?;

    let max_interations: usize = datax.len();

    let mut plot_data: Vec<(f64, Vec<f64>)> = Vec::with_capacity(max_interations);

    // Calculating both the magnitudes along x and y, and summing the magnitude.
    // Effectively a geometric mean, proper way to construct a PSD
    for n in 0..max_interations / 2 {
        let freq: f64 = n as f64 * sample_rate / max_interations as f64;
        let mag_x_sq: f64 = datax[n].norm_sqr();
        let mag_y_sq: f64 = datay[n].norm_sqr();
        let combined: f64 = (mag_x_sq + mag_y_sq).sqrt();

        plot_data.push((freq.log10(), vec![combined.log10()]));
        wtr.write_record(&[freq.to_string(), combined.to_string()])?;
    }

    wtr.flush()?;

    let _: complot::Plot = (
        plot_data.into_iter(),
        complot::complot!(
            plot_string, 
            xlabel = "Frequency (log10)", 
            ylabel = "Squared 2D FT Magnitude (log10)",
            title = format!("Plotting: {}", suffix)
        )
    ).into();

    Ok(())
}
