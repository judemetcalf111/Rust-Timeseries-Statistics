// memmap2

fn process_hdf5_chunks(path: &Path, dataset_name: &str) -> Result<(), Box<dyn Error>> {

    parent_path_buf = if let Some(parent_dir) = path.parent() {
        // Create a mutable PathBuf from the parent directory
        parent_dir.to_path_buf()
    } else {
        ;
    };

    println!("Processing HDF5 in chunks...");
    let file = hdf5::File::open(path)?;
    let dataset = file.dataset(dataset_name)?;
    
    let shape = dataset.shape(); // e.g., [100000]
    let total_len = shape[0];
    let mut offset = 0;
    let mut chunk_count = 0;

    while offset < total_len {
        // Determine slice range
        let end = std::cmp::min(offset + CHUNK_SIZE, total_len);
        let len = end - offset;

        // Read ONLY the slice from disk
        // s![offset..end] is the ndarray macro for slicing
        let chunk_data: Vec<f64> = dataset.read_slice_1d(s![offset..end])?.to_vec();

        let mut buffer: Vec<Complex<f64>> = chunk_data
            .into_iter()
            .map(|x| Complex { re: x, im: 0.0 })
            .collect();

        // Zero pad last chunk if necessary
        while buffer.len() < CHUNK_SIZE {
             buffer.push(Complex::default());
        }

        process_fft_chunk(&mut buffer, chunk_count);
        
        offset += CHUNK_SIZE;
        chunk_count += 1;
    }

    Ok(())
}
