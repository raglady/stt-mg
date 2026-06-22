mod args;
mod settings;

pub use args::*;

pub use settings::*;
use std::path::{Path, PathBuf};

use ndarray::ArcArray2;
use old_ndarray::Axis as OldAxis;
use stt_train::{Signal, get_pncc_features_with_delta};
use tokio::fs::read_dir;
use wavers::{AsNdarray, Wav};

use rubato::{
    Resampler,
    SincFixedIn,
    SincInterpolationParameters, // Note the "Sinc" prefix in 0.16.x
    SincInterpolationType,
    WindowFunction,
};

cfg_select! {
    feature = "f64" => {
        pub type Float = f64;
    }
    _ => {
        pub type Float = f32;
    }
}

/// vec_signal is an array of 16000Hz signal from many audio samples
pub fn map_phone_pncc(phone: &str, vec_signal: &[Signal]) -> (String, Vec<ArcArray2<Float>>) {
    let vec_pncc = vec_signal
        .iter()
        .map(|signal| get_pncc_features_with_delta(signal).into_shared())
        .collect::<Vec<_>>();
    (phone.to_string(), vec_pncc)
}

/// Return the sample rate and the vec![vec![data]channel]
pub fn lire_fichier_wav(
    chemin: &Path,
) -> Result<(Vec<Vec<Float>>, i32), Box<dyn std::error::Error>> {
    let mut wav: Wav<Float> = Wav::from_path(chemin)?;

    let (samples_channels, sample_rate) = wav
        .as_ndarray()
        .unwrap_or_else(|_| panic!("File {:?} is corrupt", chemin));
    let vec_of_vecs: Vec<Vec<Float>> = samples_channels
        .map_axis(OldAxis(0), |channel| channel.to_vec())
        .to_vec();

    Ok((vec_of_vecs, sample_rate))
}

pub async fn tanisa_wav(dossier: &str) -> tokio::io::Result<Vec<PathBuf>> {
    let chemin_dossier = Path::new(dossier);

    let mut fichiers_wav = Vec::new();

    let mut dir_content = read_dir(chemin_dossier).await?;
    while let Some(entry) = dir_content.next_entry().await? {
        let path = entry.path();

        if path.is_file()
            && let Some(ext) = path.extension()
            && ext.eq_ignore_ascii_case("wav")
        {
            fichiers_wav.push(path);
        }
    }

    fichiers_wav.sort();

    Ok(fichiers_wav)
}

/// Resamples multi-channel audio to exactly 16,000 Hz using high-end sinc resampling.
pub fn resample_to_16k(input: &[Vec<Float>], source_sample_rate: u32) -> Vec<Vec<Float>> {
    const TARGET_SR: u32 = 16_000;

    if source_sample_rate == TARGET_SR {
        return input.to_vec();
    }

    let num_channels = input.len();
    if num_channels == 0 || input[0].is_empty() {
        return Vec::new();
    }

    let ratio = TARGET_SR as Float / source_sample_rate as Float;
    let input_frames = input[0].len();

    // High-fidelity parameters for 16kHz target
    let params = SincInterpolationParameters {
        sinc_len: 512,
        f_cutoff: 0.925,
        interpolation: SincInterpolationType::Cubic,
        oversampling_factor: 256, // Pushed to 128 for a smoother LUT
        window: WindowFunction::BlackmanHarris2, // BH2 has better side-lobe rejection
    };

    let mut resampler = SincFixedIn::<Float>::new(
        ratio as f64,
        2.0, // Max ratio deviation tolerance
        params,
        input_frames,
        num_channels,
    )
    .expect("Failed to create sinc resampler");

    // rubato's process returns Result<Vec<Vec<T>>, ResampleError>
    resampler.process(input, None).expect("Resampling failed")
}

pub fn merge(input: &[Vec<Float>]) -> Vec<Float> {
    if input.is_empty() {
        return Vec::new();
    }
    let inner_len = input[0].len();
    let mut output = Vec::new();
    (0..inner_len).for_each(|i| {
        (0..input.len()).for_each(|j| {
            output.push(input[j][i]);
        });
    });
    output
}
