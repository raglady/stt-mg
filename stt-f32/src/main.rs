use std::{
    collections::{BTreeMap, VecDeque},
    io::BufRead,
    path::PathBuf,
    process,
    sync::Arc,
    time::Duration,
};

use chrono::{Local, NaiveDate};
use clap::Parser;
use common::{
    Args, CommonSettings, lire_fichier_wav, map_phone_pncc, merge, resample_to_16k, tanisa_wav,
};
use config::{ConfigBuilder, builder::DefaultState};
use cpal::{
    InputCallbackInfo, OutputCallbackInfo, Sample, StreamError, SupportedStreamConfig,
    available_hosts, default_host,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use env_logger::Env;
use ndarray::Array2;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use stt_train::{
    Signal, get_pncc_features_with_delta,
    monophone::{MonoPhone, bigram::Bigram},
    real_time::real_time_decode,
    to_mono,
    traits::{baum_welch_trait::BaumWelchTrait, viterbi_trait::ViterbiTrait},
};

use log::info;
use tokio::{
    fs::{File, read_dir},
    io::{AsyncReadExt, AsyncWriteExt},
    sync::{Mutex, RwLock, broadcast, watch},
    task::JoinSet,
};

#[tokio::main]
async fn main() -> tokio::io::Result<()> {
    let args = Args::parse();

    let config_path = if let Some(config_file) = args.config_file {
        PathBuf::from(config_file)
    } else {
        // loading configuration
        let config_dir = std::env::var("CONFIG_DIR")
            .ok()
            .map_or(PathBuf::from("."), PathBuf::from);
        config_dir.join("config.toml")
    };

    let config_builder: ConfigBuilder<DefaultState> = config::Config::builder()
        .add_source(config::File::from(config_path))
        // Add in settings from environment variables (with a prefix of APP and '__' as separator)
        // E.g. `APP_APPLICATION__PORT=5001 would set `Settings.application.port`
        .add_source(
            config::Environment::with_prefix("APP")
                .prefix_separator("_")
                .separator("__")
                .try_parsing(true),
        );

    let settings: Arc<CommonSettings> = Arc::new(
        config_builder
            .build()
            .unwrap()
            .try_deserialize::<CommonSettings>()
            .unwrap(),
    );

    // Initialize the logger
    let env = Env::default()
        .filter_or("LOG_LEVEL", "info")
        .write_style_or("LOG_STYLE", "auto");
    env_logger::init_from_env(env);

    info!("App starting ...");

    let train_monophone_data: Arc<RwLock<BTreeMap<String, Vec<Signal>>>> =
        Arc::new(RwLock::new(BTreeMap::new()));

    // read train dir
    info!("Gathering train audio");
    let mut train_dir = read_dir(&settings.stt_settings.monophone_training.train_dir).await?;
    let mut join_set: JoinSet<Result<(), tokio::io::Error>> = JoinSet::new();

    while let Some(class) = train_dir.next_entry().await? {
        let train_monophone_data_clone = train_monophone_data.clone();
        join_set.spawn(async move {
            if class.file_type().await?.is_dir() {
                let wav_list = tanisa_wav(class.path().to_str().unwrap()).await?;
                let wav_vec: Vec<Vec<f32>> = wav_list
                    .iter()
                    .map(|wav_path| {
                        let audio = lire_fichier_wav(wav_path).unwrap();
                        let resampled = resample_to_16k(&audio.0, audio.1.try_into().unwrap());
                        to_mono(&merge(&resampled), audio.0.len().try_into().unwrap())
                    })
                    .collect();
                if !wav_vec.is_empty() {
                    let mut guard = train_monophone_data_clone.write().await;
                    guard.insert(class.file_name().into_string().unwrap(), wav_vec);
                    drop(guard);
                }
            }
            Ok(())
        });
    }

    // Join all tasks as they finish
    while let Some(res) = join_set.join_next().await {
        let _ = res?;
        //println!("Task finished: {:?}", res.unwrap());
    }

    let guard_train = train_monophone_data.read().await;

    // initial training
    let hash_phone_pncc: BTreeMap<String, _> = guard_train
        .iter()
        .map(|(phone, value)| map_phone_pncc(phone, value))
        .collect();

    if settings.stt_settings.monophone_training.enable {
        let phoneme_vec: Vec<String> =

        // load phoneme
        {
            let mut phoneme_file = File::open(&settings.stt_settings.storage.phonemes_file).await?;
            let mut phoneme_str = String::new();
            phoneme_file.read_to_string(&mut phoneme_str).await?;
            ron::from_str(&phoneme_str).unwrap()
        };
        // load bigram_log_prob
        let bigram_log_prob: Array2<f32> = {
            let mut log_prob_bigram_file =
                File::open(&settings.stt_settings.storage.log_prob_bigram_file).await?;
            let mut bigra_log_prob_str = String::new();
            log_prob_bigram_file
                .read_to_string(&mut bigra_log_prob_str)
                .await?;
            ron::from_str(&bigra_log_prob_str).unwrap()
        };

        hash_phone_pncc.par_iter().for_each(|(k, v)| {
            v.iter().for_each(|arr| {
                if arr.is_any_nan() {
                    println!("error nan on phone {}", k);
                } else if arr.is_any_infinite() {
                    println!("error infinite on phone {}", k);
                }
            });
        });

        let (global_mean, global_var) = MonoPhone::compute_global_params(
            &hash_phone_pncc
                .values()
                .flatten()
                .cloned()
                .collect::<Vec<_>>(),
        );

        let mut monophone = MonoPhone::new(
            &Bigram::new(&phoneme_vec, &bigram_log_prob),
            settings.stt_settings.monophone_training.tolerance,
            settings.stt_settings.monophone_training.convergence,
            global_mean.view(),
            global_var.view(),
        );

        for (phone, _mfccs) in hash_phone_pncc.iter() {
            monophone.flat_start(
                phone,
                // mfccs,
                &hash_phone_pncc
                    .values()
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>(),
            );
        }

        for _ in 1..settings.stt_settings.monophone_training.component_per_state {
            monophone
                .baum_welch(
                    Arc::new(
                        hash_phone_pncc
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect::<BTreeMap<_, _>>(),
                    ),
                    3,
                )
                .await;

            monophone
                .get_phone_hmm_gmm_mut()
                .iter_mut()
                .for_each(|(_, v)| v.get_states_mut().iter_mut().for_each(|gmm| gmm.split_up()));
        }

        monophone
            .baum_welch(
                Arc::new(
                    hash_phone_pncc
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect::<BTreeMap<_, _>>(),
                ),
                settings.stt_settings.monophone_training.iteration,
            )
            .await;

        monophone.generate_fn_model();

        {
            info!("Saving monophone model");
            let mut json_model =
                File::create(&settings.stt_settings.storage.monophone_modele_file).await?;
            json_model
                .write_all(ron::to_string(&monophone).unwrap().as_bytes())
                .await?;
            json_model.sync_all().await.unwrap();
            info!("Monophone model saved");
        }
    }

    if settings.stt_settings.predict.enable || settings.stt_settings.predict.real_time {
        let model = {
            info!("Loading model");

            let mut ron_model =
                File::open(&settings.stt_settings.storage.monophone_modele_file).await?;
            let mut string_model = String::new();
            ron_model.read_to_string(&mut string_model).await?;

            let monophone = ron::from_str::<MonoPhone>(&string_model).unwrap();
            info!("Model loaded");
            Arc::new(monophone) as Arc<dyn ViterbiTrait>
        };

        // let vec_model_guard = vec_model.read().unwrap();
        // info!("debut prediction: {}", Local::now().format("%H:%M:%S%.3f"));

        if settings.stt_settings.predict.enable {
            let to_predict = Arc::new(RwLock::new(BTreeMap::new()));

            let mut predict_dir = read_dir(&settings.stt_settings.predict.predict_dir).await?;
            let mut join_set = JoinSet::new();

            while let Some(p) = predict_dir.next_entry().await? {
                let to_predict_clone = to_predict.clone();
                let wav_path = p.path();
                if wav_path.is_file()
                    && let Some(ext) = wav_path.extension()
                    && ext.eq_ignore_ascii_case("wav")
                {
                    join_set.spawn(async move {
                        let audio = lire_fichier_wav(wav_path.as_path()).unwrap();
                        let resampled = resample_to_16k(&audio.0, audio.1.try_into().unwrap());
                        let mut guard = to_predict_clone.write().await;
                        guard.insert(
                            wav_path.file_name().unwrap().display().to_string(),
                            to_mono(&merge(&resampled), audio.0.len().try_into().unwrap()),
                        );
                        drop(guard);
                    });
                }
            }

            // Join all tasks as they finish
            while let Some(res) = join_set.join_next().await {
                res?;
                //println!("Task finished: {:?}", res.unwrap());
            }

            let to_predict_clone = to_predict.clone();
            let guard = to_predict_clone.read().await;

            for (filename, signal) in guard.iter() {
                let buffer = Arc::new(RwLock::new(Vec::new()));
                let mfcc = get_pncc_features_with_delta(signal).to_shared();

                let output = model
                    .clone()
                    .viterbi_beam_search(buffer, mfcc, settings.stt_settings.predict.beam_size)
                    .await;

                println!("{} '{}'", filename, output);
            }

            drop(guard);
        }
        // info!("fin prediction: {}", Local::now().format("%H:%M:%S%.3f"));

        if settings.stt_settings.predict.real_time {
            real_time_audio(model.clone(), settings.clone()).await;
        }
    }

    if settings.general.end_press_key {
        println!("Manindria bokotra iray raha hiala");
        let _ = std::io::stdin().lock().lines().next();
    }

    Ok(())
}

async fn real_time_audio(model: Arc<dyn ViterbiTrait>, settings: Arc<CommonSettings>) {
    let available_host = available_hosts();
    info!("available_hosts: {:?}", available_host);

    let default_host = default_host();
    let devices = default_host.devices().expect("Cannot get all devices");
    let devices_name: Vec<String> = devices
        .map(|d| d.description().unwrap().name().to_string())
        .collect();
    info!("devices name: {:?}", devices_name);

    let input_devices_name: Vec<String> = default_host
        .input_devices()
        .expect("cannot get input devices")
        .map(|d| d.description().unwrap().name().to_string())
        .collect();
    info!("input devices name: {:?}", input_devices_name);

    let output_devices_name: Vec<String> = default_host
        .output_devices()
        .expect("cannot get output devices")
        .map(|d| d.description().unwrap().name().to_string())
        .collect();
    info!("output devices name: {:?}", output_devices_name);

    let default_input_device = default_host
        .default_input_device()
        .expect("cannot get default input devices");
    info!(
        "default input devices name: {:?}",
        default_input_device.description().unwrap().name()
    );

    let default_output_device = default_host
        .default_output_device()
        .expect("cannot get default output devices");
    info!(
        "default output devices name: {:?}",
        default_output_device.description().unwrap().name()
    );

    let default_input_config = default_input_device
        .default_input_config()
        .expect("cannot get default input config");
    //  let supported_input_config = default_output_device.supported_input_configs().expect("cannot get default output config").next().unwrap();
    let input_config = SupportedStreamConfig::new(
        1,
        16_000,
        *default_input_config.buffer_size(),
        cpal::SampleFormat::F32,
    );
    info!("default input config: {:?}", default_input_config);
    info!("input config: {:?}", input_config);

    let (storage_tx, storage_rx) = watch::channel(Vec::new());

    let input_stream = default_input_device
        .build_input_stream(
            &input_config.into(),
            move |data, info| input_callback(storage_tx.clone(), data, info),
            input_error,
            None,
        )
        .unwrap();

    info!("go");
    input_stream.play().unwrap();
    let storage_rx_clone = storage_rx.clone();
    let model_clone = model.clone();
    tokio::spawn(async move {
        let (output_tx, mut output_rx) = broadcast::channel(100);

        tokio::spawn(async move {
            real_time_decode(
                model_clone,
                storage_rx_clone,
                output_tx,
                Arc::new(settings.stt_settings.clone()),
            )
            .await
        });

        loop {
            if let Ok(output) = output_rx.recv().await {
                println!("{:?}", output);
            }
        }
    });

    let _ = std::io::stdin().lock().lines().next();

    let record_duration = Duration::from_secs(3);
    let start = std::time::Instant::now();
    while start.elapsed() < record_duration {
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn input_callback(storage_tx: watch::Sender<Vec<f32>>, data: &[f32], _info: &InputCallbackInfo) {
    storage_tx.send(data.to_vec()).unwrap();
    //    println!("input callback info: {:?}", info);
    //    println!("input data: {:?}", data);
}

fn input_error(stream_error: StreamError) {
    println!("input stream error: {:?}", stream_error);
}

#[allow(dead_code)]
async fn output_callback(
    storage: Arc<Mutex<VecDeque<f32>>>,
    data: &mut [f32],
    _info: &OutputCallbackInfo,
) {
    let mut storage_guard = storage.lock().await;
    for d in data.iter_mut() {
        if storage_guard.len() > 1 {
            *d = storage_guard.pop_front().unwrap();
        } else {
            *d = Sample::EQUILIBRIUM;
        }
    }
    //    println!("output callback info: {:?}", info);
}

#[allow(dead_code)]
fn output_error(stream_error: StreamError) {
    println!("output stream error: {:?}", stream_error);
}
