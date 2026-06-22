# STT MG
This is a speech to text software using hmm gmm algo and written in rust.
There are 4 features:
- f32: use float 32
- f64: use float 64
- cuda: use cuda andd float 64
- wgpu: use wgpu and float 32

These feature are choosed according your need and hardware.

It is still using monophone, there are not yet a model for the triphone.

The monophone audio should be in wav format and normalized to 3db.
Inside the training dir, each phoneme should have a directory with it's name, for example: a, e, i, z, k.
And inside each phoneme directory, there should be all wav file of this phoneme.
An exception is for the SIL, it's directory name should be `foana`

There is a predict folder for the wav file prediction.
And at the last, it support real time decoding.

# Config file
The training configurations are :
- tolerance
- convergence
- number of gmm mixture
- iteration

The predict main configuration is the beam size.

## Prerequisite
To use the package stt-cuda, you need to have nvcc installed and available in your PATH.

## To launch
```
cargo run --release -p stt-f64 -- --config-file config-f64.toml
```
