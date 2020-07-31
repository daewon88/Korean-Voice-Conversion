import os
import numpy as np
import tensorflow as tf
from model import *
from preprocess import *

sampling_rate = 16000
num_mcep = 24
frame_period = 5.0
n_frames = 128

model_dir = './model'
validation_A_dir = './test_A'
validation_B_dir = './test_B'
validation_AB_output_dir = './test_output_AB'
validation_BA_output_dir = './test_output_BA'

if validation_A_dir is not None:
    if not os.path.exists(validation_BA_output_dir):
        os.makedirs(validation_BA_output_dir)

if validation_B_dir is not None:
    if not os.path.exists(validation_AB_output_dir):
        os.makedirs(validation_AB_output_dir)

model = CycleGAN(num_features=num_mcep,n_frames=n_frames)
model.load(model_dir)


mcep_normalization_params = np.load(os.path.join(model_dir, 'mcep_normalization.npz'))
coded_sps_A_mean = mcep_normalization_params['mean_A']
coded_sps_A_std = mcep_normalization_params['std_A']
coded_sps_B_mean = mcep_normalization_params['mean_B']
coded_sps_B_std = mcep_normalization_params['std_B']

logf0s_normalization_params = np.load(os.path.join(model_dir, 'logf0s_normalization.npz'))
log_f0s_mean_A = logf0s_normalization_params['mean_A']
log_f0s_std_A = logf0s_normalization_params['std_A']
log_f0s_mean_B = logf0s_normalization_params['mean_B']
log_f0s_std_B = logf0s_normalization_params['std_B']


if validation_A_dir is not None:
    print('Generating Validation Data B from A...')
    for file in os.listdir(validation_A_dir):
        filepath = os.path.join(validation_A_dir, file)
        wav, _ = librosa.load(filepath, sr=sampling_rate, mono=True)
        wav = wav_padding(wav=wav, sr=sampling_rate, frame_period=frame_period, multiple=4)
        f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
        f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_mean_A, std_log_src=log_f0s_std_A,
                                        mean_log_target=log_f0s_mean_B, std_log_target=log_f0s_std_B)
        coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
        coded_sp_norm = (coded_sp - coded_sps_A_mean) / coded_sps_A_std
        coded_sp_converted_norm = model.test(inputs=np.array([coded_sp_norm]), direction='A2B')[0]
        coded_sp_converted = coded_sp_converted_norm * coded_sps_B_std + coded_sps_B_mean
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted, dtype=np.double)
        decoded_sp_converted = world_decode_spectral_envelop(coded_sp=coded_sp_converted, fs=sampling_rate)
        wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted, ap=ap,
                                                    fs=sampling_rate, frame_period=frame_period)
        librosa.output.write_wav(os.path.join(validation_AB_output_dir, os.path.basename(file)), wav_transformed,
                                    sampling_rate)


if validation_B_dir is not None:
    print('Generating Validation Data A from B...')
    for file in os.listdir(validation_B_dir):
        filepath = os.path.join(validation_B_dir, file)
        wav, _ = librosa.load(filepath, sr=sampling_rate, mono=True)
        wav = wav_padding(wav=wav, sr=sampling_rate, frame_period=frame_period, multiple=4)
        f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
        f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_mean_B, std_log_src=log_f0s_std_B,
                                        mean_log_target=log_f0s_mean_A, std_log_target=log_f0s_std_A)
        coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
        coded_sp_norm = (coded_sp - coded_sps_B_mean) / coded_sps_B_std
        coded_sp_converted_norm = model.test(inputs=np.array([coded_sp_norm]), direction='B2A')[0]
        coded_sp_converted = coded_sp_converted_norm * coded_sps_A_std + coded_sps_A_mean
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted, dtype=np.double)
        decoded_sp_converted = world_decode_spectral_envelop(coded_sp=coded_sp_converted, fs=sampling_rate)
        wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted, ap=ap,
                                                    fs=sampling_rate, frame_period=frame_period)
        librosa.output.write_wav(os.path.join(validation_BA_output_dir, os.path.basename(file)), wav_transformed,
                                    sampling_rate)



