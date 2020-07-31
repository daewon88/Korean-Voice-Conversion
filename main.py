from preprocess import *
from model import *
import os
import time
import pickle
import librosa

num_epochs = 5000
mini_batch_size = 1
generator_learning_rate = 0.0002
generator_learning_rate_decay = generator_learning_rate / 200000
discriminator_learning_rate = 0.00005
discriminator_learning_rate_decay = discriminator_learning_rate / 200000
sampling_rate = 16000
num_mcep = 24
frame_period = 5.0
n_frames = 128
np.random.seed(8)

train_A_dir = './train_A'
train_B_dir = './train_B'
model_dir = './model'
validation_A_dir = './validation_A'
validation_B_dir = './validation_B'
validation_AB_output_dir = './validation_out_AB'
validation_BA_output_dir = './validation_out_BA'


if os.path.isfile('coded_sps_A_norm.pickle') and os.path.isfile('coded_sps_B_norm.pickle'):
    with open('coded_sps_A_norm.pickle','rb') as f:
        coded_sps_A_norm = pickle.load(f)
    with open('coded_sps_B_norm.pickle','rb') as f:
        coded_sps_B_norm = pickle.load(f)

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

else:

    print('Preprocessing Data...')
    start_time = time.time()
    wavs_A = load_wavs(wav_dir=train_A_dir, sr=sampling_rate)
    wavs_B = load_wavs(wav_dir=train_B_dir, sr=sampling_rate)
    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = world_encode_data(wavs=wavs_A, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)
    f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = world_encode_data(wavs=wavs_B, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)
    log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)
    log_f0s_mean_B, log_f0s_std_B = logf0_statistics(f0s_B)

    print('Log Pitch A')
    print('Mean: %f, Std: %f' % (log_f0s_mean_A, log_f0s_std_A))
    print('Log Pitch B')
    print('Mean: %f, Std: %f' % (log_f0s_mean_B, log_f0s_std_B))

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transoform(coded_sps=coded_sps_A)
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = coded_sps_normalization_fit_transoform(coded_sps=coded_sps_B)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    np.savez(os.path.join(model_dir, 'logf0s_normalization.npz'), mean_A=log_f0s_mean_A, std_A=log_f0s_std_A, mean_B=log_f0s_mean_B, std_B=log_f0s_std_B)
    np.savez(os.path.join(model_dir, 'mcep_normalization.npz'), mean_A=coded_sps_A_mean, std_A=coded_sps_A_std, mean_B=coded_sps_B_mean, std_B=coded_sps_B_std)

    if validation_A_dir is not None:
        if not os.path.exists(validation_BA_output_dir):
            os.makedirs(validation_BA_output_dir)

    if validation_B_dir is not None:
        if not os.path.exists(validation_AB_output_dir):
            os.makedirs(validation_AB_output_dir)

    with open('coded_sps_A_norm.pickle','wb') as f:
        pickle.dump(coded_sps_A_norm, f)
    with open('coded_sps_B_norm.pickle','wb') as f:
        pickle.dump(coded_sps_B_norm, f)

    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Preprocessing Done.')
    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60),
                                                                    (time_elapsed % 60 // 1)))


model = CycleGAN(num_features=num_mcep,n_frames=n_frames)
for epoch in range(num_epochs):
    print('Epoch: %d' % epoch)
    start_time_epoch = time.time()
    dataset_A, dataset_B = sample_train_data(dataset_A=coded_sps_A_norm, dataset_B=coded_sps_B_norm, n_frames=n_frames)
    n_samples = dataset_A.shape[0]
    print('Samples: ', n_samples)

    for i in range(n_samples // mini_batch_size):
        num_iterations = n_samples // mini_batch_size * epoch + i
        if num_iterations == 10000:
            model.combined_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5), 
                                         loss=['mse','mse','mae','mae','mae','mae'], loss_weights=[1, 1, 10, 10, 0, 0])
            print('recompile!!')
        
        if num_iterations > 200000:
            generator_learning_rate = max(0, generator_learning_rate - generator_learning_rate_decay)
            discriminator_learning_rate = max(0, discriminator_learning_rate - discriminator_learning_rate_decay)
            tf.keras.backend.set_value(model.d_A.optimizer.learning_rate, discriminator_learning_rate)
            tf.keras.backend.set_value(model.d_B.optimizer.learning_rate, discriminator_learning_rate)
            tf.keras.backend.set_value(model.combined_model.optimizer.learning_rate, generator_learning_rate)

        else:
            tf.keras.backend.set_value(model.d_A.optimizer.learning_rate, discriminator_learning_rate)
            tf.keras.backend.set_value(model.d_B.optimizer.learning_rate, discriminator_learning_rate)
            tf.keras.backend.set_value(model.combined_model.optimizer.learning_rate,generator_learning_rate)
        
       
        start = i * mini_batch_size
        end = (i + 1) * mini_batch_size
        input_A = dataset_A[start:end]
        input_B = dataset_B[start:end]
        dA_loss, dB_loss, generator_loss = model.train(input_A,input_B,mini_batch_size)
        if i%50 == 0 :
            print("Iteration : ", num_iterations)
            print("dA_loss:{}, dB_loss:{}, generator_loss:{}".format(dA_loss, dB_loss, generator_loss))

    end_time_epoch = time.time()
    time_elapsed_epoch = end_time_epoch - start_time_epoch
    print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))

    if epoch % 50 == 0:
        model.save(model_dir)
        
    if validation_A_dir is not None:
        if epoch % 50 == 0:
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
        if epoch % 50 == 0:
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

