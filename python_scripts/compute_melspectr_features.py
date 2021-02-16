from scipy import stats
import numpy as np
import soundfile as sf
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse
import resampy
import librosa
import logmelspectr_params as params
import matplotlib.pyplot as plt


default_melspectr_dest_folder = "/nas/home/cborrelli/tripletloss_bot/features/melspectr"


def read_audio(audio_filename):
    audio, sr = sf.read(audio_filename, dtype='int16')
    assert audio.dtype == np.int16, 'Bad sample type: %r' % audio.dtype
    samples = audio / 32768.0  # Convert to [-1.0, +1.0]

    # Stereo to mono
    if len(samples.shape) > 1:
        samples = samples.mean(axis=1)

    return samples, sr


def compute_melspectrogram(arg, melspectr_dest_root, audio_folder):
    """ Compute the spectrogram of a signal
    using the function scipy.signal.spectrogram
    """
    # Read audio
    index = arg[0]
    row = arg[1]
    audio_filename = row["audio_filename"] + '.flac'
    audio, fs = read_audio(audio_filename=os.path.join(audio_folder, audio_filename))
    selection_dur = 1 * fs
    len_audio = len(audio)

    if len_audio < selection_dur:
        np.pad(audio, pad_width= selection_dur - len_audio, mode='constant')
    # Select one second
    mid_index = len_audio // 2
    audio = audio[ mid_index - selection_dur//2 : mid_index + selection_dur//2]
    # Peak normalization
    audio = audio / np.max(audio)
    # Params
    n_fft = 512
    num_mel_bins = 256
    lower_edge_hertz = 0.0
    upper_edge_hertz = fs // 2
    mel = librosa.feature.melspectrogram(y=audio, sr=fs,
                                       n_fft=n_fft, win_length=n_fft, hop_length=n_fft,
                                       n_mels=num_mel_bins, fmin=lower_edge_hertz, fmax=upper_edge_hertz)


    mel_norm = stats.zscore(mel, axis=0)
    melspectr_out_name = os.path.join(melspectr_dest_root, row['audio_filename']+'.npy')

    np.save(melspectr_out_name, mel_norm)
    return


def compute_features(audio_folder, txt_path, melspectr_dest_root, data_subset):
    # Open dataset df
    df = pd.read_csv(txt_path, sep=" ", header=None)
    df.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
    df = df.drop(columns="null")

    # Prepare parallel execution
    args_list = list(df.iterrows())
    dest_subset_folder = '{}'.format(data_subset)
    melspectr_dest_subset_root = os.path.join(melspectr_dest_root, dest_subset_folder)

    if not os.path.exists(melspectr_dest_subset_root):
        os.makedirs(melspectr_dest_subset_root)

    print("Save in {}".format(melspectr_dest_subset_root))
    #compute_melspectrogram(args_list[0], melspectr_dest_root=melspectr_dest_subset_root, audio_folder=audio_folder)


    compute_features_partial = partial(compute_melspectrogram, melspectr_dest_root=melspectr_dest_subset_root,
                        audio_lsfolder=audio_folder)
    # Run parallel execution
    pool = Pool(cpu_count() // 2)
    _ = list(tqdm(pool.imap(compute_features_partial, args_list), total=len(args_list)))
    return


if __name__ == '__main__':
    # parse input arguments
    os.nice(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('--melspectr_dest_folder', type=str, required=False, default=default_melspectr_dest_folder)
    parser.add_argument('--data_subset', type=str, required=True)


    args = parser.parse_args()
    logmelspectr_dest_folder = args.melspectr_dest_folder
    data_subset = args.data_subset

    audio_folder = '/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_{}/flac'.format(data_subset)

    if data_subset != 'train':
        txt_path = '/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{}.trl.txt'.format(
            data_subset)
    else:
        txt_path = '/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{}.trn.txt'.format(
            data_subset)


    compute_features(audio_folder, txt_path, logmelspectr_dest_folder, data_subset)
