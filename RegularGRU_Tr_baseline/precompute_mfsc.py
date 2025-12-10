# precompute_mfsc.py
# ---------------------------------------------------
# Precomputes MFSC features using the EXACT dataset
# structure used in your scattering precompute script.
# ---------------------------------------------------

import os
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
import librosa
from scipy import signal

# ------------------------------------------
# Load SUBJECT LIST (same as scattering)
# ------------------------------------------
EMG_SUBJECT_LIST = np.load(
    r"D:\Omar\work GJU\Codes\code4AVE-Speech\CLS_emg_only_new\emg_subject6.npy",
    allow_pickle=True
).tolist()

# ------------------------------------------
# Output features folder
# ------------------------------------------
SAVE_DIR = "./precomputed_mfsc"
os.makedirs(SAVE_DIR, exist_ok=True)


# ------------------------------------------
# ORIGINAL FILTER FUNCTION
# ------------------------------------------
def filter_emg(raw):
    fs = 1000
    b1, a1 = signal.iirnotch(50, 30, fs)
    b2, a2 = signal.iirnotch(150, 30, fs)
    b3, a3 = signal.iirnotch(250, 30, fs)
    b4, a4 = signal.iirnotch(350, 30, fs)
    b5, a5 = signal.butter(4, [10/(fs/2), 400/(fs/2)], 'bandpass')

    x = signal.filtfilt(b1, a1, raw, axis=1)
    x = signal.filtfilt(b2, a2, x, axis=1)
    x = signal.filtfilt(b3, a3, x, axis=1)
    x = signal.filtfilt(b4, a4, x, axis=1)
    x = signal.filtfilt(b5, a5, x, axis=1)
    return x


# ------------------------------------------
# ORIGINAL MFSC
# ------------------------------------------
def compute_mfsc(x):
    x = x[:, 250:, :]
    n_mels = 36
    sr = 1000
    channel_list = []

    for j in range(x.shape[-1]):
        mfsc_x = np.zeros((x.shape[0], 36, n_mels))

        for i in range(x.shape[0]):
            sig = np.asfortranarray(x[i, :, j])
            mel = librosa.feature.melspectrogram(
                y=sig,
                sr=sr,
                n_mels=n_mels,
                n_fft=200,
                hop_length=50,
                power=2.0
            )
            mel = librosa.power_to_db(mel).T
            mfsc_x[i, :, :] = mel

        mfsc_x = mfsc_x[..., None]
        channel_list.append(mfsc_x)

    out = np.concatenate(channel_list, axis=-1)
    mu = out.mean()
    std = out.std() + 1e-9
    out = (out - mu) / std
    return out.transpose(0, 3, 1, 2)  # (1, C, 36, 36)


# ------------------------------------------
# PROCESS ONE SUBJECT
# ------------------------------------------
def process_subject(subject_path):
    subject_id = os.path.basename(subject_path)

    for session in os.listdir(subject_path):
        sess_path = os.path.join(subject_path, session)
        if not os.path.isdir(sess_path):
            continue

        save_dir = os.path.join(SAVE_DIR, subject_id, session)
        os.makedirs(save_dir, exist_ok=True)

        files = [f for f in os.listdir(sess_path) if f.endswith(".mat")]
        if not files:
            continue

        for file in tqdm(files, desc=f"[MFSC] {subject_id}/{session}"):
            out_path = os.path.join(save_dir, file.replace(".mat", ".npy"))
            if os.path.exists(out_path):
                continue

            mat = loadmat(os.path.join(sess_path, file))["data"]
            mat = mat[None, ...]           # (1, T, C)

            filt = filter_emg(mat)
            mfsc = compute_mfsc(filt)

            label = int(file.split(".")[0])
            np.save(out_path, {"feat": mfsc, "label": label})


# ------------------------------------------
# MAIN
# ------------------------------------------
for subj in tqdm(EMG_SUBJECT_LIST, desc="ALL SUBJECTS"):
    process_subject(subj)

print("✓ MFSC generation complete.")
