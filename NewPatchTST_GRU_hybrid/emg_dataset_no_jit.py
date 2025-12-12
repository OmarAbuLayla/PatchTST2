# emg_dataset_no_jit.py
# -----------------------------------------------------
# Loads MFSC features from precomputed_mfsc/<subject>/<session>/<file>.npy
# Same as jitter version, but WITHOUT any jitter.
# -----------------------------------------------------

import os
import numpy as np
import torch
import random

# Load subject list (Linux path)
EMG_SUBJECT_LIST = np.load(
    r"/mnt/d/Omar/work GJU/Codes/code4AVE-Speech/CLS_emg_only_new/fix/emg_subject.npy",
    allow_pickle=True
).tolist()

# Correct absolute MFSC path on C drive
MFSC_root = r"/mnt/c/EMG_data/precomputed_mfsc"


class MyDataset:
    def __init__(self, set_name, dataset_root):
        self.set_name = set_name
        self.file_list = self.build_file_list()
        print(f"{set_name} samples:", len(self.file_list))

    def build_file_list(self):
        subject_list = EMG_SUBJECT_LIST

        # 70 train, 10 val, 20 test
        if self.set_name == "train":
            subjects = subject_list[:70]
        elif self.set_name == "val":
            subjects = subject_list[70:80]
        else:
            subjects = subject_list[80:]

        entries = []
        for subject_path in subjects:
            subject_id = os.path.basename(subject_path)

            for session in os.listdir(subject_path):
                sess_path = os.path.join(MFSC_root, subject_id, session)

                if not os.path.isdir(sess_path):
                    continue

                for file in os.listdir(sess_path):
                    if file.endswith(".npy"):
                        full_path = os.path.join(sess_path, file)
                        entries.append(full_path)

        random.shuffle(entries)
        return entries

    def __getitem__(self, idx):
        npy_path = self.file_list[idx]

        data = np.load(npy_path, allow_pickle=True).item()
        feat = data["feat"]        # (1, C, 36, 36)
        label = data["label"]

        return torch.FloatTensor(feat), label, npy_path

    def __len__(self):
        return len(self.file_list)
