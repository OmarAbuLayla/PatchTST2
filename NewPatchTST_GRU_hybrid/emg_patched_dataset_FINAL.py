# ------------------------------------------------------------
#  Patchified MFSC Dataset (Correct Subject/Session Traversal)
#  FINAL VERSION — aligned with emg_dataset_no_jit.py
# ------------------------------------------------------------

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# IMPORT the old working dataset information
from emg_dataset_no_jit import EMG_SUBJECT_LIST, MFSC_root


class PatchTSTDatasetFinal(Dataset):
    """
    Fully correct dataset:
    - Uses original subject/session traversal from emg_dataset_no_jit (works 100%)
    - Extracts per-channel MFSC patches
    - Outputs (L, 216) tokens + correct label
    """

    def __init__(self, split, patch_time=6, patch_freq=6,
                 stride_time=3, stride_freq=3):

        self.split = split
        self.pt = patch_time
        self.pf = patch_freq
        self.st = stride_time
        self.sf = stride_freq

        self.file_list = self.build_file_list()
        print(f"{split} samples: {len(self.file_list)}")

    # --------------------------------------------------------
    # MATCH SUBJECT SPLIT EXACTLY LIKE OLD DATASET
    # --------------------------------------------------------
    def build_file_list(self):

        subjects = EMG_SUBJECT_LIST

        if self.split == "train":
            split_subjects = subjects[:70]    # subjects 0–69
        elif self.split == "val":
            split_subjects = subjects[70:80]  # 70–79
        else:
            split_subjects = subjects[80:100] # 80–99

        file_paths = []

        # 🔥 CRITICAL PART: use REAL subject directory paths (old behavior)
        for subject_path in split_subjects:

            subject_id = os.path.basename(subject_path)
            mfsc_subject_root = os.path.join(MFSC_root, subject_id)

            if not os.path.isdir(mfsc_subject_root):
                print("WARNING: missing MFSC folder:", mfsc_subject_root)
                continue

            # Traverse MFSC sessions just like old dataset
            for session in sorted(os.listdir(mfsc_subject_root)):
                sess_dir = os.path.join(mfsc_subject_root, session)

                if not os.path.isdir(sess_dir):
                    continue

                # Append all MFSC files
                for f in sorted(os.listdir(sess_dir)):
                    if f.endswith(".npy"):
                        file_paths.append(os.path.join(sess_dir, f))

        random.shuffle(file_paths)
        return file_paths

    # --------------------------------------------------------
    # Extract correct per-channel patches (6,36,36 → L,216)
    # --------------------------------------------------------
    def patchify(self, mfsc):
        """
        mfsc shape: (6,36,36)
        return: (L, 216)
        """

        C = 6
        pt, pf = self.pt, self.pf
        st, sf = self.st, self.sf

        channel_patches = []

        for c in range(C):
            x = mfsc[c].unsqueeze(0).unsqueeze(0)  # (1,1,36,36)

            p = F.unfold(
                x,
                kernel_size=(pt, pf),
                stride=(st, sf)
            )  # → (1, pt*pf=36, L)

            p = p.squeeze(0).transpose(0, 1)  # → (L, 36)
            channel_patches.append(p)

        # Concatenate across channels → (L, 216)
        patches = torch.cat(channel_patches, dim=1)
        return patches

    # --------------------------------------------------------
    def __getitem__(self, idx):
        path = self.file_list[idx]
        data = np.load(path, allow_pickle=True).item()

        mfsc = torch.tensor(data["feat"], dtype=torch.float32).squeeze(0)  # (6,36,36)
        label = int(data["label"])

        patches = self.patchify(mfsc)  # (L,216)

        return patches, label, path

    def __len__(self):
        return len(self.file_list)
