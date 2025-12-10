import os
import numpy as np

#DATA_ROOT = r"D:\Omar\work GJU\Codes\AVE-Speech" #for windows
DATA_ROOT = r"/mnt/d/Omar/AVE-Speech_treated_15VC_noNorm" 

splits = ["Train", "Val", "Test"]

subject_paths = []

for split in splits:
    split_dir = os.path.join(DATA_ROOT, split, "EMG")
    if not os.path.isdir(split_dir):
        continue
    for subject in sorted(os.listdir(split_dir)):
        subject_path = os.path.join(split_dir, subject)
        if os.path.isdir(subject_path):
            subject_paths.append(subject_path)

np.save("emg_subject.npy", subject_paths)

print("Saved emg_subject.npy with", len(subject_paths), "subjects.")
