# coding: utf-8
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from lr_scheduler import AdjustLR

# ✅ Correct dataset import
from emg_patched_dataset_FINAL import PatchTSTDatasetFinal

# Hybrid PatchTST → GRU model
from emg_model_patchtst_3 import EMG_PatchTST_GRU_Hybrid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 5
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# -------------------------------------------------------------
# DATA LOADER (uses PatchTSTDatasetFinal)
# -------------------------------------------------------------
def data_loader(args):
    dsets = {
        split: PatchTSTDatasetFinal(
            split,
            patch_time=args.patch_time,
            patch_freq=args.patch_freq,
            stride_time=args.stride_time,
            stride_freq=args.stride_freq,
        )
        for split in ['train', 'val', 'test']
    }

    dset_loaders = {
        split: torch.utils.data.DataLoader(
            dsets[split],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True
        )
        for split in ['train', 'val', 'test']
    }

    dset_sizes = {split: len(dsets[split]) for split in ['train', 'val', 'test']}
    print(f"\nStatistics: train: {dset_sizes['train']}, "
          f"val: {dset_sizes['val']}, test: {dset_sizes['test']}")

    return dset_loaders, dset_sizes



def reload_model(model, logger, path=""):
    if not path:
        logger.info("Training from scratch.")
        return model

    model_dict = model.state_dict()
    pretrained_dict = torch.load(path, map_location="cpu")

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    logger.info("*** model loaded ***")
    return model


def showLR(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]


# -------------------------------------------------------------
# TRAIN / VAL LOOP (already compatible with patch tokens)
# -------------------------------------------------------------
def train_test(model, dset_loaders, criterion, epoch, phase,
               optimizer, args, logger, use_gpu, save_path):

    model.train() if phase == 'train' else model.eval()

    if phase == 'train':
        logger.info('-' * 10)
        logger.info(f'Epoch {epoch}/{args.epochs - 1}')
        logger.info(f'Current Learning rate: {showLR(optimizer)}')

    running_loss = 0.
    running_corrects = 0
    running_all = 0

    for batch_idx, batch in enumerate(dset_loaders[phase]):

        if len(batch) == 3:
            inputs, targets, _ = batch
        else:
            inputs, targets = batch

        inputs = inputs.float().to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)

            if outputs.dim() == 3:
                outputs = outputs.mean(dim=1)

            loss = criterion(outputs, targets)

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        _, preds = outputs.max(1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == targets.data).item()
        running_all += len(inputs)

        if batch_idx == 0:
            since = time.time()
            logger.info(f"[DEBUG] inputs shape: {inputs.shape}, outputs shape: {outputs.shape}")

        elif batch_idx % args.interval == 0:
            print(
                f'Process: [{running_all}/{len(dset_loaders[phase].dataset)} '
                f'({100. * batch_idx / (len(dset_loaders[phase]) - 1):.0f}%)] '
                f'Loss: {running_loss / running_all:.4f} '
                f'Acc:{running_corrects / running_all:.4f} '
                f'Cost time:{time.time() - since:.0f}s',
                end='\r'
            )

    logger.info(
        f'{phase} Epoch:\t{epoch:2}\t'
        f'Loss: {running_loss / len(dset_loaders[phase].dataset):.4f}\t'
        f'Acc:{running_corrects / len(dset_loaders[phase].dataset):.4f}\n'
    )

    if phase == 'train':
        torch.save(
            model.state_dict(),
            os.path.join(save_path, f'{args.mode}_emg_{epoch+1}.pt')
        )

    return model


# -------------------------------------------------------------
# TEST EVAL
# -------------------------------------------------------------
def evaluate_test(model, dset_loaders, criterion, args, logger):
    model.eval()
    running_corrects = 0
    running_all = 0

    with torch.no_grad():
        for batch in dset_loaders["test"]:
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch

            inputs = inputs.float().to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            if outputs.dim() == 3:
                outputs = outputs.mean(dim=1)

            _, preds = outputs.max(1)
            running_corrects += torch.sum(preds == targets).item()
            running_all += len(targets)

    avg_acc = running_corrects / running_all
    logger.info(f"FINAL TEST ACCURACY: {avg_acc:.4f}")
    return avg_acc



# -------------------------------------------------------------
# MAIN TRAINING FUNCTION
# -------------------------------------------------------------
def test_adam(args, use_gpu):

    save_path = f"./{args.mode}"
    os.makedirs(save_path, exist_ok=True)

    log_file = os.path.join(save_path, f"{args.mode}_emg_{args.lr}.txt")
    logger = logging.getLogger("LOG")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(logging.FileHandler(log_file, mode='a'))
    logger.addHandler(logging.StreamHandler())

    # ----------- LOAD DATASETS -----------
    dset_loaders, _ = data_loader(args)

    # ----------- DEBUG: PRINT PATHS -----------
    logger.info("\n[DEBUG] First 5 TRAIN samples:")
    for i in range(5):
        _, lbl, pth = dset_loaders['train'].dataset[i]
        logger.info(f"  TRAIN[{i}] → label={lbl}, path={pth}")

    logger.info("\n[DEBUG] First 5 VAL samples:")
    for i in range(5):
        _, lbl, pth = dset_loaders['val'].dataset[i]
        logger.info(f"  VAL[{i}] → label={lbl}, path={pth}")

    logger.info("\n[DEBUG] First 5 TEST samples:")
    for i in range(5):
        _, lbl, pth = dset_loaders['test'].dataset[i]
        logger.info(f"  TEST[{i}] → label={lbl}, path={pth}")

    # Infer input_dim (should be 216)
    sample_inputs, sample_targets, _ = next(iter(dset_loaders['train']))
    input_dim = sample_inputs.shape[-1]
    logger.info(f"[INFO] Inferred patch_dim (input_dim) = {input_dim}")

    # Build hybrid model
    model = EMG_PatchTST_GRU_Hybrid(
        input_dim=input_dim,
        num_classes=args.nClasses
    ).to(device)

    model = reload_model(model, logger, args.path)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler = AdjustLR(optimizer, [args.lr], sleep_epochs=20, half=5)

    # ----------- Tiny overfit test -----------
    logger.info("Running tiny overfit sanity check...")
    model.train()
    x_small = sample_inputs[:8].float().to(device)
    y_small = sample_targets[:8].to(device)
    for step in range(20):
        optimizer.zero_grad()
        out_small = model(x_small)
        loss_small = criterion(out_small, y_small)
        loss_small.backward()
        optimizer.step()
        acc_small = (out_small.argmax(1) == y_small).float().mean().item()
        logger.info(f"[OVERFIT] step {step:02d}: loss={loss_small:.4f}, acc={acc_small:.3f}")
    model.eval()

    # ----------- FULL TRAINING LOOP -----------
    for epoch in range(args.epochs):
        scheduler.step(epoch)
        train_test(model, dset_loaders, criterion, epoch, 'train', optimizer, args, logger, use_gpu, save_path)
        train_test(model, dset_loaders, criterion, epoch, 'val', optimizer, args, logger, use_gpu, save_path)

    logger.info("----- FINAL TEST -----")
    evaluate_test(model, dset_loaders, criterion, args, logger)



# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AVE-Speech EMG PatchTST-GRU Hybrid Training")

    parser.add_argument("--nClasses", default=101, type=int)
    parser.add_argument("--path", default="")
    parser.add_argument("--mode", default="patchtst_gru_hybrid")

    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--interval", default=100, type=int)

    # Patch parameters
    parser.add_argument("--patch-time", default=6, type=int)
    parser.add_argument("--patch-freq", default=6, type=int)
    parser.add_argument("--stride-time", default=3, type=int)
    parser.add_argument("--stride-freq", default=3, type=int)

    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    test_adam(args, use_gpu)



if __name__ == "__main__":
    main()
