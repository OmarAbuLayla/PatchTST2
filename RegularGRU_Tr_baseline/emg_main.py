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

from lr_scheduler import *
from emg_model_with_trans import *   # contains GRU + Transformer backend
from emg_dataset_no_jit import *    # MFSC loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 5
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def data_loader(args):
    dsets = {x: MyDataset(x, args.dataset) for x in ['train', 'val', 'test']}

    dset_loaders = {
        x: torch.utils.data.DataLoader(
            dsets[x],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True
        ) for x in ['train', 'val', 'test']
    }

    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
    print(f"\nStatistics: train: {dset_sizes['train']}, val: {dset_sizes['val']}, test: {dset_sizes['test']}")
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
# OLD PRINTING BEHAVIOR EXACTLY AS YOUR ORIGINAL CODE
# -------------------------------------------------------------
def train_test(model, dset_loaders, criterion, epoch, phase, optimizer, args, logger, use_gpu, save_path):

    if phase in ['val', 'test']:
        model.eval()
    else:
        model.train()

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

        inputs = inputs.reshape(inputs.shape[0], inputs.shape[2], inputs.shape[3], -1)
        inputs = inputs.float().to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        if outputs.dim() == 3:
            outputs = torch.mean(outputs, dim=1)

        _, preds = torch.max(F.softmax(outputs, dim=1).data, 1)
        loss = criterion(outputs, targets)

        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == targets.data).item()
        running_all += len(inputs)

        if batch_idx == 0:
            since = time.time()
        elif batch_idx % args.interval == 0:
            print(
                f'Process: [{running_all:5}/{len(dset_loaders[phase].dataset):5} '
                f'({100. * batch_idx / (len(dset_loaders[phase]) - 1):.0f}%)] '
                f'Loss: {running_loss / running_all:.4f} '
                f'Acc:{running_corrects / running_all:.4f} '
                f'Cost time:{time.time() - since:5.0f}s',
                end='\r'
            )

    print()
    logger.info(
        f'{phase} Epoch:\t{epoch:2}\t'
        f'Loss: {running_loss / len(dset_loaders[phase].dataset):.4f}\t'
        f'Acc:{running_corrects / len(dset_loaders[phase].dataset):.4f}\n'
    )

    if phase == 'train':
        torch.save(model.state_dict(), os.path.join(
            save_path, f'{args.mode}_emg_{epoch+1}.pt'))

    return model


# -------------------------------------------------------------
# FIXED TEST FUNCTION (ONLY FIX APPLIED)
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

            inputs = inputs.reshape(inputs.shape[0], inputs.shape[2], inputs.shape[3], -1)
            inputs = inputs.float().to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            # 🔥 FIX ADDED HERE
            if outputs.dim() == 3:     # (B,T,C)
                outputs = torch.mean(outputs, dim=1)

            if outputs.dim() == 1:     # (C,) → (1,C)
                outputs = outputs.unsqueeze(0)
            # 🔥 END FIX

            _, preds = torch.max(outputs.data, 1)
            running_corrects += torch.sum(preds == targets).item()
            running_all += len(targets)

    avg_acc = running_corrects / running_all if running_all else 0.0
    logger.info(f"FINAL TEST ACCURACY: {avg_acc:.4f}")
    return avg_acc


# -------------------------------------------------------------
def test_adam(args, use_gpu):
    save_path = f"./{args.mode}_{'everyframe' if args.every_frame else 'lastframe'}"
    os.makedirs(save_path, exist_ok=True)

    log_file = os.path.join(save_path, f"{args.mode}_emg_{args.lr}.txt")
    logger = logging.getLogger("LOG")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(log_file, mode='a'))
    logger.addHandler(logging.StreamHandler())

    model = emg_model(
        mode=args.mode,
        inputDim=256,
        hiddenDim=512,
        nClasses=args.nClasses,
        frameLen=36,
        every_frame=args.every_frame
    ).to(device)

    model = reload_model(model, logger, args.path)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)

    dset_loaders, _ = data_loader(args)
    scheduler = AdjustLR(optimizer, [args.lr], sleep_epochs=20, half=5)

    for epoch in range(args.epochs):
        scheduler.step(epoch)
        train_test(model, dset_loaders, criterion, epoch, 'train', optimizer, args, logger, use_gpu, save_path)
        train_test(model, dset_loaders, criterion, epoch, 'val', optimizer, args, logger, use_gpu, save_path)

    logger.info("----- FINAL TEST -----")
    evaluate_test(model, dset_loaders, criterion, args, logger)


# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AVE-Speech EMG Training")

    parser.add_argument("--nClasses", default=101, type=int)
    parser.add_argument("--path", default="")
    parser.add_argument("--dataset", default="emg")

    parser.add_argument(
        "--mode",
        type=str,
        default="finetuneGRU",
        choices=["finetuneGRU", "backendGRU", "transformer"]
    )

    parser.add_argument(
        "--every-frame",
        dest="every_frame",
        action="store_true"
    )
    parser.add_argument(
        "--last-frame",
        dest="every_frame",
        action="store_false"
    )
    parser.set_defaults(every_frame=True)

    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--batch-size", default=36, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--interval", default=100, type=int)

    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()
    test_adam(args, use_gpu)


if __name__ == "__main__":
    main()
