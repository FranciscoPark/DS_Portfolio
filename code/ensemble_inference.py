import sys
import argparse
import pandas as pd
import numpy as np
import os
import json
import wandb
from easydict import EasyDict
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import yaml  # pyyaml 외부 라이브러리
from transformers import AutoTokenizer
from parse_config import read_config
from model.model import BART_COMP, ELECTRA_COMP, BIGBIRD_COMP
from data_loader.dataset import get_collator, Test_Dataset, Short_Dataset
from data_loader.dataset import read_json_class
from utils.util import seed_everything
import pytorch_lightning as pl

CFG = read_config()
seed_everything(42)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
idx2class = read_json_class()

total_logits = torch.tensor([])
total_ids = torch.tensor([])

for model_name in ["kobart", "koelectra", "kobigbird"]:
    print(f"\nNow predciting with {model_name}\n")
    if model_name == "kobart":
        model = BART_COMP.load_from_checkpoint(CFG.kobart_model_ckpt)
        tokenizer = AutoTokenizer.from_pretrained(CFG.kobart_model_name)
    elif model_name == "koelectra":
        model = ELECTRA_COMP.load_from_checkpoint(CFG.koelectra_model_ckpt)
        tokenizer = AutoTokenizer.from_pretrained(CFG.koelectra_model_name)
    elif model_name == "kobigbird":
        model = BIGBIRD_COMP.load_from_checkpoint(CFG.kobigbird_model_ckpt)
        tokenizer = AutoTokenizer.from_pretrained(CFG.kobigbird_model_name)

    dataset = Test_Dataset(CFG.TEST_PATH, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=get_collator(tokenizer=tokenizer),
    )
    trainer = pl.Trainer(gpus=[CFG.DEVICE], precision=32, deterministic=True,)
    predictions = trainer.predict(model, dataloaders=dataloader)
    logits = torch.tensor([])
    for prediction in predictions:
        logits = torch.cat((logits, prediction["logits"].unsqueeze(1)), dim=0)

    if total_logits.numel() == 0:
        total_logits = logits
        for prediction in predictions:
            total_ids = torch.cat((total_ids, prediction["id"]), dim=0)
    else:
        total_logits = torch.cat((total_logits, logits), dim=1)

total_logits = torch.sum(total_logits, dim=1)
total_predictions = torch.argmax(total_logits, dim=1)
total_ids = torch.add(total_ids, -1)

df = pd.read_csv(
    "/home/ubuntu/kostat-ver2/data/2. 모델개발용자료.txt", sep="|", encoding="cp949"
)
for id, pre_idx in zip(total_ids, total_predictions):
    id = id.item()
    pre_idx = str(pre_idx.item())
    df.loc[id, "digit_1"] = idx2class[pre_idx][0]
    df.loc[id, "digit_2"] = idx2class[pre_idx][1:3]
    df.loc[id, "digit_3"] = idx2class[pre_idx][1:4]

df.to_csv(CFG.csv_filepath, encoding="cp949", index=False)
