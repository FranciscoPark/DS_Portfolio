import sys
import argparse
from cmath import log
from sched import scheduler
from adamp import AdamP
import pandas as pd
import numpy as np
import os

# import torch and trasnfomers related modules
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# import custom model and configuration classes, methods
from parse_config import read_config
from logger import activate_wandb
from model.model import ElectraConcatModel, ElectraPooledModel, BIGBIRD_COMP
from utils.util import seed_everything

# import train related modules and classes
from data_loader.dataset import Industry_Dataset, get_collator
from trainer.lightning_trainer import lightning_train

# Read configuration
CFG = read_config()
seed_everything(42)
class2idx = read_config()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def make_dummy_data() -> pd.DataFrame:
    df_industry = pd.read_excel(
        "/home/ubuntu/kostat-ver2/data/한국표준산업분류(10차)_국문.xlsx", skiprows=2
    )

    # ffill for nan
    df_industry.fillna(method="ffill", inplace=True)

    # change float type data into integer type data
    df_industry[["코드.1", "코드.2", "코드.3", "코드.4"]] = df_industry[
        ["코드.1", "코드.2", "코드.3", "코드.4"]
    ].astype(int)

    df = df_industry.copy()

    df["AI_id"] = "dummy_data"
    # rename 코드 as digit_1
    df = df.rename(columns={"코드": "digit_1", "코드.1": "digit_2", "코드.2": "digit_3"})
    df = df.rename(columns={"항목명.1": "text_obj", "항목명.3": "text_mthd", "항목명.4": "text_deal"})
    # drop other columns except for ['AI_id', 'digit_1', 'digit_2', 'digit_3', 'text_obj', 'text_mthd','text_deal']
    df = df[["AI_id", "digit_1", "digit_2", "digit_3", "text_obj", "text_mthd", "text_deal"]]
    return df


# read train data
df_train = pd.read_csv(CFG.TRAIN_PATH, sep="|", encoding="cp949")

# make dummy data
df_dummy = make_dummy_data()

# concat dummy data and train data
df_train = pd.concat([df_train, df_dummy], axis=0)

print(df_train.head())
print(df_train.tail())
activate_wandb(
    wandb_repo_name=CFG.project_name,
    run_name=CFG.wandb_run_name,
    entity_name=CFG.entity_name,
    CFG=CFG,
)

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
dataset = Industry_Dataset(filepath=CFG.TRAIN_PATH, tokenizer=tokenizer)
train_dataset, dev_dataset = torch.utils.data.random_split(dataset, [900000, 100000])

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.train_batch_size,
    shuffle=True,
    num_workers=CFG.num_workers,
    collate_fn=get_collator(),
)
dev_loader = DataLoader(
    dev_dataset,
    batch_size=CFG.val_batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    collate_fn=get_collator(),
)

model = BIGBIRD_COMP()

optimizer = AdamP(model.parameters(), lr=CFG.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=CFG.gamma, verbose=True)

lightning_train(
    model=model,
    epochs=CFG.num_epochs,
    train_loader=train_loader,
    val_loader=dev_loader,
    optimizer=optimizer,
    criterion=nn.CrossEntropyLoss(),
    scheduler=lr_scheduler,
)
