import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

import json

# import modules outside of directory
import sys, os

path = os.path.abspath(os.path.join("..", "kostat-ver2"))
sys.path.append(path)
from parse_config import read_config

sys.path.remove(path)

CFG = read_config()  # Read config.yaml file
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)


def make_json_labels():
    """ make json labels and save as label """

    # TODO: 이부분 그냥 한번에 정리해서 json 파일로 정리하고 불러오는 식으로 바꾸자
    industry_classes = pd.read_excel(
        "/home/ubuntu/kostat-ver2/data/한국표준산업분류(10차)_국문.xlsx", skiprows=2
    )
    code = None
    num_classes = 0
    class2idx = {}

    for _, row in industry_classes.iterrows():
        if str(row["코드"]) != "nan":
            code = row["코드"]
            if code == "A" or code == "B":
                code += "0"
        if str(row["코드.2"]) != "nan":
            class2idx[code + str(int(row["코드.2"]))] = num_classes
            num_classes += 1
    idx2class = {int(y): x for x, y in class2idx.items()}

    # save class2idx json to file as json format
    with open(f"./class2idx.json", "w", encoding="utf-8") as f1, open(
        "./idx2class.json", "w", encoding="utf-8"
    ) as f2:
        json.dump(class2idx, f1, ensure_ascii=False, indent=4)
        json.dump(idx2class, f2, ensure_ascii=False, indent=4)
    return class2idx


def read_json_label():
    """ read json and return label dictionary """
    with open(f"./class2idx.json", "r", encoding="utf-8") as f:
        class2idx = json.load(f)
    return class2idx


def read_json_class():
    """ read json and return class dictionary """
    with open(f"./idx2class.json", "r", encoding="utf-8") as f:
        idx2class = json.load(f)
    return idx2class


class Industry_Dataset(Dataset):
    def __init__(self, filepath, tokenizer):
        csv_data = pd.read_csv(filepath, sep="|", encoding="cp949")
        self.class2idx = read_json_label()
        self.digit_1, self.digit_2, self.digit_3 = (
            csv_data["digit_1"],
            csv_data["digit_2"],
            csv_data["digit_3"],
        )
        self.text_obj, self.text_mthd, self.text_deal = (
            csv_data["text_obj"].astype("str"),
            csv_data["text_mthd"].astype("str"),
            csv_data["text_deal"].astype("str"),
        )
        self.text_total = [
            tokenizer(x + " " + y + " " + z, return_token_type_ids=False)
            for x, y, z in zip(self.text_obj, self.text_mthd, self.text_deal)
        ]
        self.label_ids = [
            self.class2idx[digit_1 + f"{digit_3:03}"]
            for digit_1, digit_3 in zip(self.digit_1, self.digit_3)
        ]

    def __len__(self):
        return len(self.digit_1)

    def __getitem__(self, idx):
        out = self.text_total[idx]
        out["label_ids"] = self.label_ids[idx]
        return out


class Test_Dataset(Dataset):
    def __init__(self, filepath, tokenizer):
        csv_data = pd.read_csv(filepath, sep="|", encoding="cp949")

        self.id = [int(ai_id[3:]) for ai_id in csv_data["AI_id"].astype("str")]
        self.text_obj, self.text_mthd, self.text_deal = (
            csv_data["text_obj"].astype("str"),
            csv_data["text_mthd"].astype("str"),
            csv_data["text_deal"].astype("str"),
        )
        self.text_total = [
            tokenizer(x + " " + y + " " + z, return_token_type_ids=False)
            for x, y, z in zip(self.text_obj, self.text_mthd, self.text_deal)
        ]

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        out = self.text_total[idx]
        out["label_ids"] = self.id[idx]
        return self.text_total[idx]


class Short_Dataset(Dataset):
    def __init__(self, filepath, tokenizer):
        csv_data = pd.read_csv(filepath, sep="|", encoding="cp949")

        self.id = [int(ai_id[3:]) for ai_id in csv_data["AI_id"].astype("str")][:1024]
        self.text_obj, self.text_mthd, self.text_deal = (
            csv_data["text_obj"].astype("str"),
            csv_data["text_mthd"].astype("str"),
            csv_data["text_deal"].astype("str"),
        )
        self.text_total = [
            tokenizer(x + " " + y + " " + z, return_token_type_ids=False)
            for x, y, z in zip(self.text_obj, self.text_mthd, self.text_deal)
        ][:1024]

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        out = self.text_total[idx]
        out["label_ids"] = self.id[idx]
        return self.text_total[idx]


def get_collator(tokenizer=tokenizer):
    return DataCollatorWithPadding(tokenizer)


# make main function
if __name__ == "__main__":
    class2idx = make_json_labels()
    print(class2idx)
