import sys
import yaml
from easydict import EasyDict


def read_config():
    # Read config.yaml file
    config_file_arg = sys.argv[1]  # config name
    if "kobart" in config_file_arg.lower():
        config_file_name = "KoBART"
    elif "koelectra" in config_file_arg.lower():
        config_file_name = "KoElectra_lightning"
    elif "kobigbird" in config_file_arg.lower():
        config_file_name = "KoBigBird_lightning"
    elif "ensemble" in config_file_arg.lower():
        config_file_name = "Ensemble"
    else:
        raise ValueError("config file name is not correct")
    # config_file_name = "Electra"
    with open(f"./configs/{config_file_name}.yaml") as infile:
        SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
        CFG = EasyDict(SAVED_CFG["CFG"])  # KoBART vs Electra or whatever you save on config.yaml
    return CFG
