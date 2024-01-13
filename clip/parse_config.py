
import yaml
from easydict import EasyDict


def read_config():
    # Read config.yaml file
    with open("C:/Users/user/Documents/GitHub/DS_Portfolio/clip/config.yaml") as infile:
        SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
        CFG = EasyDict(SAVED_CFG["CFG"]) # convert dict to EasyDict
    return CFG