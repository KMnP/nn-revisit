#!/usr/bin/env python3
"""
a bunch of helper functions for read and write data
"""
import os
import json
import numpy as np
import time
import pandas as pd

from glob import glob
from typing import List, Union
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None


def get_newt_names(newt_feature_dir):
    files = glob(newt_feature_dir + "/*.pkl")
    feature_fp = files[0]
    feature_df = pd.read_pickle(feature_fp)
    newt_names = list(set(feature_df["name"]))
    return newt_names


def get_index_path(feature_path, index_folder, data_name):
    # setup index path
    model_type = os.path.basename(feature_path).split(".")[0]
    index_dir = os.path.join(index_folder, data_name)
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    index_path = os.path.join(index_dir, model_type)
    return index_path


def get_feature_path(cfg):
    feature_fp = os.path.join(
        cfg.DATA.DATAPATH, f"{cfg.DATA.FEATURE}.pkl")
    if not os.path.exists(feature_fp):
        raise ValueError(f"did not find features at location {feature_fp}")
    return feature_fp


def get_feature_df(cfg):
    return pd.read_pickle(get_feature_path(cfg))


def save_or_append_df(out_path, df):
    if os.path.exists(out_path):
        previous_df = pd.read_pickle(out_path)
        df = pd.concat([previous_df, df], ignore_index=True)
    df.to_pickle(out_path)
    print(f"Saved output at {out_path}")


def read_jsonl(json_file: str) -> List:
    """
    Read json data into a list of dict.
    Each file is composed of a single object type, one JSON-object per-line.
    Args:
        json_file (str): path of specific json file
    Returns:
        data (list): list of dicts
    """
    start = time.time()
    data = []
    with open(json_file) as fin:
        for line in fin:
            line_contents = json.loads(line)
            data.append(line_contents)
    end = time.time()
    elapse = end - start
    if elapse > 1:
        print("\tLoading {} takes {:.2f} seconds.".format(
            json_file, elapse))
    return data


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            # return super(MyEncoder, self).default(obj)

            raise TypeError(
                "Unserializable object {} of type {}".format(obj, type(obj))
            )


def write_json(data: Union[list, dict], outfile: str) -> None:
    json_dir, _ = os.path.split(outfile)
    if json_dir and not os.path.exists(json_dir):
        os.makedirs(json_dir)

    with open(outfile, 'w') as f:
        json.dump(data, f, cls=JSONEncoder, ensure_ascii=False, indent=2)


def read_json(filename: str) -> Union[list, dict]:
    """read json files"""
    with open(filename, "rb") as fin:
        data = json.load(fin, encoding="utf-8")
    return data


def pil_loader(path: str) -> Image.Image:
    """load an image from path, and suppress warning"""
    # to avoid crashing for truncated (corrupted images)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
