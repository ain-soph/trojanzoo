# -*- coding: utf-8 -*-

import os
import csv
import argparse 
import numpy as np 
import pandas as pd


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def reverse_index(li: list, index_str: str): 
    return len(li) - 1 - li[::-1].index(index_str)

def collect_results(path: str, extension: str = ".txt", benign_acc: float = 95.37): 
    if not path.endswith(extension): 
        print("Invalid data directory passed to collect_result function, exit.")
        exit()
    basename = os.path.splitext(os.path.basename(path))[0].split("_")
    percent = float(basename[-1])
    attack = str("_".join(basename[0:-1]))

    with open(path, "r") as raw: 
        raw_data = list(raw)

    raw_data_truncated = [i[0:22] for i in raw_data]
    last_occur_index = reverse_index(raw_data_truncated, "best result update!\n")
    best_acc = [float(i) for i in raw_data[last_occur_index + 1].split() if isfloat(i)][0]
    validate_clean_index = reverse_index(raw_data_truncated[:last_occur_index], "Validate Clean:       ")
    validate_clean_acc = [float(i.replace(",", "")) for i in raw_data[validate_clean_index].split() if isfloat(i.replace(",", ""))][1]

    return [attack, percent, best_acc, validate_clean_acc, benign_acc-validate_clean_acc]

def collect_directory(path: str, extension: str = ".txt"): 
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(extension)]
    # print(files)
    return [collect_results(path=file) for file in files]


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', dest='directory')
    parser.add_argument('-s', '--save', dest='save', action='store_true')
    parser.add_argument('-sd', '--save_directory', dest='save_dir', default='./')
    args = parser.parse_args()
    colnames = ["Attack Name", "Percent", "Attack ACC", "Validate Clean", "Difference"]

    if os.path.isdir(args.directory): 
        result = collect_directory(args.directory)
    elif os.path.isfile(args.directory): 
        result = collect_results(args.directory)
    else: 
        print("Invalid directory parameter, exit. ")
        exit()

    if args.save: 
        results_df = pd.DataFrame(result)
        results_df.columns = colnames
        results_df.sort_values(by=["Attack Name", "Percent"], inplace=True)
        results_df.to_csv(os.path.join(args.save_dir, "saved.csv"), index=False)

    print("CSV file saved")
