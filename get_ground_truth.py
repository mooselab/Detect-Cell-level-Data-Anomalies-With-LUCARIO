import sys
import pandas as pd
def read_csv_dataset(dataset_path):
    """
    This method reads a dataset from a csv file path.
    """
    dataframe = pd.read_csv(dataset_path, sep=",", header="infer", encoding="utf-8", dtype=str,
                                keep_default_na=False, low_memory=False)
    return dataframe


def get_dataframes_difference(dataframe_1, dataframe_2):
    """
    This method compares two dataframes and returns the different cells.
    """
    if dataframe_1.shape != dataframe_2.shape:
        sys.stderr.write("Two compared datasets do not have equal sizes!\n")
    difference_dataframe = dataframe_1.where(dataframe_1.values != dataframe_2.values).notna()
    return difference_dataframe

datasets = ["hosp_1k", "flights", "beers", "rayyan", "movies_1", "hosp_100k", "hosp_10k", "tax"]
datasets = ['food']
for dataset in datasets:
    print(dataset)
    dirty_df = read_csv_dataset('./datasets/%s/dirty.csv'%dataset)
    clean_df = read_csv_dataset('./datasets/%s/clean.csv'%dataset)
    diff_df = get_dataframes_difference(dirty_df, clean_df)
    diff_df.to_csv('./datasets/%s/ground_truth.csv'%dataset, index=None)