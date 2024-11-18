from detectors.pattern_alter import PatternDetector
from detectors.range import RangeDetector
from detectors.utils import Utils
import pandas as pd
import numpy as np
import string
from sklearn.cluster import KMeans
import numpy as np
import math
import scipy.stats as stats
import re
import json

def get_dataframe(path):
    dataframe = pd.read_csv(path, sep=",", header="infer", encoding="utf-8",
                                        keep_default_na=False, low_memory=False, index_col=None)
    return dataframe
    
# Get all the datasets
datasets = ['beers', 'flights', 'hospital', 'HOSP-10k', 'HOSP-100k', 'movies_1']
datasets = ['hospital']
coverage_rate = 0
multiplier = 3
# Predefine null types
null_types = ['', 'NULL', 'None', 'N/A', 'NaN']

for dataset in datasets:
    dirty_file = './datasets/%s/'%dataset + 'dirty.csv'
    ground_truth = './results/GroundTruth/%s_groundtruth.csv'%dataset
    df_dirty = get_dataframe(dirty_file)
    df_gt = get_dataframe(ground_truth)
    print(df_gt.shape[0], df_gt.shape[1])
    print(np.sum(df_gt.values)/(df_gt.shape[0]*df_gt.shape[1]))
    constraints = {}
    anomalies = {}
    for column in df_dirty.columns:
        coverage_rate = 0.95
        constraints[column] = {}
        anomalies[column] = {}

        # Type constraint
        # Filter null cells
        wrangled_column = [item for item in df_dirty[column] if item not in null_types]
        column_type, wrangled_column = Utils.column_type_constraint(wrangled_column)
        constraints[column]['type_constraint'] = column_type
        
        # Range constraint detection
        range_detector = RangeDetector(wrangled_column, coverage_rate, multiplier)
        constraints[column]['categorical_constraint'] = range_detector.categorical_range
        constraints[column]['numerical_constraint'] = range_detector.numerical_range

        # Pattern detection
        pattern_detector = PatternDetector(wrangled_column, coverage_rate)
        constraints[column]['pattern_constraint'] = pattern_detector.pattern_constraints
        
        # print(constraints[column])
    # Store the constraints
    with open('./results/LUCARIO/constraints/%s_LUCARIO.json'%dataset, "w") as json_file:
        json.dump(constraints, json_file, indent=4)