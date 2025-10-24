from detectors.pattern import PatternDetector
from detectors.range import RangeDetector
from detectors.utils import Utils
import pandas as pd
import numpy as np
import json
import time

def get_dataframe(path):
    dataframe = pd.read_csv(path, sep=",", header="infer", encoding="utf-8",
                            keep_default_na=False, low_memory=False, index_col=None)
    return dataframe
    
# Get all the datasets
datasets = ['beers', 'flights', 'hospital', 'HOSP-10K', 'HOSP-100K', 'movies_1']
coverage_rate = 0.95
multiplier = 3
# Predefine null types
null_types = ['', 'NULL', 'None', 'N/A', 'NaN']

for dataset in datasets:
    dirty_file = './datasets/%s/'%dataset + 'dirty.csv'
    ground_truth = './results/GroundTruth/%s_groundtruth.csv'%dataset
    df_dirty = get_dataframe(dirty_file)
    df_gt = get_dataframe(ground_truth)
    constraints = {}
    anomalies = {}
    t1 = time.time()
    for column in df_dirty.columns:
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
        
    t2 = time.time()
    print(f"Constraint inference time for {dataset}: {t2-t1:.3f}s")
    # Store the constraints
    with open('./results/LUCARIO/constraints/%s_LUCARIO.json'%dataset, "w") as json_file:
        json.dump(constraints, json_file, indent=4)
