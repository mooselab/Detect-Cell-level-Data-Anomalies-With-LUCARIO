from detectors.utils import Utils
import pandas as pd
import numpy as np
import re
import json

def get_dataframe(path):
    dataframe = pd.read_csv(path, sep=",", header="infer", encoding="utf-8",
                            keep_default_na=False, low_memory=False, index_col=None)
    return dataframe
    
# Get all the datasets
datasets = ['beers', 'flights', 'hospital', 'HOSP-10K', 'HOSP-100K', 'movies_1']
coverage_rate = 0.95
# Predefine null types
null_types = ['', 'NULL', 'None', 'N/A', 'NaN']

for dataset in datasets:
    dirty_file = './datasets/%s/'%dataset + 'dirty.csv'
    ground_truth = './results/GroundTruth/%s_groundtruth.csv'%dataset
    # Open the constraint file and read the JSON data
    with open('./results/LUCARIO/constraints/%s_LUCARIO.json'%dataset, "r") as json_file:
        constraints = json.load(json_file)

    df_dirty = get_dataframe(dirty_file)
    df_gt = get_dataframe(ground_truth)
    anomalies = {}
    for column in df_dirty.columns:
        type_violations, categorical_violations, numerical_violations, pattern_violations = [], [], [], []
        # Flag the anomalies
        delete = 0
        for record in df_dirty[column]:
            # Type checker
            type_violations.append(Utils.type_anomaly_detector(record, constraints[column]['type_constraint']))
            # Categorical (including null)
            if record in null_types: 
                delete += 1
                categorical_violations.append(True)
            elif constraints[column]['categorical_constraint'] != None:
                categorical_violations.append(not record in constraints[column]['categorical_constraint'])
            else: categorical_violations.append(False)

            # Numerical
            if constraints[column]['numerical_constraint'] != None:
                min_range, max_range = constraints[column]['numerical_constraint']
                try:
                    numerical_violations.append(float(record)<min_range or float(record)>max_range)
                # Not applicable to strings
                except: numerical_violations.append(False)
            else: numerical_violations.append(False)

            # Pattern
            if constraints[column]['pattern_constraint'] != []:
                pattern_matched = False
                for pattern in constraints[column]['pattern_constraint']:
                    if re.fullmatch(pattern, str(record)):
                        pattern_matched = True
                        pattern_violations.append(False)
                        break
                if not pattern_matched: 
                    pattern_violations.append(True)
            else: pattern_violations.append(False)
        # Whether we should take and update the numerical range
        if sum(numerical_violations)/len(df_dirty[column]) > 1-coverage_rate:
            numerical_violations = [False for _ in numerical_violations]
            constraints[column]['numerical_constraint'] = None
            # Store the new constraints
            with open('./results/LUCARIO/constraints/%s_LUCARIO.json'%dataset, "w") as json_file:
                json.dump(constraints, json_file, indent=4)
            
        anomalies[column] = [a or b or c or d for a, b, c, d in zip(type_violations, categorical_violations, numerical_violations, pattern_violations)]
        
    # Store the result
    pd.DataFrame(anomalies).to_csv('./results/LUCARIO/anomalies/%s_LUCARIO_detection.csv'%dataset, index=None)
    pred_df = pd.DataFrame(anomalies)
    # Create a boolean mask where the condition holds
    fp_mask = (pred_df == True) & (df_gt == False)
    tp_mask = (pred_df == True) & (df_gt == True)
    fn_mask = (pred_df == False) & (df_gt == True)

    # Count the number of True values in the mask
    fp_count = fp_mask.sum().sum()
    tp_count = tp_mask.sum().sum()
    fn_count = fn_mask.sum().sum()

    p = tp_count/(tp_count+fp_count)
    r = tp_count/(tp_count+fn_count)
    f = (2*p*r)/(p+r)
    print('%s: %.2f %.2f %.2f'%(dataset, p, r, f))

