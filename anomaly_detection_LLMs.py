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
for llm in ['GPT-5', 'Llama3']:
    print(f"Anomaly detection results of {llm}:")
    for dataset in datasets:
        dirty_file = f'./datasets/{dataset}/' + 'dirty.csv'
        ground_truth = f'./results/GroundTruth/{dataset}_groundtruth.csv'
        # Open the constraint file and read the JSON data
        with open(f'./results/{llm}/constraints/{dataset}_{llm}.json', "r") as json_file:
            constraints = json.load(json_file)

        df_dirty = get_dataframe(dirty_file)
        df_gt = get_dataframe(ground_truth)
        anomalies = {}
        for column in df_dirty.columns:
            # Check whether the column has constraints inferred
            if column not in constraints:
                anomalies[column] = [False] * len(df_dirty)
                continue
            
            type_violations, categorical_violations, numerical_violations, pattern_violations = [], [], [], []
            # Flag the anomalies
            delete = 0
            for record in df_dirty[column]:
                # Type checker
                type_violations.append(Utils.type_anomaly_detector(record, constraints[column]['type_constraint']))
                # Categorical (including null)
                if constraints[column]['categorical_constraint'] != None:
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
                if constraints[column]['pattern_constraint'] != [] and constraints[column]['pattern_constraint'] != None:
                    pattern_matched = False
                    for pattern in constraints[column]['pattern_constraint']:
                        try:
                            if re.fullmatch(pattern, str(record)):
                                pattern_matched = True
                                pattern_violations.append(False)
                                break
                        except:
                            # Invalid pattern, no violation detected
                            pattern_violations.append(False)
                    if not pattern_matched: 
                        pattern_violations.append(True)
                else: pattern_violations.append(False)
                
            anomalies[column] = [a or b or c or d for a, b, c, d in zip(type_violations, categorical_violations, numerical_violations, pattern_violations)]
        
        # Store the result
        # pd.DataFrame(anomalies).to_csv(f'./results/{llm}/anomalies/{dataset}_{llm}_detection.csv', index=None)
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
