import pandas as pd
from openai import OpenAI
import sys
import time

def log_output(message, log_file_path="./results/GPT-5/constraints_output.txt"):
    # log_output to console
    print(message)
    sys.stdout.flush() # Ensure immediate console output

    # Write to log file
    with open(log_file_path, "a") as f:
        f.write(message + "\n")
        f.flush() # Ensure immediate file write

client = OpenAI()



datasets = ['beers', 'flights', 'hospital', 'HOSP-10K', 'HOSP-100K', 'movies_1']

for dataset in datasets:
    log_output(f"Processing dataset: {dataset} with GPT-5")
    # Read the original CSV file
    df = pd.read_csv(f'./datasets/{dataset}/dirty.csv', encoding="utf-8", keep_default_na=False)
    df_gt = pd.read_csv(f'./datasets/{dataset}/ground_truth.csv', encoding="utf-8", keep_default_na=False)
    preds = df.copy()
    cols = df.columns.tolist()
    for col in cols:
        log_output(f'Column name: {col}')
        preds[col] = [False] * len(preds)
        # Get the unique values in each column
        dist_val = [str(item) for item in set(df[col])]
        t1 = time.time()
        response = client.responses.create(
            model="gpt-5",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "I will provide you a data column (potentially with anomalies) with its title. "
                            "If applicable, infer the type constraints, regex pattern constraints, and range constraints for only the normal data in each column. "
                            "Return ONLY a JSON file between ```...```. An example for 'state' column is: \n"
                            "```'state':\{'type_constraint':'String','categorical_constraint':['DC','AL,'NV'],'numerical_constraint':null,'pattern_constraint':['[A-Z]{2}']\}\n```"
                            "Another example for 'age' column is: \n"
                            "```'age':\{'type_constraint':'Numerical','categorical_constraint':null,'numerical_constraint':[0,120],'pattern_constraint':['\d\{1,3\}']\}\n```"
                            "The column is as follows:\n"
                            f"{col}: {dist_val}"
                        }
                    ]
                }
            ]
        )
        t2 = time.time()
        log_output(response.output_text)
        log_output(f'Inference time: {t2-t1:.3f}s')