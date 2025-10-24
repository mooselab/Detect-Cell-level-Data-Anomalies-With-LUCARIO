import pandas as pd

datasets = ['beers', 'flights', 'hospital', 'HOSP-10K', 'HOSP-100K', 'movies_1']

for sdc in ['rt', 'st']:
    for dataset in datasets:
        print(f"Processing dataset: {dataset} with SDC rules: {sdc}")
        # Read the original CSV file
        df = pd.read_csv(f'./datasets/{dataset}/dirty.csv', encoding="utf-8", keep_default_na=False)
        df_gt = pd.read_csv(f'./results/GroundTruth/{dataset}_groundtruth.csv', encoding="utf-8", keep_default_na=False)
        preds = df.copy()
        cols = df.columns.tolist()
        for col in cols:
            preds[col] = [False] * len(preds)
        # Read the predictions
        try:
            predictions = pd.read_csv(f'./results/AutoTest/detected_outliers/{sdc}_learnt_sdc_on_{dataset}.csv', keep_default_na=False)
            # Get the column name
            for _, row in predictions.iterrows():
                col = row['header']
                outliers = set([item.strip()[1:-1] for item in row['outlier'][1:-1].split(',')])
                # Mark the outliers in the original dataframe
                for i, item in enumerate(df[col]):
                    if item in outliers:
                        preds[col][i] = True
        # No outliers detected
        except FileNotFoundError:
            print(f"No anomaly is found on {dataset}.")
        # Output the results
        preds.to_csv(f'./results/AutoTest/anomalies/{dataset}_AutoTest_{sdc}_detection.csv', index=None)

        # Create a boolean mask where the condition holds
        fp_mask = (preds == True) & (df_gt == False)
        tp_mask = (preds == True) & (df_gt == True)
        fn_mask = (preds == False) & (df_gt == True)

        # Count the number of True values in the mask
        fp_count = fp_mask.sum().sum()
        tp_count = tp_mask.sum().sum()
        fn_count = fn_mask.sum().sum()

        p = tp_count/(tp_count+fp_count)
        r = tp_count/(tp_count+fn_count)
        f = (2*p*r)/(p+r)
        print('%s: %.2f %.2f %.2f'%(dataset, p, r, f))
