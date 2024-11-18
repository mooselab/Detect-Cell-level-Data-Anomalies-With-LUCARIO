# Detect-Cell-level-Data-Anomalies-With-LUCARIO
### The repository contains the replication package for the paper "Detect Cell-level Data Anomalies With LUCARIO".
### LUCARIO: Learning Unsupervised, Cell-level Anomaly Detector for Regex Incompatibility and Outlier
<img src="https://archives.bulbagarden.net/media/upload/4/42/0448Lucario.png?download" alt="lucario" width="100" height="100">

## Introduction
The workflow of our tool is shown in the following graph ("Str" stands for string type columns, "Num" stands for numerical type columns, and "Mix" stands for mixed type columns):

![image](./images/approach_overview.png?raw=true)

LUCARIO follows the guidance of the coverage rate $r_{cov}$ (e.g., the portion of records should be covered by the inferred constraints) to infer constraints for each column. The default value of $r_cov$ is fixed to 95\%, assuming that at least 95\% 
data in each column are healthy. 

## Dependencies

- Python >= 3.8

- Pandas == 1.5.3

- Numpy == 1.24.3

- Scikit-learn == 1.3.2

- Matplotlib == 3.7.5

## Usage
### Constraint Inference
Run ``constraint_inference.py`` to infer the constraints in each dataset. The coverage rate ($r_cov$) and the MAD multiplier can be modified in this file according to domain knowledge. The code will iterate through all the columns in the specified datasets and generate a JSON-formatted constraint file under the ``results/LUCARIO/constraints`` folder; the constraints can be explicitly validated and maintained by users. During modification, please set the undesired constraints to "null" instead of deleting the entry. Here's an example:

```
{
    "id": {
        "type_constraint": "String",
        "categorical_constraint": null,
        "numerical_constraint": null,
        "pattern_constraint": [
            "tt[0-9]{7}"
        ]
    },
    ......
}
```

### Anomaly Detection
After obtaining the constraints, one can easily detect data anomalies using the generated rules. Run ``anomaly_detection.py`` to detect the anomalies. The detection CSV results will be stored under the ``results/LUCARIO/anomalies`` folder.
