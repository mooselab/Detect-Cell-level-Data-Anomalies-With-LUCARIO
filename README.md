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

## Experiment Replication
The rules inferred by the baseline tools (except dBoost, since its anomaly detection is not based on rule inference) on our benchmark datasets are stored under the ``results`` folder. To replicate the results of statistic-based tools (i.e., [dBoost](https://github.com/cpitclaudel/dBoost), [Deequ](https://github.com/awslabs/deequ), [TFDV](https://github.com/tensorflow/data-validation), and [FlashProfile](https://github.com/SaswatPadhi/FlashProfileDemo/tree/master)) and [Auto-Test](https://github.com/qixuchen/AutoTest/tree/main), please follow the instructions in their official repositories. 

To replicate the results of GPT-5 and Llama3, please install OpenAI and ollama and run ``get_constraints.py`` under their folders. The prompt template used for constraint inference is as follows:

```
'I will provide you a data column (potentially with anomalies) with its title. '
'If applicable, infer the type constraints, regex pattern constraints, and range constraints for only the normal data in each column. '
'Return ONLY a JSON file between ```...```. An example for "state" column is: \n'
'```"state":\{"type_constraint":"String","categorical_constraint":["DC","AL","NV"],"numerical_constraint":null,"pattern_constraint":["[A-Z]\{2\}"]\}\n```'
'Another example for "age" column is: \n'
'```"age":\{"type_constraint":"Numerical","categorical_constraint":null,"numerical_constraint":[0,120],"pattern_constraint":["\d\{1,3\}"]\}\n```'
'The column is as follows:\n'
f'{col}: {dist_val}'
```
