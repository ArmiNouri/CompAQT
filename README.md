# CompAQT

## Introduction
CompAQT is a method to improve compositional generalization for multi-step quantitative reasoning for question answering over tabular data. This repository contains:
* A dataset composed of multi-step quatitative reasoning samples from four previously published datasets.
* Code and instructions on how to apply the CompAQT method to QA models.

## Dataset
You can download the data from [this link](https://drive.google.com/file/d/1DCitTop_SKVPgq5fekF8VtYWOR07UB3J/view?usp=sharing).

The dataset is composed of four previously released datasets that have been filtered and processed to focus on multi-step quantitative reasoning.
* FinQA ([paper](https://aclanthology.org/2021.emnlp-main.300/), [github repo](https://github.com/czyssrs/finqa))
* TAT-QA ([paper](https://aclanthology.org/2021.acl-long.254.pdf), [github repo](https://github.com/NExTplusplus/TAT-QA))
* HiTab ([paper](https://arxiv.org/abs/2108.06712), [github repo](https://github.com/microsoft/HiTab))
* MultiHiertt ([paper](https://github.com/psunlpgroup/MultiHiertt), [github repo](https://aclanthology.org/2022.acl-long.454/))

All datasets except FinQA have been filtered to only include samples that require quantitative reasoning. The samples have also been reformatted to match the FinQA format. 

Each sample has the following format:
```
{
    "source": the original source of the dataset (`finqa`, `tatqa`, `hitab`, or `multihiertt`)
    "pre_text": the text before the table
    "post_text": the text after the table
    "table_ori": the original table, represented as a nested array
    "table": the normalized table, where the first row represents the column headers and the left-most column represents the row headers
    "id": unique example id; the id matches the id of each sample in the original dataset 

    "qa": {
        "question": the question
        "program": the reasoning program
        "gold_inds": the gold supporting facts
        "exe_ans": the gold execution result
    }
}
```

## Data seletion
To select a particular dataset, set the `source` parameter in `generator/config.py` to one these options: `finqa|tatqa|hitab|multihiertt|all`

## Setting up the environment
You can set up the environment by installing all requirements: `pip install -r requirements.txt`

## To run the FinQA model
The code is largely adapted from [FinQA](https://github.com/czyssrs/finqa). 
All configurations are modifiable within `generator/config.py`
* First navigate to the generator: `cd generator`
* Run `chmod +x run_finqa.sh`
* To run FinQA with CompAQT: `./run_finqa.sh`

## To run the PVN model
* First navigate to the generator: `cd generator`
* Run `chmod +x run_pvn.sh`
* To run FinQA with CompAQT: `./run_pvn.sh`