# Edit Aware Representation Learning (EARL)

This is the official repository accompanying the paper [Edit Aware Representation Learning via Levenshtein Prediction](https://aclanthology.org/2023.insights-1.6/). If you use this code, please consider citing our paper as follows.

```bibtex
@inproceedings{marrese-taylor-etal-2023-edit,
    title = "Edit Aware Representation Learning via {L}evenshtein Prediction",
    author = "Marrese-taylor, Edison  and
      Reid, Machel  and
      Solano, Alfredo",
    booktitle = "The Fourth Workshop on Insights from Negative Results in NLP",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.insights-1.6",
    pages = "53--58",
    abstract = "We propose a novel approach that employs token-level Levenshtein operations to learn a continuous latent space of vector representations to capture the underlying semantic information with regard to the document editing process. Though our model outperforms strong baselines when fine-tuned on edit-centric tasks, it is unclear if these results are due to domain similarities between fine-tuning and pre-training data, suggesting that the benefits of our proposed approach over regular masked language-modelling pre-training are limited.",
}
```

# Installation

```bash
cd code/fairseq
pip install --editable ./
pip install einops
pip install transformers
pip install python-Levenshtein
```

Create a folder to host the data. In our machine we used `~/data/early`, please make sure to adjust our scripts if you decide to use a different path.

# Pre-training

## Data

Prepare data using:

```bash
python convert_peer_to_tsv.py --data ~/data/early/x.jsonl
bash detokenize_tsv_dataset.sh  ~/data/early/x.tsv
python convert_wikiedits_to_tsv.py --data ~/data/early/wikiedits
```

Preprocessing data (compute vocabularies, split and and binarize) using:

```bash
cd ~/data/early
bash preprocess.sh x.tsv.detok 80000 20000
bash preprocess.sh insertions.tsv.detok 9616458 2747560 
bash preprocess.sh deletions.tsv.detok 6546673 1870478 
bash preprocess.sh wikiedits 0 0
```

## Training

Please refer to the `train.sh` for a reference script, and check the files inside the `jobs` folder for details of the exact hyperparameter settings we used when pre-training models on our cluster.
  
    
# Downstream tasks

## Preparing Data

*For MNLI (from GLUE tasks)*

1. Download the GLUE data with the script here https://github.com/nyu-mll/GLUE-baselines#downloading-glue (the RoBERTa script does not work anymore, check https://github.com/pytorch/fairseq/issues/3840)

  ```bash
  git clone https://github.com/nyu-mll/GLUE-baselines
  python download_glue_data.py --data_dir glue_data --tasks all
  ```

2. Preprocess the data using the RoBERTa example script

  ```bash
  bash code/fairseq/examples/roberta/preprocess_GLUE_tasks.sh ALL
  ```

*For PAWS*

1. Download paws by running:
   
  ```bash
  cd glue_data
  mkdir paws
  cd paws
  wget https://storage.googleapis.com/paws/english/paws_wiki_labeled_final.tar.gz
  tar -xvzf paws_wiki_labeled_final.tar.gz
  ```

2. Process the data using `convert_paws.sh /path/to/glue_data`


## Running/Training Models


Use our command, which is based on the RoBERTA documentation, as follows:
  
  ```bash
  fairseq-hydra-train --config-dir configs --config-name <config_name> task.data=/path/to/data-bin checkpoint.restore_file=/path/to/roberta/model.pt checkpoint.save_dir=/output/path/ | tee -a /output/path/train.log 
  ```
Where:
 - `<config_name>`: one of the names of the files inside the `configs` folder. For example, use `x_default` to fine-tune on the WikiEditsMix dataset. 


# Sumary of Changes Introduced to `fairseq`

- Models:

  - `roberta.py`: `forward()` function, enable multiple heads at the same time
  - `roberta.py`: `get_classification_head()` add the `reduce` parameter to allow for for sequence-labeling tasks (such as the levenshtein prediction)
   
- Tasks:
  - `lenveshtein_prediction.py`: the new task, based on `sentence_prediction.py`

- Criterions:
  - `lenveshtein_prediction.py`: the loss for the new task, based on `sentence_prediction.py`
