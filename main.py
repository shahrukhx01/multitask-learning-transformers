import logging
import torch
import nltk
import numpy as np
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from tqdm import tqdm as tqdm1

import transformers
from accelerate import Accelerator
from filelock import FileLock
from transformers import set_seed
from transformers.file_utils import is_offline_mode
from utils.arguments import parse_args
from multitask_model import MultitaskModel
from preprocess import convert_to_features
from multitask_data_collator import MultitaskTrainer, NLPDataCollator
from multitask_eval import multitask_eval_fn

logger = logging.getLogger(__name__)


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def main():
    args = parse_args()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).

    dataset_dict = {
        "quora_keyword_pairs": load_dataset(
            "multitask_dataloader.py",
            data_files={
                "train": "data/quora_keyword_pairs/train.tsv",
                "validation": "data/quora_keyword_pairs/dev.tsv",
            },
        ),
        "spaadia_squad_pairs": load_dataset(
            "multitask_dataloader.py",
            data_files={
                "train": "data/spaadia_squad_pairs/train.csv",
                "validation": "data/spaadia_squad_pairs/val.csv",
            },
        ),
    }

    for task_name, dataset in dataset_dict.items():
        print(task_name)
        print(dataset_dict[task_name]["train"][0])
        print()

    model_name = "roberta-base"
    multitask_model = MultitaskModel.create(
        model_name=model_name,
        model_type_dict={
            "quora_keyword_pairs": transformers.AutoModelForSequenceClassification,
            "spaadia_squad_pairs": transformers.AutoModelForSequenceClassification,
        },
        model_config_dict={
            "quora_keyword_pairs": transformers.AutoConfig.from_pretrained(
                model_name, num_labels=2
            ),
            "spaadia_squad_pairs": transformers.AutoConfig.from_pretrained(
                model_name, num_labels=2
            ),
        },
    )
    if model_name.startswith("roberta-"):
        print(multitask_model.encoder.embeddings.word_embeddings.weight.data_ptr())
        print(
            multitask_model.taskmodels_dict[
                "quora_keyword_pairs"
            ].roberta.embeddings.word_embeddings.weight.data_ptr()
        )
        print(
            multitask_model.taskmodels_dict[
                "spaadia_squad_pairs"
            ].roberta.embeddings.word_embeddings.weight.data_ptr()
        )
    else:
        print("Exercise for the reader: add a check for other model architectures =)")

    convert_func_dict = {
        "quora_keyword_pairs": convert_to_features,
        "spaadia_squad_pairs": convert_to_features,
    }

    columns_dict = {
        "quora_keyword_pairs": ["input_ids", "attention_mask", "labels"],
        "spaadia_squad_pairs": ["input_ids", "attention_mask", "labels"],
    }

    features_dict = {}
    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                convert_func_dict[task_name],
                batched=True,
                load_from_cache_file=False,
            )
            print(
                task_name,
                phase,
                len(phase_dataset),
                len(features_dict[task_name][phase]),
            )
            features_dict[task_name][phase].set_format(
                type="torch",
                columns=columns_dict[task_name],
            )
            print(
                task_name,
                phase,
                len(phase_dataset),
                len(features_dict[task_name][phase]),
            )

    train_dataset = {
        task_name: dataset["train"] for task_name, dataset in features_dict.items()
    }

    trainer = MultitaskTrainer(
        model=multitask_model,
        args=transformers.TrainingArguments(
            output_dir="./models/multitask_model",
            overwrite_output_dir=True,
            learning_rate=1e-5,
            do_train=True,
            num_train_epochs=1,
            # Adjust batch size if this doesn't fit on the Colab GPU
            per_device_train_batch_size=8,
            save_steps=3000,
        ),
        data_collator=NLPDataCollator(),
        train_dataset=train_dataset,
    )
    trainer.train()
    ## evaluate dataset
    multitask_eval_fn(multitask_model, model_name, dataset_dict)


if __name__ == "__main__":
    main()