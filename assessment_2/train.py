import json
import logging
import os
import random
import sys

import torch
import transformers
from tqdm import tqdm
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from data_args import DataArguments
from dataset import AbsoluteDataset, collate_fn
from engine import CustomTrainer, compute_metrics_fn
from model import Model
from model_args import ModelArguments

torch.set_float32_matmul_precision("high")
logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this"
                " behavior, change the `--output_dir` or add `--overwrite_output_dir` to train"
                " from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        # transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load dataset
    print("Loading dataset...")
    ROLES = set()
    role_pad_idx = 0

    files = []
    for file in tqdm(os.listdir(data_args.data_folder)):
        json_path = os.path.join(data_args.data_folder, file)
        files.append(json_path)

        with open(json_path, "r") as f:
            data = json.load(f)

        for node in data["nodes"]:
            ROLES.add(node["role"])
    ROLES = ["<PAD_TOKEN>"] + list(ROLES)

    # split train val
    random.seed(42)
    random.shuffle(files)
    train_files = files[: int(len(files) * 0.8)]
    val_files = files[int(len(files) * 0.8) :]

    train_dataset = AbsoluteDataset(
        train_files,
        roles=ROLES,
        data_folder=data_args.data_folder,
        split="train",
    )
    val_dataset = AbsoluteDataset(
        val_files,
        roles=ROLES,
        data_folder=data_args.data_folder,
        split="valid",
    )

    model = Model(
        hidden_channels=[int(x) for x in model_args.hidden_channels.split(",")],
        graph_in=model_args.graph_in,
        graph_out=model_args.graph_out,
        hidden_dim=model_args.hidden_dim,
        dropout=model_args.dropout,
        num_roles=len(ROLES),
        role_pad_idx=role_pad_idx,
    )
    compute_metrics = compute_metrics_fn

    # Initialize trainer
    print("Initializing model...")

    if last_checkpoint is None and model_args.resume is not None:
        logger.info(f"Loading {model_args.resume} ...")
        checkpoint = torch.load(model_args.resume, "cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint.pop("state_dict")
        # checkpoint = {k[6:]: v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)

        # if "fc.weight" in checkpoint:
        #     model.fc.load_state_dict(
        #         {"weight": checkpoint["fc.weight"], "bias": checkpoint["fc.bias"]}
        #     )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print("Start training...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    if last_checkpoint is not None or model_args.resume is not None:
        trainer.evaluate()

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
