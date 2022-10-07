#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Union
import numpy as np
import torch
from transformers.file_utils import PaddingStrategy
import transformers
from modeling_deberta_mtl import DebertaForCompassMTL
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from data_collator_mtl_mlm import DataCollatorForLanguageModelingStd, DataCollatorForLanguageModeling
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.trainer_utils import get_last_checkpoint
from data_choice_mtl_sep_opt_fix_no_opt_big import MultipleChoiceMTLMixDataset, MultipleChoiceDataset, Split, task_dict
# transformers.logging.set_verbosity_error()
from transformers.integrations import WandbCallback, rewrite_logs

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_task_pred: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_task_embedding: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task: Optional[str] = field(default="all",
                                metadata={"help": "The input training data file (a text file)."})
    eval_task: Optional[str] = field(default=None,
                                     metadata={"help": "The input training data file (a text file)."})
    mlm_style: Optional[str] = field(default="std",
                                     metadata={"help": "The input training data file (a text file)."})
    data_dir: Optional[str] = field(default="../corpus/",
                                    metadata={"help": "The input training data file (a text file)."})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    maximum: Optional[int] = field(
        default=100000,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If passed, sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If passed, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    mlm_weight: float = field(
        default=1.0, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    mlm_clone: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to the maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )


class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_metrics = TrainMetric()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        model_metrics = outputs.my_metrics
        self.train_metrics.update(model_metrics)
        self.state.train_metrics = self.train_metrics

        return (loss, outputs) if return_outputs else loss

@dataclass
class TrainMetric():
    loss_ce: torch.FloatTensor = field(default=torch.tensor(0.))
    loss_bce: torch.FloatTensor = field(default=torch.tensor(0.))
    loss_mlm: torch.FloatTensor = field(default=torch.tensor(0.))

    def update(self, metric_dict):
        for k, v in metric_dict.items(): setattr(self, k, v)


class MyCallback(WandbCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            logs = rewrite_logs(logs)
            logs.update(vars(state.train_metrics))
            self._wandb.log({**logs, "train/global_step": state.global_step})

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    special_tokens = ["[task: %s]" % str(task) for task in task_dict]
    if data_args.eval_task is None:
        tokenizer.add_tokens(special_tokens)
    print("*******total tasks*********", len(special_tokens))
    if model_args.use_task_pred:
        replace_task_prefix = True
    else:
        replace_task_prefix = False

    model = DebertaForCompassMTL.from_pretrained(
        model_args.model_name_or_path,
        mlm_weight = data_args.mlm_weight,
        mlm_clone=data_args.mlm_clone,
        model_type=model_args.model_type,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    task_list = task_dict
    if training_args.do_train:
        if data_args.eval_task:
            train_dataset = (
                MultipleChoiceDataset(
                    data_dir=data_args.data_dir,
                    tokenizer=tokenizer,
                    mtl_tasks=task_list,
                    task=data_args.eval_task,
                    replace_task_prefix=replace_task_prefix,
                    maximum=data_args.maximum,
                    max_seq_length=max_seq_length,
                    overwrite_cache=data_args.overwrite_cache,
                    mode=Split.train,
                )
                if training_args.do_eval
                else None
            )
        else:
            train_dataset = (
                MultipleChoiceMTLMixDataset(
                    data_dir=data_args.data_dir,
                    tokenizer=tokenizer,
                    task=data_args.task,
                    replace_task_prefix=replace_task_prefix,
                    maximum=data_args.maximum,
                    max_seq_length=max_seq_length,
                    overwrite_cache=data_args.overwrite_cache,
                    mode=Split.train,
                )
                if training_args.do_train
                else None
            )

    if training_args.do_eval:
        eval_dataset = (
            MultipleChoiceDataset(
                data_dir=data_args.data_dir,
                tokenizer=tokenizer,
                mtl_tasks=task_list,
                task=data_args.eval_task,
                replace_task_prefix=replace_task_prefix,
                maximum=data_args.maximum,
                max_seq_length=max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
                mode=Split.dev,
            )
            if training_args.do_eval
            else None
        )

    # Metric
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

    if data_args.eval_task:
        data_collator = None
    else:
        if data_args.mlm_style == "must_prefix":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm_clone=data_args.mlm_clone,
                mlm_probability=data_args.mlm_probability,
            )
        else:
            data_collator = DataCollatorForLanguageModelingStd(
                tokenizer=tokenizer,
                mlm_clone=data_args.mlm_clone,
                mlm_probability=data_args.mlm_probability,
            )

    # Initialize our Trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[MyCallback]
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint, ignore_keys_for_eval=["my_metrics"])
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(ignore_keys=["my_metrics"])
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
