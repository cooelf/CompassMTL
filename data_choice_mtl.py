# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension """

import sys
import csv
import glob
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import tqdm
from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available
import torch
from torch.utils.data.dataset import Dataset
import random
import datasets
from random import choice

logger = logging.getLogger(__name__)
import numpy as np
csv.field_size_limit(sys.maxsize)
@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]

std_task_dict = {
    "rainbow": ["anli", "cosmosqa", "hellaswag", "physicaliqa", "socialiqa", "winogrande", "csqa2"],
    "glue": ["cola", "mnli", "mrpc", "sst", "qqp", "qnli","rte"],
}
fast_task_dict = {
    "lex_glue": ["case_hold", "ecthr_a", "ecthr_b", "scotus", "eurlex", "ledgar", "unfair_tos"],
    "single": ["dream", "commonsense_qa", "quail", "quartz", "wiqa", "qasc", "sciq"],
    "ai2_arc": ["ARC-Easy", "ARC-Challenge"],
    "domain": ["zapsdcn/chemprot", "zapsdcn/rct-20k", "zapsdcn/hyperpartisan_news", "zapsdcn/imdb", "hrithikpiyush/acl-arc", "vannacute/AmazonReviewHelpfulness", "ag_news"],
    "super_glue": ["boolq", "cb", "copa"]
}

task_full_dict = list(std_task_dict.values()) + list(fast_task_dict.values())
task_dict = []
for task_list in task_full_dict:
    task_dict.extend(task_list)
task_dict = set(task_dict)

list_articles = ["Article 2", "Article 3", "Article 5", "Article 6", "Article 8", "Article 9", "Article 10", "Article 11", "Article 14", "Article 1 of Protocol 1"]
list_issue_areas = ["Criminal Procedure", "Civil Rights", "First Amendment", "Due Process", "Privacy", "Attorneys", "Unions", "Economic Activity", "Judicial Power", "Federalism", "Interstate Relations", "Federal Taxation", "Miscellaneous", "Private Action"]
with open("eurovoc_descriptors.json", "r") as fp:
    dict_concepts = json.load(fp)
list_provision = ["Adjustments", "Agreements", "Amendments", "Anti-Corruption Laws", "Applicable Laws", "Approvals", "Arbitration", "Assignments", "Assigns", "Authority", "Authorizations", "Base Salary", "Benefits", "Binding Effects", "Books", "Brokers", "Capitalization", "Change In Control", "Closings", "Compliance With Laws", "Confidentiality", "Consent To Jurisdiction", "Consents", "Construction", "Cooperation", "Costs", "Counterparts", "Death", "Defined Terms", "Definitions", "Disability", "Disclosures", "Duties", "Effective Dates", "Effectiveness", "Employment", "Enforceability", "Enforcements", "Entire Agreements", "Erisa", "Existence", "Expenses", "Fees", "Financial Statements", "Forfeitures", "Further Assurances", "General", "Governing Laws", "Headings", "Indemnifications", "Indemnity", "Insurances", "Integration", "Intellectual Property", "Interests", "Interpretations", "Jurisdictions", "Liens", "Litigations", "Miscellaneous", "Modifications", "No Conflicts", "No Defaults", "No Waivers", "Non-Disparagement", "Notices", "Organizations", "Participations", "Payments", "Positions", "Powers", "Publicity", "Qualifications", "Records", "Releases", "Remedies", "Representations", "Sales", "Sanctions", "Severability", "Solvency", "Specific Performance", "Submission To Jurisdiction", "Subsidiaries", "Successors", "Survival", "Tax Withholdings", "Taxes", "Terminations", "Terms", "Titles", "Transactions With Affiliates", "Use Of Proceeds", "Vacations", "Venues", "Vesting", "Waiver Of Jury Trials", "Waivers", "Warranties", "Withholdings"]
list_unfair = ["Limitation of liability", "Unilateral termination", "Unilateral change", "Content removal", "Contract by using", "Choice of law", "Jurisdiction", "Arbitration"]
list_chemprot = ['INHIBITOR', 'ANTAGONIST', 'AGONIST', 'DOWNREGULATOR', 'PRODUCT-OF', 'SUBSTRATE', 'INDIRECT-UPREGULATOR', 'UPREGULATOR', 'INDIRECT-DOWNREGULATOR', 'ACTIVATOR', 'AGONIST-ACTIVATOR', 'AGONIST-INHIBITOR', 'SUBSTRATE_PRODUCT-OF']
list_rct = ['OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS', 'BACKGROUND']
list_arc = ['introduction', 'experiments', "none", 'conclusion', 'related work', 'method']
list_ag = ['World', 'Sports', 'Business', 'Sci/Tech']


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_ids: List[List[int]]
    mlm_ids: List[int]
    attention_mask: Optional[List[List[int]]]
    mlm_masks: Optional[List[int]]
    token_type_ids: Optional[List[List[int]]]
    mlm_types: Optional[List[int]]
    label: Optional[int]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

class MultipleChoiceMTLMixDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[InputFeatures]

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        task: str,
        replace_task_prefix: bool,
        maximum: None,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: Split = Split.train,
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.task = task
        self.replace_task_prefix = replace_task_prefix
        self.maximum = maximum
        self.max_seq_length = max_seq_length
        self.overwrite_cache = overwrite_cache
        self.mode = mode
        self._build()

    def _build(self):
        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        self.features = []
        total_examples = 0
        if self.task == "all":
            mtl_tasks = list(std_task_dict.keys()) + list(fast_task_dict.keys())
        elif self.task == "glue":
            mtl_tasks = ["glue"]
        elif self.task == "rainbow":
            mtl_tasks = ["rainbow"]
        elif self.task == "lex_glue":
            mtl_tasks = ["lex_glue"]
        else:
            mtl_tasks = None
        mtl_tasks = set(mtl_tasks)
        for m_task in mtl_tasks:
            if m_task in std_task_dict:
                tasks = std_task_dict[m_task]
                for task in tasks:
                    processor = processors[task]()
                    task_data_dir = os.path.join(self.data_dir, task)
                    cached_features_file = os.path.join(
                        task_data_dir,
                        f"cached_mtl_std_{self.mode.value}_{self.tokenizer.__class__.__name__}_{self.max_seq_length}__{self.maximum}_{task}"
                    )
                    lock_path = cached_features_file + ".lock"
                    with FileLock(lock_path):
                        if os.path.exists(cached_features_file) and not self.overwrite_cache:
                            logger.info(f"Loading {task} features from cached file {cached_features_file}")
                            print(f"Loading {task} features from cached file {cached_features_file}")
                            # continue
                            features = torch.load(cached_features_file)
                        else:
                            logger.info(f"Creating {task} features from dataset file at {cached_features_file}")
                            print(f"Creating {task} features from dataset file at {cached_features_file}")
                            label_list = processor.get_labels()
                            if self.mode == Split.dev:
                                examples = processor.get_dev_examples(task_data_dir, task, mtl=True)
                            elif self.mode == Split.test:
                                examples = processor.get_test_examples(task_data_dir, task, mtl=True)
                            else:
                                examples = processor.get_train_examples(task_data_dir, task, mtl=True)
                            logger.info(f"{task} Training examples: {len(examples)}")
                            features = convert_examples_to_features(
                                task,
                                examples,
                                self.maximum,
                                label_list,
                                self.max_seq_length,
                                self.tokenizer,
                            )
                            logger.info(f"Saving {task} features into cached file {cached_features_file}")
                            torch.save(features, cached_features_file)
                    # features = features[:self.maximum]
                    total_examples += len(features)
                    self.features.append(features)
            elif m_task in fast_task_dict:
                tasks = fast_task_dict[m_task]
                for task in tasks:
                    task_data_dir = os.path.join(self.data_dir, task)
                    if "/" in task:
                        pre_task_dir = task_data_dir = os.path.join(self.data_dir, task.split("/")[0])
                        if not os.path.exists(pre_task_dir):
                            os.mkdir(pre_task_dir)
                    if not os.path.exists(task_data_dir):
                        os.mkdir(task_data_dir)
                    cache_task_name = task.replace("/", "")
                    cached_features_file = os.path.join(
                        task_data_dir,
                        f"cached_mtl_std_{self.mode.value}_{self.tokenizer.__class__.__name__}_{self.max_seq_length}__{self.maximum}_{cache_task_name}"
                    )
                    lock_path = cached_features_file + ".lock"
                    with FileLock(lock_path):
                        if os.path.exists(cached_features_file) and not self.overwrite_cache:
                            logger.info(f"Loading {task} features from cached file {cached_features_file}")
                            print(f"Loading {task} features from cached file {cached_features_file}")
                            # continue
                            features = torch.load(cached_features_file)
                        else:
                            logger.info(f"Creating features from dataset file at {cached_features_file}")
                            print(f"Creating features from dataset file at {cached_features_file}")
                            if m_task == 'lex_glue':
                                dataset = datasets.load_dataset('lex_glue', task)
                            elif m_task == 'ai2_arc':
                                dataset = datasets.load_dataset('ai2_arc', task)
                            elif m_task == 'domain':
                                dataset = datasets.load_dataset(task)
                            elif m_task == 'super_glue':
                                dataset = datasets.load_dataset('super_glue', task)
                            else:
                                dataset = datasets.load_dataset(task)
                            if self.mode == Split.dev:
                                examples = dataset['validation']
                            elif self.mode == Split.test:
                                examples = dataset['test']
                            elif self.mode == Split.train:
                                examples = dataset['train']
                            logger.info("Training examples: %s", len(examples))
                            features = convert_examples_to_features_fast(
                                task,
                                examples,
                                self.maximum,
                                self.max_seq_length,
                                self.tokenizer,
                            )
                            logger.info("Saving features into cached file %s", cached_features_file)
                            torch.save(features, cached_features_file)
                    total_examples += len(features)
                    self.features.append(features)
        self.mixture(self.features, maximum=self.maximum)
        print("total examples: ", total_examples, ", total mixtures: ", self.num_data)

    def mixture(self, task_data, maximum=None, seed=None):
        self.rng = np.random.RandomState(seed)
        # Create random order of tasks
        # Using size-proportional sampling
        task_choice_list = []
        for i, data in enumerate(task_data):
            # Examples-proportional mixing in T5
            if maximum is None:
                num_task_data = len(data)
            else:
                num_task_data = min(len(data), maximum)
            task_choice_list += [i] * num_task_data
        self.num_data = len(task_choice_list)
        task_choice_list = np.array(task_choice_list)
        self.rng.shuffle(task_choice_list)
        # Add index into each dataset
        counters = {}
        self.task_choice_list = []
        for i in range(len(task_choice_list)):
            idx = counters.get(task_choice_list[i], 0)
            self.task_choice_list.append((task_choice_list[i], idx))
            counters[task_choice_list[i]] = idx + 1
        print(counters)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        task_idx, example_idx = self.task_choice_list[index]
        task = self.features[task_idx]
        example = task[example_idx]
        input_ids = example.input_ids
        mlm_ids = example.mlm_ids
        mlm_masks = example.mlm_masks
        mlm_types = example.mlm_types
        attention_mask = example.attention_mask
        token_type_ids = example.token_type_ids
        labels = example.label
        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids, "labels": labels,
                "mlm_ids": mlm_ids, "mlm_masks": mlm_masks, "mlm_types": mlm_types}

class MultipleChoiceDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[InputFeatures]

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        tasks,
        maximum: 1000000,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: Split = Split.train,
    ):

        def get_key(dict, value):
            return [k for k, v in dict.items() if value in v][0]

        std_full_tasks = []
        for task_list in std_task_dict.values():
            std_full_tasks.extend(task_list)

        self.features = []
        tasks = tasks.split(",")
        for task in tasks:

            task_data_dir = os.path.join(data_dir, task)
            if "/" in task:
                pre_task_dir = os.path.join(data_dir, task.split("/")[0])
                if not os.path.exists(pre_task_dir):
                    os.mkdir(pre_task_dir)
            if not os.path.exists(task_data_dir):
                os.mkdir(task_data_dir)
            cache_task_name = task.replace("/", "")

            cached_features_file = os.path.join(
                task_data_dir,
                f"cached_big_single_{mode.value}_{tokenizer.__class__.__name__}_{max_seq_length}_{maximum}_{cache_task_name}"
            )
            # task = task.replace("%", "/")
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):
                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading {task} features from cached file {cached_features_file}")
                    print(f"Loading {task} features from cached file {cached_features_file}")
                    features = torch.load(cached_features_file)
                else:
                    if task in std_full_tasks:
                        processor = processors[task]()
                        label_list = processor.get_labels()
                        if mode == Split.dev:
                            examples = processor.get_dev_examples(task_data_dir, task, mtl=True)
                        elif mode == Split.test:
                            examples = processor.get_test_examples(task_data_dir, task, mtl=True)
                        else:
                            examples = processor.get_train_examples(task_data_dir, task, mtl=True)
                        logger.info(f"Training examples: {len(examples)}")
                        features = convert_examples_to_features(
                            task,
                            examples,
                            maximum,
                            label_list,
                            max_seq_length,
                            tokenizer,
                        )
                    else:
                        m_task = get_key(fast_task_dict, task)
                        logger.info(f"Creating features from dataset file at {cached_features_file}")
                        print(f"Creating features from dataset file at {cached_features_file}")
                        if m_task == 'lex_glue':
                            dataset = datasets.load_dataset('lex_glue', task)
                        elif m_task == 'ai2_arc':
                            dataset = datasets.load_dataset('ai2_arc', task)
                        elif m_task == 'domain':
                            dataset = datasets.load_dataset(task)
                        elif m_task == 'super_glue':
                            dataset = datasets.load_dataset('super_glue', task)
                        else:
                            dataset = datasets.load_dataset(task)
                        if mode == Split.dev:
                            # work for ag-news
                            if 'validation' in dataset:
                                examples = dataset['validation']
                            else:
                                examples = dataset['test']
                        elif mode == Split.test:
                            examples = dataset['test']
                        elif mode == Split.train:
                            examples = dataset['train']
                        logger.info("Training examples: %s", len(examples))
                        features = convert_examples_to_features_fast(
                            task,
                            examples,
                            maximum,
                            max_seq_length,
                            tokenizer,
                        )
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(features, cached_features_file)
            self.features.extend(features)
        # print("aaa")
        # logger.info(f"Saving features into cached file {cached_features_file}")
        # torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        example = self.features[index]
        input_ids = example.input_ids
        attention_mask = example.attention_mask
        token_type_ids = example.token_type_ids
        labels = example.label
        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids, "labels": labels}

class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} train")
        high = os.path.join(data_dir, "train/high")
        middle = os.path.join(data_dir, "train/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "train", task, mtl)

    def get_dev_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} dev")
        high = os.path.join(data_dir, "dev/high")
        middle = os.path.join(data_dir, "dev/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "dev", task, mtl)

    def get_test_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} test")
        high = os.path.join(data_dir, "test/high")
        middle = os.path.join(data_dir, "test/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "test", task, mtl)

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, "r", encoding="utf-8") as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines

    def _create_examples(self, lines, set_type, task, mtl):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            race_id = f"{set_type}-{data_raw['race_id']}"
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):
                truth = str(ord(data_raw["answers"][i]) - ord("A"))
                question = data_raw["questions"][i]
                options = data_raw["options"][i]
                examples.append(
                    InputExample(
                        example_id=race_id,
                        contexts=["[task: %s]" % str(task) + ' ' + article] * 4,
                        question=question,
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth,
                    )
                )
        return examples

class ANLIProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} train")
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), self._read_labels(os.path.join(data_dir, "train-labels.lst")), task, mtl)

    def get_dev_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} dev")
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), self._read_labels(os.path.join(data_dir, "dev-labels.lst")), task, mtl)

    def get_test_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} test")
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), None, task, mtl)

    def get_labels(self):
        """See base class."""
        return ["1", "2"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                data_raw = json.loads(line)
                lines.append(data_raw)
            return lines

    def _read_labels(self, input_file):
        with open(input_file, "r") as f:
            return [line.strip() for line in f]

    def _create_examples(self, lines, labels, task, mtl):
        """Creates examples for the training and dev sets."""
        if labels is None:
            labels = ["1"] * len(lines)
        if mtl:
            examples = [
                InputExample(
                    example_id=line["story_id"],
                    contexts=["[task: %s]" % str(task) + ' ' + line["obs1"]] * 4,
                    question=line["obs2"],
                    endings=[line["hyp1"], line["hyp2"], "N/A", "N/A"],
                    label=labels[idx],
                )
                for idx, line in enumerate(lines)  # we skip the line with the column names
            ]
        else:
            examples = [
                InputExample(
                    example_id=line["story_id"],
                    contexts=["[task: %s]" % str(task) + ' ' + line["obs1"]] * 2,
                    question=line["obs2"],
                    endings=[line["hyp1"], line["hyp2"]],
                    label=labels[idx],
                )
                for idx, line in enumerate(lines)  # we skip the line with the column names
            ]
        return examples

class CosmosQAProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} train")
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), self._read_labels(os.path.join(data_dir, "train-labels.lst")), task, mtl)

    def get_dev_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} dev")
        return self._create_examples(self._read_json(os.path.join(data_dir, "valid.jsonl")), self._read_labels(os.path.join(data_dir, "valid-labels.lst")), task, mtl)

    def get_test_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} test")
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), None, task, mtl)

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                data_raw = json.loads(line)
                lines.append(data_raw)
            return lines

    def _read_labels(self, input_file):
        with open(input_file, "r") as f:
            return [line.strip() for line in f]

    def _create_examples(self, lines, labels, task, mtl):
        """Creates examples for the training and dev sets."""
        if labels is None:
            labels = ["0"] * len(lines)
        examples = [
            InputExample(
                example_id=line["id"],
                contexts=["[task: %s]" % str(task) + ' ' + line["context"]] * 4,
                question=line["question"],
                endings=[line["answer0"], line["answer1"], line["answer2"], line["answer3"]],
                label=labels[idx],
            )
            for idx, line in enumerate(lines)  # we skip the line with the column names
        ]

        return examples

class HellaswagProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} train")
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), self._read_labels(os.path.join(data_dir, "train-labels.lst")), task , mtl)

    def get_dev_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} dev")
        return self._create_examples(self._read_json(os.path.join(data_dir, "valid.jsonl")), self._read_labels(os.path.join(data_dir, "valid-labels.lst")), task , mtl)

    def get_test_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} test")
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), None, task , mtl)

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                data_raw = json.loads(line)
                lines.append(data_raw)
            return lines

    def _read_labels(self, input_file):
        with open(input_file, "r") as f:
            return [line.strip() for line in f]

    def _create_examples(self, lines, labels, task , mtl):
        """Creates examples for the training and dev sets."""
        if labels is None:
            labels = ["0"] * len(lines)
        examples = [
            InputExample(
                example_id=line["ind"],
                contexts=["[task: %s]" % str(task) + ' ' + line["ctx"] + ' ' + option for option in line["ending_options"]],
                question='',
                endings=[line["activity_label"]]*4,
                label=labels[idx],
            )
            for idx, line in enumerate(lines)  # we skip the line with the column names
        ]

        return examples

class PIQAProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} train")
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), self._read_labels(os.path.join(data_dir, "train-labels.lst")), task , mtl)

    def get_dev_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} dev")
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), self._read_labels(os.path.join(data_dir, "dev-labels.lst")), task , mtl)

    def get_test_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} test")
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), None, task , mtl)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                data_raw = json.loads(line)
                lines.append(data_raw)
            return lines

    def _read_labels(self, input_file):
        with open(input_file, "r") as f:
            return [line.strip() for line in f]

    def _create_examples(self, lines, labels, task , mtl):
        """Creates examples for the training and dev sets."""
        if labels is None:
            labels = ["0"] * len(lines)
        if mtl:
            examples = [
                InputExample(
                    example_id=line["id"],
                    contexts=["[task: %s]" % str(task) + ' ' + line["goal"]] * 4,
                    question='',
                    endings=[line["sol1"], line["sol2"], "N/A", "N/A"],
                    label=labels[idx],
                )
                for idx, line in enumerate(lines)  # we skip the line with the column names
            ]
        else:
            examples = [
                InputExample(
                    example_id=line["id"],
                    contexts=["[task: %s]" % str(task) + ' ' + line["goal"]] * 2,
                    question="",
                    endings=[line["sol1"], line["sol2"]],
                    label=labels[idx],
                )
                for idx, line in enumerate(lines)  # we skip the line with the column names
            ]

        return examples

class SocialQAProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} train")
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), self._read_labels(os.path.join(data_dir, "train-labels.lst")), task , mtl)

    def get_dev_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} dev")
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), self._read_labels(os.path.join(data_dir, "dev-labels.lst")), task , mtl)

    def get_test_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} test")
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), None, task , mtl)

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                data_raw = json.loads(line)
                lines.append(data_raw)
            return lines

    def _read_labels(self, input_file):
        with open(input_file, "r") as f:
            return [line.strip() for line in f]

    def _create_examples(self, lines, labels, task , mtl):
        """Creates examples for the training and dev sets."""
        if labels is None:
            labels = ["1"] * len(lines)
        if mtl:
            examples = [
                InputExample(
                    example_id="",
                    contexts=["[task: %s]" % str(task) + ' ' + line["context"]] * 4,
                    question=line["question"],
                    endings=[line["answerA"], line["answerB"], line["answerC"], "N/A"],
                    label=labels[idx],
                )
                for idx, line in enumerate(lines)  # we skip the line with the column names
            ]
        else:
            examples = [
                InputExample(
                    example_id="",
                    contexts=["[task: %s]" % str(task) + ' ' + line["context"]] * 3,
                    question=line["question"],
                    endings=[line["answerA"], line["answerB"], line["answerC"]],
                    label=labels[idx],
                )
                for idx, line in enumerate(lines)  # we skip the line with the column names
            ]

        return examples

class WinoProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} train")
        return self._create_examples(self._read_json(os.path.join(data_dir, "train_xl.jsonl")), self._read_labels(os.path.join(data_dir, "train_xl-labels.lst")), task , mtl)

    def get_dev_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} dev")
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), self._read_labels(os.path.join(data_dir, "dev-labels.lst")), task , mtl)

    def get_test_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} test")
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), None, task , mtl)

    def get_labels(self):
        """See base class."""
        return ["1", "2"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                data_raw = json.loads(line)
                lines.append(data_raw)
            return lines

    def _read_labels(self, input_file):
        with open(input_file, "r") as f:
            return [line.strip() for line in f]

    def _create_examples(self, lines, labels, task , mtl):
        """Creates examples for the training and dev sets."""
        if labels is None:
            labels = ["1"] * len(lines)
        if mtl:
            examples = [
                InputExample(
                    example_id=line["qID"],
                    contexts=["[task: %s]" % str(task) + ' ' + line["sentence"]] * 4,
                    question='',
                    endings=[line["option1"], line["option2"], "N/A", "N/A"],
                    label=labels[idx],
                )
                for idx, line in enumerate(lines)  # we skip the line with the column names
            ]
        else:
            examples = [
                InputExample(
                    example_id=line["qID"],
                    contexts=["[task: %s]" % str(task) + ' ' + line["sentence"]] * 2,
                    question='',
                    endings=[line["option1"], line["option2"]],
                    label=labels[idx],
                )
                for idx, line in enumerate(lines)  # we skip the line with the column names
            ]

        return examples

class CSQA2Processor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} train")
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "", task, mtl)

    def get_dev_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} dev")
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "", task, mtl)

    def get_test_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} test")
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), None, task, mtl)

    def get_labels(self):
        """See base class."""
        return ["yes", "no"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                data_raw = json.loads(line)
                lines.append(data_raw)
            return lines

    def _read_labels(self, input_file):
        with open(input_file, "r") as f:
            return [line.strip() for line in f]

    def _create_examples(self, lines, labels, task, mtl):
        """Creates examples for the training and dev sets."""
        if labels is None:
            examples = [
                InputExample(
                    example_id=line["id"],
                    contexts=["[task: %s]" % str(task) + ' ' + line["question"]] * 2,
                    question="",
                    endings=["yes", "no"],
                    label="yes",
                )
                for idx, line in enumerate(lines)  # we skip the line with the column names
            ]
        else:
            if mtl:
                examples = [
                    InputExample(
                        example_id=line["id"],
                        contexts=["[task: %s]" % str(task) + ' ' + line["question"]] * 4,
                        question="",
                        endings=["yes", "no", "N/A", "N/A"],
                        label=line["answer"],
                    )
                    for idx, line in enumerate(lines)  # we skip the line with the column names
                ]
            else:
                examples = [
                    InputExample(
                        example_id=line["id"],
                        contexts=["[task: %s]" % str(task) + ' ' + line["question"]] * 2,
                        question="",
                        endings=["yes", "no"],
                        label=line["answer"],
                    )
                    for idx, line in enumerate(lines)  # we skip the line with the column names
                ]

        return examples

class MrpcProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} train")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", task , mtl)

    def get_dev_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} dev")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", task , mtl)

    def get_test_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} test")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test.csv", task , mtl)

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, task , mtl):
        """Creates examples for the training and dev sets."""
        if mtl:
            examples = [
                InputExample(
                    example_id="%s-%s" % (set_type, idx),
                    contexts=["[task: %s]" % str(task) + ' ' + line[3]] * 4,
                    question=line[4],
                    endings=["the sentences in the pair are semantically not equivalent", "the sentences in the pair are semantically equivalent", "N/A", "N/A"],
                    label=line[0]
                )
                for idx, line in enumerate(lines[1:])  # we skip the line with the column names
            ]
        else:
            examples = [
                InputExample(
                    example_id="%s-%s" % (set_type, idx),
                    contexts=["[task: %s]" % str(task) + ' ' + line[3]] * 2,
                    question=line[4],
                    endings=["the sentences in the pair are semantically not equivalent", "the sentences in the pair are semantically equivalent"],
                    label=line[0]
                )
                for idx, line in enumerate(lines[1:])  # we skip the line with the column names
            ]
        return examples

class MNLIProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} train")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", task , mtl)

    def get_dev_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} dev")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev", task , mtl)

    def get_test_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} test")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test.csv", task , mtl)

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type, task , mtl):
        """Creates examples for the training and dev sets."""
        if mtl:
            examples = [
                InputExample(
                    example_id=line[0],
                    contexts=["[task: %s]" % str(task) + ' ' + line[8]] * 4,
                    question=line[9],
                    endings=["the inference relation between the two sentences is contradiction", "the inference relation between the two sentences is entailment", "the inference relation between the two sentences is neutral", "N/A"],
                    label=line[-1]
                )
                for line in lines[1:]  # we skip the line with the column names
            ]
        else:
            examples = [
                InputExample(
                    example_id=line[0],
                    contexts=["[task: %s]" % str(task) + ' ' + line[8]] * 3,
                    question=line[9],
                    endings=["the inference relation between the two sentences is contradiction", "the inference relation between the two sentences is entailment", "the inference relation between the two sentences is neutral"],
                    label=line[-1]
                )
                for line in lines[1:]  # we skip the line with the column names
            ]
        return examples

class CoLAProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} train")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", task , mtl)

    def get_dev_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} dev")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", task , mtl)

    def get_test_examples(self, data_dir, task=None, mtl=False):
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} test")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test.csv", task , mtl)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, task , mtl):
        """Creates examples for the training and dev sets."""
        if mtl:
            examples = [
                InputExample(
                    example_id="%s-%s" % (set_type, idx),
                    contexts=["[task: %s]" % str(task) + ' ' + line[3]] * 4,
                    question='',
                    endings=["the grammar of the sentence is unacceptable", "the grammar of the sentence is acceptable", "N/A", "N/A"],
                    label=line[1]
                )
                for idx, line in enumerate(lines)  # we skip the line with the column names
            ]
        else:
            examples = [
                InputExample(
                    example_id="%s-%s" % (set_type, idx),
                    contexts=["[task: %s]" % str(task) + ' ' + line[3]] * 2,
                    question='',
                    endings=["the grammar of the sentence is unacceptable", "the grammar of the sentence is acceptable"],
                    label=line[1]
                )
                for idx, line in enumerate(lines)  # we skip the line with the column names
            ]
        return examples

class SST2Processor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir, task=None, mtl=False) :
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} train")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", task , mtl)

    def get_dev_examples(self, data_dir, task=None, mtl=False) :
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} dev")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", task , mtl)

    def get_test_examples(self, data_dir, task=None, mtl=False) :
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} test")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test.csv", task , mtl)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, task , mtl):
        """Creates examples for the training and dev sets."""
        if mtl:
            examples = [
                InputExample(
                    example_id="%s-%s" % (set_type, idx),
                    contexts=["[task: %s]" % str(task) + ' ' + line[0]] * 4,
                    question='',
                    endings=["the sentiment is negative", "the sentiment is positive", "N/A", "N/A"],
                    label=line[1]
                )
                for idx, line in enumerate(lines[1:])  # we skip the line with the column names
            ]
        else:
            examples = [
                InputExample(
                    example_id="%s-%s" % (set_type, idx),
                    contexts=["[task: %s]" % str(task) + ' ' + line[0]] * 2,
                    question='',
                    endings=["the sentiment is negative", "the sentiment is positive"],
                    label=line[1]
                )
                for idx, line in enumerate(lines[1:])  # we skip the line with the column names
            ]
        return examples

class QqpProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir, task=None, mtl=False) :
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} train")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", task , mtl)

    def get_dev_examples(self, data_dir, task=None, mtl=False) :
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} dev")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", task , mtl)

    def get_test_examples(self, data_dir, task=None, mtl=False) :
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} test")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test.csv", task , mtl)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, task , mtl):
        """Creates examples for the training and dev sets."""
        if mtl:
            examples = [
                InputExample(
                    example_id="%s-%s" % (set_type, line[0]),
                    contexts=["[task: %s]" % str(task) + ' ' + line[3]] * 4,
                    question=line[4],
                    endings=["The paired questions are not semantically duplicate", "The paired questions are semantically duplicate", "N/A", "N/A"],
                    label=line[5]
                )
                for idx, line in enumerate(lines[1:])  # we skip the line with the column names
            ]
        else:
            examples = [
                InputExample(
                    example_id="%s-%s" % (set_type, line[0]),
                    contexts=["[task: %s]" % str(task) + ' ' + line[3]] * 2,
                    question=line[4],
                    endings=["The paired questions are not semantically duplicate", "The paired questions are semantically duplicate"],
                    label=line[5]
                )
                for idx, line in enumerate(lines[1:])  # we skip the line with the column names
            ]
        return examples

class QnliProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir, task=None, mtl=False) :
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} train")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", task , mtl)

    def get_dev_examples(self, data_dir, task=None, mtl=False) :
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} dev")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", task , mtl)

    def get_test_examples(self, data_dir, task=None, mtl=False) :
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} test")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test.csv", task , mtl)

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type, task , mtl):
        """Creates examples for the training and dev sets."""
        if mtl:
            examples = [
                InputExample(
                    example_id="%s-%s" % (set_type, line[0]),
                    contexts=["[task: %s]" % str(task) + ' ' + line[1]] * 4,
                    question=line[2],
                    endings=["the context sentence contains the answer to the question", "the context sentence does not contains the answer to the question", "N/A", "N/A"],
                    label=line[-1]
                )
                for idx, line in enumerate(lines[1:])  # we skip the line with the column names
            ]
        else:
            examples = [
                InputExample(
                    example_id="%s-%s" % (set_type, line[0]),
                    contexts=["[task: %s]" % str(task) + ' ' + line[1]] * 2,
                    question=line[2],
                    endings=["the context sentence contains the answer to the question", "the context sentence does not contains the answer to the question"],
                    label=line[-1]
                )
                for idx, line in enumerate(lines[1:])  # we skip the line with the column names
            ]
        return examples

class RTEProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir, task=None, mtl=False) :
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} train")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", task , mtl)

    def get_dev_examples(self, data_dir, task=None, mtl=False) :
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} dev")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", task , mtl)

    def get_test_examples(self, data_dir, task=None, mtl=False) :
        """See base class."""
        logger.info(f"LOOKING AT {data_dir} test")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test.csv", task , mtl)

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type, task , mtl):
        """Creates examples for the training and dev sets."""
        if mtl:
            examples = [
                InputExample(
                    example_id="%s-%s" % (set_type, line[0]),
                    contexts=["[task: %s]" % str(task) + ' ' + line[1]] * 4,
                    question=line[2],
                    endings=["the inference relation between the two sentences is entailment", "the inference relation between the two sentences is not entailment", "N/A", "N/A"],
                    label=line[-1]
                )
                for idx, line in enumerate(lines[1:])  # we skip the line with the column names
            ]
        else:
            examples = [
                InputExample(
                    example_id="%s-%s" % (set_type, line[0]),
                    contexts=["[task: %s]" % str(task) + ' ' + line[1]] * 2,
                    question=line[2],
                    endings=["the inference relation between the two sentences is entailment", "the inference relation between the two sentences is not entailment"],
                    label=line[-1]
                )
                for idx, line in enumerate(lines[1:])  # we skip the line with the column names
            ]
        return examples

def convert_examples_to_features(
    task: str,
    examples: List[InputExample],
    maximum: int,
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index > maximum:
            break
        if ex_index % 10000 == 0:
            logger.info(f"Writing example {ex_index} of {len(examples)}")
        choices_inputs = []
        task_inputs = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context

            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                # if example.question == "":
                #     text_b = example.question + " " + tokenizer.eos_token + " " + ending
                # else:
                #     text_b = example.question + " " + ending
                text_b = example.question + " " + tokenizer.eos_token + " " + ending

            inputs = tokenizer(text_a, text_b, add_special_tokens=True, max_length=max_length, truncation='longest_first',
                padding='max_length', return_attention_mask=True, return_token_type_ids=True)

            if ending_idx == 0:
                mlm_inputs = tokenizer(text_a, example.question, add_special_tokens=True, max_length=max_length, truncation='longest_first',
                padding='max_length', return_attention_mask=True, return_token_type_ids=True)

            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            choices_inputs.append(inputs)

        label = label_map[example.label]

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        features.append(
            InputFeatures(
                example_id=task+str(ex_index),
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                mlm_ids=mlm_inputs["input_ids"],
                mlm_masks=mlm_inputs["attention_mask"],
                mlm_types=mlm_inputs["token_type_ids"],
                label=label
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features

def mapping_examples(task, example):
    if task == "case_hold":
        context = example['context']
        question = ''
        labels = example['label']
        endings = example['endings']
        choice_list = [i for i in range(0, 5) if i != labels]
        random_del_id = choice(choice_list)
        endings.pop(random_del_id)
        assert random_del_id != labels
        if labels > random_del_id:
            labels = labels - 1
        label_id = labels
    elif task == "ecthr_a" or task == "ecthr_b":
        context = " ".join(example['text'])
        question = ''
        labels = example['labels']
        if len(labels) == 0:
            right_ending = "there is no violation"
        else:
            right_ending = 'the case violates' + ' ' + list_articles[random.choice(labels)]
        wrong_ending = random.sample([art for idx, art in enumerate(list_articles) if idx not in labels], 3)
        wrong_ending = ['the case violates' + ' ' + art for art in wrong_ending]
        endings = [right_ending] + wrong_ending
        random.shuffle(endings)
        label_id = endings.index(right_ending)
    elif task == "scotus":
        context = example['text']
        question = ''
        labels = example['label']
        right_ending = 'the relevant issue area is' + ' ' + list_issue_areas[labels]
        wrong_ending = random.sample([art for idx, art in enumerate(list_issue_areas) if idx != labels], 3)
        wrong_ending = ['the relevant issue area is' + ' ' + art for art in wrong_ending]
        endings = [right_ending] + wrong_ending
        random.shuffle(endings)
        label_id = endings.index(right_ending)
    elif task == "eurlex":
        context = example['text']
        question = ''
        labels = example['labels']
        if len(labels) == 0:
            right_ending = "there is no EuroVoc concept label"
        else:
            random_choice = str(random.choice(labels))
            while random_choice not in dict_concepts:
                print("dirty data", random_choice, labels)
                random_choice = str(random.choice(labels))
            right_ending = 'the EuroVoc concept label is' + ' ' + dict_concepts[random_choice]['en']
        wrong_ending = random.sample([dict_concepts[key]['en'] for key in dict_concepts.keys() if key not in labels], 3)
        wrong_ending = ['the EuroVoc concept label is' + ' ' + art for art in wrong_ending]
        endings = [right_ending] + wrong_ending
        random.shuffle(endings)
        label_id = endings.index(right_ending)
    elif task == "ledgar":
        context = example['text']
        question = ''
        labels = example['label']
        right_ending = 'the main topic of the contract provision is' + ' ' + list_provision[labels]
        wrong_ending = random.sample([art for idx, art in enumerate(list_provision) if idx != labels], 3)
        wrong_ending = ['the main topic of the contract provision is' + ' ' + art for art in wrong_ending]
        endings = [right_ending] + wrong_ending
        random.shuffle(endings)
        label_id = endings.index(right_ending)
    elif task == "unfair_tos":
        context = example['text']
        question = ''
        labels = example['labels']
        if len(labels) == 0:
            right_ending = "there is no unfair contractual term"
        else:
            right_ending = 'the term potentially violate user rights' + ' ' + list_unfair[random.choice(labels)]
        wrong_ending = random.sample([art for idx, art in enumerate(list_unfair) if idx not in labels], 3)
        wrong_ending = ['the term potentially violate user rights' + ' ' + art for art in wrong_ending]
        endings = [right_ending] + wrong_ending
        random.shuffle(endings)
        label_id = endings.index(right_ending)
    elif task == "dream":
        context = " ".join(example['dialogue'])
        question = example['question']
        endings = example['choice']
        answer = example['answer']
        endings.append('N/A')
        label_id = endings.index(answer)
    elif task == "commonsense_qa":
        context = example['question_concept']
        question = example['question']
        answerKey = example['answerKey']
        choices = example['choices']
        endings = choices['text']
        label_idx = choices['label'].index(answerKey)
        choice_list = [i for i in range(0, 5) if i != label_idx]
        random_del_id = choice(choice_list)
        endings.pop(random_del_id)
        assert random_del_id != label_idx
        if label_idx > random_del_id:
            label_idx = label_idx - 1
        label_id = label_idx
    elif task == "quail":
        context = example['context']
        question = example['question']
        endings = example['answers']
        label_id = example['correct_answer_id']
    elif task == "quartz":
        context = example['para']
        question = example['question']
        choices = example['choices']
        answerKey = example['answerKey']
        endings = choices["text"] + ["N/A", "N/A"]
        label_idx = choices["label"].index(answerKey)
        label_id = label_idx
    elif task == "wiqa":
        context = example['question_stem']
        question = ". ".join(example['question_para_step'])
        choices = example['choices']
        answerKey = example['answer_label_as_choice']
        endings = choices["text"] + ["N/A"]
        label_idx = choices["label"].index(answerKey)
        label_id = label_idx
    elif task == "qasc":
        context = ''
        question = example['question']
        answerKey = example['answerKey']
        choices = example['choices']
        endings = choices['text']
        label_idx = choices['label'].index(answerKey)
        while len(endings) > 4:
            choice_list = [i for i in range(0, len(endings)) if i != label_idx]
            random_del_id = choice(choice_list)
            endings.pop(random_del_id)
            assert random_del_id != label_idx
            if label_idx > random_del_id:
                label_idx = label_idx - 1
        label_id = label_idx
    elif task == "sciq":
        context = example['support']
        question = example['question']
        endings = [example['correct_answer'], example['distractor1'], example['distractor2'], example['distractor3']]
        answerKey = example['correct_answer']
        random.shuffle(endings)
        label_id = endings.index(answerKey)
    elif task == "ARC-Easy" or task == "ARC-Challenge":
        context = ''
        question = example['question']
        answerKey = example['answerKey']
        choices = example['choices']
        endings = choices['text']
        label_idx = choices['label'].index(answerKey)
        while len(endings) < 4:
            endings.append("N/A")
        while len(endings) > 4:
            choice_list = [i for i in range(0, len(endings)) if i != label_idx]
            random_del_id = choice(choice_list)
            endings.pop(random_del_id)
            assert random_del_id != label_idx
            if label_idx > random_del_id:
                label_idx = label_idx - 1
        label_id = label_idx
    elif task == "swag":
        context = ''
        question = example['startphrase']
        endings = [example['ending0'], example['ending1'], example['ending2'], example['ending3']]
        label_id = example['label']
    elif task == "zapsdcn/chemprot":
        context = example['text']
        question = ''
        label = example['label']
        right_ending = 'the relation classification is' + ' ' + label
        wrong_ending = random.sample([art for art in list_chemprot if art != label], 3)
        wrong_ending = ['the relation classification is' + ' ' + art for art in wrong_ending]
        endings = [right_ending] + wrong_ending
        random.shuffle(endings)
        label_id = endings.index(right_ending)
    elif task == "zapsdcn/rct-20k":
        context = example['text']
        question = ''
        label = example['label']
        right_ending = 'the abstract sent. roles is' + ' ' + label
        wrong_ending = random.sample([art for art in list_rct if art != label], 3)
        wrong_ending = ['the abstract sent. roles is' + ' ' + art for art in wrong_ending]
        endings = [right_ending] + wrong_ending
        random.shuffle(endings)
        label_id = endings.index(right_ending)
    elif task == "zapsdcn/hyperpartisan_news":
        context = example['text']
        question = ''
        label = example['label']
        if label == 'true':
            right_ending = 'there is partisanship'
        else:
            right_ending = 'there is no partisanship'
        endings = ["there is partisanship", "there is no partisanship", "N/A", "N/A"]
        random.shuffle(endings)
        label_id = endings.index(right_ending)
    elif task == "zapsdcn/imdb":
        context = example['text']
        question = ''
        label = example['label']
        if label == 1:
            right_ending = 'the sentiment is positive'
        else:
            right_ending = 'the sentiment is negative'
        endings = ["the sentiment is positive", "the sentiment is negative", "N/A", "N/A"]
        random.shuffle(endings)
        label_id = endings.index(right_ending)
    elif task == "hrithikpiyush/acl-arc":
        context = example['text']
        question = ''
        label = str(example['section_name'])
        right_ending = 'the citation intent is' + ' ' + label
        wrong_ending = random.sample([art for art in list_arc if art != label], 3)
        wrong_ending = ['the citation intent is' + ' ' + art for art in wrong_ending]
        endings = [right_ending] + wrong_ending
        random.shuffle(endings)
        label_id = endings.index(right_ending)
    elif task == "vannacute/AmazonReviewHelpfulness":
        context = example['text']
        question = ''
        label = example['label']
        right_ending = 'the review is ' + label
        endings = ["the review is helpful", "the review is unhelpful", "N/A", "N/A"]
        random.shuffle(endings)
        label_id = endings.index(right_ending)
    elif task == "ag_news":
        context = example['text']
        question = ''
        label = example['label']
        right_ending = 'the news topic is' + ' ' + list_ag[label]
        wrong_ending = random.sample([art for idx, art in enumerate(list_ag) if idx != label], 3)
        wrong_ending = ['the news topic is' + ' ' + art for art in wrong_ending]
        endings = [right_ending] + wrong_ending
        random.shuffle(endings)
        label_id = endings.index(right_ending)
    elif task == "boolq":
        context = example['passage']
        question = example['question']
        label = example['label']
        if label == 1:
            right_ending = 'True'
        else:
            right_ending = 'False'
        endings = ["True", "False", "N/A", "N/A"]
        random.shuffle(endings)
        label_id = endings.index(right_ending)
    elif task == "cb":
        context = example['premise']
        question = example['hypothesis']
        label = example['label']
        if label == 0:
            right_ending = "the inference relation between the two sentences is entailment"
        elif label == 1:
            right_ending = "the inference relation between the two sentences is contradiction"
        elif label == 2:
            right_ending = "the inference relation between the two sentences is neutral"
        else:
            right_ending = "N/A"
        endings = ["the inference relation between the two sentences is entailment",
                   "the inference relation between the two sentences is contradiction",
                   "the inference relation between the two sentences is neutral",
                   "N/A"]
        random.shuffle(endings)
        label_id = endings.index(right_ending)
    elif task == "copa":
        context = example['question']
        question = example['premise']
        endings = [example['choice1'], example['choice2'], "N/A", "N/A"]
        label_id = example['label']
    else:
        context = None
        question = None
        endings = None
        label_id = None

    return context, question, endings, label_id

def convert_examples_to_features_fast(
    task: str,
    examples: List[InputExample],
    maximum,
    max_length: int,
    tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index > maximum:
            break
        if ex_index % 10000 == 0:
            logger.info(f"Writing example {ex_index} of {len(examples)}")
        choices_inputs = []
        context, question, endings, label_id = mapping_examples(task, example)
        if len(endings) != 4:
            print("****************")
        assert len(endings) == 4
        for ending_idx, ending in enumerate(endings):
            text_a = "[task: %s]" % str(task) + ' ' + context

            if question.find("_") != -1:
                # this is for cloze question
                text_b = question.replace("_", ending)
            else:
                text_b = question + " " + tokenizer.eos_token + " " + ending

            inputs = tokenizer(text_a, text_b, add_special_tokens=True, max_length=max_length, truncation='longest_first',
                pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=True)

            if ending_idx == 0:
                mlm_inputs = tokenizer(text_a, question, add_special_tokens=True, max_length=max_length, truncation='longest_first',
                pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=True)

            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            choices_inputs.append(inputs)

        # label = label_map[example.label]

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        features.append(
            InputFeatures(
                example_id=task+str(ex_index),
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                mlm_ids=mlm_inputs["input_ids"],
                mlm_masks=mlm_inputs["attention_mask"],
                mlm_types=mlm_inputs["token_type_ids"],
                label=label_id,
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features

processors = {"race": RaceProcessor,
              "anli": ANLIProcessor,
              "cosmosqa": CosmosQAProcessor,
              "hellaswag": HellaswagProcessor,
              "physicaliqa": PIQAProcessor,
              "socialiqa": SocialQAProcessor,
              "winogrande": WinoProcessor,
              "csqa2": CSQA2Processor,
              "cola": CoLAProcessor,
              "mnli": MNLIProcessor,
              "mrpc": MrpcProcessor,
              "sst": SST2Processor,
              "qqp": QqpProcessor,
              "qnli": QnliProcessor,
              "rte": RTEProcessor,
              }