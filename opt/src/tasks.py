from .templates import *
from .utils import temp_seed
import json
import os
from datasets import load_dataset
from dataclasses import dataclass
from typing import List, Union
import string
import random
import datasets
import sys
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# ---- HF offline/cache shim (drop-in, keeps your call-sites unchanged) ----
import os as _os
import logging as _logging
from datasets import DownloadConfig as _DownloadConfig
from datasets import load_dataset as _orig_load_dataset

def _try_load(path, name, kwargs):
    return _orig_load_dataset(path, **kwargs) if name is None else _orig_load_dataset(path, name, **kwargs)

def load_dataset(*args, **kwargs):
    """
    Wrapper that:
      1) uses $HF_DATASETS_CACHE if set,
      2) respects offline mode ($HF_DATASETS_OFFLINE / local_files_only),
      3) retries with sensible alternates (DROP, BoolQ, WiC, RTE, ReCoRD),
      4) also tries lowercase (path/name) as a last resort.
    """
    cache_dir = _os.environ.get("HF_DATASETS_CACHE")
    if cache_dir and "cache_dir" not in kwargs:
        kwargs["cache_dir"] = cache_dir

    # Offline mode (no network on compute nodes)
    offline_env = str(_os.environ.get("HF_DATASETS_OFFLINE", "")).strip()
    if offline_env in ("1", "true", "True", "yes", "YES"):
        kwargs.setdefault("download_mode", "reuse_dataset_if_exists")
        kwargs["download_config"] = kwargs.get("download_config", _DownloadConfig(local_files_only=True))

    if not args:
        raise TypeError("load_dataset wrapper requires at least the dataset path")
    path = args[0]
    name = args[1] if len(args) >= 2 else None

    alternates = [(path, name)]

    # SQuAD
    if path == "squad" and name is None:
        alternates += [("rajpurkar/squad", None)]

    # DROP
    if path == "drop" and name is None:
        alternates += [("allenai/drop", None)]

    # BoolQ
    if (path, name) == ("boolq", None):
        alternates += [("super_glue", "boolq")]

    # WiC
    if (path, name) == ("super_glue", "wic"):
        alternates += [("wic", None)]

    # RTE
    if (path, name) == ("super_glue", "rte"):
        alternates += [("glue", "rte")]

    # ReCoRD: primary is super_glue/record; add helpful alternates
    if (path, name) == ("record", None):
        alternates += [("super_glue", "record")]
    if (path, name) == ("super_glue", "ReCoRD"):  # handle capitalized config
        alternates += [("super_glue", "record")]

    # Last-resort: try lowercase path/name if user passed caps
    lp = path.lower() if isinstance(path, str) else path
    ln = name.lower() if isinstance(name, str) else name
    if (lp, ln) != (path, name):
        alternates += [(lp, ln)]

    last_err = None
    for p, n in alternates:
        try:
            return _try_load(p, n, kwargs)
        except Exception as e:
            last_err = e

    _logging.getLogger(__name__).warning(
        "load_dataset failed for %s. If running offline, pre-cache on a login node and set "
        "HF_DATASETS_CACHE + HF_DATASETS_OFFLINE. Last error: %r", alternates, last_err
    )
    raise last_err
# ---- end shim ----




def get_task(task_name):
    aa = task_name.split("__")
    if len(aa) == 2:
        task_group, subtask = aa
    else:
        task_group = aa[0]
        subtask = None
    class_ = getattr(sys.modules[__name__], f"{task_group}Dataset")
    instance = class_(subtask)
    return instance


@dataclass
class Sample:
    id: int = None
    data: dict = None
    correct_candidate: Union[str, List[str]] = None
    candidates: List[str] = None


class Dataset:
    mixed_set = False
    train_sep = "\n\n"
    generation = False # whether this is a generation task

    def __init__(self, subtask=None, **kwargs) -> None:
        self.subtask = subtask
    
    def get_task_name(self):
        return self.subtask
        
    def load_dataset():
        raise NotImplementedError
    
    def get_template(self, template_version=0):
       templates = {0: Template}
       return templates[template_version]
   
    def build_sample(self, example):
        return 
     
    def sample_train_sets(self, num_train=32, num_dev=None, num_eval=None, num_train_sets=None, seed=None):
        if seed is not None:
            # one train/demo set using the designated seed
            seeds = [seed]
        elif num_train_sets is not None:
            # num_train_sets train/demo sets
            seeds = list(range(num_train_sets))
        else: 
            # one train/demo set per evaluation sample
            assert num_dev is None # not supported
            len_valid_samples = len(self.samples["valid"]) if num_eval is None else num_eval
            with temp_seed(0):
                seeds = np.random.randint(0, 10000, len_valid_samples)

        train_samples = [] 
        for i, set_seed in enumerate(seeds):
            if self.mixed_set:
                raise NotImplementedError
                train_samples.append(self.sample_subset(data_split="valid", seed=set_seed, num=num_train, exclude=i))
            else:
                if num_dev is not None:
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_train+num_dev)) # dev set is included at the end of train set
                    if num_train + num_dev > len(self.samples["train"]):
                        logger.warn("num_train + num_dev > available training examples")
                else:
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_train))
                if num_dev is not None:
                    logger.info(f"Sample train set {len(train_samples[-1])}/{len(self.samples['train'])}")
                    logger.info(f"... including dev set {num_dev} samples")
        return train_samples

    def sample_subset(self, data_split="train", seed=0, num=100, exclude=None):
        with temp_seed(seed):
            samples = self.samples[data_split] 
            lens = len(samples)
            index = np.random.permutation(lens).tolist()[:num if exclude is None else num+1]
            if exclude is not None and exclude in index:
                index.remove(exclude)
            else:
                index = index[:num]
            return [samples[i] for i in index]
    
    @property
    def valid_samples(self):
        return self.samples["valid"]
    
    @property
    def test_samples(self):
        return self.samples["test"]



class SST2Dataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        d = load_dataset('glue', 'sst2')
        train_d = d["train"]
        validation_d = d["validation"] 
        
        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]
        test_samples = [self.build_sample(ex) for ex in validation_d]
        
        self.samples = {"train": train_samples, "valid": valid_samples, "test": test_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])
        
    def get_template(self, template_version=0):
        return {0: SST2Template}[template_version]()
        
    
class CopaDataset(Dataset):
    train_sep = "\n\n"
    mixed_set = False

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        d = load_dataset('super_glue', "copa")
        train_examples = d["train"]
        valid_examples = d["validation"]
    
        train_samples = [self.build_sample(example) for example in train_examples]
        valid_samples = [self.build_sample(example) for example in valid_examples]
        test_samples = [self.build_sample(ex) for ex in valid_examples]

        self.samples = {"train": train_samples, "valid": valid_samples, "test": test_samples} 
    
    # for generative tasks, candidates are []
    def build_sample(self, example):
        sample = \
            Sample(
                id=example["idx"],
                data=example,
                candidates=[example["choice1"], example["choice2"]],
                correct_candidate=example[f"choice{example['label'] + 1}"],
            )
        
        return sample
        
    def get_template(self, template_version=0):
        return {0: CopaTemplate}[template_version]()


class BoolQDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("boolq")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=["Yes", "No"],
                correct_candidate="Yes" if example["answer"] else "No",
            )
        
        return sample
    
    def get_template(self, template_version=2):
        return {0: BoolQTemplate, 1: BoolQTemplateV2, 2: BoolQTemplateV3}[template_version]()


class MultiRCDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "multirc")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: MultiRCTemplate}[template_version]()


class CBDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "cb")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1, 2],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: CBTemplate}[template_version]()


class WICDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "wic")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: WICTemplate}[template_version]()


class WSCDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "wsc.fixed")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: WSCTemplate}[template_version]()


class ReCoRDDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "record")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=example['entities'],
                correct_candidate=example['answers']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: ReCoRDTemplateGPT3}[template_version]()


class RTEDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "rte")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: RTETemplate}[template_version]()

 
class SQuADDataset(Dataset):
    metric_name = "f1"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()
        
    def load_dataset(self):
        dataset = load_dataset("squad")
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples, "test": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example['answers']['text']
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "title": example['title'],
                "context": example['context'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )
        
    def get_template(self, template_version=0):
        return {0: SQuADv2Template}[template_version]()


class DROPDataset(Dataset):
    metric_name = "f1"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()
        
    def load_dataset(self):
        dataset = load_dataset("drop")
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example['answers_spans']['spans']
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "context": example['passage'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )
        
    def get_template(self, template_version=0):
        return {0: DROPTemplate}[template_version]()
