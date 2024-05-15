from copy import deepcopy
import numpy as np
import torch,random
from .image_pair_data import ImagePairDataset
# from .image_pair_data_reverse import  ImagePairReverseDataset
# from .image_pair_data_lq import ImagePairDatasetLQ,ImagePairDatasetSyn
# from .single_image_data import SingleImageDataset
from .image_test_data import ImgTestDataset
from .image_sequence_data import ImgCataDataset
from .fetcher import PrefetchDataLoader
from functools import partial


def build_dataset(dataset_opt):
    """Build dataset from options.

    Args:
        dataset_opt (dict): Configuration for dataset. It must contain:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_opt = deepcopy(dataset_opt)
    dataset_type = dataset_opt.pop('type')
    # print(dataset_type)
    if dataset_type in ['ImagePairDataset','ImgCataDataset','ImgTestDataset','SingleImageDataset','ImagePairDatasetSyn','BareImageDataset']:
      if dataset_type=='ImagePairDataset':
        dataset = ImagePairDataset(dataset_opt)
      elif dataset_type=='ImgCataDataset':
        dataset = ImgCataDataset(dataset_opt)
      elif dataset_type=='ImgTestDataset':
        dataset = ImgTestDataset(dataset_opt)
      # elif dataset_type=='SingleImageDataset':
      #   dataset = SingleImageDataset(dataset_opt)
      # elif dataset_type=='ImagePairDatasetSyn':
      #   dataset = ImagePairDatasetSyn(dataset_opt)
      # elif dataset_type=='BareImageDataset':
      #   dataset = BareImageDataset(dataset_opt)
    else:
      dataset = None
    
    return dataset


def build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    """Build dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None
    """
    phase = dataset_opt['phase']
    rank=0 # set to 0 for one gpu training
    if phase == 'train':
        if dist:  # distributed training
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True)
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
    elif phase in ['val', 'test']:  # validation
        dataloader_args = dict(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    else:
        raise ValueError(f'Wrong dataset phase: {phase}. ' "Supported ones are 'train', 'val' and 'test'.")

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)
    dataloader_args['persistent_workers'] = dataset_opt.get('persistent_workers', False)

    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode == 'cpu':  # CPUPrefetcher
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    else:
        # prefetch_mode=None: Normal dataloader
        # prefetch_mode='cuda': dataloader for CUDAPrefetcher
        return torch.utils.data.DataLoader(**dataloader_args)


def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)