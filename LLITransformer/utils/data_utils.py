import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from utils.imagenet import get_data

def get_loader(cfg):
    train_dataset = get_data(
        train=True,
        psf_dir=cfg.dir.psf_dir,
        filename_list_dir=cfg.dir.train_filename_dir,
        label_list_dir=cfg.dir.train_labels_dir
    )
    
    val_dataset = get_data(
        train=False,
        psf_dir=cfg.dir.psf_dir,
        filename_list_dir=cfg.dir.val_filename_dir,
        label_list_dir=cfg.dir.val_labels_dir
    )

    train_sampler = DistributedSampler(train_dataset,shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.train_batch_size,
                              num_workers=4*cfg.train.GPU_num,
                              pin_memory=True,
                              drop_last=False,
                              sampler=train_sampler,
                              prefetch_factor=2)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4*cfg.train.GPU_num,
                            pin_memory=True,
                            drop_last=False)

    return train_loader, val_loader
