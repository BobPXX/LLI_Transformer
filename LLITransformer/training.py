import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import random
import logging
import numpy as np
from datetime import timedelta
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from utils.scheduler import WarmupCosineSchedule
from utils.data_utils import get_loader
import torch.distributed as dist
from tensorboardX import SummaryWriter

from model import LLI_Transformer

writer=SummaryWriter('log')
logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(cfg, model):
    torch.save(model.module.state_dict(), cfg.dir.save_model_dir)
    #torch.save(model.state_dict(), cfg.basic.save_model_dir)

def load_model(cfg,model):
    loaded_dict = torch.load(cfg.dir.load_model_dir)
    model_dict = model.state_dict()
    loaded_dict = {k: v for k, v in loaded_dict.items() if k in model_dict}
    model_dict.update(loaded_dict)
    model.load_state_dict(model_dict)

def setup(cfg):
    model = LLI_Transformer(num_classes=cfg.train.num_classes)
    
    num_params = count_parameters(model)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    
    if cfg.train.load==True:
        load_model(cfg,model)
  
    return model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(cfg):
    torch.manual_seed(cfg.basic.seed)
    torch.cuda.manual_seed_all(cfg.basic.seed)
    np.random.seed(cfg.basic.seed)
    random.seed(cfg.basic.seed)
    torch.backends.cudnn.deterministic = True


def valid(model, val_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()
    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(val_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    loss_fct = torch.nn.CrossEntropyLoss()
    loss_fct.cuda()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.cuda() for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x.to(torch.float))

            eval_loss = loss_fct(logits, y.to(torch.long))
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    return accuracy


def train(cfg):
    """ Train the model """
    model=setup(cfg)
    model.cuda()
    model = torch.nn.DataParallel(model)

    # Prepare dataset
    train_loader, val_loader = get_loader(cfg)

    # Prepare optimizer and scheduler
    if cfg.optimizer.optimizer=='SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.optimizer.learning_rate,
                                momentum=0.9,
                                weight_decay=cfg.optimizer.weight_decay)
    if cfg.optimizer.optimizer=='AdamW':
        optimizer = torch.optim.AdamW(model.parameters(),
                                lr=cfg.optimizer.learning_rate,
                                weight_decay=cfg.optimizer.weight_decay)
    t_total = cfg.train.num_steps

    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=cfg.scheduler.warmup_steps, t_total=t_total)

    model.zero_grad()
    losses = AverageMeter()
    loss_fct = torch.nn.CrossEntropyLoss()
    loss_fct.cuda()
    global_step, best_acc = 0, 0

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", cfg.train.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", cfg.train.train_batch_size)
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.cuda() for t in batch)
            x, y = batch

            logits = model(x.to(torch.float))

            loss = loss_fct(logits, y.to(torch.long))
            loss.backward()
            losses.update(loss.item())

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if cfg.scheduler.use==True:
                scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
            )
            if global_step % cfg.train.eval_every == 0:
                accuracy = valid(model, val_loader, global_step)
                if best_acc < accuracy:
                    save_model(cfg, model)
                    best_acc = accuracy
                model.train()

            if global_step % t_total == 0:
                break
        losses.reset()

        if global_step % t_total == 0:
            break

    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    dist.init_process_group(backend='nccl')
    cfg = OmegaConf.load('imagenet_configs.yaml')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    set_seed(cfg)

    model = setup(cfg)
    model.cuda()
    dummy_input=torch.rand(1,3,224,224).cuda()
    with SummaryWriter(comment='LLITransformer') as w:
        w.add_graph(model,(dummy_input.to(torch.float),))


    train(cfg)


if __name__ == "__main__":
    main()
