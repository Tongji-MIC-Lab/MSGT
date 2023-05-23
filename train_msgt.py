"""
Training script. Should be pretty adaptable to whatever.
"""
import argparse
import os
import shutil

import multiprocessing
import numpy as np
import pandas as pd
import torch
from allennlp.common.params import Params
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
from torch.nn import DataParallel
from torch.nn.modules import BatchNorm2d
from tqdm import tqdm

from dataloaders.vcr_plus import VCR, VCRLoader
#from dataloaders.vcr_attribute_new_tag import VCR, VCRLoader
from utils.pytorch_misc import time_batch, save_checkpoint, clip_grad_norm, \
    restore_checkpoint, print_para, restore_best_checkpoint

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

# This is needed to make the imports work
from allennlp.models import Model
import models

#os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,8,9'
#################################
#################################
######## Data loading stuff
#################################
#################################


parser = argparse.ArgumentParser(description='train')
parser.add_argument(
    '-params',
    dest='params',
    help='Params location',
    type=str,
)
parser.add_argument(
    '-rationale',
    action="store_true",
    help='use rationale',
)
parser.add_argument(
    '-folder',
    dest='folder',
    help='folder location',
    type=str,
)
parser.add_argument(
    '-restore',
    dest='restore',
    action="store_true",
    help="reload checkpoint"
)
parser.add_argument(
    '-train',
    dest='train',
    action="store_true",
    help="train checkpoint"
)
parser.add_argument(
    '-test',
    dest='test',
    action="store_true",
    help="test checkpoint"
)
parser.add_argument(
    '-no_tqdm',
    dest='no_tqdm',
    action='store_true',
)

args = parser.parse_args()

params = Params.from_file(args.params)
train, val, test = VCR.splits(mode='rationale' if args.rationale else 'answer',
                              embs_to_load=params['dataset_reader'].get('embs', 'bert_da'),
                              only_use_relevant_dets=params['dataset_reader'].get('only_use_relevant_dets', False))
NUM_GPUS = torch.cuda.device_count()
NUM_CPUS = multiprocessing.cpu_count()
if NUM_GPUS == 0:
    raise ValueError("you need gpus!")

def _to_gpu(td):
    if NUM_GPUS > 1:
        return td
    for k in td:
        if k != 'metadata':
            td[k] = {k2: v.cuda(non_blocking=True) for k2, v in td[k].items()} if isinstance(td[k], dict) else td[k].cuda(
                non_blocking=True)
    return td
num_workers = 64#(4 * NUM_GPUS if NUM_CPUS == 32 else 2*NUM_GPUS)-1
print(f"Using {num_workers} workers out of {NUM_CPUS} possible", flush=True)
loader_params = {'batch_size': 96 // NUM_GPUS, 'num_gpus':NUM_GPUS, 'num_workers':num_workers}
train_loader = VCRLoader.from_dataset(train, **loader_params)
val_loader = VCRLoader.from_dataset(val, **loader_params)
test_loader = VCRLoader.from_dataset(test, **loader_params)

ARGS_RESET_EVERY = 100
print("Loading {} for {}".format(params['model'].get('type', 'WTF?'), 'rationales' if args.rationale else 'answer'), flush=True)
model = Model.from_params(vocab=train.vocab, params=params['model'])
# for submodule in model.detector.backbone.modules():
#     if isinstance(submodule, BatchNorm2d):
#         submodule.track_running_stats = False

model = DataParallel(model).cuda() if NUM_GPUS > 1 else model.cuda()
optimizer = Optimizer.from_params([x for x in model.named_parameters() if (x[1].requires_grad and ("ffn" not in x[0] and "self_attention" not in x[0]))],
                                  params['trainer']['optimizer'])
                                  
optimizer2 = Optimizer.from_params([x for x in model.named_parameters() if (x[1].requires_grad and ("ffn" in x[0] or "self_attention" in x[0]))],
                                  params['trainer']['optimizer2'])

lr_scheduler_params = params['trainer'].pop("learning_rate_scheduler", None)
scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params) if lr_scheduler_params else None

lr_scheduler_params2 = params['trainer'].pop("learning_rate_scheduler2", None)
scheduler2 = LearningRateScheduler.from_params(optimizer2, lr_scheduler_params2) if lr_scheduler_params2 else None


if os.path.exists(args.folder) and args.restore:
    print('restore is True')
    print("Found folder! restoring", flush=True)
    start_epoch, val_metric_per_epoch = restore_checkpoint(model, optimizer, serialization_dir=args.folder,
                                                          learning_rate_scheduler=scheduler)
else:
    print("Making directories")
    os.makedirs(args.folder, exist_ok=True)
    start_epoch, val_metric_per_epoch = 0, []
    shutil.copy2(args.params, args.folder)

param_shapes = print_para(model)
num_batches = 0

if args.train:
    print('It is training!!!!!')
    for epoch_num in range(start_epoch, params['trainer']['num_epochs'] + start_epoch):
        train_results = []
        norms = []
        model.train()
        for b, (time_per_batch, batch) in enumerate(time_batch(train_loader if args.no_tqdm else tqdm(train_loader), reset_every=ARGS_RESET_EVERY)):
            
            batch = _to_gpu(batch)
            optimizer.zero_grad()
            output_dict = model(**batch)
            loss = output_dict['loss'].mean()
            loss.backward()

            num_batches += 1
            if scheduler:
                scheduler.step_batch(num_batches)
                scheduler2.step_batch(num_batches)

            norms.append(
                clip_grad_norm(model.named_parameters(), max_norm=params['trainer']['grad_norm'], clip=True, verbose=False)
            )
            optimizer.step()
            optimizer2.step()

            train_results.append(pd.Series({'epoch': epoch_num,
                                            'loss': output_dict['loss'].mean().item(),
                                            'cls_loss': output_dict['cls_loss'].mean().item(),
                                            'b_cls_loss': output_dict['b_cls_loss'].mean().item(),
                                            'accuracy': (model.module if NUM_GPUS > 1 else model).get_metrics(
                                                reset=(b % ARGS_RESET_EVERY) == 0)[
                                                'accuracy'],
                                            'lr':optimizer.state_dict()['param_groups'][0]['lr'],
                                            'lr2':optimizer2.state_dict()['param_groups'][0]['lr']
                                            }))
            if b % ARGS_RESET_EVERY == 0 and b > 0:
                norms_df = pd.DataFrame(pd.DataFrame(norms[-ARGS_RESET_EVERY:]).mean(), columns=['norm']).join(
                    param_shapes[['shape', 'size']]).sort_values('norm', ascending=False)

                print("e{:2d}b{:5d}/{:5d}. norms: \nsumm:\n{}\n~~~~~~~~~~~~~~~~~~\n".format(
                    epoch_num, b, len(train_loader),
                    pd.DataFrame(train_results[-ARGS_RESET_EVERY:]).mean(),
                ), flush=True)

        print("---\nTRAIN EPOCH {:2d}:\n{}\n----".format(epoch_num, pd.DataFrame(train_results).mean()))
        val_probs = []
        val_labels = []
        val_loss_sum = 0.0
        model.eval()
        for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
            with torch.no_grad():
                batch = _to_gpu(batch)
                output_dict = model(**batch)
                val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
                val_labels.append(batch['label'].detach().cpu().numpy())
                val_loss_sum += output_dict['loss'].mean().item() * batch['label'].shape[0]
        val_labels = np.concatenate(val_labels, 0)
        val_probs = np.concatenate(val_probs, 0)
        val_loss_avg = val_loss_sum / val_labels.shape[0]

        val_metric_per_epoch.append(round(float(np.mean(val_labels == val_probs.argmax(1))),6))
        if scheduler:
            scheduler.step(val_metric_per_epoch[-1], epoch_num)
            scheduler2.step(val_metric_per_epoch[-1], epoch_num)

        print("Val epoch {} has acc {:.3f} and loss {:.3f}".format(epoch_num, val_metric_per_epoch[-1], val_loss_avg),
              flush=True)
        print(val_metric_per_epoch)
        if int(np.argmax(val_metric_per_epoch)) < (len(val_metric_per_epoch) - 1 - params['trainer']['patience']):
            print("Stopping at epoch {:2d}".format(epoch_num))
            break
        save_checkpoint(model, optimizer, args.folder, epoch_num, val_metric_per_epoch,
                        is_best=int(np.argmax(val_metric_per_epoch)) == (len(val_metric_per_epoch) - 1))

if args.test:
    print('It is validation!!!!')
    print("STOPPING. now running the best model on the validation set", flush=True)
    # Load best
    restore_best_checkpoint(model, args.folder)
    model.eval()
    val_probs = []
    val_labels = []
    for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
    #for b, (time_per_batch, batch) in enumerate(time_batch(test_loader)):
        with torch.no_grad():
            batch = _to_gpu(batch)
            output_dict = model(**batch)
            val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
            val_labels.append(batch['label'].detach().cpu().numpy())
    val_labels = np.concatenate(val_labels, 0)
    val_probs = np.concatenate(val_probs, 0)
    acc = float(np.mean(val_labels == val_probs.argmax(1)))
    print("Final val accuracy is {:.3f}".format(acc))
    np.save(os.path.join(args.folder, f'valpreds.npy'), val_probs)

