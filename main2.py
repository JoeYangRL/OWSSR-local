
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
import logging.config

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch import cuda
from utils2 import set_models, set_seed, ModelEMA
import copy
from trainer2 import train
from parse import set_parser

logger = logging.getLogger(__name__)


"""
# used args paramters
args.world_size: how many gpus you want to use [check]
args.port [check]
args.seed: set random seed [check]
args.eval_only: if only evaluate [check]
args.now_step: current task step (start counting from 1) [check]
args.output_path: path to checkpoint [check]
"""


def main(rank, args):

    ##############################################################################
    # setup
    if args.eval_only:
        log_name = 'step%i_test.log'%(args.now_step)
    else:
        log_name = 'step%i_train.log'%(args.now_step)
    logger.info(f"{log_name}")
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
    log_dict = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'form01': {
                'format': "%(asctime)s - %(levelname)s - %(name)s -   %(message)s", 
                'datefmt': "%m/%d/%Y %H:%M:%S",
            }
        },
        'handlers': {
            'ch': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'form01',
                'stream': 'ext://sys.stderr'
            },
            'fh': {
                'class': 'logging.FileHandler',
                'level': 'INFO',
                'formatter': 'form01',
                'filename': os.path.join(args.output_path, log_name),
                'mode': 'w'
            }
        },
        'root': {
            'handlers': ['ch', 'fh'],
            'level': 'INFO',
        }
    }
    logging.config.dictConfig(log_dict)
    args.rank = rank
    
    if rank == 0:
        logger.info("This is cosine method")
        logger.info("---------- start setup -------------------")


    global best_acc
    global best_acc_val

    
    
    device = torch.device("cuda:{}".format(rank) if torch.cuda.is_available() else "cpu" )
    torch.cuda.set_device(device)
    if args.world_size > 1:
        dist.init_process_group(backend='nccl', world_size=args.world_size, rank=rank)
    
    args.device = device
    if rank == 0:
        logger.info(dict(args._get_kwargs()))
    if args.seed is not None:
        set_seed(args)
    """
    #disable summarywriter
    if rank == 0:
        os.makedirs(args.output_path, exist_ok=True)
        args.writer = SummaryWriter(args.output_path)"""

    ##############################################################################
    # prepare model
    if rank == 0:
        logger.info("---------- start loading model -------------------")

    model, optimizer, scheduler = set_models(args)
    logger.info(f"{model}")
    logger.info("Total params: {:.6f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    if args.use_ema:
        ema_model = ModelEMA(args, model, args.ema_decay)
        logger.info("activate ema model")
    else:
        ema_model = None
        logger.info("deactivate ema model")
    
    if args.now_step > 1:
        logger.info("==> Resuming from checkpoint..")
        checkpoint = torch.load(os.path.join(args.output_path, 'model_best_step%i.pth.tar'%(args.now_step-1)))
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
            old_model = copy.deepcopy(ema_model.ema)
        else:
            old_model = copy.deepcopy(model)
        for para in old_model.parameters():
            para.requires_grad = False
        old_model.eval()
        if args.base_task_cls != 0:
            num_cls = int(args.base_task_cls + (args.now_step-1) * ((100-args.base_task_cls) / (args.steps)))
            args.cls_per_step = int((100-args.base_task_cls) / (args.steps))
        else:
            num_cls = int((100/args.steps) * args.now_step)
            args.cls_per_step = int(100/args.steps)
        # change output dim, reset optimizer & scheduler
        model.change_output_dim(args, num_cls)
        model.to(args.device)
        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if args.opt == 'sgd':
            optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                              momentum=0.9, nesterov=args.nesterov)
        elif args.opt == 'adam':
            optimizer = optim.Adam(grouped_parameters, lr=2e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch-args.warmup_epoch, args.lr)
        if args.use_ema:
            ema_model = ModelEMA(args, model, args.ema_decay)
    else:
        old_model = None
    
    #用DDP包装原来的model
    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank],
            output_device=rank, find_unused_parameters=True)

    model.zero_grad()
    if not args.eval_only:
        if rank == 0:
            logger.info("---------- enter training process -------------------")
            logger.info(f"  Task = cifar100_B{args.base_task_cls}_{args.steps}steps_{args.num_labeled}num-labeled")
            logger.info(f"  Num Epochs = {args.epoch}")
            logger.info(f"  Batch size per GPU = {args.batch_size}")
            logger.info(f"  Total train batch size = {args.batch_size*args.world_size}")
        train(args, model, old_model, ema_model, optimizer, scheduler, rank)
    """
    #TODO: add eval_only code
    else:
        logger.info("***** Running Evaluation *****")
        logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
        eval_model(args, labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
             ood_loaders, model, ema_model)"""


if __name__ == '__main__':
    #main()
    args = set_parser()
    
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port
        mp.spawn(main, nprocs=args.world_size, args=(args,))
    else:
        main(0, args)