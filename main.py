import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import tensorboard_logger as tb_logger
from utils.cifar100 import load_cifar100
from utils.utils_algo import *

from utils.ComCon import ComSifted, ComSoft, ComWeight
from utils.cifar10 import load_cifar10
from utils.resnet import SupConResNet

'''
train: python main_weight_copy.py --multiprocessing-distributed
python main.py --multiprocessing-distributed
python main_cifar10.py --multiprocessing-distributed
python main_cifar10_copy.py --multiprocessing-distributed
python train.py --multiprocessing-distributed
python train_copy.py --multiprocessing-distributed
'''

parser = argparse.ArgumentParser(description='PyTorch implementation of MulCo')
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'cifar100'],
                    help='dataset name (cifar10)')
parser.add_argument('--num-class', default=10, type=int,
                    help='number of class')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--num', default=5, type=int, help='number of complementary labels')
parser.add_argument('--lam', default=1, type=float, help='dynamic weight for ConLoss')
parser.add_argument('--exp-dir', default='experiment', type=str,
                    help='experiment directory for saving checkpoints and logs')
parser.add_argument('--seed', default=123, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--method', default='ComWeight', type=str, choices=['ComSifted', 'ComWeight', "ComSoft"],
                    help='the method for representation')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=['resnet18'],
                    help='network architecture')
parser.add_argument('-j', '--workers', default=32, type=int,
                    help='number of data loading workers (default: 32)')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('--dim', default=128, type=int,
                    help='embedding dimension')
parser.add_argument('--moco_queue', default=8192, type=int,
                    help='queue size; number of negative samples')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='momentum for updating momentum encoder')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10002', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--t', default=0.05, type=float,
                    help='softmax temperature (default: 0.05)')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    help='print frequency (default: 100)')

parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--schedule', default=[100, 150], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')



def main():
    print(args)
    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    model_path = '{method}_{ds}_lr{lr}_wd{wd}_sd{seed}_lam{lam}_c{num}_queue{queue}_t{t}_{b}'.format(
        method=args.method,
        ds=args.dataset,
        lr=args.lr,
        wd=args.weight_decay,
        seed=args.seed,
        lam=args.lam,
        num=args.num,
        queue=args.moco_queue,
        t=args.t,
        b=args.batch_size
    )
    args.exp_dir = os.path.join(args.exp_dir, model_path)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    cudnn.benchmark = True
    args.gpu = gpu
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.method == "ComSifted":
        model = ComSifted(args, SupConResNet)
        criterion = nn.NLLLoss().cuda(args.gpu)
    elif args.method == "ComWeight":
        model = ComWeight(args, SupConResNet)
        criterion = nn.NLLLoss().cuda(args.gpu)
    else:
        model = ComSoft(args, SupConResNet)
        criterion = nn.NLLLoss().cuda(args.gpu)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # set optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.dataset == 'cifar100':
        train_loader, train_sampler, test_loader = load_cifar100(batch_size=args.batch_size, num=args.num)
    elif args.dataset == 'cifar10':
        args.num_class = 10
        train_loader, train_sampler, test_loader = load_cifar10(batch_size=args.batch_size, num=args.num)
    else:
        raise NotImplementedError("You have chosen an unsupported dataset. Please check and try again.")
    # this train loader is the complementary label training loader

    if args.gpu == 0:
        logger = tb_logger.Logger(logdir=os.path.join(args.exp_dir, 'tensorboard'), flush_secs=2)
    else:
        logger = None

    print('\nStart Training\n')

    best_acc = 0
    mmc = 0  # mean max confidence
    save_table = np.zeros(shape=(args.epochs, 4))

    for epoch in range(args.start_epoch, args.epochs):
        # scheduler_warmup.step(epoch)
        is_best = False
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)
        acc_train, cls_loss, con_loss, time = train(train_loader, model, criterion, optimizer, epoch, args, logger)

        acc_test, top_5 = test(model, test_loader, args, epoch, logger)
        print('Epoch: {}. Train Acc: {}. Te Acc: {}. Time: {}.'.format(epoch + 1, acc_train, acc_test, time))
        save_table[epoch, :] = epoch + 1, acc_train, acc_test, top_5

        if not os.path.exists("./rebuttal/{method}/".format(method=args.method)):
            os.makedirs("./rebuttal/{method}/".format(method=args.method))

        np.savetxt("./rebuttal/{method}/{ds}_lr{lr}_wd{wd}_sd{seed}_lam{lam}_c{num}_queue{queue}_t{t}_{b}.csv".format(
            method=args.method, ds=args.dataset, lr=args.lr, wd=args.weight_decay, seed=args.seed, lam=args.lam,
            num=args.num, queue=args.moco_queue, t=args.t, b=args.batch_size), save_table, delimiter=',', fmt='%2.2f')

        if acc_test > best_acc:
            best_acc = acc_test
            is_best = True
            torch.save(model.state_dict(), "{}/best_model.tar".format(args.exp_dir))


def train(train_loader, model, criterion, optimizer, epoch, args, tb_logger):
    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    loss_cls_log = AverageMeter('Loss@Cls', ':2.2f')
    loss_cont_log = AverageMeter('Loss@Cont', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, acc_cls, loss_cls_log, loss_cont_log],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (img_q, img_k, com_label, true_label) in enumerate(train_loader):

        data_time.update(time.time() - end)
        img_q, img_k, com_label = img_q.cuda(), img_k.cuda(), com_label.cuda()
        Y_true = true_label.long().detach().cuda()
        # for showing training accuracy and will not be used when training

        cls_out, con_out, con_target, out_k = model(img_q, img_k, com_label, args)

        # UB-LOG
        comp_loss = log_loss(cls_out, (1 - com_label).clone())
        con_loss = criterion(con_out, con_target)  # contrastive loss

        lam = min((epoch / 100) * args.lam, args.lam)
        loss = comp_loss + lam*con_loss

        loss_cls_log.update(comp_loss.item())
        loss_cont_log.update(con_loss.item())

        # log accuracy
        acc = accuracy(cls_out, Y_true)[0]
        acc_cls.update(acc[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)

    if args.gpu == 0:
        tb_logger.log_value('Train Acc', acc_cls.avg, epoch)
        tb_logger.log_value('Classification Loss', loss_cls_log.avg, epoch)
        tb_logger.log_value('Contrastive Loss', loss_cont_log.avg, epoch)
    return acc_cls.avg/args.world_size, loss_cls_log.avg/args.world_size, loss_cont_log.avg/args.world_size, data_time.sum


def test(model, test_loader, args, epoch, tb_logger):
    with torch.no_grad():
        print('==> Evaluation...')
        model.eval()
        top1_acc = AverageMeter("Top1")
        top5_acc = AverageMeter("Top5")
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images, args, eval_only=True)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1_acc.update(acc1[0])
            top5_acc.update(acc5[0])

        # average across all processes
        acc_tensors = torch.Tensor([top1_acc.avg, top5_acc.avg]).cuda(args.gpu)
        dist.all_reduce(acc_tensors)
        acc_tensors /= args.world_size

        print('Accuracy is %.2f%% (%.2f%%)' % (acc_tensors[0], acc_tensors[1]))
        if args.gpu == 0:
            tb_logger.log_value('Top1 Acc', acc_tensors[0], epoch)
            tb_logger.log_value('Top5 Acc', acc_tensors[1], epoch)
    return acc_tensors[0], acc_tensors[1]


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def log_loss(outputs, partialY):
    k = partialY.shape[1]
    can_num = partialY.sum(dim=1).float()

    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * partialY

    average_loss = - ((k-1)/(k-can_num) * torch.log(final_outputs.sum(dim=1))).mean()
    return average_loss

if __name__ == '__main__':
    args = parser.parse_args()

    print("lam:{}, num:{}, method:{}, data:{}, lr:{}, wd:{}, dir:{}, t:{}, seed:{}".format(
        args.lam, args.num, args.method, args.dataset, args.lr, args.weight_decay, args.exp_dir,
        args.t, args.seed))
    main()

