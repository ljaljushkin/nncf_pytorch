import os
import shutil
import time
from datetime import datetime
from urllib.request import urlretrieve

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from nncf import NNCFConfig
from nncf.torch import create_compressed_model
from nncf.torch import register_default_init_args


def download_tinyImg200(path,
                        url='http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                        tarname='tiny-imagenet-200.zip'):
    if not os.path.exists(path):
        os.mkdir(path)
    archive_path = os.path.join(path, tarname)
    urlretrieve(url, archive_path)
    print(archive_path)
    import zipfile
    zip_ref = zipfile.ZipFile(archive_path, 'r')
    zip_ref.extractall()
    zip_ref.close()


DATASET_DIR = 'tiny-imagenet-200'
DOWNLOAD_DIR = '/media/ssd'
WORKING_DIR = '/home/nlyalyus/Developer/sandbox/tmp/int8_tutorial'

if not os.path.exists(DATASET_DIR):
    download_tinyImg200(DOWNLOAD_DIR)

d = datetime.now()
run_id = '{:%Y-%m-%d__%H-%M-%S}'.format(d)
LOG_DIR = os.path.join(WORKING_DIR, run_id)


def main():
    num_classes = 200  # 200 is for tiny-imagenet, default is 1000 for imagenet
    init_lr = 1e-4
    batch_size = 256
    workers = 4
    image_size = 64
    data = DATASET_DIR
    start_epoch = 0
    epochs = 4

    tb = SummaryWriter(LOG_DIR)

    # create model
    model = models.resnet18(pretrained=True)
    # update the last FC layer for tiny-imagenet number of classes
    model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    model.cuda()
    model.fc = model.fc.cuda()

    # Data loading code
    train_dir = os.path.join(data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [80000, 20000])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    best_acc1 = 0
    acc1 = 0
    for epoch in range(start_epoch, epochs):
        # adjust_learning_rate(optimizer, epoch, init_lr)

        # train for one epoch with nncf
        train(train_loader, model, criterion, optimizer, epoch, tb)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, tb, epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc1': acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    print(f'Accuracy of FP32 model: {acc1:.3f}')

    nncf_config_dict = {
        "input_info": {
            "sample_size": [1, 3, image_size, image_size]
        },
        "compression": {
            "algorithm": "quantization",  # specify the algorithm here
        }
    }
    # Load a configuration file to specify compression
    nncf_config = NNCFConfig.from_dict(nncf_config_dict)
    # Provide data loaders for compression algorithm initialization, if necessary
    nncf_config = register_default_init_args(nncf_config, train_loader)

    compression_ctrl, model = create_compressed_model(model, nncf_config)

    # evaluate on validation set after initialization of quantization (POT case)
    acc1 = validate(val_loader, model, criterion, tb, epoch=4)
    print(f'Accuracy of initialized INT8 model: {acc1:.3f}')

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = init_lr * 0.1

    # train for one epoch with NNCF
    train(train_loader, model, criterion, optimizer, epoch=4, tb=tb)

    # evaluate on validation set after Quantization-Aware Training (QAT case)
    acc1 = validate(val_loader, model, criterion, tb, epoch=5)
    print(f'Accuracy of tuned INT8 model: {acc1:.3f}')

    compression_ctrl.export_model("resnet_int8.onnx")


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(LOG_DIR, filename))
    if is_best:
        shutil.copyfile(os.path.join(LOG_DIR, filename), os.path.join(LOG_DIR, 'model_best.pth.tar'))


def train(train_loader, model, criterion, optimizer, epoch, tb):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda()
            target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do opt step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print_frequency = 50
        if i % print_frequency == 0:
            progress.display(i)

        global_step = len(train_loader) * epoch
        tb.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], i + global_step)
        tb.add_scalar("train/loss", losses.avg, i + global_step)
        tb.add_scalar("train/top1", top1.avg, i + global_step)
        tb.add_scalar("train/top5", top5.avg, i + global_step)


def validate(val_loader, model, criterion, tb, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print_frequency = 10
            if i % print_frequency == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        tb.add_scalar("val/loss", losses.avg, len(val_loader) * epoch)
        tb.add_scalar("val/top1", top1.avg, len(val_loader) * epoch)
        tb.add_scalar("val/top5", top5.avg, len(val_loader) * epoch)

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 4 epochs"""
    lr = init_lr * (0.1 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


main()
