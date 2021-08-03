import json
import os
import shutil
import time
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

from nncf import NNCFConfig
from nncf.torch import create_compressed_model
from nncf.torch import register_default_init_args


def download_tinyImg200(path,
                        url='http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                        tarname='tiny-imagenet-200.zip'):
    if not os.path.exists(path):
        os.mkdir(path)
    urlretrieve(url, os.path.join(path, tarname))
    print(os.path.join(path, tarname))
    import zipfile
    zip_ref = zipfile.ZipFile(os.path.join(path, tarname), 'r')
    zip_ref.extractall()
    zip_ref.close()


if not os.path.exists('tiny-imagenet-200'):
    download_tinyImg200('.')

PATH = '/home/nlyalyus/Developer/sandbox/tmp/int8_tutorial'


def main(params):
    arch = params['arch']
    num_classes = params['num_classes']
    init_lr = params['init_lr']
    # momentum = params['momentum']
    # weight_decay = params['weight_decay']
    batch_size = params['batch_size']
    workers = params['workers']
    pretrained = params['pretrained']
    resume = params['resume']
    checkpoint_compressed = params['checkpoint_compressed']
    data = params['data']
    evaluate = params['evaluate']
    start_epoch = params['start_epoch']
    epochs = params['epochs']

    use_nncf = params['use_nncf']
    nncf_config_file = params['nncf_config_file']

    best_acc1 = 0

    # create model
    if pretrained:
        print("=> using pre-trained model '{}'".format(arch))
        model = models.__dict__[arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(arch))
        model = models.__dict__[arch]()
    # update the last FC layer for tiny-imagenet number of classes
    model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        print('using GPU')
        model.cuda()
        model.fc = model.fc.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # TODO: add the parameter in config files for optimizer type
    # optimizer = torch.optim.SGD(model.parameters(), init_lr,
    #                             momentum=momentum,
    #                             weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)

            start_epoch = checkpoint['epoch']
            print('resumed start_epoch', start_epoch)
            best_acc1 = checkpoint['best_acc1']

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}, best_acc1 {:6.2f})"
                  .format(resume, checkpoint['epoch'], checkpoint['best_acc1']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    # Data loading code
    traindir = os.path.join(data, 'train')
    valdir = os.path.join(data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(224),
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

    if use_nncf:
        # Load a configuration file to specify compression
        nncf_config = NNCFConfig.from_json(nncf_config_file)
        # Provide data loaders for compression algorithm initialization, if necessary
        nncf_config = register_default_init_args(nncf_config, train_loader, criterion)
        # Apply the specified compression algorithms to the model
        print('=> compressing the model with {}'.format(nncf_config_file))
        compression_ctrl, model = create_compressed_model(model, nncf_config)

    if evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch, init_lr)

        if use_nncf:
            # update compression scheduler state at the begin of the epoch
            compression_ctrl.scheduler.epoch_step()
            # train for one epoch with nncf
            train(train_loader, model, criterion, optimizer, epoch, use_nncf, compression_ctrl=compression_ctrl)
        else:
            # train for one epoch without nncf
            train(train_loader, model, criterion, optimizer, epoch, use_nncf, compression_ctrl=None)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not use_nncf:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, use_nncf, compression_ctrl):
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

        if use_nncf:
            compression_ctrl.scheduler.step()

        if torch.cuda.is_available():
            images = images.cuda()
            target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        if use_nncf:
            compression_loss = compression_ctrl.loss()
            loss += compression_loss

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

        print_frequency = 10
        if i % print_frequency == 0:
            progress.display(i)


def validate(val_loader, model, criterion):
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

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(PATH, filename))
    if is_best:
        shutil.copyfile(os.path.join(PATH, filename), os.path.join(PATH, 'model_best.pth.tar'))


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
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
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


def create_json_files(batch_size, input_size):
    """
    Define configurations for compression algorithms
    Create the json files
    Return the configurations as dictinary objects
    """

    config_dir = 'config_files'
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    def write_json(json_obj, json_name):
        with open(os.path.join(config_dir, json_name), 'w') as jsonFile:
            json.dump(json_obj, jsonFile)

    # Define config objects below
    configs = {}

    # Quantization int8
    # https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md
    configs['quantization.json'] = {

        "input_info": {
            "sample_size": [batch_size, 3, input_size, input_size]
        },

        "epochs": 1,  # number of epochs to tune

        "optimizer": {
            "base_lr": 1e-5  # learning rate for the optimizer during tuning
        },

        "compression": {
            "algorithm": "quantization",  # specify the algorithm here
        }
    }

    # Filter pruning
    # https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Pruning.md
    configs['pruning.json'] = {

        "input_info": {
            "sample_size": [batch_size, 3, input_size, input_size]
        },

        "epochs": 1,  # number of epochs to tune

        "optimizer": {
            "base_lr": 1e-3  # learning rate for the optimizer during tuning
        },

        "compression": {
            "algorithm": "filter_pruning",
        }
    }

    # Sparsity
    # https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Sparsity.md
    configs['sparsity.json'] = {

        "input_info": {
            "sample_size": [batch_size, 3, input_size, input_size]
        },

        "epochs": 50,  # number of epochs to tune

        "optimizer": {
            "base_lr": 1e-3  # learning rate for the optimizer during tuning
        },

        "compression": {
            "algorithm": "magnitude_sparsity",
            "sparsity_init": 0.1,
            # Initial value of the sparsity level applied to the model in 'create_compressed_model' function
            "params": {
                "schedule": "multistep",
                # The type of scheduling to use for adjusting the target sparsity level
                "multistep_steps": [
                    # A list of scheduler steps at which to transition to the next scheduled sparsity level (multistep scheduler only).
                    5,
                    10,
                    20,
                    30,
                    40
                ],
                "multistep_sparsity_levels": [
                    # Levels of sparsity to use at each step of the scheduler as specified in the 'multistep_steps' attribute. The first sparsity level will be applied immediately, so the length of this list should be larger than the length of the 'steps' by one."
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6
                ],
            }
        }
    }

    # create json files, that will be used by nncf later
    for config_key, config_val in configs.items():
        write_json(config_val, config_key)

    return configs


params = {}

params['arch'] = 'resnet18'
params['num_classes'] = 200  # 200 is for tiny-imagenet, default is 1000 for imagenet
params['data'] = 'tiny-imagenet-200'  # path to data

params['init_lr'] = 1e-4
params['batch_size'] = 256
params['workers'] = 4
params['start_epoch'] = 0  # updated automatically if training is resumed
params['epochs'] = 15  # 15 is for full precision training, will be updated (increased) in case of tuning with nncf
params['pretrained'] = True  # pretrained model on Imagenet

params['resume'] = PATH + 'model_best.pth.tar'  # path to latest checkpoint (or None)
params['checkpoint_compressed'] = False

params['evaluate'] = False

params['use_nncf'] = True

if params['use_nncf']:
    # create all config files
    configs = create_json_files(params['batch_size'], input_size=224)

    # choose config file
    algorithm_config = 'quantization.json'
    params['nncf_config_file'] = 'config_files/' + algorithm_config

    # update tune params to fit certain compression algorithm
    params['init_lr'] = configs[algorithm_config]['optimizer']['base_lr']
    params['epochs'] = params['epochs'] + configs[algorithm_config]['epochs']

# Run certain algorithm once
main(params)

# Iterate over algorithms
algorithm_configs = ['quantization.json', 'pruning.json', 'sparsity.json']
for algorithm_config in algorithm_configs:
    # Run the tuning procedure:
    # print(algorithm_config)
    # params['nncf_config_file'] = 'config_files/' + algorithm_config
    # update tune params
    # main(params)
    pass
