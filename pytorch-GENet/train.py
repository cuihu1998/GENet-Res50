import argparse
import json
import os
import time
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.model_zoo as model_zoo
from tensorboardX import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from models import *
from utils import *
import torchvision.datasets as datasets
from models.genet import GENet
import torch.nn as nn
import tqdm

writer = SummaryWriter()

parser = argparse.ArgumentParser(description='Pruning')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--GPU', default='0', type=str, help='GPU to use')
parser.add_argument('--path', default='/data/tmp/imagenet_train', type=str, help='')
parser.add_argument('--save_file', default='saveto', type=str, help='save file for checkpoints')
parser.add_argument('--print_freq', '-p', default=1, type=int, metavar='N', help='print frequency (default: 10)')

# Learning specific arguments
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-bt', '--test_batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('-lr', '--learning_rate', default=.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay factor')
parser.add_argument('-epochs', '--no_epochs', default=5, type=int, metavar='epochs', help='no. epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--nesterov', default=False, type=bool, help='yesterov?')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float, metavar='W', help='weight decay')
parser.add_argument('--eval', '-e', action='store_true', help='resume from checkpoint')
parser.add_argument('--dataset', choices=['cifar10','cifar100'], default = 'imagenet')
parser.add_argument('--epoch_step', default='[60,120,160]', type=str, help='json list with epochs to drop lr on')

# Net specific
parser.add_argument('--mlp', default=False, type=bool, help='mlp?')
parser.add_argument('--extra_params', default=True, type=bool, help='extraparams?')
parser.add_argument('--extent', default=0, type=int, help='Extent for pooling')


args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.cuda.set_device(0)

if not os.path.exists('checkpoints/'):
    os.makedirs('checkpoints/')

model = GENet(num_classes=1000, mlp=args.mlp, extra_params=args.extra_params)

get_no_params(model)
model.to(device)

print('Standard Aug')

transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ]),
])

transform_train = transforms
transform_val = transforms


traindir = os.path.join(args.path)
train = datasets.ImageFolder(traindir, transform)

trainloader = torch.utils.data.DataLoader(
    train, batch_size=args.batch_size, shuffle=False, num_workers=4)


error_history = []
epoch_step = json.loads(args.epoch_step)

def train():
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(trainloader):

        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device)

        output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err2 = get_error(output.detach(), target, topk=(1, 2))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top2.update(err2.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top2.val:.3f} ({top2.avg:.3f})'.format(
                epoch, i, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top2=top2))

    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_top1', top1.avg, epoch)
    writer.add_scalar('train_top2', top2.avg, epoch)

if __name__ == '__main__':

    filename = 'checkpoints/%s.t7' % args.save_file
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=epoch_step, gamma=args.lr_decay_ratio)

    for epoch in range(args.no_epochs):
        scheduler.step()
        print('Epoch %d:' % epoch)
        print('Learning rate is %s' % [v['lr'] for v in optimizer.param_groups][0])
        # train for one epoch
        train()
        # # # evaluate on validation set
        print("training done.")