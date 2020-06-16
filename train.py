from data import *
from utils.augmentations import SSDAugmentation, SSDAugmentationLight
from layers.modules import MultiBoxLoss
from layers.box_utils import match
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import cv2 as cv

VOC_ROOT = ''
torch.set_printoptions(threshold=5000)
# torch.set_printoptions(profile="full")

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO', 'CUSTOM'],
                    type=str, help='VOC, COCO or CUSTOM')
parser.add_argument('--train_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--test_root', default="",
                    help='Test data root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--random_weights', default=False, type=bool,
                    help='Start with random weights')
args = parser.parse_args()

    
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'COCO':
        if args.train_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify train_root if specifying dataset')
            print("WARNING: Using default COCO train_root because " +
                  "--train_root was not specified.")
            args.train_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.train_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.train_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying train_root')
        cfg = voc
        dataset = VOCDetection(root=args.train_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    elif args.dataset == 'CUSTOM':

        # Init train dataset
        if args.train_root == VOC_ROOT:
            parser.error('Must specify train_root for custom datasets')
        cfg = custom
        dataset = CustomDetection(root=args.train_root,
                                transform=SSDAugmentationLight(cfg['min_dim'], 
                                                         MEANS))

        # Init test dataset
        if(args.test_root == ""):
            print("No test dataset specified")
        else:
            test_dataset = CustomDetection(root=args.test_root,
                                    transform=SSDAugmentationLight(cfg['min_dim'], 
                                                            MEANS))



    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    elif args.random_weights:
        pass
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            with torch.no_grad():
                targets = [Variable(ann.cuda()) for ann in targets]
        else:
            images = Variable(images)
            with torch.no_grad():
                targets = [Variable(ann) for ann in targets]

        # for img in images:
        #     img = img.numpy().transpose(1,2,0)
        #     img = img.astype(np.uint8)
        #     cv.imshow("image", img)
        #     cv.waitKey()
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()

        loss_l, loss_c = criterion(out, targets)
        # print(loss_l)
        # print(loss_c)
        loss = loss_l + loss_c
        print(loss)
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

        if args.visdom:
            update_vis_plot(iteration, loss_l.item(), loss_c.item(),
                            iter_plot, epoch_plot, 'append')

        # TODO: remove second line and uncomment first
        # if iteration != 0 and iteration % 500 == 0:
        if iteration % 500 == 0:

            if args.test_root != "":
                print("Evaulating weights against test dataset")
                measure_iou(net, test_dataset)

            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, 'ssd300_COCO_' +
                       repr(iteration) + '.pth'))



        # if iteration != 0 and iteration % 50 == 0:
        #     torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, 'ssd300_COCO_last.pth'))

    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')

def measure_iou(net, test_dataset):

    num_images = len(test_dataset)

    intersection_50 = 0
    intersection_25 = 0
    union_50 = 0
    union_25 = 0

    for i in range(num_images):
        im, gts, h, w = test_dataset.pull_item(i)
        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        out = net(x)
        # Apparently I need to call detect() to get output in a format I understand
        out = net.detect(out[0], net.softmax(out[1]), out[2].type(type(x.data))).data
        # dets = out[0].detach().numpy()
        # conf = out[1].detach().numpy()

        dets = out[0, 1, :, 1:5]
        conf = out[0, 1, :, 0]



        i50, u50 = get_intersection_union(gts, dets, conf, 0.50)
        i25, u25 = get_intersection_union(gts, dets, conf, 0.25)
        intersection_50 += i50
        union_50 += u50
        intersection_25 += i25
        union_25 += u25

        # if u25 != 0:
        #     print("individual iou: {}".format(i25/u25))
        # else:
        #     print("individual iou: {}".format(0))

        # img = im.numpy().transpose(1,2,0)
        # # print(img.shape)
        # # print(img)
        # img = np.ascontiguousarray(img, dtype=np.uint8)
        # h, w, ch = img.shape
        # r, g, b = cv.split(img)
        # img = cv.merge((b, g, r))

        # for i in range(len(dets[:, 0])):
        #     det = dets[i]
        #     c = conf[i]

        #     if(c > 0.25):
        #         cv2.rectangle(img, (int(det[0]*w), int(det[1]*h)), (int(det[2]*w), int(det[3]*h)), (0,0,255), 2)

        # for gt in gts:
        #     cv2.rectangle(img, (int(gt[0]*w), int(gt[1]*h)), (int(gt[2]*w), int(gt[3]*h)), (0,255,255), 2)

        # cv.imshow("img", img)
        # cv.waitKey()

    if union_50 == 0:
        iou_50 = 0
    else:
        iou_50 = intersection_50/union_50

    if union_25 == 0:
        iou_25 = 0
    else:
        iou_25 = intersection_25/union_25

    print("IOU @ 50\% conf: {}".format(iou_50) )
    print("IOU @ 25\% conf: {}".format(iou_25) )

def get_intersection_union(gts, dets, conf, conf_thresh):
    # print("intersection")
    intersection = 0
    for i in range(len(dets[:,0])):
        detbb = dets[i]

        # print(conf)

        if conf[i] > conf_thresh:
            for gtbb in gts:
                # print(gtbb)
                # print(detbb)

                ixmin = np.maximum(gtbb[0], detbb[0])
                iymin = np.maximum(gtbb[1], detbb[1])
                ixmax = np.minimum(gtbb[2], detbb[2])
                iymax = np.minimum(gtbb[3], detbb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)

                intersection += iw * ih

    # print("individual inter: {}".format(intersection))
    

    union = 0
    for i in range(len(dets[:,0])):
        detbb = dets[i]
        if conf[i] > conf_thresh:
                union += ((detbb[2] - detbb[0]) * (detbb[3] - detbb[1]))

    for gtbb in gts:
        union += ((gtbb[2] - gtbb[0]) * (gtbb[3] - gtbb[1]))

    union -= intersection

    # print("individual union: {}".format(union))

    return intersection, union

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
