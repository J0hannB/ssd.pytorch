"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import *
from utils.augmentations import SSDAugmentation, SSDAugmentationLight
from data import custom as customConfig
# from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

VOC_ROOT = ''


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.voc_root, '%s.txt')
imgpath = os.path.join(args.voc_root, '%s.jpg')
imgsetpath = os.path.join('/home/jonathan/darknet/grassTestV4/cfg/test-setAside.txt')
YEAR = '2007'
devkit_path = args.voc_root + 'VOC' + YEAR
dataset_mean = (104, 117, 123)
set_type = 'test'


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    # tree = ET.parse(filename)
    objects = []
    print("reading from " + filename)
    target = []

    transform = YOLOAnnotationTransform()
    with open(filename) as file:
        for line in file:
            target.append(line)
            # print(line)

    target = transform(target, 1, 1)
    for row in target:
        print(row)
        x1 = int(row[0] * 1368)
        y1 = int(row[1] * 912)
        x2 = int(row[2] * 1368)
        y2 = int(row[3] * 912)
        label = int(row[4])
        obj_struct = {}
        obj_struct['name'] = "Class %d" % label
        obj_struct['bbox'] = [x1,y1,x2,y2]
        obj_struct['difficult'] = False
        # print(obj_struct)
        objects.append(obj_struct)
    # for obj in tree.findall('object'):
    #     obj_struct = {}
    #     obj_struct['name'] = obj.find('name').text
    #     obj_struct['pose'] = obj.find('pose').text
    #     obj_struct['truncated'] = int(obj.find('truncated').text)
    #     obj_struct['difficult'] = int(obj.find('difficult').text)
    #     bbox = obj.find('bndbox')
    #     obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
    #                           int(bbox.find('ymin').text) - 1,
    #                           int(bbox.find('xmax').text) - 1,
    #                           int(bbox.find('ymax').text) - 1]
    #     objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(['Class-0']):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, id_list in enumerate(dataset.ids):
                index = id_list[0]
                quadrant = id_list[1]
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:d} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, quadrant, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir='output', use_07=True):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(['Class-0']):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(
           filename, annopath, imgsetpath.format(set_type), cls, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            # recs[imagename] = parse_rec(annopath % (imagename))
            recs[imagename] = parse_rec(os.path.splitext(imagename)[0] + '.txt')
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    # print(recs)
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        # print(R)
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[os.path.splitext(os.path.split(imagename)[1])[0]] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

        # print(class_recs)
        # if len(bbox) > 0:
            # img = cv2.imread(imagename)
            # for box in bbox:
            #     print(box)
            #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            # cv2.imshow(imagename, img)
            # cv2.waitKey()

    # print(class_recs)

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        # print(lines)
        splitlines = [x.strip().split(' ') for x in lines]
        print(splitlines)
        image_ids = [x[0] for x in splitlines]
        quadrants = [int(x[1]) for x in splitlines]
        confidence = np.array([float(x[2]) for x in splitlines])
        BB = np.array([[float(z) for z in x[3:]] for x in splitlines])

        print(BB)
        # print(image_ids)

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        quadrants = [quadrants[x] for x in sorted_ind]
        print(image_ids)

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        for imagename in imagenames:
            image_id = os.path.splitext(os.path.split(imagename)[1])[0]
            img = cv2.imread(imagename)

            height, width, channels = img.shape
            print(img.shape)

            for d in range(nd):
                # print(image_id)
                # print(image_ids[d])
                if image_ids[d] == image_id:
                    quadrant = quadrants[d]
                    print(quadrant)
                    if confidence[d] > args.confidence_threshold and quadrant == 0:
                        bb = BB[d, :]
                        print(bb)

                        # if quadrant == 1:
                        #     bb[0] += width//2
                        #     bb[2] += widht//2
                        # elif quadrant == 2:
                        #     bb[1] += height//2
                        #     bb[3] += height//2
                        # elif quadrant == 3:
                        #     bb[0] += width//2
                        #     bb[2] += widht//2
                        #     bb[1] += height//2
                        #     bb[3] += height//2



                        cv2.rectangle(img, (int(bb[0]/2), int(bb[1]/2)), (int(bb[2]/2), int(bb[3]/2)), (0,0,255), 2)

            cv2.imshow("image", img)
            cv2.waitKey()



        for d in range(nd):
            R = class_recs[image_ids[d]]
            # print(R)
            bb = BB[d, :].astype(float)
            # print(bb)



            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = -1 # np.cumsum(fp)
        tp = -1 #np.cumsum(tp)
        rec = -1 #tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = -1#tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = -1 #voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap

def test_net_custom(save_folder, net, cuda, dataset, transform, top_k, imsize=300, thresh=0.05):

    quadrants = 4
    num_images = len(dataset) // quadrants

    idq = dataset.ids

    for i in range(num_images):
        partial_imgs = []
        partial_gts = []
        partial_detections = []

        for j in range(quadrants):
            im, gt, h, w = dataset.pull_item(i*quadrants+j)

            x = Variable(im.unsqueeze(0))
            if args.cuda:
                x = x.cuda()
            detections = net(x).data

            im = im.numpy().transpose(1,2,0)
            im = np.ascontiguousarray(im, dtype=np.uint8)
            partial_imgs.append(im)
            partial_gts.append(gt)
            partial_detections.append(detections)

        img_id = idq[i*quadrants][0]

        img_path = os.path.join(args.voc_root, img_id + '.jpg')

        img = cv2.imread(img_path)

        h, w, c = img.shape

        # skip j = 0, because it's the background class
        for j in range(len(partial_detections)):
            detection = partial_detections[j]
            gt = partial_gts[j]
            quad = j

            for bb in gt:
                bb[0] *= w // 2
                bb[2] *= w // 2
                bb[1] *= h // 2
                bb[3] *= h // 2

                if quad == 1:
                    bb[0] += w // 2
                    bb[2] += w // 2
                elif quad == 2:
                    bb[1] += h // 2
                    bb[3] += h // 2
                elif quad == 3:
                    bb[1] += h // 2
                    bb[3] += h // 2
                    bb[0] += w // 2
                    bb[2] += w // 2


                cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0,255,255), 2)




            for k in range(1, detections.size(1)):
                dets = detection[0, k, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= w // 2
                boxes[:, 2] *= w // 2
                boxes[:, 1] *= h // 2
                boxes[:, 3] *= h // 2

                if quad == 1:
                    boxes[:, 0] += w // 2
                    boxes[:, 2] += w // 2
                elif quad == 2:
                    boxes[:, 1] += h // 2
                    boxes[:, 3] += h // 2
                elif quad == 3:
                    boxes[:, 1] += h // 2
                    boxes[:, 3] += h // 2
                    boxes[:, 0] += w // 2
                    boxes[:, 2] += w // 2

                scores = dets[:, 0].cpu().numpy()

                # print(scores)
                # print("drawing detections")

                for k in range(boxes.size(1)):

                    if scores[k] >= args.confidence_threshold:
                        box = boxes[k, :]
                        print(box)

                        # cv2.rectangle(img, (0, 0), (5, 5), (0,0,255), 2)
                        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 2)


        cv2.imshow("full", img)
        cv2.waitKey()




def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(customConfig["num_classes"])]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)
        print(gt)

        # print(im.dtype)

        img = im.numpy().transpose(1,2,0)
        # print(img.shape)
        # print(img)
        img = np.ascontiguousarray(img, dtype=np.uint8)
        # print(img.shape)
        # cv2.imshow("image", img)
        # cv2.waitKey()

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        # print(x)
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)


        # print("drawing ground truth")

        # for j in range(len(gt)):
        #     box = gt[j, 0:4]
        #     print(box)
        #     cv2.rectangle(img, (int(box[0]*300), int(box[1]*300)), (int(box[2]*300), int(box[3]*300)), (255,0,255), 2)



        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            # print(scores)
            # print("drawing detections")

            # for k in range(boxes.size(1)):

            #     if scores[k] >= args.confidence_threshold:
            #         box = boxes[k, :]
            #         print(box)

            #         # cv2.rectangle(img, (0, 0), (5, 5), (0,0,255), 2)
            #         cv2.rectangle(img, (int(box[0]*300.0/w), int(box[1]*300.0/h)), (int(box[2]*300.0/w), int(box[3]*300.0/h)), (0,0,255), 2)

            all_boxes[j][i] = cls_dets


            # if image_ids[d] == image_id and confidence[d] > args.confidence_threshold:
            #     bb = BB[d, :]
            #     print(bb)
            #     cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0,0,255), 2)


        # cv2.imshow("image", img)
        # cv2.waitKey()
      
        # image_id = os.path.splitext(os.path.split(imagename)[1])[0]

            # print(image_id)
            # print(image_ids[d])


        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir)


if __name__ == '__main__':
    print(args)
    # load net
    num_classes = customConfig["num_classes"]                   # +1 for background
    net = build_ssd('test', 300, num_classes)            # initialize SSD

    if args.cuda:
        net.load_state_dict(torch.load(args.trained_model))
    else:
        net.load_state_dict(torch.load(args.trained_model, map_location=torch.device('cpu')))
    net.eval()
    print('Finished loading model!')
    # load data
    # dataset = VOCDetection(args.voc_root, [('2007', set_type)],
    #                        BaseTransform(300, dataset_mean),
    #                        VOCAnnotationTransform())

    dataset = CustomDetection(args.voc_root, transform=SSDAugmentationLight(customConfig['min_dim'], 
                                                         MEANS))
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net_custom(args.save_folder, net, args.cuda, dataset,
             BaseTransform(net.size, dataset_mean), args.top_k, 300,
             thresh=args.confidence_threshold)
