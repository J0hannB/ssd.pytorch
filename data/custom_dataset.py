"""Custom Dataset Classes

"""
from .config import HOME
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


class YOLOAnnotationTransform(object):
    """Transforms a YOLO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_list=None):
        self.class_list = class_list

    def __call__(self, target, width, height):

        res = []
        for row in target:
            row = row.split()
            clss = float(row[0])
            bbox = row[1:5]
            x = float(bbox[0])
            y = float(bbox[1])
            # due to my conversion script w and h can sometimes be negative. Account for that here with abs()
            w = abs(float(bbox[2]))
            h = abs(float(bbox[3]))
            bboxf = list()
            bboxf.append(x-w/2)
            bboxf.append(y-h/2)
            bboxf.append(x+w/2)
            bboxf.append(y+h/2)

            #for i in range(4):
            #    if bboxf[i] > 1.0 or bboxf[i] < 0.0:
            #        print("Invalid gt dimension: {}".format(bboxf[i]))
            # for dim in bbox:
            #     dim = float(dim)
            #     bboxf.append(dim)

            resRow = bboxf
            resRow.append(clss)
            res.append(resRow)


        # print(res)

        return res # [[xmin, ymin, xmax, ymax, label_ind], ... ]


        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        # res = []
        # for obj in target.iter('object'):
        #     difficult = int(obj.find('difficult').text) == 1
        #     if not self.keep_difficult and difficult:
        #         continue
        #     name = obj.find('name').text.lower().strip()
        #     bbox = obj.find('bndbox')

        #     pts = ['xmin', 'ymin', 'xmax', 'ymax']
        #     bndbox = []
        #     for i, pt in enumerate(pts):
        #         cur_pt = int(bbox.find(pt).text) - 1
        #         # scale height or width
        #         cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
        #         bndbox.append(cur_pt)
        #     label_idx = self.class_to_ind[name]
        #     bndbox.append(label_idx)
        #     res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        #     # img_id = target.find('filename').text[:-4]

        # return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class CustomDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
    """

    def __init__(self, root,
                 transform=None, target_transform=YOLOAnnotationTransform()
                 ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = osp.join(root, '%s.txt')
        self._imgpath = osp.join(root, '%s.jpg')
        self.name = 'Custom'

        self.ids = list()
        files = os.listdir(root)
        for file in files:
            if osp.splitext(file)[1] == '.jpg':
                # 4 quadrants for each image
                self.ids.append([osp.splitext(file)[0], 0])
                self.ids.append([osp.splitext(file)[0], 1])
                self.ids.append([osp.splitext(file)[0], 2])
                self.ids.append([osp.splitext(file)[0], 3])

        print("Loaded {} images into dataset".format(len(self.ids)))

        
        # for (year, name) in image_sets:
        #     rootpath = osp.join(self.root, 'VOC' + year)
        #     for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
        #         self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):

        img_id = self.ids[index][0]
        quadrant_idx = self.ids[index][1]

        # print(img_id)
        # target = ET.parse(self._annopath % img_id).getroot()
        target = list()
        with open(self._annopath % img_id) as annoFile:
            for line in annoFile:
                target.append(line)


        img = cv2.imread(self._imgpath % img_id)
        
        if img is None:
            img_id = self.ids[index+1][0]
            quadrant_idx = self.ids[index+1][1]

            # print(img_id)
            # target = ET.parse(self._annopath % img_id).getroot()
            target = list()
            with open(self._annopath % img_id) as annoFile:
                for line in annoFile:
                    target.append(line)


            img = cv2.imread(self._imgpath % img_id)
#            return None, None, None, None

        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)



        # This will not work if num_workers is not 0
        # print(width//2)
        # print(height//2)
        # print(img.shape)

        # print(quadrant_idx)


        if quadrant_idx == 0:
            yMin = 0
            yMax = height//2
            xMin = 0
            xMax = width//2
        if quadrant_idx == 1:
            yMin = 0
            yMax = height//2
            xMin = width//2
            xMax = width
        if quadrant_idx == 2:
            yMin = height//2
            yMax = height
            xMin = 0
            xMax = width//2
        if quadrant_idx == 3:
            yMin = height//2
            yMax = height
            xMin = width//2
            xMax = width

        # print(img.shape)
        # print("min/max: (%f, %f) (%f, %f)" % (xMin, yMin, xMax, yMax))

        goodTargets = []

        for i, box in enumerate(target):

            x1 = box[0] * width
            y1 = box[1] * height
            x2 = box[2] * width
            y2 = box[3] * height

            # print(box)
            # print("(%f, %f) (%f, %f)" % (x1, y1, x2, y2))
            if x1 <= xMin:
                if x2 <= xMin:
                    continue
                x1 = xMin
            if y1 <= yMin:
                if y2 <= yMin:
                    continue
                y1 = yMin
            if x2 >= xMax:
                if x1 >= xMax:
                    continue
                x2 = xMax
            if y2 >= yMax:
                if y1 >= yMax:
                    continue
                y2 = yMax



            # print("(%f, %f) (%f, %f)" % (x1, y1, x2, y2))
            box[0] = (x1-xMin) / (xMax-xMin)
            box[1] = (y1-yMin) / (yMax-yMin)
            box[2] = (x2-xMin) / (xMax-xMin)
            box[3] = (y2-yMin) / (yMax-yMin)
            # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            goodTargets.append(box)

        if len(goodTargets) == 0:
            # print("Appending false target to targetless image")
            goodTargets.append([0.0,0.0,1.0/width,1.0/height,0.0])
            # goodTargets.append([0.0,0.0,0.0,0.0,0.0])

        for t in goodTargets:
            if t[0] > 1.0 or t[0] < 0.0:
                print("invalid target in image " + img_id)
                print(t)
            if t[1] > 1.0 or t[1] < 0.0:
                print("invalid target in image " + img_id)
                print(t)
            if t[2] > 1.0 or t[2] < 0.0:
                print("invalid target in image " + img_id)
                print(t)
            if t[3] > 1.0 or t[3] < 0.0:
                print("invalid target in image " + img_id)
                print(t)
        target = goodTargets
        # print(target)
        # cv2.imshow("full Image", img)

        img = img[yMin:yMax, xMin:xMax]

        # for t in target:
        #     cv2.rectangle(img, (int(t[0]*width//2), int(t[1]*height//2)), (int(t[2]*width//2), int(t[3]*height//2)), (0,255,255), 2)


        # print(img.shape)

        # cv2.imshow("partial image", img)
        # cv2.waitKey(1000)
        # cv2.waitKey()

        # print("target from custom_dataset.py")
        # print(target)


        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        # print(type(width))
        # print(type(height))

        # print("updated: " + str(quadrant_idx))


        im = torch.from_numpy(img).permute(2, 0, 1)
        # print(im)
        return im, target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index][0]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index][0]
        # anno = ET.parse(self._annopath % img_id).getroot()
        anno = list()
        with open(self._annopath % img_id) as annoFile:
            for line in annoFile:
                anno.append(line)
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
