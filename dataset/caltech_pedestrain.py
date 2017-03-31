import os
import numpy as np
import glob
from imdb import Imdb
import xml.etree.ElementTree as ET
from evaluate.eval_voc import voc_eval
import cv2

def load_caltech(image_set, devkit_path, shuffle=False):
    """
        wrapper function for loading pascal voc dataset

        Parameters:
        ----------
        image_set : str
            train, trainval...
        devkit_path : str
            root directory of dataset
        shuffle : bool
            whether to shuffle initial list

        Returns:
        ----------
        Imdb
    """
    image_set = [y.strip() for y in image_set.split(',')]
    assert image_set, "No image_set specified"

    imdbs = []
    for s in image_set:
        imdbs.append(Caltech_Pedestrian(s, devkit_path, shuffle, is_train=True))
    if len(imdbs) > 1:
        return ConcatDB(imdbs, shuffle)
    else:
        return imdbs[0]

class Caltech_Pedestrian(Imdb):
    """
        Implementation of Imdb for Caltech pedestrian datasets

        Parameters:
        ----------
        image_set : str
            set to be used, can be train, val, trainval, test
        devkit_path : str
            devkit path of Caltech pedestrian dataset
        shuffle : boolean
            whether to initial shuffle the image list
        is_train : boolean
            if true, will load annotations
    """
    def __init__(self, image_set, devkit_path, shuffle=False, is_train=False):
        super(Caltech_Pedestrian, self).__init__('caltech-pedestrian_' + image_set)
        self.image_set = image_set
        self.devkit_path = devkit_path  # caltech-pedestrian
        assert os.path.exists(self.devkit_path), 'self.devkit_path does not exist: {}'.format(self.devkit_path)
        #self.data_path = os.path.join(devkit_path, 'images')   # caltech-pedestrian/data/images
        self.data_path = self.devkit_path
        assert os.path.exists(self.data_path), 'self.data_path does not exist: {}'.format(self.data_path)
        self.annotation_path = os.path.join(devkit_path, 'annotations')
        assert os.path.exists(self.annotation_path), 'self.annotation_path does not exist: {}'.format(self.annotation_path)
        self.extension = '.png'
        self.is_train = is_train

        self.classes = ['person']

        # todo: check what is padding, comp_id
        self.config = {#'use_difficult': True,
                       'comp_id': 'comp4',
                       'padding': 56}

        self.num_classes = len(self.classes)

        # retrieve annotations (labels) files (.xml)
        self.annotation_set = self._load_image_set_index(shuffle)
        if self.is_train:
            # load annotations (labels) from xml
            self.labels = self._load_image_labels()

        # count number of images after parsing xml files
        # as some images don't contain the object class included in the training
        self.num_images = len(self.annotation_set)

    def _load_image_set_index(self, shuffle):
        """
        find out which indexes correspond to given image set (train or val)

        Parameters:
        ----------
        shuffle : boolean
            whether to shuffle the image list
        Returns:
        ----------
        entire list of images specified in the setting
        """

        print 'preparing [%s] data (reading xml annotations)...' % self.image_set

        # retrieve all annotation files for every image in annotation_path
        annotation_full_set = [os.path.basename(xml_path)
                           for xml_path in sorted(glob.glob('%s/*.xml' % self.annotation_path))]
        annotation_set = []
        if self.image_set == 'train':    # set00 ~ set04 are for training
            for img_name in annotation_full_set:
                set_id = int(img_name.split('_')[0].strip('set'))
                #if set_id < 5:
                if set_id < 1:
                    annotation_set.append(img_name)
        elif self.image_set == 'val':
            for img_name in annotation_full_set:
                set_id = int(img_name.split('_')[0].strip('set'))
                #if set_id == 5:
                if set_id == 1:
                    annotation_set.append(img_name)
        elif self.image_set == 'trainval': # set00 ~ set05 are for training + val
            for img_name in annotation_full_set:
                set_id = int(img_name.split('_')[0].strip('set'))
                #if set_id <= 5:
                if set_id <= 1:
                    annotation_set.append(img_name)
        elif self.image_set == 'test':
            for img_name in annotation_full_set:
                set_id = int(img_name.split('_')[0].strip('set'))
                #if set_id > 5:
                if set_id > 1:
                    annotation_set.append(img_name)
        else:
            raise NotImplementedError, "check if self.image_set is either " \
                                       "train, val, trainval, or test. " + \
                                       self.image_set + " not supported"

        if shuffle:
            print 'shuffling data as asked...'
            np.random.shuffle(annotation_set)

        print 'preparing [%s] data (reading xml annotations)...totally %d...Done!' % (self.image_set, len(annotation_set))

        return annotation_set

    def _load_image_labels(self):
        """
        preprocess all ground-truths

        Returns:
        ----------
        labels packed in [num_images x max_num_objects x 5] tensor
        """
        print 'parsing annotations...'
        temp = []
        max_objects = 0

        # todo: caltech dataset contains two classes, 'person' and 'people'
        # todo: here only count the one labeled as 'person'
        person_count = 0

        annotation_set = []
        # load ground-truth from xml annotations
        for label_file in self.annotation_set:
            tree = ET.parse(os.path.join(self.annotation_path, label_file))
            root = tree.getroot()
            size = root.find('size')
            width = float(size.find('width').text)
            height = float(size.find('height').text)
            label = []

            for obj in root.iter('object'):
                cls_name = obj.find('name').text
                if cls_name not in self.classes:
                    continue
                cls_id = self.classes.index(cls_name)
                xml_box = obj.find('bndbox')
                xmin = float(xml_box.find('xmin').text) / width
                ymin = float(xml_box.find('ymin').text) / height
                xmax = float(xml_box.find('xmax').text) / width
                ymax = float(xml_box.find('ymax').text) / height
                label.append([cls_id, xmin, ymin, xmax, ymax])
                person_count += 1

            if len(label) > 0:
                annotation_set.append(label_file)
                temp.append(np.array(label))
                max_objects = max(max_objects, len(label))

        # update annotation set containing object classes of interest
        self.annotation_set = annotation_set

        # add padding to labels so that the dimensions match in each batch
        # todo: why need padding?
        # TODO: design a better way to handle label padding

        assert max_objects > 0, "No objects found for any of the images"
        #assert max_objects <= self.config['padding'], "# obj exceed padding"
        self.config['padding'] = max_objects
        self.padding = self.config['padding']
        labels = []
        for label in temp:
            label = np.lib.pad(label, ((0, self.padding - label.shape[0]), (0,0)),
                               'constant', constant_values=(-1, -1))
            labels.append(label)

        print 'parsing annotations...Done! Total %d person counts' % person_count
        print '=============='
        print '# of the effective annotation file: %d (as getting rid of "people" class)' % len(self.annotation_set)
        print 'max. amount of objects in an image: %d' % max_objects
        print '=============='
        return np.array(labels)

    @property
    # todo: what is this for?
    def cache_path(self):
        """
        make a directory to store all caches

        Returns:
        ---------
            cache path
        """
        cache_path = os.path.join(os.path.dirname(__file__), '..', 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        #return self.labels[index, :, :]
        return self.labels[index]

    def image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.annotation_set is not None, "Dataset not initialized"
        name = self.annotation_set[index]   # e.g. 'set00_V010_img00577.xml'
        set_name, video_name, xml_name = name.split('_')
        img_name = os.path.splitext(xml_name)[0] + self.extension
        img_path = os.path.join(self.data_path, set_name, video_name, img_name)
        assert os.path.exists(img_path), 'Path does not exist: {}'.format(img_path)

        return img_path

    def evaluate_detections(self, detections):
        raise NotImplementedError, 'evaluate_detections to be implemented!'

