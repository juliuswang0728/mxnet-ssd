import xml.etree.ElementTree as ET
import os
import re
import json
import glob
import cv2 as cv

def indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def write_xml_structure(data, img_shape, image_name, output_folder):
    height, width, channels = img_shape
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "kitti/training/image_2"
    ET.SubElement(annotation, "filename").text = image_name

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "kitti-usa"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(channels)

    # write annotation of each object
    for datum in data:
        datum[-1] = datum[-1].strip('\n')
        x_min, y_min, x_max, y_max = float(datum[4]), float(datum[5]), float(datum[6]), float(datum[7])
        x_min, y_min, x_max, y_max = int(x_min + 0.5), int(y_min + 0.5), int(x_max + 0.5), int(y_max + 0.5)

        label = datum[0]
        label = 'person' if label == 'Pedestrian' else label
        trunc, occl = datum[1], datum[2]

        object = ET.SubElement(annotation, "object")
        ET.SubElement(object, "name").text = label
        bndbox = ET.SubElement(object, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(x_min)
        ET.SubElement(bndbox, "ymin").text = str(y_min)
        ET.SubElement(bndbox, "xmax").text = str(x_max)
        ET.SubElement(bndbox, "ymax").text = str(y_max)

        ET.SubElement(object, "occl").text = str(occl)
        ET.SubElement(object, "trunc").text = str(trunc)

    # write final built tree
    tree = ET.ElementTree(annotation)
    indent(annotation)
    filename = '%s/%s.xml' % (output_folder, image_name)
    tree.write(filename, encoding="utf-8", xml_declaration=True)

def convert(image_folder, label_folder, output_folder):
    if not os.path.exists(image_folder):
        raise OSErrexceptionor, 'path [%s] not exists!' % image_folder
    if not os.path.exists(label_folder):
        raise OSErrexceptionor, 'path [%s] not exists!' % label_folder
    if not os.path.exists(output_folder):
        print '%s not exists, so creating one...' % output_folder
        os.mkdir(output_folder)

    print 'converting kitti label files [in %s] into VOC-compatible XML files' % label_folder
    print 'annotations output to [%s]...' % output_folder

    n_objects = 0

    for label in sorted(glob.glob(label_folder + '/*.txt')):
        #label_file = os.path.split(label)[1]
        label_file = os.path.basename(label)
        image_name = label_file.split('.')[0]
        image_path = image_folder + '/' + image_name + '.png'
        if not os.path.exists(image_path):
            raise OSErrexceptionor, 'image path [%s] not exists!' % image_path
        print image_path
        with open(label, 'r') as f:
            img = cv.imread(image_path)

            data = [line.split(' ') for line in f.readlines()]
            write_xml_structure(data, img.shape, image_name, output_folder)

            '''is_person = False
            for datum in data:
                datum[-1] = datum[-1].strip('\n')
                if datum[0] == 'Pedestrian':
                    is_person = True
                    x, y, w, h = float(datum[4]), float(datum[5]), float(datum[6]) - float(datum[4]), float(datum[7]) - float(datum[5])
                    x, y, w, h = int(x + 0.5), int(y + 0.5), int(w + 0.5), int(h + 0.5)
                    print x, y, w, h
                    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    n_objects += 1
            if is_person:
                cv.imshow('annotations', img)
                cv.waitKey(0)'''

    '''for base in bases:
        # read video folders V001, V002, ...etc
        set_name = os.path.basename(base)
        for video_dir in sorted(glob.glob(base + '/*')):
            video_name = os.path.basename(video_dir)

            folder = '/%s/%s' % (set_name, video_name)

            print video_dir, video_name
            # read images within video folders

            for image_path in sorted(glob.glob(video_dir + '/*.png')):
                image_name = os.path.basename(image_path)
                n_frame = re.search('img([0-9]+)\.png', image_name).groups()[0]
                n_frame = str(int(n_frame))

                # if this frame has any annotation
                if n_frame in annotations[set_name][video_name]['frames']:
                    #print image_name, annotations[set_name][video_name]['frames'][n_frame]
                    data = annotations[set_name][video_name]['frames'][n_frame]
                    img = cv.imread(image_path)
                    # write one xml file for each image
                    write_xml_structure(data, img.shape, folder, image_name, output_folder)

                    for datum in data:
                        x, y, w, h = [int(v) for v in datum['pos']]
                        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                        n_objects += 1
                    #cv.imshow('annotations', img)
                    #cv.waitKey(0)
'''
    print 'convertion done...! total objects: %d' % n_objects
    return

if __name__ == '__main__':
    convert('../data/kitti/training/image_2',
            '../data/kitti/training/label_2',
            '../data/kitti/training/annotations')