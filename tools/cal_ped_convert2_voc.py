import xml.etree.ElementTree as ET
import os
import re
import json
import glob
import cv2 as cv

set_names = ['set%02d' % i for i in range(0, 11)]
set_names = ['set00', 'set01', 'set02']
base = '/home/juliuswang/Projects/mxnet-ssd/data/caltech-pedestrian/'
bases = [base + set_name for set_name in set_names]

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

def write_xml_structure(data, img_shape, folder, image_name, output_folder):
    height, width, channels = img_shape
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "caltech-pedestrian/images%s" % folder
    ET.SubElement(annotation, "filename").text = image_name

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "caltech-pedestrian"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(channels)

    # write annotation of each object
    for datum in data:
        x, y, w, h = [int(v) for v in datum['pos']]
        start_frame, end_frame = datum['str'], datum['end']
        is_occl, is_hide = datum['occl'], datum['hide']
        label = datum['lbl']

        object = ET.SubElement(annotation, "object")
        ET.SubElement(object, "name").text = label
        ET.SubElement(object, "truncated").text = '1'
        bndbox = ET.SubElement(object, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(x)
        ET.SubElement(bndbox, "ymin").text = str(y)
        ET.SubElement(bndbox, "xmax").text = str(x + w)
        ET.SubElement(bndbox, "ymax").text = str(y + h)

        ET.SubElement(object, "str").text = str(start_frame)
        ET.SubElement(object, "end").text = str(end_frame)
        ET.SubElement(object, "occl").text = str(is_occl)
        ET.SubElement(object, "hide").text = str(is_hide)

    # write final built tree
    tree = ET.ElementTree(annotation)
    indent(annotation)
    set_name, video_name = folder.split('/')[1::]
    filename = '%s/%s_%s_%s.xml' % \
                (output_folder, set_name, video_name, os.path.splitext(image_name)[0])
    tree.write(filename, encoding="utf-8", xml_declaration=True)

def convert(json_filename, output_folder):
    print 'converting %s into VOC-compatible XML files' % json_filename
    annotations = json.load(open(json_filename))

    n_objects = 0
    # iterate through set00, set01, ...folders
    for base in bases:
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

    print 'convertion done...!'
    return

if __name__ == '__main__':
    convert('../data/cal_ped_annotations.json',
            '../data/caltech-pedestrian/annotations')