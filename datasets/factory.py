# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
import os

import numpy as np

# Set up voc_<year>_<split>
# for year in ['2007', '2012']:
#   for split in ['train', 'val', 'trainval', 'test']:
#     name = 'voc_{}_{}'.format(year, split)
#     __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
for year in ['2007']:
  for split in ['train', 'val', 'trainval', 'test', 'mytest', 'myval']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))

  # print('----------------Debug-----------------')
  # print(__sets[name]().gt_roidb()[0])
  # print('----------------Debug-----------------')
  
  # print('-----------------------Debug-----------------------') 
  # index = 'rs00118_jpg.rf.090032231010a32e48bdf2fb70fc9d1c' 
  # filename = os.path.join(__sets[name]()._data_path, 'Annotations', index + '.xml')
  # import xml.etree.ElementTree as ET
  # tree = ET.parse(filename)
  # objs = tree.findall('object')
  # num_objs = len(objs)

  # boxes = np.zeros((num_objs, 4), dtype=np.uint16)
  # gt_classes = np.zeros((num_objs), dtype=np.int32)
  # overlaps = np.zeros((num_objs, __sets[name]().num_classes), dtype=np.float32)
  # seg_areas = np.zeros((num_objs), dtype=np.float32)
  # ishards = np.zeros((num_objs), dtype=np.int32)

  # for ix, obj in enumerate(objs):
  #     diffc = obj.find('difficult')
  #     difficult = 0 if diffc == None else int(diffc.text)
  #     ishards[ix] = difficult

  #     class_name = obj.find('name').text.strip()

  #     if class_name in __sets[name]()._classes:
  #         cls = __sets[name]()._class_to_ind[obj.find('name').text.strip()]
  #         gt_classes[ix] = cls
  #     else:
  #         cls = 0
  #         gt_classes[ix] = 0

  # print('-----------------------Debug-----------------------')
  # print(gt_classes)
  # print('-----------------------Debug-----------------------')
  # print('-----------------------Debug-----------------------')

  return __sets[name]()

def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())