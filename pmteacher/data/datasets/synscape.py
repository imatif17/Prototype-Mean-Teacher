import contextlib
import datetime
import io
import json
import functools
import logging
import numpy as np
import os
from detectron2.utils.comm import get_world_size
import shutil
import multiprocessing as mp

import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.build import print_instances_class_histogram


logger = logging.getLogger(__name__)

def create_dict(images, json_path, image_path, mapper):
    with open(f'{json_path}/{images}.json') as f:
        x = json.load(f)
    data_dict = {}
    data_dict["file_name"] = f'{image_path}/{images}.png'
    data_dict["height"] = x['camera']['intrinsic']['resy']
    data_dict["width"] = x['camera']['intrinsic']['resx']
    data_dict["image_id"] = images

    boxes = x['instance']['bbox2d'].keys()
    annos = []
    for item in boxes:
        if (x['instance']['class'][item] in mapper.keys()):
            anno = {}
            anno["bbox"] = [x['instance']['bbox2d'][item]['xmin'] * data_dict["width"],
                    x['instance']['bbox2d'][item]['ymin'] * data_dict["height"],
                    x['instance']['bbox2d'][item]['xmax'] * data_dict["width"],
                    x['instance']['bbox2d'][item]['ymax'] * data_dict["height"]
            ]
            anno["bbox_mode"] = 0
            anno["category_id"] = mapper[x['instance']['class'][item]]
            annos.append(anno)
    data_dict['annotations'] = annos
    return data_dict

def get_synapse_dataset(name, json_file, image_root):
    meta = MetadataCatalog.get(name)
    thing_classes = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    meta.thing_classes = thing_classes
    logger.info("Loading Synsapse dataset")
    from cityscapesscripts.helpers.labels import labels
    labels = [l for l in labels if l.hasInstances and not l.ignoreInEval]
    dataset_id_to_contiguous_id = {l.id: idx for idx, l in enumerate(labels)}
    meta.thing_dataset_id_to_contiguous_id = dataset_id_to_contiguous_id

    images = list(range(1, 25001))
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))
    dataset_dicts = pool.map(
                    functools.partial(create_dict, json_path=json_file, image_path=image_root, mapper = dataset_id_to_contiguous_id),
                    images,   
                )
    print_instances_class_histogram(dataset_dicts, MetadataCatalog.get("synscapes").thing_classes)
    pool.close()
    return dataset_dicts

def synapse_register(name, metadata, json_file, image_root):
    """
    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    
    DatasetCatalog.register(name, lambda: get_synapse_dataset(name, json_file, image_root))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )