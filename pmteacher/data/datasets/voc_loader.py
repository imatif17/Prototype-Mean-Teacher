import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union
import functools
import multiprocessing as mp
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from detectron2.utils.comm import get_world_size

CLASS_NAMES = (
    'traffic light', 'traffic sign', 'car', 'person', 'bus', 'truck', 'rider', 'bike', 'motor', 'train'
)


# fmt: on

def load_bdd_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    #classes = []
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileid = np.loadtxt(f, dtype=str)

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))
    dataset_dicts = pool.map(
                    functools.partial(create_dict, annotation_dirname=annotation_dirname,
                    dirname = dirname, split = split, class_names = class_names),
                    fileid,   
                )
    pool.close()
    return dataset_dicts
    

def create_dict(fileid, annotation_dirname, dirname, split, class_names):
    anno_file = os.path.join(annotation_dirname, fileid + ".xml")
    jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

    with PathManager.open(anno_file) as f:
        tree = ET.parse(f)

    r = {
        "file_name": jpeg_file,
        "image_id": fileid,
        "height": int(tree.findall("./size/height")[0].text),
        "width": int(tree.findall("./size/width")[0].text),
    }
    instances = []

    for obj in tree.findall("object"):
        cls = obj.find("name").text
        bbox = obj.find("bndbox")
        bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]

        bbox[0] -= 1.0
        bbox[1] -= 1.0
        if cls in class_names:
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
    r["annotations"] = instances
    return r

def register_bdd(name, dirname, split, year, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_bdd_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )