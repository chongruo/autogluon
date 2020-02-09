from .base import DatasetBase
from ....core import *
import os

from gluoncv import data as gdata
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric


class CustomCOCODetectionBase(gdata.COCODetection):
    """MS COCO detection dataset.
    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/coco'
        Path to folder storing the dataset.
    splits : list of str, default ['instances_val2017']
        Json annotations name.
        Candidates can be: instances_val2017, instances_train2017.
    transform : callable, default None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.
        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    min_object_area : float
        Minimum accepted ground-truth area, if an object's area is smaller than this value,
        it will be ignored.
    skip_empty : bool, default is True
        Whether skip images with no valid object. This should be `True` in training, otherwise
        it will cause undefined behavior.
    use_crowd : bool, default is True
        Whether use boxes labeled as crowd instance.
    """
    
    def __init__(self, classes=None, root=os.path.join('~', '.mxnet', 'datasets', 'coco'), 
                 splits=('instances_val2017',), transform=None, min_object_area=0,
                 skip_empty=True, use_crowd=True):

        if classes:
            self._set_class(classes)  
        super(CustomCOCODetectionBase, self).__init__(root)

    @classmethod
    def _set_class(cls, classes):
        cls.CLASSES = classes


@obj()
class CustomCOCODetection(DatasetBase):
    """Built-in class to work with the well-known COCO dataset for object detection. 
    
    Returns
    -------
    Dataset object that can be passed to `task.fit()`, which is actually an :class:`autogluon.space.AutoGluonObject`. 
    To interact with such an object yourself, you must first call `Dataset.init()` to instantiate the object in Python.
    """
    def __init__(self, root, splits, name, classes, data_shape, Train, **kwargs):
        super().__init__()
    
        if Train:
            self.dataset = CustomCOCODetectionBase(classes=classes,
                                                root=root,
                                                splits=splits,
                                                use_crowd=False)
        else:
            self.dataset = CustomCOCODetectionBase(classes=classes,
                                                root=root,
                                                splits=splits,
                                                skip_empty=False)


        def get_metric(data_shape):
            def metric_fn(val_dataset):
                return COCODetectionMetric(val_dataset,  ''+ '_eval', cleanup=True,
                                           data_shape=(data_shape, data_shape))
            return metric_fn

        self.metric = get_metric(data_shape)
        
    
    def get_dataset_and_metric(self):
        return (self.dataset, self.metric)
    
    def get_classes(self):
        return self.dataset.classes 
    
    def get_dataset_name(self):
        return 'coco'




