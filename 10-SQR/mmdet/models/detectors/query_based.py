from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from .sparse_rcnn import SparseRCNN


@DETECTORS.register_module()
class QueryBased(SparseRCNN):
    '''
    We hack and build our model into Sparse RCNN framework implementation
    in mmdetection.
    '''
