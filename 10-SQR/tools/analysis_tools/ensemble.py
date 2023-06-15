import argparse

import mmcv
from mmcv import Config, DictAction

from mmdet.datasets import build_dataset
import numpy as np
from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from mmcv.ops.nms import batched_nms
import torch, torchvision


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                     'results saved in pkl format')
    parser.add_argument('config', help='Config of the model')
    parser.add_argument('--pkls', nargs="+", default=["a", "b"], help='Results in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='Evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    assert args.eval or args.format_only, (
        'Please specify at least one operation (eval/format the results) with '
        'the argument "--eval", "--format-only"')
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)

    pkls = args.pkls
    outputs_list = []
    for pkl in pkls:
        outputs_list.append(mmcv.load(pkl))

    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        raise NotImplementedError
        dataset.format_results(outputs, **kwargs)
    if args.eval:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))

        outputs = ensemble(outputs_list)

        print(dataset.evaluate(outputs, **eval_kwargs))


def ensemble(outs):
    img_num = len(outs[0])
    out_num = len(outs)
    final = []
    base = outs[0]
    other = outs[1:]
    # collect all to base
    for out in other:
        for imgid, img in enumerate(out):
            for clsid, cls in enumerate(img):
                base[imgid][clsid] = np.concatenate((base[imgid][clsid], cls), axis=0)

    nms_cfg = {'type': 'nms', 'iou_threshold': 0.7}

    for imgid, img in enumerate(base):
        img_new = []
        for cls in img:
            boxes = torch.Tensor(cls[:, 0:4]).contiguous()
            scores = torch.Tensor(cls[:, 4]).contiguous()
            idxs = torch.Tensor([1]*len(scores)).contiguous()
            dets, keep = batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=True)

            img_new.append(dets.numpy())
        final.append(img_new)


    return final

if __name__ == '__main__':
    main()
