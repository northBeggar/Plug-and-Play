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
from mmdet.datasets.api_wrappers import COCOeval
import pickle5

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

    #pkl = mmcv.load(args.pkls[0])
    with open(args.pkls[0], "rb") as f:
        pkl = pickle5.load(f)

    #outs5 = postproc_deform(pkl, 5)
    #outs4 = postproc_deform(pkl, 4)
    #outs3 = postproc_deform(pkl, 3)
    #outs2 = postproc_deform(pkl, 2)
    #outs1 = postproc_deform(pkl, 1)
    #outs0 = postproc_deform(pkl, 0)

    # outs = organize(pkl)
    outs5 = postproc(pkl, 5)
    outs4 = postproc(pkl, 4)
    outs3 = postproc(pkl, 3)
    outs2 = postproc(pkl, 2)
    outs1 = postproc(pkl, 1)
    outs0 = postproc(pkl, 0)

    iou_thrs = 0.75

    cocoEval5 = preeval(base=outs5, dataset=dataset, iou_thrs=iou_thrs)
    cocoEval4 = preeval(base=outs4, dataset=dataset, iou_thrs=iou_thrs)
    cocoEval3 = preeval(base=outs3, dataset=dataset, iou_thrs=iou_thrs)
    cocoEval2 = preeval(base=outs2, dataset=dataset, iou_thrs=iou_thrs)
    cocoEval1 = preeval(base=outs1, dataset=dataset, iou_thrs=iou_thrs)
    cocoEval0 = preeval(base=outs0, dataset=dataset, iou_thrs=iou_thrs)

    evalImgs5 = fangyi(base=cocoEval5, supts=[cocoEval4,
                                              cocoEval3,
                                              cocoEval2,
                                              cocoEval1,
                                              cocoEval0,
                                              ])

    #cocoEval5.eval = motivation_verification(evalImgs5,
    #                                         [evalImgs5],
    #                                         cocoEval5.params)
    #cocoEval5.summarize()


def fangyi(base, supts):
    # FP, FN, TP, TN = 0, grief1, grief2 = 0, sorrow = 0
    # for an image i,
    #   for a query q, stage6 prediction is q6
    #       if q6 is FP: FP += 1
    #           check if q1-5 exists TP or FP with lower score, if true, grief1+=1,
    #       if q6 is TP: TP += 1
    #           check if q1-5 exists same TP but with higher score, if true, sorrow+=1
    #   for a gt
    #       if gt is FN: FN += 1
    #           check if q1-5 exists TP, if true, grief2+=1,
    #
    # sorrow rate = sorrow / TP: among all TP, how many of them could have a higher confident score
    # the higher, the worse
    # grief rate = grief / (FP + FN): among all mistakes that q6 predicts, so many of them could have a correct detection
    # the higher, the worse
    # if q6 is a TN:
    # check if q1-5 exists TN and score < q6, if true, sorrow+=1

    p = base.params
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)

    base.params = p
    base._prepare()
    # loop through images, area range, max detection number
    base.ious = {(imgId, catId): base.computeIoU(imgId, catId) \
                     for imgId in p.imgIds
                     for catId in p.catIds}
    for supt in supts:
        supt.params = p
        supt._prepare()
        supt.ious = {(imgId, catId): supt.computeIoU(imgId, catId) \
                     for imgId in p.imgIds
                     for catId in p.catIds}

    maxDet = p.maxDets[-1]
    aRng = p.areaRng[0]

    res = []
    FP, FN, TP, TN, grief, sorrow, all_pred, all_gt, grief1, grief2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    FP2, FN2, TP2, TN2, = 0, 0, 0, 0

    for imgId in p.imgIds:
        total_q_num = 0
        for catId in p.catIds:
            gt = base._gts[imgId, catId]
            dt = base._dts[imgId, catId]
            if len(gt) == 0 and len(dt) == 0:
                res.append(None)
                continue

            for g in gt:
                if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                    g['_ignore'] = 1
                else:
                    g['_ignore'] = 0

            # sort dt highest score first, sort gt ignore last
            gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
            gt = [gt[i] for i in gtind]
            dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
            dt = [dt[i] for i in dtind[0:maxDet]]
            iscrowd = [int(o['iscrowd']) for o in gt]
            # load computed ious
            ious = base.ious[imgId, catId][:, gtind] if len(base.ious[imgId, catId]) > 0 else base.ious[
                imgId, catId]

            T = len(p.iouThrs)
            G = len(gt)
            D = len(dt)

            gtm = np.zeros((T, G))  # gtm positive number == TP number; gtm 0 number == FN number
            dtm = np.zeros((T, D))  # dtm positive number == TP number; dtm 0 number == FP number
            gtIg = np.array([g['_ignore'] for g in gt])
            dtIg = np.zeros((T, D))
            if not len(ious) == 0:
                for tind, t in enumerate(p.iouThrs):
                    for dind, d in enumerate(dt):
                        # information about best match so far (m=-1 -> unmatched)
                        iou = min([t, 1 - 1e-10])
                        m = -1
                        for gind, g in enumerate(gt):
                            # if this gt already matched, and not a crowd, continue
                            if gtm[tind, gind] > 0 and not iscrowd[gind]:
                                continue
                            # if dt matched to reg gt, and on ignore gt, stop
                            if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                                break
                            # continue to next gt unless better match made
                            if ious[dind, gind] < iou:
                                continue
                            # if match successful and best so far, store appropriately
                            iou = ious[dind, gind]
                            m = gind
                            g_reserve = g
                            iou_reserve = iou
                        # if match made store id of match for both dt and gt
                        if m == -1:
                            FP2 += 1
                            if check_supts_FP(imgId, catId, d, supts, t):
                                grief1 += 1
                                grief += 1
                            continue
                        dtIg[tind, dind] = gtIg[m]
                        if gtIg[m] == 0 and not (d['area'] < aRng[0] or d['area'] > aRng[1]):
                            TP2 += 1
                            if check_supts_TP(imgId, catId, g_reserve, d, supts, iou_reserve):
                                sorrow += 1
                        dtm[tind, dind] = gt[m]['id']
                        gtm[tind, m] = d['id']
            else:
                for dind, d in enumerate(dt):
                    FP2 += 1
                    if check_supts_FP(imgId, catId, d, supts, t):
                        grief1 += 1
                        grief += 1

            #if not len(ious) == 0:
            for gind, g in enumerate(gt):
                if gtm[tind, gind] == 0 and gtIg[gind] == 0:
                    FN2 += 1
                    if check_supts_FN(imgId, catId, g, supts, t):
                        grief2 += 1
                        grief += 1

            # set unmatched detections outside of area range to ignore
            a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]).reshape((1, len(dt)))
            dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))

            res.append({
                'image_id': imgId,
                'category_id': catId,
                'aRng': aRng,
                'maxDet': maxDet,
                'dtIds': [d['id'] for d in dt],
                'gtIds': [g['id'] for g in gt],
                'dtMatches': dtm,
                'gtMatches': gtm,
                'dtScores': [d['score'] for d in dt],
                'gtIgnore': gtIg,
                'dtIgnore': dtIg,})

            tps = np.logical_and(dtm, np.logical_not(dtIg))
            fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
            npig = np.count_nonzero(gtIg == 0) # number of gt
            fns = npig - tps.sum()
            fns2 = np.logical_and(np.logical_not(gtm), np.logical_not(gtIg))
            assert fns == fns2.sum()
            tps2 = np.logical_and(gtm, np.logical_not(gtIg))
            assert tps.sum() == tps2.sum()
            TP += tps.sum()
            FP += fps.sum()
            all_gt += npig
            all_pred = TP + FP
            FN += fns

    print('TP, FP, all_gt, FN', TP, FP, all_gt, FN)
    print('TP2:{}, FP2:{}, FN2:{}, sorrow:{}, sorrow rate:{}, grief1:{}, grief2:{}, grief:{}, grief1 rate{}, grief2 rate{} , grief rate{}'.format(
                TP2, FP2, FN2, sorrow, sorrow/TP2, grief1, grief2, grief, grief1/FP2, grief2/FN2, grief/(FP2 + FN2)))
    return res


def check_supts_FN(imgId, catId, g, supts, t):
    gt_realid = g['id']
    p = supts[0].params
    maxDet = p.maxDets[-1]
    aRng = p.areaRng[0]
    for supt in supts:
        gt = supt._gts[imgId, catId]
        dt = supt._dts[imgId, catId]
        if len(gt) == 0 and len(dt) == 0:
            continue

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = supt.ious[imgId, catId][:, gtind] if len(supt.ious[imgId, catId]) > 0 else supt.ious[
            imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)

        gtm = np.zeros((T, G))  # gtm positive number == TP number; gtm 0 number == FN number
        dtm = np.zeros((T, D))  # dtm positive number == TP number; dtm 0 number == FP number
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                        g_reserve = g
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    if gtIg[m] == 0 and not (d['area'] < aRng[0] or d['area'] > aRng[1]):
                        # TP
                        if g_reserve['id'] == gt_realid:
                            if d['score'] > 0.1:
                                return True
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']

    return False


def check_supts_FP(imgId, catId, d, supts, t):
    qry_id = d['fangyi_query']
    base_score = d['score']
    p = supts[0].params
    maxDet = p.maxDets[-1]
    aRng = p.areaRng[0]

    for supt in supts:
        gt = supt._gts[imgId, catId]
        dt = supt._dts[imgId, catId]
        if len(gt) == 0 and len(dt) == 0:
            continue

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = supt.ious[imgId, catId][:, gtind] if len(supt.ious[imgId, catId]) > 0 else supt.ious[
            imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)

        gtm = np.zeros((T, G))  # gtm positive number == TP number; gtm 0 number == FN number
        dtm = np.zeros((T, D))  # dtm positive number == TP number; dtm 0 number == FP number
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                        g_reserve = g
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        # FP
                        if qry_id == d['fangyi_query']:
                            if base_score > d['score']:
                                return True
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    if gtIg[m] == 0 and not (d['area'] < aRng[0] or d['area'] > aRng[1]):
                        if d['fangyi_query'] == qry_id:
                            return True
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
        else:
            for dind, d in enumerate(dt):
                # FP
                if qry_id == d['fangyi_query']:
                    if base_score > d['score']:
                        return True

    return False


def check_supts_TP(imgId, catId, g, d, supts, t):
    gt_realid = g['id']
    qry_id = d['fangyi_query']
    base_score = d['score']
    p = supts[0].params
    maxDet = p.maxDets[-1]
    aRng = p.areaRng[0]

    for supt in supts:
        gt = supt._gts[imgId, catId]
        dt = supt._dts[imgId, catId]
        if len(gt) == 0 or len(dt) == 0:
            continue

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = supt.ious[imgId, catId][:, gtind] if len(supt.ious[imgId, catId]) > 0 else supt.ious[imgId, catId]

        gtIg = np.array([g['_ignore'] for g in gt])
        if not len(ious) == 0:
            for dind, d in enumerate(dt):
                if not d['fangyi_query'] == qry_id:
                    continue
                for gind, g in enumerate(gt):
                    if not g['id'] == gt_realid:
                        continue
                    iou = ious[dind, gind]
                    if iou >= t:
                        if d['score'] > base_score:
                            return True

    return False


def preeval(base, dataset, iou_thrs):
    cocoGt = dataset.coco
    preds = dataset._det2json(base)
    cocoDt = cocoGt.loadRes(preds)

    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')  # initialize a class here
    cocoEval.params.catIds = dataset.cat_ids
    cocoEval.params.imgIds = dataset.img_ids
    cocoEval.params.maxDets = [300]
    cocoEval.params.iouThrs = [iou_thrs]
    cocoEval.params.areaRng = [[0, 10000000000.0]]
    cocoEval.params.areaRngLbl = ['all']

    return cocoEval


def motivation_verification(base, others, p):
    k_list = list(range(80))
    i_list = list(range(5000))
    T, A, M = 1, 1, 1
    R = 101
    K = 80
    precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
    recall = -np.ones((T, K, A, M))
    scores = -np.ones((T, R, K, A, M))
    for k in k_list:
        Nk = k * 5000
        E = [base[k + i*80] for i in i_list]
        E = [e for e in E if not e is None]
        dtScores = np.concatenate([e['dtScores'] for e in E])

        inds = np.argsort(-dtScores, kind='mergesort')
        dtScoresSorted = dtScores[inds]

        dtm = np.concatenate([e['dtMatches'] for e in E], axis=1)[:, inds]
        dtIg = np.concatenate([e['dtIgnore'] for e in E], axis=1)[:, inds]
        gtIg = np.concatenate([e['gtIgnore'] for e in E])
        npig = np.count_nonzero(gtIg == 0)

        if npig == 0:
            continue
        tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
        fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

        tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
        fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
        for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
            tp = np.array(tp)
            fp = np.array(fp)
            nd = len(tp)
            rc = tp / npig
            pr = tp / (fp+tp+np.spacing(1))
            q  = np.zeros((R,))
            ss = np.zeros((R,))

            if nd:
                recall[t,k,0,0] = rc[-1]
            else:
                recall[t,k,0,0] = 0

            pr = pr.tolist(); q = q.tolist()

            for i in range(nd-1, 0, -1):
                if pr[i] > pr[i-1]:
                    pr[i-1] = pr[i]

            inds = np.searchsorted(rc, p.recThrs, side='left')
            try:
                for ri, pi in enumerate(inds):
                    q[ri] = pr[pi]
                    ss[ri] = dtScoresSorted[pi]
            except:
                pass
            precision[t,:,k,0,0] = np.array(q)
            scores[t,:,k,0,0] = np.array(ss)

    eval = {
            'counts': [T, R, K, A, M],
            'precision': precision,
            'recall':   recall,
            'scores': scores,
    }
    return eval

def organize(outs):
    # FP, FN, TP, TN = 0, grief = 0, sorrow = 0
    # for an image i,
    #   for a query q, stage6 prediction is q6
    #       if q6 is FP: FP += 1
    #           check if q1-5 exists TP or FP with lower score, if true, grief+=1,
    #       if q6 is TP: TP += 1
    #           check if q1-5 exists same TP but with higher score, if true, sorrow+=1
    #   for a gt
    #       if gt is FN: FN += 1
    #           check if q1-5 exists TP, if true, grief+=1,
    #
    # sorrow rate = sorrow / TP: among all TP, how many of them could have a higher confident score
    # the higher, the worse
    # grief rate = grief / (FP + FN): among all mistakes that q6 predicts, so many of them could have a correct detection
    # the higher, the worse
    # if q6 is a TN:
    # check if q1-5 exists TN and score < q6, if true, sorrow+=1

    allres = []
    for img in outs:
        scale_factor = img['scale_factor']
        l_det_bboxes, l_det_labels = [], []
        for stage in range(6):
            cls_score, bboxes_list = img[stage]
            cls_score_per_img = torch.sigmoid(torch.Tensor(cls_score))
            scores_per_img, labels_per_img = torch.max(cls_score_per_img, dim=1)
            bbox_pred_per_img = torch.Tensor(bboxes_list)
            bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)

            queryid = torch.Tensor(list(range(len(scores_per_img)))).float()
            det_bboxes = torch.cat([bbox_pred_per_img, scores_per_img[:, None], labels_per_img[:, None].float(), queryid[:, None]], dim=1)  # （100， 5）
            det_labels = labels_per_img  # （100，）
            l_det_bboxes.append(det_bboxes[:, :, None])
            l_det_labels.append(det_labels[:, None])

        bboxes = torch.cat(l_det_bboxes, dim=2).permute(0, 2, 1).contiguous().numpy()
        labels = torch.cat(l_det_labels, dim=1).numpy()

        res = [bboxes[labels[:, -1] == i, :] for i in range(80)]
        allres.append(res)

    return allres

def postproc(outs, stageid=5, mmdet_implementation=False):
    res = []
    for img in outs:
        cls_score, bboxes_list = img[stageid]
        cls_score_per_img = torch.sigmoid(torch.Tensor(cls_score))

        if mmdet_implementation:
            scores_per_img, topk_indices = cls_score_per_img.flatten(0, 1).topk(100, sorted=False)
            labels_per_img = topk_indices % 80
            bbox_pred_per_img = torch.Tensor(bboxes_list)[topk_indices // 80]
        else:
            scores_per_img, labels_per_img = torch.max(cls_score_per_img, dim=1)
            bbox_pred_per_img = torch.Tensor(bboxes_list)

        scale_factor = img['scale_factor']
        bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
        queryid = torch.Tensor(list(range(len(scores_per_img)))).float()
        det_bboxes = [torch.cat([bbox_pred_per_img, scores_per_img[:, None], labels_per_img[:, None].float(), queryid[:, None]], dim=1)]
        det_labels = [labels_per_img]
        res.extend([bbox2result(det_bboxes[i], det_labels[i], 80) for i in range(1)])

    return res

def postproc_deform(outs, stageid=5, mmdet_implementation=False):
    res = []
    for img in outs:
        cls_score, bboxes_list = img[stageid]
        cls_score = torch.sigmoid(torch.Tensor(cls_score[0]))
        bbox_pred = torch.Tensor(bboxes_list[0])
        img_shape = img['img_shape']
        scale_factor = img['scale_factor']

        scores, det_labels = torch.max(cls_score, dim=1)
        bbox_pred = torch.Tensor(bbox_pred)

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

        det_bboxes /= det_bboxes.new_tensor(scale_factor)
        # det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        scores_per_img = scores
        labels_per_img = det_labels
        bbox_pred_per_img = det_bboxes

        queryid = torch.Tensor(list(range(len(scores_per_img)))).float()
        det_bboxes = [torch.cat([bbox_pred_per_img, scores_per_img[:, None], labels_per_img[:, None].float(), queryid[:, None]], dim=1)]
        det_labels = [labels_per_img]
        res.extend([bbox2result(det_bboxes[i], det_labels[i], 80) for i in range(1)])

    return res


def bbox_cxcywh_to_xyxy(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)


if __name__ == '__main__':
    main()
