# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import pdb
import re
from collections import OrderedDict
from functools import partial
from glob import glob
from multiprocessing import Pool
from tempfile import TemporaryDirectory
from typing import List, Optional, Sequence, Union

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet.registry import METRICS
from . import _polyiou


class VectorDouble(object):
    thisown = property(
        lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')

    def iterator(self):
        return _polyiou.VectorDouble_iterator(self)

    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _polyiou.VectorDouble___nonzero__(self)

    def __bool__(self):
        return _polyiou.VectorDouble___bool__(self)

    def __len__(self):
        return _polyiou.VectorDouble___len__(self)

    def __getslice__(self, i, j):
        return _polyiou.VectorDouble___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _polyiou.VectorDouble___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _polyiou.VectorDouble___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _polyiou.VectorDouble___delitem__(self, *args)

    def __getitem__(self, *args):
        return _polyiou.VectorDouble___getitem__(self, *args)

    def __setitem__(self, *args):
        return _polyiou.VectorDouble___setitem__(self, *args)

    def pop(self):
        return _polyiou.VectorDouble_pop(self)

    def append(self, x):
        return _polyiou.VectorDouble_append(self, x)

    def empty(self):
        return _polyiou.VectorDouble_empty(self)

    def size(self):
        return _polyiou.VectorDouble_size(self)

    def swap(self, v):
        return _polyiou.VectorDouble_swap(self, v)

    def begin(self):
        return _polyiou.VectorDouble_begin(self)

    def end(self):
        return _polyiou.VectorDouble_end(self)

    def rbegin(self):
        return _polyiou.VectorDouble_rbegin(self)

    def rend(self):
        return _polyiou.VectorDouble_rend(self)

    def clear(self):
        return _polyiou.VectorDouble_clear(self)

    def get_allocator(self):
        return _polyiou.VectorDouble_get_allocator(self)

    def pop_back(self):
        return _polyiou.VectorDouble_pop_back(self)

    def erase(self, *args):
        return _polyiou.VectorDouble_erase(self, *args)

    def __init__(self, *args):
        _polyiou.VectorDouble_swiginit(self, _polyiou.new_VectorDouble(*args))

    def push_back(self, x):
        return _polyiou.VectorDouble_push_back(self, x)

    def front(self):
        return _polyiou.VectorDouble_front(self)

    def back(self):
        return _polyiou.VectorDouble_back(self)

    def assign(self, n, x):
        return _polyiou.VectorDouble_assign(self, n, x)

    def resize(self, *args):
        return _polyiou.VectorDouble_resize(self, *args)

    def insert(self, *args):
        return _polyiou.VectorDouble_insert(self, *args)

    def reserve(self, n):
        return _polyiou.VectorDouble_reserve(self, n)

    def capacity(self):
        return _polyiou.VectorDouble_capacity(self)

    __swig_destroy__ = _polyiou.delete_VectorDouble


_polyiou.VectorDouble_swigregister(VectorDouble)


def iou_poly(p, q):
    return _polyiou.iou_poly(p, q)


def parse_gt(filename):
    objects = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
        for splitline in splitlines:
            object_struct = {}
            object_struct['name'] = splitline[8]
            if len(splitline) == 9:
                object_struct['difficult'] = 0
            elif len(splitline) == 10:
                object_struct['difficult'] = int(splitline[9])
            object_struct['bbox'] = [
                int(float(splitline[0])),
                int(float(splitline[1])),
                int(float(splitline[4])),
                int(float(splitline[5]))
            ]
            w = int(float(splitline[4])) - int(float(splitline[0]))
            h = int(float(splitline[5])) - int(float(splitline[1]))
            object_struct['area'] = w * h
            objects.append(object_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    recs = {}
    for imagename in imagenames:
        recs[imagename] = parse_gt(annopath.format(imagename))
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        records = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in records])
        difficult = np.array([x['difficult'] for x in records]).astype(bool)
        det = [False] * len(records)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox, 'difficult': difficult, 'det': det}

    detfile = detpath.format(classname)
    if os.path.getsize(detfile) == 0:
        return 0, 0

    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    bboxes = np.array([[float(z) for z in x[2:]] for x in splitlines])

    sorted_ind = np.argsort(-confidence)
    bboxes = bboxes[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        record = class_recs[image_ids[d]]
        bb = bboxes[d, :].astype(float)
        ovmax = -np.inf
        bbgt = record['bbox'].astype(float)

        if bbgt.size > 0:
            ixmin = np.maximum(bbgt[:, 0], bb[0])
            iymin = np.maximum(bbgt[:, 1], bb[1])
            ixmax = np.minimum(bbgt[:, 2], bb[2])
            iymax = np.minimum(bbgt[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (bbgt[:, 2] - bbgt[:, 0] + 1.) *
                   (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not record['difficult'][jmax]:
                if not record['det'][jmax]:
                    tp[d] = 1.
                    record['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec[-1], ap


def dota_eval(
        detpath,
        class_list,
        annopath=r'/data/my_code/dataset/DOTA/test/labelTxt/{:s}.txt',
        imagesetfile=r'/data/my_code/dataset/DOTA/test/testset.txt'):
    detpath += r'/{:s}.txt'

    max_len = max([len(x) for x in class_list])

    map_value, ap_list = 0.0, []
    print_msg = []
    for classname in class_list:
        _, ap = voc_eval(
            detpath,
            annopath,
            imagesetfile,
            classname,
            ovthresh=0.5,
            use_07_metric=True)
        print_msg.append(f'{classname:<{max_len}}: {ap * 100:.2f}')
        map_value += ap
        ap_list.append(ap)

    map_value = map_value / len(class_list) * 100
    print_msg.append(f'mAP: {map_value:.2f}')

    ap_list = np.array(ap_list)
    return print_msg, ap_list


nms_thresh = 0.3


def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])


def GetFileFromThisRootDir(dir_path, ext=None):
    allfiles = []
    need_ext_filter = ext is not None
    for root, _, files in os.walk(dir_path):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if need_ext_filter and extension in ext:
                allfiles.append(filepath)
            elif not need_ext_filter:
                allfiles.append(filepath)
    return allfiles


def py_cpu_nms_poly_fast(dets, thresh):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = VectorDouble([
            dets[i][0], dets[i][1], dets[i][2], dets[i][3],
            dets[i][4], dets[i][5], dets[i][6], dets[i][7]
        ])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou
        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except Exception:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]

        order = order[inds + 1]
    return keep


def nmsbynamedict(nameboxdict, nms, thresh):
    nameboxnmsdict = {x: [] for x in nameboxdict}
    for imgname in nameboxdict:
        keep = nms(np.array(nameboxdict[imgname]), thresh)
        outdets = []
        for index in keep:
            outdets.append(nameboxdict[imgname][index])
        nameboxnmsdict[imgname] = outdets
    return nameboxnmsdict


def poly2origpoly(poly, x, y, rate):
    origpoly = []
    for i in range(int(len(poly) / 2)):
        tmp_x = float(poly[i * 2] + x) / float(rate)
        tmp_y = float(poly[i * 2 + 1] + y) / float(rate)
        origpoly.append(tmp_x)
        origpoly.append(tmp_y)
    return origpoly


def mergesingle(dstpath, nms, fullname):
    name = custombasename(fullname)
    dstname = os.path.join(dstpath, name + '.txt')
    with open(fullname, 'r') as f_in:
        nameboxdict = {}
        lines = f_in.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
        for splitline in splitlines:
            subname = splitline[0]
            splitname = subname.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, subname)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])

            pattern2 = re.compile(r'__([\d+\.]+)__\d+___')
            rate = re.findall(pattern2, subname)[0]

            confidence = splitline[1]
            poly = list(map(float, splitline[2:]))
            origpoly = poly2origpoly(poly, x, y, rate)
            det = origpoly
            det.append(confidence)
            det = list(map(float, det))
            if oriname not in nameboxdict:
                nameboxdict[oriname] = []
            nameboxdict[oriname].append(det)
        nameboxnmsdict = nmsbynamedict(nameboxdict, nms, nms_thresh)
        with open(dstname, 'w') as f_out:
            for imgname in nameboxnmsdict:
                for det in nameboxnmsdict[imgname]:
                    confidence = det[-1]
                    bbox = det[0:-1]
                    outline = imgname + ' ' + str(confidence) + ' ' + ' '.join(map(str, bbox))
                    f_out.write(outline + '\n')


def merge_result(srcpath, dstpath):
    pool = Pool(16)
    filelist = GetFileFromThisRootDir(srcpath)
    mergesingle_fn = partial(mergesingle, dstpath, py_cpu_nms_poly_fast)
    pool.map(mergesingle_fn, filelist)


def rewrite(srcpath):
    txt_list = glob(f'{srcpath}/*.txt')
    for txt_file in txt_list:
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            line = line.strip('\n').split(' ')
            for index in sorted([4, 5, 8, 9], reverse=True):
                del line[index]
            new_lines.append(' '.join(line) + '\n')

        with open(txt_file, 'w') as f:
            f.writelines(new_lines)


def merge_and_eval(
        srcpath,
        dstpath,
        class_list,
        annopath=r'/data/my_code/dataset/DOTA/test/labelTxt/{:s}.txt',
        imagesetfile=r'/data/my_code/dataset/DOTA/test/testset.txt'):
    merge_result(srcpath, dstpath)
    rewrite(dstpath)
    return dota_eval(dstpath, class_list, annopath=annopath, imagesetfile=imagesetfile)


def bbox_to_poly(bbox: np.ndarray) -> List[float]:
    x1, y1, x2, y2 = bbox.tolist()
    return [x1, y1, x2, y1, x2, y2, x1, y2]


@METRICS.register_module()
class DOTAMetric(BaseMetric):
    default_prefix: Optional[str] = 'dota'

    def __init__(self,
                 ann_dir: Optional[str] = None,
                 imagesetfile: Optional[str] = None,
                 metric: Union[str, List[str]] = 'mAP',
                 eval_mode: str = '11points',
                 iou_thr: float = 0.5,
                 use_07_metric: bool = True,
                 merge_patches: bool = True,
                 nms_iou_thr: float = 0.3,
                 minus_one: bool = True,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric should be one of {allowed_metrics}, but got {metric}.')
        assert eval_mode in ['area', '11points'], \
            'Unrecognized mode, only "area" and "11points" are supported'

        self.ann_dir = ann_dir
        self.imagesetfile = imagesetfile
        self.metric = metric
        self.eval_mode = eval_mode
        self.iou_thr = iou_thr
        self.use_07_metric = use_07_metric
        self.merge_patches = merge_patches
        self.nms_iou_thr = nms_iou_thr
        self.minus_one = minus_one

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            img_id = str(data_sample.get('img_id', ''))
            img_path = data_sample.get('img_path', None)

            pred = data_sample['pred_instances']
            pred_bboxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()

            self.results.append(
                dict(
                    img_id=img_id,
                    img_path=img_path,
                    bboxes=pred_bboxes,
                    scores=pred_scores,
                    labels=pred_labels,
                ))

    def _infer_ann_dir(self, results: list) -> Optional[str]:
        for result in results:
            img_path = result.get('img_path', None)
            if img_path is None:
                continue
            norm_path = os.path.normpath(str(img_path))
            parts = norm_path.split(os.sep)
            if 'JPEGImages' in parts:
                jpeg_index = parts.index('JPEGImages')
                root = os.sep.join(parts[:jpeg_index])
                ann_dir = os.path.join(root, 'labelTxt')
                return ann_dir
        return None

    def _build_imagesetfile(self, ann_dir: str, out_file: str) -> str:
        image_names = sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(ann_dir)
            if name.endswith('.txt'))
        with open(out_file, 'w') as f:
            for name in image_names:
                f.write(name + '\n')
        return out_file

    def compute_metrics(self, results: list) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()
        class_names = self.dataset_meta['classes']

        ann_dir = self.ann_dir or self._infer_ann_dir(results)
        if ann_dir is None:
            raise RuntimeError(
                'DOTAMetric requires `ann_dir`, or valid `img_path` metadata '
                'to infer annotation directory.')

        with TemporaryDirectory(prefix='dota_eval_') as tmp_dir:
            class_txt = [[] for _ in class_names]

            for result in results:
                img_id = result['img_id']
                bboxes = result['bboxes']
                scores = result['scores']
                labels = result['labels']

                for bbox, score, label in zip(bboxes, scores, labels):
                    label = int(label)
                    if label < 0 or label >= len(class_names):
                        continue
                    poly = bbox_to_poly(np.asarray(bbox, dtype=np.float32))
                    bbox_str = ' '.join(str(x) for x in poly)
                    class_txt[label].append(f'{img_id} {float(score)} {bbox_str}\n')

            for class_name, txt in zip(class_names, class_txt):
                with open(os.path.join(tmp_dir, f'{class_name}.txt'), 'w') as f:
                    f.writelines(txt)

            imagesetfile = self.imagesetfile
            if imagesetfile is None:
                imagesetfile = self._build_imagesetfile(
                    ann_dir, os.path.join(tmp_dir, 'imageset.txt'))

            annopath = os.path.join(ann_dir, '{:s}.txt')
            print_msg, ap_array = merge_and_eval(
                tmp_dir,
                tmp_dir,
                class_names,
                annopath=annopath,
                imagesetfile=imagesetfile)

        for msg in print_msg:
            logger.info(msg)

        ap_list = [float(x) for x in np.asarray(ap_array).tolist()]
        mean_ap = float(np.mean(ap_list)) if len(ap_list) > 0 else 0.0

        eval_results = OrderedDict()
        eval_results['mAP'] = round(mean_ap, 3)
        eval_results['classwise_AP'] = [round(float(ap), 3) for ap in ap_list]

        return eval_results
