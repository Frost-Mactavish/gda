_base_ = ["dior.py"]

dataset_type = "DOTADataset"
data_root = "/data/my_code/dataset/DOTA_xml"

backend_args = None

custom_hooks = [
    dict(
        type="SaveBestResultHook", key_indicator="dota/mAP", save_file="best_result.txt"
    )
]

model = dict(
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
        ),
        rcnn=dict(max_per_img=2000),
    ),
)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1333, 1024), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(1333, 1024), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            datasets=[
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file=data_root + "/ImageSets/Main/trainval.txt",
                    data_prefix=dict(sub_data_root=data_root + "/Annotations"),
                    filter_cfg=dict(filter_empty_gt=False, min_size=5, bbox_min_size=5),
                    pipeline=train_pipeline,
                    backend_args=backend_args,
                )
            ],
        ),
    ),
)

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "/ImageSets/Main/test.txt",
        data_prefix=dict(sub_data_root=data_root + "/Annotations"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
val_evaluator = dict(
    type="DOTAMetric",
    ann_dir="/data/my_code/dataset/DOTA/test/labelTxt",
    imagesetfile="/data/my_code/dataset/DOTA/test/testset.txt",
    iou_thr=0.5,
    use_07_metric=True,
    merge_patches=True,
    nms_iou_thr=0.3,
)
test_dataloader = val_dataloader
test_evaluator = val_evaluator
