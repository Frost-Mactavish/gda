_base_ = ["dior.py"]

dataset_type = "VOCDataset"
data_root = "/data/my_code/dataset/VOC"
work_dir = "log/VOC/JOINT"

backend_args = None

model = dict(train_cfg=dict(rpn_proposal=dict(max_per_img=1000)))

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1000, 600), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(1000, 600), keep_ratio=True),
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
                    filter_cfg=dict(filter_empty_gt=True, min_size=5, bbox_min_size=5),
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
test_dataloader = val_dataloader
