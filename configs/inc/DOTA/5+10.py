_base_ = ["../_base_/dota.py"]

task = "5+10"
step = 1
ori_num_classes = 5
num_classes = 15

dataset_type = "DOTADataset"
data_root = f"/data/my_code/dataset/DOTA_xml/{task}"

ori_config = "configs/base/DOTA/5.py"
ori_checkpoint = "log/DOTA/5+10/STEP0/best_model.pth"

work_dir = f"log/DOTA/{task}/STEP{step}"
confidence_cache = f"{work_dir}/confidence_cache.pth"
load_from = load_from_weight = f"{work_dir}/temp.pth"

backend_args = None

model = dict(
    ori_setting=dict(
        ori_checkpoint_file=ori_checkpoint,
        ori_num_classes=ori_num_classes,
        num_classes=num_classes,
        ori_config_file=ori_config,
        load_from_weight=load_from_weight,
    ),
    current_dataset_setting=dict(
        dataset="voc",
        train_val=f"{data_root}/task{step}_trainval.txt",
        img_dir="/data/my_code/dataset/DOTA_xml/JPEGImages",
        use_class_special=True,
        cache_path=confidence_cache,
    ),
    roi_head=dict(bbox_head=dict(num_classes=num_classes)),
)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1333, 1024), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]
train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            datasets=[
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file=data_root + f"/task{step}_trainval.txt",
                    data_prefix=dict(sub_data_root=data_root + f"/task{step}_trainval"),
                    filter_cfg=dict(filter_empty_gt=False, min_size=5, bbox_min_size=5),
                    pipeline=train_pipeline,
                    backend_args=backend_args,
                )
            ]
        )
    )
)
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + f"/task{step}_test.txt",
        data_prefix=dict(sub_data_root=data_root + f"/task{step}_test"),
    )
)
test_dataloader = val_dataloader

optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1, decay_mult=1.0)})
)
