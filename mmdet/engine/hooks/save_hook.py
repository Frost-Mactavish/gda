import os
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class SaveBestResultHook(Hook):
    def __init__(self, key_indicator="pascal_voc/mAP", save_file="best_result.txt"):
        self.key_indicator = key_indicator
        self.save_file = save_file
        self.best_score = None

    def after_val_epoch(self, runner, metrics):
        if self.key_indicator not in metrics:
            return

        current_score = metrics[self.key_indicator]
        
        if self.best_score is None or current_score > self.best_score:
            self.best_score = current_score
            save_path = os.path.join(runner.work_dir, self.save_file)

            prefix = self.key_indicator.rsplit("/", 1)[0]
            mAP = metrics[self.key_indicator]
            ap_list = metrics.get(f"{prefix}/classwise_AP", [])

            step = runner.cfg.get("step")
            if step > 0:
                task = str(runner.cfg.get("task"))

                num_old_classes = int(task.split("+")[0])
                num_classes = int(runner.cfg.num_classes)
                num_new_classes = num_classes - num_old_classes

                old_map = sum(ap_list[:num_old_classes]) / num_old_classes
                new_map = sum(ap_list[num_old_classes:]) / num_new_classes

                with open(save_path, "w") as f:
                    f.write(f"Best Epoch: {runner.epoch}\n")
                    f.write(f"Eval Results:\n")
                    f.write(f"  mAP: {mAP * 100:.1f}\n")
                    f.write(f"  mAP_Old: {old_map * 100:.1f}\n")
                    f.write(f"  mAP_New: {new_map * 100:.1f}\n")
                    f.write(f"  AP: {list(ap_list)}\n")
            else:
                with open(save_path, "w") as f:
                    f.write(f"Best Epoch: {runner.epoch}\n")
                    f.write(f"Eval Results:\n")
                    f.write(f"  mAP: {mAP * 100:.1f}\n")
                    f.write(f"  AP: {list(ap_list)}\n")

            # save necessary state_dict and meta.dataset_meta for storage efficiency
            model = runner.model
            if hasattr(model, "module"):
                model = model.module

            checkpoint = {
                "meta": {"dataset_meta": runner.train_dataloader.dataset.metainfo},
                "state_dict": model.state_dict(),
            }

            ckpt_path = os.path.join(runner.work_dir, "best_model.pth")
            torch.save(checkpoint, ckpt_path)
