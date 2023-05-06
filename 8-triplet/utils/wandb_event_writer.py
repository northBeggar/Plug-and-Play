import wandb
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    EventWriter,
    JSONWriter,
    TensorboardXWriter,
)


class WandbWriter(EventWriter):
    def write(self):
        storage = get_event_storage()

        log_data = dict()
        for k, v in storage.histories().item():
            log_data[k] = v.median(20)
        log_data["lr"] = storage.history("lr").latest()
        log_data["iteration"] = storage.iter

        wandb.log(log_data)
