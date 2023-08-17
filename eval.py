import os

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate, to_absolute_path

from src.model.trainer_forecast import Trainer as Model


@hydra.main(version_base=None, config_path="./conf/", config_name="config")
def main(conf):
    pl.seed_everything(conf.seed)

    checkpoint = to_absolute_path(conf.checkpoint)
    assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"

    model = Model.load_from_checkpoint(checkpoint, pretrained_model=None)

    trainer = pl.Trainer(
        logger=False,
        accelerator="gpu",
        devices=conf.gpus,
        max_epochs=1,
        limit_val_batches=conf.limit_val_batches,
        limit_test_batches=conf.limit_test_batches,
    )

    datamodule: pl.LightningDataModule = instantiate(conf.datamodule, test=conf.test)

    if not conf.test:
        trainer.validate(model, datamodule)
    else:
        trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
