from os.path import basename, join
import sys,os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import torch
from commode_utils.callback import PrintEpochResultCallback, UploadCheckpointCallback
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


def train(model: LightningModule, data_module: LightningDataModule,
          config: DictConfig):

    model_name = model.__class__.__name__
    dataset_name = basename(config.dataset.name)
    # tensorboard logger
    tensorlogger = TensorBoardLogger(config.ts_logger_path)
    # define model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=join(tensorlogger.log_dir, "checkpoints"),
        monitor="val_recall",
        mode='max',
        filename="{epoch:02d}-{step:02d}-{val_f1:.4f}-{val_loss:.4f}-{val_recall:.4f}",
        every_n_epochs=config.hyper_parameters.val_every_step,  #every_n_val_epochs=config.hyper_parameters.val_every_step,
        save_top_k=5,
    )
    upload_weights = UploadCheckpointCallback(
        join(tensorlogger.log_dir, "checkpoints"))

    early_stopping_callback = EarlyStopping(patience=config.hyper_parameters.patience,
                                            monitor="val_loss",
                                            verbose=True,
                                            mode="min")

    lr_logger = LearningRateMonitor("step")
    print_epoch_results = PrintEpochResultCallback(split_symbol="_",
                                                   after_test=False)

    gpu =1 if torch.cuda.is_available() else None
    trainer = Trainer(
        max_epochs=config.hyper_parameters.n_epochs,
        gradient_clip_val=config.hyper_parameters.clip_norm,
        deterministic=False, # True
        check_val_every_n_epoch=config.hyper_parameters.val_every_step,
        log_every_n_steps=config.hyper_parameters.log_every_n_steps,
        logger=[tensorlogger],
        gpus=gpu,
        enable_progress_bar=True,  # progress_bar_refresh_rate=config.hyper_parameters.progress_bar_refresh_rate,
        callbacks=[
            lr_logger, early_stopping_callback, checkpoint_callback,
            print_epoch_results, upload_weights
        ],
        resume_from_checkpoint=config.hyper_parameters.resume_from_checkpoint,
    )

    trainer.fit(model=model, datamodule=data_module)
    trainer.save_checkpoint(join(tensorlogger.log_dir, "checkpoints","final_model.ckpt"))
    trainer.test(model=model)
if __name__ == "__main__":
    train()
