import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
import wandb

from omegaconf import OmegaConf
import hydra
from data import DataModule
from model import ColaModel

class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule=datamodule
    def on_validation_end(self, trainer, pl_module):
        # can be done on complete dataset also
        val_batch = next(iter(self.datamodule.val_dataloader()))
        sentences = val_batch["sentence"]

        # get the predictions
        logits = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        preds = torch.argmax(logits, 1)
        labels = val_batch["label"]

        # predicted and labelled data
        df = pd.DataFrame(
            {"Sentence": sentences, "Label": labels.numpy(), "Predicted": preds.numpy()}
        )

        # wrongly predicted data
        wrong_df = df[df["Label"] != df["Predicted"]]

        # Logging wrongly predicted dataframe as a table
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )

@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    cola_data = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    cola_model = ColaModel(cfg.model.name)
    # print(OmegaConf.to_yaml(cfg))

    root_dir = hydra.utils.get_original_cwd()
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{root_dir}/models",
        filename="best-checkpoint",
        monitor="valid/loss",
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    wandb_logger = WandbLogger(project="MLOps Basics", name='cola')#, offline=True)

    trainer = pl.Trainer(
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=cfg.training.max_epochs,
        # logger=pl.loggers.TensorBoardLogger(save_dir="logs/", name='cola'),
        logger=wandb_logger,
        callbacks=[checkpoint_callback,SamplesVisualisationLogger(cola_data), early_stopping_callback],
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
    )

    trainer.fit(cola_model,cola_data)

if __name__ == "__main__":
    main()
