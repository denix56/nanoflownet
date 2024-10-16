import torch

from lightning.pytorch.cli import LightningCLI


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    cli = LightningCLI(run=False)  # True by default
    # you'll have to call fit yourself:
    ckpt_path = None
    if 'continue' in cli.config:
        ckpt_path = cli.config['continue']
    cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)
