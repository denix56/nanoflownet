from typing import List, Optional, Literal

import numpy as np
import cv2

import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from src import flow_vis

from .nanoflownet_torch import NanoFlowNet
from src.loss_functions_torch import MultiScaleEndPointError, FocalLoss, EdgeDetailAggregateLoss


class NanoFlowNetModule(L.LightningModule):
    def __init__(self, detail_guidance: str, lr: float, loss_weights: List[float], dataset_size: List[int],
                 compile_mode: str = None):
        super().__init__()
        self.lr = lr
        self.loss_weights = loss_weights
        self.compile_mode = compile_mode

        self.net = NanoFlowNet()
        self.example_input_array = {
            'image': torch.rand(1, dataset_size[0], dataset_size[1], 2)
        }

        if self.compile_mode is not None and self.compile_mode != 'none':
            print("compile_mode", self.compile_mode)
            self.net = torch.compile(self.net, mode=self.compile_mode)
            print("Model compiled.")

        self.msep_loss = MultiScaleEndPointError()

        if detail_guidance == 'off':
            self.guide_loss = None
        elif detail_guidance == 'motion_boundaries':
            self.guide_loss = FocalLoss(from_logits=True, reduction='sum')
        elif detail_guidance == 'edge_detect':
            self.guide_loss = EdgeDetailAggregateLoss()
        else:
            raise ValueError(f'Unknown guidance: {detail_guidance}')

    def forward(self, image):
        return self.net(image)

    def training_step(self, batch, batch_idx):
        x, target = batch['image'], batch['target']
        out = self.net(x)
        loss = 0
        for o, w in zip(out[:-1], self.loss_weights):
            loss += w * self.msep_loss(o, target[:, :2])
        self.log('train/msep_loss_total', loss)
        if self.guide_loss is not None:
            guide_loss = self.guide_loss(out[-1], target[:, -1:])
            loss = loss + self.loss_weights[-1] * guide_loss
            self.log('train/guide_loss', guide_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch['image'], batch['target']
        output = self.net(x)
        if isinstance(output, tuple) or isinstance(output, list):
            output = output[0]

        # Log images
        if batch_idx == 0:
            output_vis = output.to(torch.float32)
            for k in range(len(output_vis)):
                output_flow = output_vis[k, :2].permute(1, 2, 0).cpu().numpy()
                flo_rgb = flow_vis.flow_to_color(output_flow, convert_to_bgr=False)

                gt_flow = target[k, :2].permute(1, 2, 0).cpu().numpy()
                gt_flo_rgb = flow_vis.flow_to_color(gt_flow, convert_to_bgr=False)

                flo_rgb = cv2.resize(flo_rgb, (gt_flo_rgb.shape[1], gt_flo_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

                img = np.concatenate([flo_rgb, gt_flo_rgb], axis=1).transpose(2, 0, 1)
                for logger in self.trainer.loggers:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        tb_logger.add_image(f'val/opt_flow/{k}', img, global_step=self.global_step)

        loss = self.msep_loss(output, target[:, :2])
        self.log('val/msep_loss', loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt
