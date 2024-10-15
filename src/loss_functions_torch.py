import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import focal_loss


def end_point_error(input_flow, target_flow):
    input_flow = F.interpolate(input_flow, size=target_flow.shape[2:], mode='bilinear', align_corners=True)
    return torch.mean(torch.sqrt(torch.sum((input_flow - target_flow) ** 2, dim=-1)))


class MultiScaleEndPointError(nn.Module):
    def __init__(self, weights=None):
        super(MultiScaleEndPointError, self).__init__()
        if weights is None:
            weights = [1]
        self.weights = weights

    def forward(self, input_flow, target_flow):
        if type(input_flow) not in [tuple, list]:
            input_flow = [input_flow]
        loss = 0
        for output, weight in zip(input_flow, self.weights):
            loss += weight * end_point_error(output, target_flow)
        return loss


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
    from_logits: bool = False,
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    if from_logits:
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    else:
        p = inputs
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="none", from_logits: bool = False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(self, inputs, targets):
        return focal_loss(inputs, targets, self.alpha, self.gamma, self.reduction, from_logits=self.from_logits)


class EdgeDetailAggregateLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()

        laplacian = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]])
        laplacian = torch.stack((laplacian, laplacian), dim=-1).view(3, 3, 2, 1)
        self.add_buffer('laplacian', laplacian)

        fuse_kernel = torch.tensor([[6.0 / 10], [3.0 / 10], [1.0 / 10]]).view(1, 1, 3, 1)
        self.add_buffer('fuse_kernel', fuse_kernel)

    def forward(self, network_output: torch.Tensor, targets: torch.Tensor):
        targets = targets[:, :, :, 0:2]

        boundary_targets = F.conv2d(targets, self.laplacian, stride=1, padding='same')
        boundary_targets = torch.clamp(boundary_targets, 0, 1)
        boundary_targets = (boundary_targets > 0.1).to(boundary_targets.dtype)

        boundary_targets_x2 = F.conv2d(targets, self.laplacian, stride=2, padding='same')
        boundary_targets_x2 = torch.clamp(boundary_targets_x2, 0, 1)
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, scale_factor=2, mode='bilinear', align_corners=True)
        boundary_targets_x2_up = (boundary_targets_x2_up > 0.1).to(boundary_targets.dtype)

        boundary_targets_x4 = F.conv2d(targets, self.laplacian, stride=4, padding='same')
        boundary_targets_x4 = torch.clamp(boundary_targets_x4, 0, 1)
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, scale_factor=4, mode='bilinear', align_corners=True)
        boundary_targets_x4_up = (boundary_targets_x4_up > 0.1).to(boundary_targets.dtype)
        boundary_targets_pyramid = torch.cat((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim=-1)

        boundary_targets_pyramid = F.conv2d(boundary_targets_pyramid, self.fuse_kernel, stride=1, padding='same')
        boundary_targets_pyramid = (boundary_targets_pyramid > 0.1).to(boundary_targets_pyramid.dtype)

        return focal_loss(network_output, boundary_targets_pyramid, reduction='sum')
