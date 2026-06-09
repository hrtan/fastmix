# Copyright (c) 2023, Tri Dao.

import torch
import torch.nn as nn
# import xentropy_cuda_lib

# `all_gather_into_tensor` and `reduce_scatter_tensor` are new placeholders for
# `_all_gather_base` and `_reduce_scatter_base`. They require the most recent
# version of PyTorch. The following 2 lines are for backward compatibility with
# older PyTorch.
if "all_gather_into_tensor" not in dir(torch.distributed):
    torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base



class SoftmaxCrossEntropyLossFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits,
        labels,
        smoothing=0.0,
        ignored_index=-100,
        inplace_backward=False,
        process_group=None,
    ):
        """
        logits: (batch, vocab_size)
        labels: (batch,)
        If process_group is not None, we're doing Tensor Parallel: each process is responsible for
        one part of the vocab. The loss needs to be aggregated across processes.
        """
        batch, vocab_size = logits.shape
        assert labels.shape == (batch,)
        world_size = 1 if process_group is None else torch.distributed.get_world_size(process_group)
        ctx.total_classes = world_size * vocab_size

        if world_size == 1:
            losses = SoftmaxCrossEntropyLossFn._compute_loss(logits, labels, smoothing, ignored_index)
            labels_local = labels
            # Calculate lse for single - process scenario
            lse = torch.logsumexp(logits, dim=-1)
        else:
            rank = torch.distributed.get_rank(process_group)
            vocab_start_index, vocab_end_index = rank * vocab_size, (rank + 1) * vocab_size

            # Create a mask of valid vocab ids (1 means it needs to be masked).
            labels_mask = (labels < vocab_start_index) | (labels >= vocab_end_index)
            ignored_mask = labels == ignored_index
            labels_local = torch.where(ignored_mask, labels, labels - vocab_start_index)

            local_logits = logits
            losses = SoftmaxCrossEntropyLossFn._compute_loss(local_logits, labels_local, smoothing, ignored_index, ctx.total_classes)

            lse_local = torch.logsumexp(local_logits, dim=-1)
            assert lse_local.shape == (batch,)
            assert losses.shape == (batch,)
            losses.masked_fill_(ignored_mask, 0)

            lse_allgather = torch.empty(
                world_size, batch, dtype=lse_local.dtype, device=lse_local.device
            )
            torch.distributed.all_gather_into_tensor(
                lse_allgather, lse_local.contiguous(), group=process_group
            )
            handle_losses = torch.distributed.all_reduce(
                losses, op=torch.distributed.ReduceOp.SUM, group=process_group, async_op=True
            )
            lse = torch.logsumexp(lse_allgather, dim=0)

            rank_per_sample = torch.div(labels, vocab_size, rounding_mode="floor")
            lse_local = lse_allgather[
                rank_per_sample, torch.arange(batch, device=lse_allgather.device)
            ]

            handle_losses.wait()
            if smoothing == 0.0:
                losses += lse - lse_local
            else:
                losses += (1 - smoothing) * (lse - lse_local) + smoothing * (
                    lse - lse_allgather.sum(dim=0)
                )
            losses.masked_fill_(ignored_mask, 0)

        ctx.save_for_backward(logits, lse, labels_local)
        ctx.smoothing = smoothing
        ctx.ignored_index = ignored_index
        ctx.inplace_backward = inplace_backward
        return losses

    @staticmethod
    def _compute_loss(logits, labels, smoothing, ignored_index, total_classes=None):
        if total_classes is None:
            total_classes = logits.size(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        if smoothing == 0.0:
            losses = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        else:
            nll_loss = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            smooth_loss = -log_probs.sum(dim=-1) / total_classes
            losses = (1 - smoothing) * nll_loss + smoothing * smooth_loss
        losses.masked_fill_(labels == ignored_index, 0)
        return losses

    @staticmethod
    def backward(ctx, grad_loss):
        logits, lse, labels = ctx.saved_tensors
        grad_loss = grad_loss.contiguous()
        grad_loss.masked_fill_(labels == ctx.ignored_index, 0)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        grad_logits = probs
        index = labels.unsqueeze(-1)
        grad_logits.scatter_(dim=-1, index=index, src=grad_logits.gather(dim=-1, index=index) - 1)
        if ctx.smoothing > 0:
            grad_logits = (1 - ctx.smoothing) * grad_logits - ctx.smoothing / ctx.total_classes

        grad_logits = grad_logits * grad_loss.unsqueeze(-1)
        return grad_logits, None, None, None, None, None, None

# class SoftmaxCrossEntropyLossFn(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         logits,
#         labels,
#         smoothing=0.0,
#         ignored_index=-100,
#         inplace_backward=False,
#         process_group=None,
#     ):
#         """
#         logits: (batch, vocab_size)
#         labels: (batch,)
#         If process_group is not None, we're doing Tensor Parallel: each process is responsible for
#         one part of the vocab. The loss needs to be aggregated across processes.
#         """
#         batch, vocab_size = logits.shape
#         assert labels.shape == (batch,)
#         world_size = 1 if process_group is None else torch.distributed.get_world_size(process_group)
#         ctx.total_classes = world_size * vocab_size

#         if world_size == 1:
#             losses, lse = xentropy_cuda_lib.forward(logits, labels, smoothing)
#             losses.masked_fill_(labels == ignored_index, 0)
#             labels_local = labels
#         else:
#             rank = torch.distributed.get_rank(process_group)
#             vocab_start_index, vocab_end_index = rank * vocab_size, (rank + 1) * vocab_size

#             # Create a mask of valid vocab ids (1 means it needs to be masked).
#             labels_mask = (labels < vocab_start_index) | (labels >= vocab_end_index)
#             ignored_mask = labels == ignored_index
#             labels_local = torch.where(ignored_mask, labels, labels - vocab_start_index)

#             # For tensor parallel cross entropy with smoothing, we want to pass in the total number
#             # of classes so that smoothing can be applied correctly. If total_classes=-1, use the
#             # last dimension of the input tensor.
#             losses, lse_local = xentropy_cuda_lib.forward(
#                 logits, labels_local, smoothing, world_size * vocab_size
#             )
#             assert lse_local.shape == (batch,)
#             assert losses.shape == (batch,)
#             losses.masked_fill_(ignored_mask, 0)
#             # For labels == ignored_index, the loss is always 0.
#             # If there's no smoothing, if labels are in the vocab of this partition, losses contains
#             # lse_local - predicted logit, and 0 otherwise.
#             # If there's smoothing=0.1, for labels in the vocab of this partition, losses contains
#             # 0.9 * (lse_local - predicted logit) + 0.1 * (lse_local - sum logit / total_classes)
#             # For labels not in the vocab of this partition, losses contains
#             # 0.1 * (lse_local - sum logit / total_classes).

#             lse_allgather = torch.empty(
#                 world_size, batch, dtype=lse_local.dtype, device=lse_local.device
#             )
#             torch.distributed.all_gather_into_tensor(
#                 lse_allgather, lse_local.contiguous(), group=process_group
#             )
#             handle_losses = torch.distributed.all_reduce(
#                 losses, op=torch.distributed.ReduceOp.SUM, group=process_group, async_op=True
#             )
#             lse = torch.logsumexp(lse_allgather, dim=0)
#             # If there's no smoothing, the total losses are lse_local - predicted_logit,
#             # we just have to subtract the lse_local and add the lse (global).
#             # If there's smoothing=0.1, the total losses are
#             # 0.9 * (lse_local - predicted_logit) + 0.1 * (sum of all lse_local - sum logit / total_classes)
#             # We want 0.9 * (lse - predicted_logit) + 0.1 * (lse - sum logit / total_classes).
#             rank_per_sample = torch.div(labels, vocab_size, rounding_mode="floor")
#             lse_local = lse_allgather[
#                 rank_per_sample, torch.arange(batch, device=lse_allgather.device)
#             ]

#             handle_losses.wait()
#             if smoothing == 0.0:
#                 losses += lse - lse_local
#             else:
#                 losses += (1 - smoothing) * (lse - lse_local) + smoothing * (
#                     lse - lse_allgather.sum(dim=0)
#                 )
#             losses.masked_fill_(ignored_mask, 0)

#         ctx.save_for_backward(logits, lse, labels_local)
#         ctx.smoothing = smoothing
#         ctx.ignored_index = ignored_index
#         ctx.inplace_backward = inplace_backward
#         return losses

#     @staticmethod
#     def backward(ctx, grad_loss):
#         logits, lse, labels = ctx.saved_tensors
#         grad_loss = grad_loss.contiguous()
#         grad_loss.masked_fill_(labels == ctx.ignored_index, 0)
#         grad_logits = xentropy_cuda_lib.backward(
#             grad_loss, logits, lse, labels, ctx.smoothing, ctx.inplace_backward, ctx.total_classes
#         )
#         return grad_logits, None, None, None, None, None, None


class FusedCrossEntropyLoss2(nn.Module):
    def __init__(
        self,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.0,
        inplace_backward=True,
        process_group=None,
    ):
        super().__init__()
        if reduction not in ["mean", "none"]:
            raise NotImplementedError("Only support reduction = 'mean' or 'none'")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.inplace_backward = inplace_backward
        self.process_group = process_group

    def forward(self, input, target):
        assert input.is_cuda and target.is_cuda
        # SoftmaxCrossEntropyLoss implicitly casts to float
        if len(input.shape) == 3:
            input = input.view(-1, input.size(-1))
            target = target.view(-1)
        loss = SoftmaxCrossEntropyLossFn.apply(
            input,
            target,
            self.label_smoothing,
            self.ignore_index,
            self.inplace_backward,
            self.process_group,
        )
        if self.reduction == "mean":
            return loss.sum() / (target != self.ignore_index).sum()
        else:
            return loss



class FusedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.0,
        inplace_backward=True,
        process_group=None,
    ):
        super().__init__()
        if reduction not in ["mean", "none"]:
            raise NotImplementedError("Only support reduction = 'mean' or 'none'")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.inplace_backward = inplace_backward
        self.process_group = process_group

    def forward(self, input, target, sample_weights=None):
        assert input.is_cuda and target.is_cuda
        # SoftmaxCrossEntropyLoss implicitly casts to float
        if len(input.shape) == 3:
            batch_size, sequence_length = input.shape[:2]
            input = input.view(-1, input.size(-1))
            target = target.view(-1)
        else:
            assert len(input.shape) == 3, "Input must be 2D or 3D tensor in loss calculator, got {}".format((input.shape))
        
        loss = SoftmaxCrossEntropyLossFn.apply(
            input,
            target,
            self.label_smoothing,
            self.ignore_index,
            self.inplace_backward,
            self.process_group,
        )
        if sample_weights is not None:
            # 扩展 sample_weights 以匹配 loss 的形状
            sample_weights = sample_weights.view(-1, 1).expand(-1, sequence_length).contiguous().view(-1)
            # 应用样本权重
            loss = loss * sample_weights
        valid_mask = (target != self.ignore_index)
        if self.reduction == "mean":
            if sample_weights is not None:
                # 计算加权平均损失
                weighted_sum_loss = (loss * valid_mask).sum()
                weighted_sum_weights = (sample_weights * valid_mask).sum()
                return weighted_sum_loss / weighted_sum_weights
            else:
                return loss.sum() / valid_mask.sum()
        else:
            return loss