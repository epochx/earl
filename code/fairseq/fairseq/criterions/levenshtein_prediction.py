# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
import logging

import torch
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from einops import rearrange

from typing import Dict, Optional, Callable
from fairseq import utils
from fairseq.logging.meters import AverageMeter
from fairseq.logging.metrics import log_custom

logger = logging.getLogger(__name__)


index2label = {
    0: "DELETE",
    1: "REPLACE",
    2: "REPLACER",
    3: "INSERT",
    4: "INSERTER",
}


# def f1(actual, predicted, label):

#     """ A helper function to calculate f1-score for the given `label` """

#     tp = np.sum((actual == label) & (predicted == label))
#     fp = np.sum((actual != label) & (predicted == label))
#     fn = np.sum((predicted != label) & (actual == label))

#     try:
#         precision = tp / (tp + fp)
#         recall = tp / (tp + fn)
#         f1 = 2 * (precision * recall) / (precision + recall)
#     except Exception as e:
#         print(e)
#         return 0
#     return f1


# def f1_macro(actual, predicted):

#     if isinstance(actual, list):
#         actual = np.asarray(actual)
#     if isinstance(predicted, list):
#         predicted = np.asarray(predicted)

#     assert actual.size == predicted.size

#     # F1 = 2 * (precision * recall) / (precision + recall)

#     # `macro` f1- unweighted mean of f1 per label
#     return np.mean(
#         [f1(actual, predicted, label) for label in np.unique(actual)]
#     )


def get_confusion_matrix(actual, predicted, label):
    tp = torch.sum(
        (actual == label).long() & (predicted == label).long()
    ).item()
    fp = torch.sum(
        (actual != label).long() & (predicted == label).long()
    ).item()
    fn = torch.sum(
        (predicted != label).long() & (actual == label).long()
    ).item()
    output = {"tp": tp, "fp": fp, "fn": fn}
    return output


# class OutputMeter(Meter):
#     """Stores all values and stores metrics at the end"""

#     def __init__(self, name, round: Optional[int] = None):
#         assert name in METRICS
#         self.name = name
#         self.pred_storage = []
#         self.tgt_storage = []
#         self.reset()

#     def reset(self):
#         self.pred_storage = []
#         self.tgt_storage = []

#     def update(self, preds, tgts):
#         self.pred_storage.extend(preds)
#         self.tgt_storage.extend(tgts)

#     def state_dict(self):
#         return {
#             "pred_storage": self.pred_storage,
#             "tgt_storage": self.tgt_storage,
#         }

#     def load_state_dict(self, state_dict):
#         self.pred_storage = state_dict["pred_storage"]
#         self.tgt_storage = state_dict["tgt_storage"]

#     @property
#     def smoothed_value(self) -> float:
#         if self.name == "macro_f1":
#             value = f1_macro(self.tgt_storage, self.pred_storage)
#             return 100 * value
#         elif self.name == "f1":
#             raise NotImplementedError


@dataclass
class LevenshteinPredictionCriterionConfig(FairseqDataclass):
    levenshtein_prediction_head_name: str = field(
        default="levenshtein_prediction_head",
        metadata={"help": "name of the levenshtein prediction head to use"},
    )

    delta_x_head_name: str = field(
        default="delta_x_head",
        metadata={"help": "name of the delta_x  head to use"},
    )


@register_criterion(
    "levenshtein_prediction", dataclass=LevenshteinPredictionCriterionConfig
)
class LevenshteinPredictionCriterion(FairseqCriterion):
    def __init__(self, cfg: LevenshteinPredictionCriterionConfig, task):
        super().__init__(task)
        self.levenshtein_prediction_head_name = (
            cfg.levenshtein_prediction_head_name
        )
        self.delta_x_head_name = cfg.delta_x_head_name
        self.source_dictionary = task.source_dictionary
        self.label_dictionary = task.label_dictionary
        self.delta_x_loss = task.cfg.delta_x_loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, "classification_heads")
            and self.levenshtein_prediction_head_name
            in model.classification_heads
        ), "model must provide levenshtein prediction head for --criterion=levenshtein_prediction"

        masked_tokens = sample["mask_target"].ne(self.padding_idx)

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if masked_tokens.device == torch.device("cpu"):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(), masked_tokens, masked_tokens.new([True]),
            )

        if self.delta_x_loss:
            head_name = [
                self.levenshtein_prediction_head_name,
                self.delta_x_head_name,
            ]
        else:
            head_name = self.levenshtein_prediction_head_name

        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=head_name,
            masked_tokens=masked_tokens,
        )

        levenshtein_logits = logits[self.levenshtein_prediction_head_name]
        mask_logits = logits["masked_lm"]

        if self.delta_x_loss:
            delta_x_logits = logits[self.delta_x_head_name]

        # MLM loss
        mask_targets = sample["mask_target"]
        nmasks = 0
        if masked_tokens is not None:
            mask_targets = mask_targets[masked_tokens]
            nmasks = masked_tokens.int().sum()

        mask_loss = F.cross_entropy(
            mask_logits.view(-1, mask_logits.size(-1)),
            mask_targets.view(-1),
            ignore_index=self.padding_idx,
            reduction="sum",
        )

        loss = mask_loss / nmasks

        # regular levenshtein loss
        # targets = model.get_targets(sample, [logits]).view(-1)
        # model.get_targets simply returns sample["target"]
        # https://github.com/pytorch/fairseq/blob/5551a1995bea28e47b388dc21fe683efee2d53f6/fairseq/models/fairseq_model.py#L59
        levenshtein_targets = sample["levenshtein_target"].view(-1)
        sample_size = levenshtein_targets.numel()

        # lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        bsz, tgt_len, _ = levenshtein_logits.shape
        lprobs = rearrange(levenshtein_logits, "b t c -> (b t) c")

        levenshtein_targets[levenshtein_targets < 5] = 105
        levenshtein_targets = levenshtein_targets - 5
        # FIXME remove 4 that are included in preprocessing <s>, <unk>, <pad>, </s>, KEEP
        lev_pred_loss = (
            F.cross_entropy(
                lprobs, levenshtein_targets, reduction="none", ignore_index=100,
            )
            .view(bsz, tgt_len)
            .mean(1)
            .sum()
        )  # TODO add padding support

        loss += lev_pred_loss

        if self.delta_x_loss:
            delta_x_targets = sample["delta_x_target"]

            lprobs = F.log_softmax(delta_x_logits, dim=-1, dtype=torch.float32)

            # transform EOS into PAD
            delta_x_targets[delta_x_targets == 2] = self.source_dictionary.pad()

            # batch_size, seq_len
            bow_lls = torch.gather(lprobs, 1, delta_x_targets)

            target_masks = (
                delta_x_targets == self.source_dictionary.pad()
            ).float()
            inv_target_masks = 1 - target_masks
            inv_target_masks = inv_target_masks.to(
                device=levenshtein_targets.device
            )

            masked_bow_lls = bow_lls * inv_target_masks

            bow_ll = masked_bow_lls.mean(1).sum()

            loss += bow_ll

        logging_output = {}

        # mha & ffn regularization update
        if (
            hasattr(model.args, "mha_reg_scale_factor")
            and model.args.mha_reg_scale_factor != 0.0
        ):
            mha_reg_loss = model._get_adaptive_head_loss()
            loss += mha_reg_loss
            logging_output.update({"mha_reg_loss": mha_reg_loss})
        if (
            hasattr(model.args, "ffn_reg_scale_factor")
            and model.args.ffn_reg_scale_factor != 0.0
        ):
            ffn_reg_loss = model._get_adaptive_ffn_loss()
            loss += ffn_reg_loss
            logging_output.update({"ffn_reg_loss": ffn_reg_loss})

        logging_output.update(
            {
                "loss": loss.data,
                "mask_loss": mask_loss,
                "lev_pred_loss": lev_pred_loss.data,
                "nmasks": nmasks,
                "ntokens": sample["ntokens"],
                "ntokens_no_pad": levenshtein_targets[
                    levenshtein_targets != 100
                ].size(0),
                "nsentences": sample_size,
                "sample_size": sample_size,
            }
        )

        if self.delta_x_loss:
            logging_output.update(
                {"delta_x_loss": bow_ll.data,}
            )

        preds = levenshtein_logits.argmax(dim=-1).view(-1)
        logging_output["ncorrect"] = (
            preds[levenshtein_targets != 100]
            == levenshtein_targets[levenshtein_targets != 100]
        ).sum()

        # adding for f1 performance
        for index in range(5):
            real_targets = levenshtein_targets[
                levenshtein_targets == index
            ].cpu()
            real_preds = preds[levenshtein_targets == index].cpu()
            label = index2label[index]
            confusion_matrix = get_confusion_matrix(
                real_targets, real_preds, index
            )
            for key, value in confusion_matrix.items():
                logging_output[f"{label}_{key}"] = value

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        mask_loss_sum = sum(log.get("mask_loss", 0) for log in logging_outputs)
        lev_pred_loss_sum = sum(
            log.get("lev_pred_loss", 0) for log in logging_outputs
        )
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)

        nmasks = sum(log.get("nmasks", 0) for log in logging_outputs)

        ntokens_no_pad = sum(
            log.get("ntokens_no_pad", 0) for log in logging_outputs
        )
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        mha_reg_loss_sum = sum(
            log.get("mha_reg_loss", 0) for log in logging_outputs
        )
        ffn_reg_loss_sum = sum(
            log.get("ffn_reg_loss", 0) for log in logging_outputs
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        metrics.log_scalar(
            "mask_loss", mask_loss_sum / nmasks / math.log(2), nmasks, round=3
        )
        metrics.log_derived(
            "mask_ppl",
            lambda meters: utils.get_perplexity(meters["mask_loss"].avg),
        )

        metrics.log_scalar(
            "lev_pred_loss",
            lev_pred_loss_sum / sample_size / math.log(2),
            sample_size,
            round=3,
        )

        if mha_reg_loss_sum:
            metrics.log_scalar(
                "mha_reg_loss",
                mha_reg_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if ffn_reg_loss_sum:
            metrics.log_scalar(
                "ffn_reg_loss",
                ffn_reg_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss",
                lev_pred_loss_sum / ntokens / math.log(2),
                ntokens,
                round=3,
            )

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy",
                100.0 * ncorrect / ntokens_no_pad,
                ntokens_no_pad,
                round=1,
            )

            f1_scores = []
            for label in index2label.values():
                tp = sum(log.get(f"{label}_tp", 0) for log in logging_outputs)
                fp = sum(log.get(f"{label}_fp", 0) for log in logging_outputs)
                fn = sum(log.get(f"{label}_fn", 0) for log in logging_outputs)

                try:
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    f1 = 2 * (precision * recall) / (precision + recall)
                    if not isinstance(f1, torch.Tensor):
                        f1 = torch.tensor(f1)

                    f1 = torch.nan_to_num(f1)

                except ZeroDivisionError as e:
                    # text = f"LABEL({label})\tTP:{tp}\tFP:{fp}\tFN:{fn}"
                    # logger.info(
                    #     f"Exception({e})\nLABEL({label})\tTP:{tp}\tFP:{fp}\tFN:{fn}"
                    # )
                    f1 = 0.0

                f1_scores.append(f1)
                metrics.log_scalar(
                    f"{label}_f1", 100.0 * f1, sample_size, round=1,
                )
            f1_scores = torch.tensor(f1_scores)
            f1_score = f1_scores.sum() / f1_scores.numel()

            metrics.log_scalar(
                "f1", 100.0 * f1_score, sample_size, round=1,
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
