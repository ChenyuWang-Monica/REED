# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss


@register_loss("unimol_global")
class UniMolLossGlobal(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.seed = task.seed
        self.dist_mean = 6.312581655060595
        self.dist_std = 3.3899264663911888

    def forward(self, model, sample, reduce=True):
        input_key = "net_input"
        target_key = "target"
        difficult_input_key = "difficult_net_input"
        difficult_target_key = "difficult_target"
        clean_input_key = "clean_net_input"

        masked_tokens = sample[target_key]["tokens_target"].ne(self.padding_idx)
        sample_size = masked_tokens.long().sum()

        difficult_masked_tokens = sample[difficult_target_key]["tokens_target"].ne(self.padding_idx)
        difficult_sample_size = difficult_masked_tokens.long().sum()

        (
            logits_encoder,
            encoder_distance,
            encoder_coord,
            x_norm,
            delta_encoder_pair_rep_norm,
            _,
        ) = model(**sample[input_key], encoder_masked_tokens=masked_tokens)

        clean_graph_rep = model(**sample[clean_input_key], output_global_rep_only=True)

        (
            difficult_logits_encoder,
            difficult_encoder_distance,
            difficult_encoder_coord,
            difficult_x_norm,
            difficult_delta_encoder_pair_rep_norm,
            _,
        ) = model(**sample[difficult_input_key], encoder_masked_tokens=difficult_masked_tokens,
                  clean_global_rep=clean_graph_rep)

        target = sample[target_key]["tokens_target"]
        if masked_tokens is not None:
            target = target[masked_tokens]
        masked_token_loss = F.nll_loss(
            F.log_softmax(logits_encoder, dim=-1, dtype=torch.float32),
            target,
            ignore_index=self.padding_idx,
            reduction="mean",
        )
        masked_pred = logits_encoder.argmax(dim=-1)
        masked_hit = (masked_pred == target).long().sum()
        masked_cnt = sample_size
        loss = masked_token_loss * self.args.masked_token_loss
        logging_output = {
            "sample_size": 1,
            "bsz": sample[target_key]["tokens_target"].size(0),
            "seq_len": sample[target_key]["tokens_target"].size(1)
                       * sample[target_key]["tokens_target"].size(0),
            "masked_token_loss": masked_token_loss.data,
            "masked_token_hit": masked_hit.data,
            "masked_token_cnt": masked_cnt,
        }

        difficult_target = sample[difficult_target_key]["tokens_target"]
        if difficult_masked_tokens is not None:
            difficult_target = difficult_target[difficult_masked_tokens]
        difficult_masked_token_loss = F.nll_loss(
            F.log_softmax(difficult_logits_encoder, dim=-1, dtype=torch.float32),
            difficult_target,
            ignore_index=self.padding_idx,
            reduction="mean",
        )
        difficult_masked_pred = difficult_logits_encoder.argmax(dim=-1)
        difficult_masked_hit = (difficult_masked_pred == difficult_target).long().sum()
        difficult_masked_cnt = difficult_sample_size
        difficult_loss = difficult_masked_token_loss * self.args.difficult_masked_token_loss
        logging_output["difficult_masked_token_loss"] = difficult_masked_token_loss.data
        logging_output["difficult_masked_token_hit"] = difficult_masked_hit.data
        logging_output["difficult_masked_token_cnt"] = difficult_masked_cnt

        if encoder_coord is not None:
            # real = mask + delta
            coord_target = sample[target_key]["coord_target"]
            masked_coord_loss = F.smooth_l1_loss(
                encoder_coord[masked_tokens].view(-1, 3).float(),
                coord_target[masked_tokens].view(-1, 3),
                reduction="mean",
                beta=1.0,
            )
            loss = loss + masked_coord_loss * self.args.masked_coord_loss
            # restore the scale of loss for displaying
            logging_output["masked_coord_loss"] = masked_coord_loss.data

            difficult_coord_target = sample[difficult_target_key]["coord_target"]
            difficult_masked_coord_loss = F.smooth_l1_loss(
                difficult_encoder_coord[difficult_masked_tokens].view(-1, 3).float(),
                difficult_coord_target[difficult_masked_tokens].view(-1, 3),
                reduction="mean",
                beta=1.0,
            )
            difficult_loss = difficult_loss + difficult_masked_coord_loss * self.args.difficult_masked_coord_loss
            # restore the scale of loss for displaying
            logging_output["difficult_masked_coord_loss"] = difficult_masked_coord_loss.data

        if encoder_distance is not None:
            dist_masked_tokens = masked_tokens
            masked_dist_loss = self.cal_dist_loss(
                sample, encoder_distance, dist_masked_tokens, target_key, normalize=True
            )
            loss = loss + masked_dist_loss * self.args.masked_dist_loss
            logging_output["masked_dist_loss"] = masked_dist_loss.data

            difficult_dist_masked_tokens = difficult_masked_tokens
            difficult_masked_dist_loss = self.cal_dist_loss(
                sample, difficult_encoder_distance, difficult_dist_masked_tokens, difficult_target_key, normalize=True
            )
            difficult_loss = difficult_loss + difficult_masked_dist_loss * self.args.difficult_masked_dist_loss
            logging_output["difficult_masked_dist_loss"] = difficult_masked_dist_loss.data

        if self.args.x_norm_loss > 0 and x_norm is not None:
            loss = loss + self.args.x_norm_loss * x_norm
            logging_output["x_norm_loss"] = x_norm.data

        if self.args.difficult_x_norm_loss > 0 and difficult_x_norm is not None:
            difficult_loss = difficult_loss + self.args.difficult_x_norm_loss * difficult_x_norm
            logging_output["difficult_x_norm_loss"] = difficult_x_norm.data

        if (
                self.args.delta_pair_repr_norm_loss > 0
                and delta_encoder_pair_rep_norm is not None
        ):
            loss = (
                    loss + self.args.delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm
            )
            logging_output[
                "delta_pair_repr_norm_loss"
            ] = delta_encoder_pair_rep_norm.data

        if (
                self.args.difficult_delta_pair_repr_norm_loss > 0
                and difficult_delta_encoder_pair_rep_norm is not None
        ):
            difficult_loss = (
                    difficult_loss + self.args.difficult_delta_pair_repr_norm_loss * difficult_delta_encoder_pair_rep_norm
            )
            logging_output[
                "difficult_delta_pair_repr_norm_loss"
            ] = difficult_delta_encoder_pair_rep_norm.data

        logging_output["loss"] = loss.data
        logging_output["difficult_loss"] = difficult_loss.data

        full_loss = loss + difficult_loss
        logging_output["full_loss"] = full_loss.data
        return full_loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        full_loss_sum = sum(log.get("full_loss", 0) for log in logging_outputs)

        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("full_loss", full_loss_sum / sample_size, sample_size, round=3)

        metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)

        masked_loss = sum(log.get("masked_token_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "masked_token_loss", masked_loss / sample_size, sample_size, round=3
        )
        difficult_masked_loss = sum(log.get("difficult_masked_token_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "difficult_masked_token_loss", difficult_masked_loss / sample_size, sample_size, round=3
        )

        masked_acc = sum(
            log.get("masked_token_hit", 0) for log in logging_outputs
        ) / sum(log.get("masked_token_cnt", 0) for log in logging_outputs)
        metrics.log_scalar("masked_acc", masked_acc, sample_size, round=3)
        difficult_masked_acc = sum(
            log.get("difficult_masked_token_hit", 0) for log in logging_outputs
        ) / sum(log.get("difficult_masked_token_cnt", 0) for log in logging_outputs)
        metrics.log_scalar("difficult_masked_acc", difficult_masked_acc, sample_size, round=3)

        masked_coord_loss = sum(
            log.get("masked_coord_loss", 0) for log in logging_outputs
        )
        if masked_coord_loss > 0:
            metrics.log_scalar(
                "masked_coord_loss",
                masked_coord_loss / sample_size,
                sample_size,
                round=3,
            )
        difficult_masked_coord_loss = sum(
            log.get("difficult_masked_coord_loss", 0) for log in logging_outputs
        )
        if difficult_masked_coord_loss > 0:
            metrics.log_scalar(
                "difficult_masked_coord_loss",
                difficult_masked_coord_loss / sample_size,
                sample_size,
                round=3,
            )

        masked_dist_loss = sum(
            log.get("masked_dist_loss", 0) for log in logging_outputs
        )
        if masked_dist_loss > 0:
            metrics.log_scalar(
                "masked_dist_loss", masked_dist_loss / sample_size, sample_size, round=3
            )
        difficult_masked_dist_loss = sum(
            log.get("difficult_masked_dist_loss", 0) for log in logging_outputs
        )
        if difficult_masked_dist_loss > 0:
            metrics.log_scalar(
                "difficult_masked_dist_loss", difficult_masked_dist_loss / sample_size, sample_size, round=3
            )

        x_norm_loss = sum(log.get("x_norm_loss", 0) for log in logging_outputs)
        if x_norm_loss > 0:
            metrics.log_scalar(
                "x_norm_loss", x_norm_loss / sample_size, sample_size, round=3
            )
        difficult_x_norm_loss = sum(log.get("difficult_x_norm_loss", 0) for log in logging_outputs)
        if difficult_x_norm_loss > 0:
            metrics.log_scalar(
                "difficult_x_norm_loss", difficult_x_norm_loss / sample_size, sample_size, round=3
            )

        delta_pair_repr_norm_loss = sum(
            log.get("delta_pair_repr_norm_loss", 0) for log in logging_outputs
        )
        if delta_pair_repr_norm_loss > 0:
            metrics.log_scalar(
                "delta_pair_repr_norm_loss",
                delta_pair_repr_norm_loss / sample_size,
                sample_size,
                round=3,
            )
        difficult_delta_pair_repr_norm_loss = sum(
            log.get("difficult_delta_pair_repr_norm_loss", 0) for log in logging_outputs
        )
        if difficult_delta_pair_repr_norm_loss > 0:
            metrics.log_scalar(
                "difficult_delta_pair_repr_norm_loss",
                difficult_delta_pair_repr_norm_loss / sample_size,
                sample_size,
                round=3,
            )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def cal_dist_loss(self, sample, dist, masked_tokens, target_key, normalize=False):
        dist_masked_tokens = masked_tokens
        masked_distance = dist[dist_masked_tokens, :]
        masked_distance_target = sample[target_key]["distance_target"][
            dist_masked_tokens
        ]
        # padding distance
        nb_masked_tokens = dist_masked_tokens.sum(dim=-1)
        masked_src_tokens = sample["net_input"]["src_tokens"].ne(self.padding_idx)
        masked_src_tokens_expanded = torch.repeat_interleave(masked_src_tokens, nb_masked_tokens, dim=0)
        #
        if normalize:
            masked_distance_target = (
                                             masked_distance_target.float() - self.dist_mean
                                     ) / self.dist_std
        masked_dist_loss = F.smooth_l1_loss(
            masked_distance[masked_src_tokens_expanded].view(-1).float(),
            masked_distance_target[masked_src_tokens_expanded].view(-1),
            reduction="mean",
            beta=1.0,
        )
        return masked_dist_loss


@register_loss("unimol_infer_global")
class UniMolInferGlobalLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.bos_idx = task.dictionary.bos()
        self.eos_idx = task.dictionary.eos()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        input_key = "net_input"
        target_key = "target"
        src_tokens = sample[input_key]["src_tokens"]
        token_mask = (src_tokens.ne(self.padding_idx) & src_tokens.ne(self.bos_idx) & src_tokens.ne(self.eos_idx))
        (
            encoder_rep,
            encoder_pair_rep,
        ) = model(**sample[input_key], features_only=True)
        sample_size = sample[input_key]["src_tokens"].size(0)
        encoder_rep_list = []
        encoder_pair_rep_list = []
        if 'pdb_id' in sample[target_key].keys():
            name_key = 'pdb_id'
        elif 'smi_name' in sample[target_key].keys():
            name_key = 'smi_name'
        else:
            raise NotImplementedError("No name key in the original data")

        for i in range(sample_size):  # rm padding bos eos token
            encoder_rep_list.append(encoder_rep[i][token_mask[i]].data.cpu().numpy())
            encoder_pair_rep_list.append(encoder_pair_rep[i][token_mask[i], :][:, token_mask[i]].data.cpu().numpy())
        logging_output = {
            "mol_repr_cls": encoder_rep[:, 0, :].data.cpu().numpy(),  # get cls token
            "atom_repr": encoder_rep_list,
            "pair_repr": encoder_pair_rep_list,
            "data_name": sample[target_key][name_key],
            "bsz": sample[input_key]["src_tokens"].size(0),
        }
        return 0, sample_size, logging_output
