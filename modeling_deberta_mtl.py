from torch import nn
import torch
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler, StableDropout, ACT2FN, DebertaV2OnlyMLMHead
from transformers import DebertaV2PreTrainedModel, DebertaV2Model
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from transformers.utils import logging
logger = logging.get_logger(__name__)

@dataclass
class MultipleChoiceModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    my_metrics: dict = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class DebertaForCompassMTL(DebertaV2PreTrainedModel):
    # _keys_to_ignore_on_load_missing = [r"position_ids"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, mlm_weight, mlm_clone, model_type):
        super().__init__(config)
        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.cls = DebertaV2OnlyMLMHead(config)
        self.mlm_weight = mlm_weight
        self.mlm_clone = mlm_clone
        self.model_type = model_type
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mlm_ids: Optional[torch.LongTensor] = None,
        mlm_masks: Optional[torch.LongTensor] = None,
        mlm_types: Optional[torch.LongTensor] = None,
        mlm_labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # flat_mlm_input_ids = mlm_input_ids.view(-1, mlm_input_ids.size(-1)) if mlm_input_ids is not None else None

        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        metrics_for_wandb = {}
        if self.model_type == "mtl":
            outputs = self.deberta(
                flat_input_ids,
                token_type_ids=flat_token_type_ids,
                attention_mask=flat_attention_mask,
                position_ids=flat_position_ids,
                inputs_embeds=flat_inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            encoder_layer = outputs[0]  # batch, seq, dim

            pooled_output = self.pooler(encoder_layer)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            reshaped_logits = logits.view(-1, num_choices)

            loss_fct = CrossEntropyLoss()
            loss_ce = loss_fct(reshaped_logits, labels)
            loss = loss_ce

        elif self.model_type == "mlm":
            outputs = self.deberta(
                mlm_ids,
                token_type_ids=mlm_types,
                attention_mask=mlm_masks,
                position_ids=flat_position_ids,
                inputs_embeds=flat_inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]  # batch, seq, dim
            reshaped_logits = self.cls(sequence_output)

            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(reshaped_logits.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            masked_lm_loss = masked_lm_loss * self.mlm_weight
            metrics_for_wandb["loss_mlm"] = masked_lm_loss
            loss = masked_lm_loss
        else:
            outputs = self.deberta(
                flat_input_ids,
                token_type_ids=flat_token_type_ids,
                attention_mask=flat_attention_mask,
                position_ids=flat_position_ids,
                inputs_embeds=flat_inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            encoder_layer = outputs[0]  # batch, seq, dim

            pooled_output = self.pooler(encoder_layer)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            reshaped_logits = logits.view(-1, num_choices)

            loss_fct = CrossEntropyLoss()
            loss_ce = loss_fct(reshaped_logits, labels)

            loss = loss_ce
            metrics_for_wandb["loss_ce"] = loss_ce

            mlm_outputs = self.deberta(
                mlm_ids,
                token_type_ids=mlm_types,
                attention_mask=mlm_masks,
                position_ids=flat_position_ids,
                inputs_embeds=flat_inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = mlm_outputs[0]  # batch, seq, dim
            prediction_scores = self.cls(sequence_output)

            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            masked_lm_loss = masked_lm_loss * self.mlm_weight
            metrics_for_wandb["loss_mlm"] = masked_lm_loss

            loss = loss + masked_lm_loss

        metrics_for_wandb["loss"] = loss

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            my_metrics=metrics_for_wandb,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )