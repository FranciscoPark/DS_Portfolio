import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
from transformers import BartModel, AutoModel

from .loss import FocalLoss

# import custom functions and classes
from parse_config import read_config
from logger import activate_wandb, compare_and_save, log_wandb
from .metric import Summary, AverageMeter, compute_metrics_custom, LightningLosses

CFG = read_config()


class BART_COMP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.transformer = BartModel.from_pretrained("gogamza/kobart-base-v2")
        self.lr = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.softmax = nn.Softmax(dim=1)
        self.train_losses = LightningLosses()
        self.train_top1_acc = torchmetrics.Accuracy()
        self.train_topk_acc = torchmetrics.Accuracy(top_k=3)
        self.train_f1 = torchmetrics.F1Score(num_classes=232, average="macro")
        self.val_losses = LightningLosses()
        self.val_top1_acc = torchmetrics.Accuracy()
        self.val_topk_acc = torchmetrics.Accuracy(top_k=3)
        self.val_f1 = torchmetrics.F1Score(num_classes=232, average="macro")
        self.test_top1_acc = torchmetrics.Accuracy()
        self.test_topk_acc = torchmetrics.Accuracy(top_k=3)
        self.test_f1 = torchmetrics.F1Score(num_classes=232, average="macro")
        self.classification_head = nn.Sequential(
            nn.Dropout(0.1, inplace=False),
            nn.Linear(768, 768, bias=True),
            nn.GELU(),
            nn.Dropout(0.1, inplace=False),
            nn.Linear(768, 232, bias=True),
        )

    def forward(self, input_ids, attention_mask):
        out = self.transformer(input_ids, attention_mask=attention_mask)["last_hidden_state"][
            :, 0, :
        ]
        out = self.classification_head(out)
        return out

    def configure_optimizers(self):
        if self.scheduler is not None:
            lr_scheduler = self.scheduler
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        if self.scheduler is None:
            return {"optimizer": self.optimizer}
        else:
            return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}

    def training_step(self, train_batch, batch_idx):
        out = self.forward(train_batch["input_ids"], train_batch["attention_mask"])
        loss = self.criterion(out, train_batch["labels"])
        prediction = torch.argmax(out, dim=1)

        self.train_losses.forward(loss.item())
        acc1 = self.train_top1_acc(prediction, train_batch["labels"])
        acck = self.train_topk_acc.forward(out, train_batch["labels"])
        f1 = self.train_f1(prediction, train_batch["labels"])

        log_wandb(
            epoch=self.current_epoch,
            step=self.global_step,  # change to actual step number
            learning_rate=self.lr,  # 제대로 되는지 확인 -> 안됨
            loss=self.train_losses.compute().item(),
            topk_integer=3,
            accuracy=self.train_top1_acc.compute().item(),
            topk_accuracy=self.train_topk_acc.compute().item(),
            f1=self.train_f1.compute().item(),
            is_train=True,
        )
        return {"loss": loss, "train_acc1": acc1, "train_acck": acck, "train_f1": f1}

    def training_epoch_end(self, outputs):
        epoch_loss = self.train_losses.compute().item()
        epoch_acc1 = self.train_top1_acc.compute().item()

        self.train_top1_acc.reset()
        self.train_topk_acc.reset()
        self.train_f1.reset()
        self.train_losses.reset()

        print(f"\ntrain_loss: {round(epoch_loss, 4)}, train_acc1: {round(epoch_acc1, 4)}")
        print()

    def validation_step(self, val_batch, batch_idx):
        out = self.forward(val_batch["input_ids"], val_batch["attention_mask"])
        loss = self.criterion(out, val_batch["labels"])
        prediction = torch.argmax(out, dim=1)

        self.val_losses.forward(loss.item())
        acc1 = self.val_top1_acc(prediction, val_batch["labels"])
        acck = self.val_topk_acc.forward(out, val_batch["labels"])
        f1 = self.val_f1(prediction, val_batch["labels"])

        log_wandb(
            epoch=self.current_epoch,
            step=self.global_step,
            learning_rate=self.lr,
            loss=self.val_losses.compute().item(),
            topk_integer=3,
            accuracy=self.val_top1_acc.compute().item(),
            topk_accuracy=self.val_topk_acc.compute().item(),
            f1=self.val_f1.compute().item(),
            is_train=False,
        )
        return {"val_loss": loss, "val_acc1": acc1, "val_acck": acck, "val_f1": f1}

    def validation_epoch_end(self, outputs):
        epoch_loss = self.val_losses.compute().item()
        epoch_acc1 = self.val_top1_acc.compute().item()

        self.val_top1_acc.reset()
        self.val_topk_acc.reset()
        self.val_f1.reset()
        self.val_losses.reset()

        print(f"\nval_loss: {round(epoch_loss, 4)}, val_acc1: {round(epoch_acc1, 4)}")
        print()
        return {
            "val_loss": epoch_loss,
            "val_acc1": epoch_acc1,
        }

    def test_step(self, test_batch, batch_idx):
        out = self.forward(test_batch["input_ids"], test_batch["attention_mask"])
        prediction = torch.argmax(out, dim=1)

        acc1 = self.test_top1_acc(prediction, test_batch["labels"])
        acck = self.test_topk_acc.forward(out, test_batch["labels"])
        f1 = self.test_f1(prediction, test_batch["labels"])

        return {"test_acc1": acc1, "test_f1": f1, "test_acck": acck}

    def predict_step(self, predict_batch, batch_idx):
        out = self.forward(predict_batch["input_ids"], predict_batch["attention_mask"])
        logits = self.softmax(out)
        prediction = torch.argmax(out, dim=1)
        return {
            "id": predict_batch["labels"],
            "prediction": prediction,
            "logits": logits,
        }


class ELECTRA_COMP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.transformer = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.lr = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.softmax = nn.Softmax(dim=1)
        self.train_losses = LightningLosses()
        self.train_top1_acc = torchmetrics.Accuracy()
        self.train_topk_acc = torchmetrics.Accuracy(top_k=3)
        self.train_f1 = torchmetrics.F1Score(num_classes=232, average="macro")
        self.val_losses = LightningLosses()
        self.val_top1_acc = torchmetrics.Accuracy()
        self.val_topk_acc = torchmetrics.Accuracy(top_k=3)
        self.val_f1 = torchmetrics.F1Score(num_classes=232, average="macro")
        self.test_top1_acc = torchmetrics.Accuracy()
        self.test_topk_acc = torchmetrics.Accuracy(top_k=3)
        self.test_f1 = torchmetrics.F1Score(num_classes=232, average="macro")
        self.classification_head = nn.Sequential(
            nn.Dropout(0.1, inplace=False),
            nn.Linear(768, 768, bias=True),
            nn.GELU(),
            nn.Dropout(0.1, inplace=False),
            nn.Linear(768, 232, bias=True),
        )

    def forward(self, input_ids, attention_mask):
        out = self.transformer(input_ids, attention_mask=attention_mask)["last_hidden_state"][
            :, 0, :
        ]
        out = self.classification_head(out)
        return out

    def configure_optimizers(self):
        if self.scheduler is not None:
            lr_scheduler = self.scheduler
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        if self.scheduler is None:
            return {"optimizer": self.optimizer}
        else:
            return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}

    def training_step(self, train_batch, batch_idx):
        out = self.forward(train_batch["input_ids"], train_batch["attention_mask"])
        loss = self.criterion(out, train_batch["labels"])
        prediction = torch.argmax(out, dim=1)

        self.train_losses.forward(loss.item())
        acc1 = self.train_top1_acc(prediction, train_batch["labels"])
        acck = self.train_topk_acc.forward(out, train_batch["labels"])
        f1 = self.train_f1(prediction, train_batch["labels"])

        log_wandb(
            epoch=self.current_epoch,
            step=self.global_step,  # change to actual step number
            learning_rate=self.lr,  # 제대로 되는지 확인 -> 안됨
            loss=self.train_losses.compute().item(),
            topk_integer=3,
            accuracy=self.train_top1_acc.compute().item(),
            topk_accuracy=self.train_topk_acc.compute().item(),
            f1=self.train_f1.compute().item(),
            is_train=True,
        )
        return {"loss": loss, "train_acc1": acc1, "train_acck": acck, "train_f1": f1}

    def training_epoch_end(self, outputs):
        epoch_loss = self.train_losses.compute().item()
        epoch_acc1 = self.train_top1_acc.compute().item()

        self.train_top1_acc.reset()
        self.train_topk_acc.reset()
        self.train_f1.reset()
        self.train_losses.reset()

        print(f"\ntrain_loss: {round(epoch_loss, 4)}, train_acc1: {round(epoch_acc1, 4)}")
        print()

    def validation_step(self, val_batch, batch_idx):
        out = self.forward(val_batch["input_ids"], val_batch["attention_mask"])
        loss = self.criterion(out, val_batch["labels"])
        prediction = torch.argmax(out, dim=1)

        self.val_losses.forward(loss.item())
        acc1 = self.val_top1_acc(prediction, val_batch["labels"])
        acck = self.val_topk_acc.forward(out, val_batch["labels"])
        f1 = self.val_f1(prediction, val_batch["labels"])

        log_wandb(
            epoch=self.current_epoch,
            step=self.global_step,
            learning_rate=self.lr,
            loss=self.val_losses.compute().item(),
            topk_integer=3,
            accuracy=self.val_top1_acc.compute().item(),
            topk_accuracy=self.val_topk_acc.compute().item(),
            f1=self.val_f1.compute().item(),
            is_train=False,
        )
        return {"val_loss": loss, "val_acc1": acc1, "val_acck": acck, "val_f1": f1}

    def validation_epoch_end(self, outputs):
        epoch_loss = self.val_losses.compute().item()
        epoch_acc1 = self.val_top1_acc.compute().item()

        self.val_top1_acc.reset()
        self.val_topk_acc.reset()
        self.val_f1.reset()
        self.val_losses.reset()

        print(f"\nval_loss: {round(epoch_loss, 4)}, val_acc1: {round(epoch_acc1, 4)}")
        print()
        return {
            "val_loss": epoch_loss,
            "val_acc1": epoch_acc1,
        }

    def test_step(self, test_batch, batch_idx):
        out = self.forward(test_batch["input_ids"], test_batch["attention_mask"])
        prediction = torch.argmax(out, dim=1)

        acc1 = self.test_top1_acc(prediction, test_batch["labels"])
        acck = self.test_topk_acc.forward(out, test_batch["labels"])
        f1 = self.test_f1(prediction, test_batch["labels"])

        return {"test_acc1": acc1, "test_f1": f1, "test_acck": acck}

    def predict_step(self, predict_batch, batch_idx):
        out = self.forward(predict_batch["input_ids"], predict_batch["attention_mask"])
        logits = self.softmax(out)
        prediction = torch.argmax(out, dim=1)
        return {
            "id": predict_batch["labels"],
            "prediction": prediction,
            "logits": logits,
        }


class BIGBIRD_COMP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.transformer = AutoModel.from_pretrained("monologg/kobigbird-bert-base")
        self.lr = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.softmax = nn.Softmax(dim=1)
        self.train_losses = LightningLosses()
        self.train_top1_acc = torchmetrics.Accuracy()
        self.train_topk_acc = torchmetrics.Accuracy(top_k=3)
        self.train_f1 = torchmetrics.F1Score(num_classes=232, average="macro")
        self.val_losses = LightningLosses()
        self.val_top1_acc = torchmetrics.Accuracy()
        self.val_topk_acc = torchmetrics.Accuracy(top_k=3)
        self.val_f1 = torchmetrics.F1Score(num_classes=232, average="macro")
        self.test_top1_acc = torchmetrics.Accuracy()
        self.test_topk_acc = torchmetrics.Accuracy(top_k=3)
        self.test_f1 = torchmetrics.F1Score(num_classes=232, average="macro")
        self.classification_head = nn.Sequential(
            nn.Dropout(0.1, inplace=False),
            nn.Linear(768, 768, bias=True),
            nn.GELU(),
            nn.Dropout(0.1, inplace=False),
            nn.Linear(768, 232, bias=True),
        )

    def forward(self, input_ids, attention_mask):
        out = self.transformer(input_ids, attention_mask=attention_mask)["pooler_output"]
        out = self.classification_head(out)
        return out

    def configure_optimizers(self):
        if self.scheduler is not None:
            lr_scheduler = self.scheduler
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        if self.scheduler is None:
            return {"optimizer": self.optimizer}
        else:
            return {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}

    def training_step(self, train_batch, batch_idx):
        out = self.forward(train_batch["input_ids"], train_batch["attention_mask"])
        loss = self.criterion(out, train_batch["labels"])
        prediction = torch.argmax(out, dim=1)

        self.train_losses.forward(loss.item())
        acc1 = self.train_top1_acc(prediction, train_batch["labels"])
        acck = self.train_topk_acc.forward(out, train_batch["labels"])
        f1 = self.train_f1(prediction, train_batch["labels"])

        log_wandb(
            epoch=self.current_epoch,
            step=self.global_step,  # change to actual step number
            learning_rate=self.lr,  # 제대로 되는지 확인 -> 안됨
            loss=self.train_losses.compute().item(),
            topk_integer=3,
            accuracy=self.train_top1_acc.compute().item(),
            topk_accuracy=self.train_topk_acc.compute().item(),
            f1=self.train_f1.compute().item(),
            is_train=True,
        )
        return {"loss": loss, "train_acc1": acc1, "train_acck": acck, "train_f1": f1}

    def training_epoch_end(self, outputs):
        epoch_loss = self.train_losses.compute().item()
        epoch_acc1 = self.train_top1_acc.compute().item()

        self.train_top1_acc.reset()
        self.train_topk_acc.reset()
        self.train_f1.reset()
        self.train_losses.reset()

        print(f"\ntrain_loss: {round(epoch_loss, 4)}, train_acc1: {round(epoch_acc1, 4)}")
        print()

    def validation_step(self, val_batch, batch_idx):
        out = self.forward(val_batch["input_ids"], val_batch["attention_mask"])
        loss = self.criterion(out, val_batch["labels"])
        prediction = torch.argmax(out, dim=1)

        self.val_losses.forward(loss.item())
        acc1 = self.val_top1_acc(prediction, val_batch["labels"])
        acck = self.val_topk_acc.forward(out, val_batch["labels"])
        f1 = self.val_f1(prediction, val_batch["labels"])

        log_wandb(
            epoch=self.current_epoch,
            step=self.global_step,
            learning_rate=self.lr,
            loss=self.val_losses.compute().item(),
            topk_integer=3,
            accuracy=self.val_top1_acc.compute().item(),
            topk_accuracy=self.val_topk_acc.compute().item(),
            f1=self.val_f1.compute().item(),
            is_train=False,
        )
        return {"val_loss": loss, "val_acc1": acc1, "val_acck": acck, "val_f1": f1}

    def validation_epoch_end(self, outputs):
        epoch_loss = self.val_losses.compute().item()
        epoch_acc1 = self.val_top1_acc.compute().item()

        self.val_top1_acc.reset()
        self.val_topk_acc.reset()
        self.val_f1.reset()
        self.val_losses.reset()

        print(f"\nval_loss: {round(epoch_loss, 4)}, val_acc1: {round(epoch_acc1, 4)}")
        print()
        return {
            "val_loss": epoch_loss,
            "val_acc1": epoch_acc1,
        }

    def test_step(self, test_batch, batch_idx):
        out = self.forward(test_batch["input_ids"], test_batch["attention_mask"])
        prediction = torch.argmax(out, dim=1)

        acc1 = self.test_top1_acc(prediction, test_batch["labels"])
        acck = self.test_topk_acc.forward(out, test_batch["labels"])
        f1 = self.test_f1(prediction, test_batch["labels"])

        return {"test_acc1": acc1, "test_f1": f1, "test_acck": acck}

    def predict_step(self, predict_batch, batch_idx):
        out = self.forward(predict_batch["input_ids"], predict_batch["attention_mask"])
        logits = self.softmax(out)
        prediction = torch.argmax(out, dim=1)
        return {
            "id": predict_batch["labels"],
            "prediction": prediction,
            "logits": logits,
        }


from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    ElectraPreTrainedModel,
    ElectraModel,
    ElectraConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput


class FCLayer(nn.Module):
    """ FC Layer for classification: https://github.com/monologg/R-BERT/blob/254aa5090c94542635b2df46ed16d596facb6556/model.py#L6-L18 """

    def __init__(self, input_dim, output_dim, dropout_rate=0.1, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = (
            nn.GELU()
        )  # electra uses gelu whereas BERT or Roberta used tanh in fully connected layer

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.activation(x)
        return self.linear(x)


# https://github.com/huggingface/transformers/blob/ddbb485c41e63dfbd7c2667e01bbe2ab5b3fe660/src/transformers/models/electra/modeling_electra.py#L932-L951
class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.gelu = nn.GELU()

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = self.gelu(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


# https://github.com/huggingface/transformers/blob/ddbb485c41e63dfbd7c2667e01bbe2ab5b3fe660/src/transformers/models/electra/modeling_electra.py#L961-L1047
class ElectraConcatModel(ElectraPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.dropout_rate = config.hidden_dropout_prob
        self.electra = ElectraModel(config)
        self.classifier = ElectraClassificationHead(config)
        self.proj_fc_layer = FCLayer(config.hidden_size * 4, config.hidden_size, self.dropout_rate)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,  # SHOULD BE SET TRUE in order to use hidden states accross attention layers
            return_dict=return_dict,
        )

        hidden_states = discriminator_outputs["hidden_states"]
        cls_concat = torch.cat(tuple([hidden_states[i][:, 0, :] for i in [-4, -3, -2, -1]]), dim=-1)
        cls_proj = self.proj_fc_layer(cls_concat)
        logits = self.classifier(cls_proj)

        loss = None
        if labels is not None:
            if "focal" in CFG.loss.lower():
                loss_fct = FocalLoss(gamma=0.5)
            elif "crossentropy" in CFG.loss.lower():
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + discriminator_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_outputs.hidden_states,
            attentions=discriminator_outputs.attentions,
        )


# https://github.com/sangHa0411/DACON-NLI/blob/643e6a9876f990cc3dbc78ff5fb24a507aa9ebf4/models/layer3.py#L68-L70
class ElectraPooledModel(ElectraPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.electra = ElectraModel(config)
        self.classifier = ElectraClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        hidden_states = outputs["hidden_states"]

        cls_output = hidden_states[-1][:, 0] * 0.6
        midterm_output1 = hidden_states[-2][:, 0] * 0.3
        midterm_output2 = hidden_states[-3][:, 0] * 0.1

        pooled_output = cls_output + midterm_output1 + midterm_output2
        logits = self.classifier(pooled_output)

        loss = None
        outputs.hidden_states = None
        if labels is not None:
            if "focal" in CFG.loss.lower():
                loss_fct = FocalLoss(gamma=0.5)
            elif "crossentropy" in CFG.loss.lower():
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

