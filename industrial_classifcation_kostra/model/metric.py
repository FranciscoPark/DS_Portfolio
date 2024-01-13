import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from torchmetrics import Metric

# Reference: https://github.com/pytorch/examples/blob/00ea159a99f5cb3f3301a9bf0baa1a5089c7e217/imagenet/main.py#L361-L450

from enum import Enum


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def top_k_acc(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    Reference for its usage: https://colab.research.google.com/drive/1dg5FXgNXUQOqGHjBziKvJfu338oBBc2l#scrollTo=nhibBF1aoE19&line=4&uniqifier=1
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def compute_metrics_huggingface(pred):
    """ 
    Metrics Computation with huggingface traininer 
    Refer to the documentation: https://huggingface.co/docs/transformers/training
    """
    labels = pred.label_ids  # answer label
    preds = pred.predictions.argmax(-1)  # prediction label
    # precision = f1 in many times for multiclass classification https://stackoverflow.com/questions/54068401/can-the-precision-recall-and-f1-be-the-same-value
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )  # macro for multiclass classification
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def compute_metrics_custom(label, output, input_topk: int = 3):
    """ 
    computes metrics for custom trainer.
    output is prediction logits, prediction is answer label 
    """
    if label.is_cuda and prediction.is_cuda:
        label = label.detach().cpu().numpy()
        prediction = prediction.detach().cpu().numpy()
    elif not label.is_cuda and not prediction.is_cuda:
        pass
    else:
        print("Error: label and prediction must be on the same device")
        print("label is on cuda: ", label.is_cuda)
        print("prediction is on cuda: ", prediction.is_cuda)
        exit(1)

    prediction = output.argmax(-1)
    batch_accuracy, batch_topk_accuracy = top_k_acc(output, label, topk=(1, input_topk))
    batch_f1_score = f1_score(label, prediction, average="macro")

    return batch_accuracy, batch_topk_accuracy, batch_f1_score


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class Metrics(object):
    """ 
    Simple version of Averaging metrics collected across batches such as loss, f1 and accuracy
    Reference: https://github.com/pytorch/examples/blob/21c240b814658e590b4fa9d4682d39831060c5b9/imagenet/main.py#L367-L385    
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LightningLosses(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("loss", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss: float):
        self.loss += torch.tensor(loss)
        self.total += torch.tensor(1)

    def compute(self):
        return self.loss.float() / self.total.float()
