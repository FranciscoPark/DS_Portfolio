import os
import torch
import wandb
from parse_config import read_config

CFG = read_config()

""" Wandb logging related functions """


def activate_wandb(wandb_repo_name: str, run_name: str, entity_name: str, CFG: dict = CFG) -> None:
    """ login and activate wandb """
    wandb.login()
    wandb.init(project=wandb_repo_name, name=run_name, entity=entity_name, config=CFG)


def compare_and_save(
    epoch: int,
    model: torch.nn.Module,
    input_accuracy: float,
    input_topk_accuracy: float,
    input_loss: float,
    input_f1: float,
    save_directory: str = CFG.result_dir,
    wandb_run_name: str = "",
    wandb_entity_name: str = CFG.entity_name,
    previous_best_accuracy: float = 0.0,
    previous_topk_accuracy: float = 0.0,
    previous_best_loss: float = 100.0,
    previous_best_f1: float = 0.0,
    *args,
    **kwargs,
):
    """
    Compare the accuracy and loss of the current model with the best model
    and save the best model
    """

    # Raise error if wandb run name is not provided
    if wandb_run_name == "":
        raise ValueError("Wandb run name is saved directory name: Please provide its name.")

    # make directory with wandb_run_name
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # construct saving path according to wandb run name
    save_path = os.path.join(save_directory, wandb_run_name)
    if not os.path.exists(os.path.join(save_directory, wandb_run_name)):
        os.makedirs(save_path)

    # compare previous metrics and save the best model according to the input metrics
    if input_accuracy > previous_best_accuracy:
        previous_best_accuracy = input_accuracy
        torch.save(
            model.state_dict(), f"{save_path}/{epoch}-best-val-accuracy-model.pt",
        )
        print("Saved model with highest validation accuracy: {:.4f}".format(previous_best_accuracy))
        wandb.log({"best_validation_accuracy": input_accuracy})

    if input_topk_accuracy > previous_topk_accuracy:
        previous_topk_accuracy = input_topk_accuracy
        print("Logging highest validation topk accuracy: {:.4f}".format(previous_topk_accuracy))
        wandb.log({"best_validation_topk_accuracy": input_topk_accuracy})

    if input_loss < previous_best_loss:
        previous_best_loss = input_loss
        torch.save(
            model.state_dict(), f"{save_path}/{epoch}-best-val-loss-model.pt",
        )
        print("Saved model with lowest validation loss: {:.4f}".format(previous_best_loss))
        wandb.log({"best_validation_loss": input_loss})

    if input_f1 > previous_best_f1:
        previous_best_f1 = input_f1
        torch.save(
            model.state_dict(), f"{save_path}/{epoch}-best-val-f1-model.pt",
        )
        print("Saved model with highest validation f1: {:.4f}".format(previous_best_f1))
        wandb.log({"best_validation_f1": input_f1})
    return previous_best_accuracy, previous_topk_accuracy, previous_best_loss, previous_best_f1


def log_wandb(
    epoch: int,
    step: int,
    learning_rate: float,
    loss: float,
    accuracy: float,
    topk_integer: int,
    topk_accuracy: float,
    f1: float,
    is_train=True,
    *args,
    **kwargs,
):
    """
    Logs the metrics to wandb
    """

    if is_train:
        wandb.log(
            {
                "train/epoch": epoch,
                "train/loss": loss,
                # "train/learning_rate": learning_rate,
                "train/accuracy": accuracy,
                f"train/top_{topk_integer}_accuracy": topk_accuracy,
                "train/f1": f1,
            },
            step=step,
        )
    else:
        wandb.log(
            {
                "validation/epoch": epoch,
                "validation/loss": loss,
                # "validation/learning_rate": learning_rate,
                "validation/accuracy": accuracy,
                f"validation/top_{topk_integer}_accuracy": topk_accuracy,
                "validation/f1": f1,
            },
            step=step,
        )
        return accuracy, loss, f1

