import torch
import os
import json
import matplotlib.pyplot as plt
import datetime

from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from isvqa_data_setup import ISVQA
from nuscenesqa_data_setup import NuScenesQA
from engine import trainjob
from models import MultiviewViltForQuestionAnswering, MultiviewViltForQuestionAnsweringBaseline
from nuscenes.nuscenes import NuScenes
from typing import List, Tuple
from prettytable import PrettyTable


device = "cuda" if torch.cuda.is_available() else "cpu"


def save_plots(results: Tuple[List[float], List[float], List[float], List[float]], path: str):
    """
    A function to save plots of the results after training is done.
    """
    plt.plot(range(1, len(results[0])+1), results[0], label="Training")
    plt.plot(range(1, len(results[2])+1), results[2], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{path}/loss.png", facecolor="white")

    plt.clf()

    plt.plot(range(1, len(results[1])+1), ([x*100 for x in results[1]]), label="Training")
    plt.plot(range(1, len(results[3])+1), ([x*100 for x in results[3]]), label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy %")
    plt.legend()
    plt.savefig(f"{path}/accuracy.png", facecolor="white")

    plt.clf()


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    cnt = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        cnt += 1
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params



def train(hyperparameters: defaultdict,
          model_variation: str,
          dataset: str,
          qa_path: str,
          nuscenes_path: str,
          path_to_save_results: str,
          path_to_save_model: str,
          title: str = None,
          pretrained_model: bool = True,
          fine_tune_all: bool = False,
          image_lvl_pos_emb: bool = True,
          best_baseline: str = None,
          scheduler_type: str = None,
          device: str = device):
    
    seed = hyperparameters["seed"] if hyperparameters["seed"] is not None else 42

    generator = torch.Generator().manual_seed(seed)  # set a generator for reproducable results
    percentage = hyperparameters["percentage"]

    # Define the paths for the train, val, and test sets
    if percentage is None or percentage == 100:
        train_path = f"{qa_path}/train_set.json"
    else:
        train_path = f"{qa_path}/train_set_{percentage}.json"
    val_path = f"{qa_path}/val_set.json"
    answers_path = f"{qa_path}/answers.json"
    
    # Load the dataset (either ISVQA or NuScenesQA)
    if dataset == "isvqa":
        train_set = ISVQA(qa_path=train_path,
                          nuscenes_path=nuscenes_path,
                          answers_path=answers_path,
                          device=device)
        
        val_set = ISVQA(qa_path=val_path,
                        nuscenes_path=nuscenes_path,
                        answers_path=answers_path,
                        device=device)

        num_answers = len(train_set.answers)
    elif dataset == "nuscenesqa":
        dataroot = nuscenes_path[:-8]
        nusc = NuScenes(version="v1.0-trainval", dataroot=dataroot, verbose=False)

        train_set = NuScenesQA(qa_path=train_path,
                               nusc=nusc,
                               nuscenes_path=nuscenes_path,
                               answers_path=answers_path,
                               device=device)
        
        val_set = NuScenesQA(qa_path=val_path,
                             nusc=nusc,
                             nuscenes_path=nuscenes_path,
                             answers_path=answers_path,
                             device=device)

        num_answers = len(train_set.answers)

    batch_size = hyperparameters["batch_size"]

    # Define the train and val dataloaders
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              generator=generator)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False)
    
    # Define some values necessary for the initialization of the model
    set_size = hyperparameters["set_size"] if hyperparameters["set_size"] is not None else 6
    img_seq_len = hyperparameters["img_seq_len"] if hyperparameters["img_seq_len"] is not None else 210
    question_seq_len = hyperparameters["question_seq_len"] if hyperparameters["question_seq_len"] is not None else 40
    emb_dim = hyperparameters["emb_dim"] if hyperparameters["emb_dim"] is not None else 768

    if emb_dim != 768 and pretrained_model == True:
        raise ValueError("For pretrained ViLT the only valid value of emd_dim is 768")

    # Define the model
    if model_variation == "baseline":
        model = MultiviewViltForQuestionAnsweringBaseline(set_size, img_seq_len, emb_dim, pretrained_model, pretrained_model, image_lvl_pos_emb)

        # If we use pretrained weights and we don't want to fine tune the whole model (we only want to learn the VQA head and the set_positional_embedding because
        # they were initialized randomly), then we set requires_grad = False for all the other parameters.
        if not fine_tune_all and pretrained_model:
            for name, parameter in model.named_parameters():
                if name != "model.vilt.model.embeddings.img_position_embedding":
                    parameter.requires_grad = False

        # Define a new VQA head based on which dataset is used. This will also set automatically requires_grad = True for the classifier of the model
        model.model.classifier = nn.Sequential(
            nn.Linear(emb_dim, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Linear(1536, num_answers)
        )
        model = model.to(device)
    elif model_variation == "double_vilt":
        model = MultiviewViltForQuestionAnswering(set_size, img_seq_len, question_seq_len, emb_dim, pretrained_model, pretrained_model, image_lvl_pos_emb,
                                                  pretrained_model_path=best_baseline).to(device)

        if not fine_tune_all and pretrained_model:
            for name, parameter in model.named_parameters():
                if name[:22] != "final_model.classifier" and name[:8] != "img_attn" and name[:10] != "preprocess":
                    parameter.requires_grad = False

        model.final_model.classifier = nn.Sequential(
            nn.Linear(emb_dim, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Linear(1536, num_answers)
        ).to(device)
    else:
        raise ValueError("model_variation should be either 'baseline' or 'double_vilt'")
    
    print("Parameters to be trained: ")
    count_parameters(model)

    weight_decay = hyperparameters["weight_decay"] if hyperparameters["weight_decay"] is not None else 0

    optimizer_name = "adam" if hyperparameters["optimizer_name"] is None else hyperparameters["optimizer_name"]

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr"], weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters["lr"], weight_decay=weight_decay)

    if scheduler_type == "steplr":
        step_size = hyperparameters["scheduler_step_size"]
        gamma = hyperparameters["scheduler_gamma"]
        if step_size is None:
            raise ValueError("hyperparameters['scheduler_step_size'] should be defined.")
        if gamma is None:
            raise ValueError("hyperparameters['scheduler_gamma'] should be defined.")
        scheduler = StepLR(optimizer=optimizer,
                           step_size=step_size,
                           gamma=gamma)
    elif scheduler_type is not None:
        raise ValueError("scheduler_type should be either 'steplr' or 'None'")
    else:
        scheduler = None

    epochs = hyperparameters["epochs"]

    grad_accum_size = hyperparameters["grad_accum_size"] if hyperparameters["grad_accum_size"] is not None else 1

    results = trainjob(model, epochs, train_loader, val_loader, optimizer, scheduler, grad_accum_size, num_answers)

    # Define a setup dictionary that will be saved together with the results, in order to be able to remeber what setup gave the corresponding results
    setup = {"model_variation": model_variation,
                "dataset": dataset,
                 "pretrained": pretrained_model,
                 "img_lvl_emb": image_lvl_pos_emb,
                 "fine_tune_all": fine_tune_all,
                 "seed": seed,
                 "percentage": percentage,
                 "emb_dim": emb_dim,
                 "epochs": epochs,
                 "optimizer": optimizer_name,
                 "lr": hyperparameters["lr"],
                 "weight_decay": weight_decay,
                 "batch_size": batch_size,
                 "grad_accum_size": grad_accum_size,
                 "best_baseline": best_baseline,
                 "scheduler": scheduler_type,
                 "scheduler_step_size": hyperparameters["scheduler_step_size"],
                 "scheduler_gamme": hyperparameters["scheduler_gamma"]
                 }

    # Save the model and the results
    if title is None or os.path.exists(f"{path_to_save_results}/{title}"):
        title = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results_folder = f"{path_to_save_results}/{title}"
    model_folder = f"{path_to_save_model}/{title}"

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(model.state_dict(), f"{model_folder}/model.pth")
    with open(f"{model_folder}/setup.json", "w") as f:
        json.dump(setup, f)

    with open(f"{results_folder}/results.json", "w") as f:
        json.dump(results, f)

    with open(f"{results_folder}/setup.json", "w") as f:
        json.dump(setup, f)

    save_plots(results=results,
               path=results_folder)
    
    print("Done!")
