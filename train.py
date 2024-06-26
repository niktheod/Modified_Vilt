import torch
import os
import json
import random
import matplotlib.pyplot as plt
import datetime

from torch import nn
from torch.utils.data import random_split, DataLoader, Subset
from collections import defaultdict
from isvqa_data_setup import ISVQA
from nuscenesqa_data_setup import NuScenesQA
from engine import trainjob
from models import MultiviewViltForQuestionAnswering
from nuscenes.nuscenes import NuScenes
from typing import List, Tuple


device = "cuda" if torch.cuda.is_available() else "cpu"


def keep_fraction(dataset, frac_keep, seed):
    """
    A function that keeps only a specific fraction of a given dataset.
    """
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    num_keep = int(frac_keep * len(indices))
    subset_indices = indices[:num_keep]
    dataset = Subset(dataset, subset_indices)

    return dataset


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



def train(hyperparameters: defaultdict,
          dataset: str,
          qa_path: str,
          nuscenes_path: str,
          path_to_save_results: str,
          path_to_save_model: str,
          title: str = None,
          pretrained_model: bool = True,
          fine_tune_all: bool = False,
          image_lvl_pos_emb: bool = True,
          device: str = device):
    
    seed = hyperparameters["seed"] if hyperparameters["seed"] is not None else 42

    generator = torch.Generator().manual_seed(seed)  # set a generator for reproducable results
    frac_keep = hyperparameters["frac_keep"] if hyperparameters["frac_keep"] is not None else 1

    # Define the paths for the train, val, and test sets
    train_path = f"{qa_path}/train_set.json"
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

    # Keep only a fraction of the train dataset if wanted
    if frac_keep < 1:
        train_set = keep_fraction(train_set, frac_keep, seed)

    batch_size = hyperparameters["batch_size"]

    # Define the train anc val dataloaders
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              generator=generator)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False)
    
    # Define some values necessary for the initialization of the model
    set_size = hyperparameters["set_size"] if hyperparameters["set_size"] is not None else 6
    seq_len = hyperparameters["seq_len"] if hyperparameters["seq_len"] is not None else 210
    emb_dim = hyperparameters["emb_dim"] if hyperparameters["emb_dim"] is not None else 768

    if emb_dim != 768 and pretrained_model == True:
        raise ValueError("For pretrained ViLT the only valid value of emd_dim is 768")

    # Define the model
    model = MultiviewViltForQuestionAnswering(set_size, seq_len, emb_dim, pretrained_model, pretrained_model, image_lvl_pos_emb).to(device)

    # If we use pretrained weights and we don't want to fine tune the whole model (we only want to learn the VQA head and the set_positional_embedding because
    # they were initialized randomly), then we set requires_grad = False for all the other parameters.
    if not fine_tune_all and pretrained_model:
        for name, parameter in model.named_parameters():
            if name != "model.vilt.model.embeddings.set_positional_embedding":
                parameter.requires_grad = False

    # Define a new VQA head based on which dataset is used. This will also set automatically requires_grad = True for the classifier of the model
    model.model.classifier = nn.Sequential(
        nn.Linear(emb_dim, 1536),
        nn.LayerNorm(1536),
        nn.GELU(),
        nn.Linear(1536, num_answers)
    ).to(device)

    weight_decay = hyperparameters["weight_decay"] if hyperparameters["weight_decay"] is not None else 0
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr"], weight_decay=weight_decay)

    epochs = hyperparameters["epochs"]

    results = trainjob(model, epochs, train_loader, val_loader, optimizer, num_answers)

    # Define a setup dictionary that will be saved together with the results, in order to be able to remeber what setup gave the corresponding results
    setup = {"dataset": dataset,
                 "pretrained": pretrained_model,
                 "img_lvl_emb": image_lvl_pos_emb,
                 "fine_tune_all": fine_tune_all,
                 "seed": seed,
                 "frac_keep": frac_keep,
                 "emb_dim": emb_dim,
                 "epochs": epochs,
                 "lr": hyperparameters["lr"],
                 "weight_decay": weight_decay
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
