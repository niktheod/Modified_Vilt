import torch
import torch.nn.functional as F
import json
import nuscenes

from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple
from utility import ViltSetProcessor, ViTSetProcessor


device = "cuda" if torch.cuda.is_available() else "cpu"


class NuScenesQA(Dataset):
    """
    A class that loads the NuScenes-QA dataset in the form of a torch.utils.data.Dataset object.
    """
    def __init__(self, qa_path: str, nusc: nuscenes.nuscenes.NuScenes, nuscenes_path: str, answers_path: str, processor_type: str, device=device) -> None:
        super().__init__()
        self.device = device
        self.qa_set = self.load_nuscenesqa(qa_path)  # a list of dictionaries that contain all the sampleTokens-Question-Answer pairs
        self.nuscenes_path = nuscenes_path
        self.nusc = nusc
        self.answers = self.get_unique_answers(answers_path)  # a list of all the unique answers in the NuScenes-QA dataset

        if processor_type == "vilt":
            self.processor = ViltSetProcessor(device=device)  # the processor that will take as input the PIL images and the question and will return
                                                          # them in a format compatible to be used as inputs for the MultivieViltModel
        elif processor_type == "vit":
            self.processor = ViTSetProcessor()  # the processor that will take as input the PIL images and the question and will return
                                                # them in a format compatible to be used as inputs for ViLT however the images will be of the size that is
                                                # compatible with ViT (224x224) because they'll be first processed by a ViT.
        else:
            raise ValueError("processor_type should be either 'vilt' or 'vit'.")

    @staticmethod
    def load_nuscenesqa(path: str) -> List[dict]:
        with open(path) as f:
            qa_set = json.load(f)

        return qa_set
    
    @staticmethod
    def get_images_for_question(qa_sample: dict, nusc: nuscenes.nuscenes.NuScenes, folder_path: str):
        """
        A method that "translates" the sample_token of a nuscenes scene into a list of 6 PIL images (one from each camera in that scene).
        """
        sample_token = qa_sample["sample_token"]
        order = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
        image_tokens = [nusc.get("sample", sample_token)["data"][direction] for direction in order]
        image_paths = [nusc.get("sample_data", token)["filename"] for token in image_tokens]
        images = [Image.open(f"{folder_path[:-8]}/{path}") for path in image_paths]

        return images
    
    @staticmethod
    def get_unique_answers(path: str) -> List[str]:
        with open(path) as f:
            answers = json.load(f)

        return answers
    
    def __len__(self) -> int:
        return len(self.qa_set)

    def __getitem__(self, index: int) -> Tuple[dict, torch.Tensor]:
        qa_sample = self.qa_set[index]
        question = qa_sample["question"]
        images = self.get_images_for_question(qa_sample, self.nusc, self.nuscenes_path)
        inputs = self.processor(images, question)  # get the final inputs that will go into the model

        answer = qa_sample["answer"]

        # Turn the answer into an one-hot representation, based on the index of it in the list with all the unique answers
        answer_idx = torch.tensor(self.answers.index(answer))
        one_hot_answer = F.one_hot(answer_idx, len(self.answers)).type(torch.float32).to(self.device)

        return inputs, one_hot_answer
