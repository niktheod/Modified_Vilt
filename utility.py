import torch

from torchvision import transforms
from typing import List
from transformers import ViltImageProcessor, BertTokenizer, ViTImageProcessor
from PIL import Image
from collections import OrderedDict


device = "cuda" if torch.cuda.is_available() else "cpu"


class ViltImageSetProcessor():
    """
    A class based on ViltImageProcessor with the difference that it handles sets of images instead of individual images.
    """
    def __init__(self) -> None:
        self.processor = ViltImageProcessor()

    def __call__(self, 
                 image_set: List[List[Image.Image]]) -> OrderedDict:

        pixel_values = []
        pixel_mask = []

        # Process the images in the set one by one and add them in the lists
        for image in image_set:
            processed_img = self.processor(image, return_tensors="pt")

            pixel_values.append(processed_img["pixel_values"])
            pixel_mask.append(processed_img["pixel_mask"])

        # Create a dictionary similar to the one that ViltImageProcessor returns, whith the difference that pixel values will be of shape 
        # (num_images, C, H, W) and pixel mask (num_images, H, W)
        processed_set = OrderedDict({
            "pixel_values": torch.stack(pixel_values),
            "pixel_mask": torch.stack(pixel_mask)
        })

        return processed_set
    

class ViltSetProcessor():
    """
    A class based on ViltProcessor with the difference that it handles pairs of sets of images and question instead of pairs of individual images and question.
    """
    def __init__(self, device=device) -> None:
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.image_set_processor = ViltImageSetProcessor()
        self.device=device

    def __call__(self,
                 image_set: List[List[Image.Image]], 
                 text: str) -> OrderedDict:
        
        # First process the question
        processed_text = self.tokenizer(text, 
                                        add_special_tokens=True, 
                                        max_length=40,
                                        padding="max_length", 
                                        return_tensors='pt',
                                        return_attention_mask=True,
                                        return_token_type_ids=True)
        
        # Second process the set of images
        processed_img = self.image_set_processor(image_set)

        # Combine them together in a dictionary like ViltProcessor does
        inputs = OrderedDict({
            "input_ids": processed_text["input_ids"].squeeze().to(self.device),
            "token_type_ids": processed_text["token_type_ids"].squeeze().to(self.device),
            "attention_mask": processed_text["attention_mask"].squeeze().to(self.device),
            "pixel_values": processed_img["pixel_values"].squeeze().to(self.device),
            "pixel_mask": processed_img["pixel_mask"].squeeze().to(self.device)
        })

        return inputs


class ViTImageSetProcessor():
    """
    A class based on ViTImageProcessor with the difference that it handles sets of images instead of individual images.
    """
    def __init__(self) -> None:
        self.processor = ViTImageProcessor(do_rescale=False)
        # self.transform = transforms.ToTensor()

    def __call__(self, 
                 image_set: List[List[Image.Image]]) -> OrderedDict:

        pixel_values = []
        pixel_mask = []

        # Process the images in the set one by one and add them in the lists
        for image in image_set:
            # image = self.transform(image)
            processed_img = self.processor(image, return_tensors="pt")

            pixel_values.append(processed_img["pixel_values"])
            pixel_mask.append(processed_img["pixel_mask"])

        # Create a dictionary similar to the one that ViltImageProcessor returns, whith the difference that pixel values will be of shape 
        # (num_images, C, H, W) and pixel mask (num_images, H, W)
        processed_set = OrderedDict({
            "pixel_values": torch.stack(pixel_values),
            "pixel_mask": torch.stack(pixel_mask)
        })

        return processed_set
    

class ViTSetProcessor():
    """
    A class based on ViltProcessor with the difference that it handles pairs of sets of images and question instead of pairs of individual images and question.
    """
    def __init__(self, device=device) -> None:
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.image_set_processor = ViTImageSetProcessor()
        self.device=device

    def __call__(self,
                 image_set: List[List[Image.Image]], 
                 text: str) -> OrderedDict:
        
        # First process the question
        processed_text = self.tokenizer(text, 
                                        add_special_tokens=True, 
                                        max_length=40,
                                        padding="max_length", 
                                        return_tensors='pt',
                                        return_attention_mask=True,
                                        return_token_type_ids=True)
        
        # Second process the set of images
        processed_img = self.image_set_processor(image_set)

        # Combine them all together in a dictionary like ViltProcessor does
        inputs = OrderedDict({
            "input_ids": processed_text["input_ids"].squeeze().to(self.device),
            "token_type_ids": processed_text["token_type_ids"].squeeze().to(self.device),
            "attention_mask": processed_text["attention_mask"].squeeze().to(self.device),
            "pixel_values": processed_img["pixel_values"].squeeze().to(self.device),
            "pixel_mask": processed_img["pixel_mask"].squeeze().to(self.device)
        })

        return inputs