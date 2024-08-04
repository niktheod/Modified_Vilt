import torch

from torch import nn
from typing import Optional
from transformers import ViltForQuestionAnswering, ViltConfig, ViTModel, ViTConfig, BertModel, BertConfig
from modified_transformers import ViltModel as Baseline, ViltConfig as ViltConfig2, ViltForQuestionAnswering as MultiviewViltForQuestionAnswering
    

class DoubleVilt(nn.Module):
    """
    A class based on ViltForQuestionAnswering, but it works with a set of images.
    """
    def __init__(self, set_size: int, img_seq_len: int, question_seq_len: int, emb_dim: int, pretrained_baseline: bool, pretrained_final_model: bool, pretrained_model_path: str = None,
                 device: str = "cuda") -> None:
        super().__init__()
        if pretrained_baseline:
            self.baseline = Baseline.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        else:
            self.baseline = Baseline(ViltConfig2())
        
        if pretrained_model_path is not None:
            pretrained_dict = torch.load(pretrained_model_path)
            model_dict = self.baseline.state_dict()
            pretrained_dict = {k[11:]: v for k, v in pretrained_dict.items() if k[11:] in model_dict}
            self.baseline.load_state_dict(pretrained_dict)

        self.img_attn = nn.MultiheadAttention(emb_dim, 12, batch_first=True)

        if pretrained_final_model:
            self.final_model = MultiviewViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        else:
            self.final_model = MultiviewViltForQuestionAnswering(ViltConfig2())

        self.set_size = set_size
        self.img_seq_len = img_seq_len
        self.question_seq_len = question_seq_len
        self.emb_dim = emb_dim
        self.device = device

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ):
        # Get the output from the first ViLT (the hidden states)
        first_output = self.baseline(input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask, head_mask, inputs_embeds, image_embeds,
                                     labels, output_attentions, output_hidden_states, return_dict)

        # Get the [CLS] tokens of the question and of each image in the image set
        idx = 0
        question_vectors = first_output.pooler_output.unsqueeze(1)
        idx += self.question_seq_len

        images = []
        for _ in range(self.set_size):
            images.append(first_output[0][:, idx])
            idx += self.img_seq_len

        # Concatenate the [CLS] tokens of the images in the image set
        images = torch.stack(images, dim=1)

        # print(question_vectors.shape)
        # print(question_vectors)
        # print(images.shape)
        # print(images)
        # raise ValueError
        
        # Get the attention scores of the question-guided attention on the images. Each score will show how relevant is each image for the question
        _, attn_scores = self.img_attn(question_vectors, images, images)

        weights = (attn_scores / attn_scores.max(dim=2)[0].unsqueeze(2)).squeeze()
        print(weights)
        weights = weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        weighted_pixel_values = weights * pixel_values
        new_pixel_mask = torch.ones_like(weighted_pixel_values[:, :, 0])
                
        return self.final_model(input_ids, attention_mask, token_type_ids, weighted_pixel_values, new_pixel_mask, labels=labels)


class ImageSetQuestionAttention(nn.Module):
    def __init__(self, pretrained_vit_version: str = "google/vit-base-patch16-224-in21k", pretrained_bert_version: str = "bert-base-uncased",
                 pretrained_vilt_version: str = "dandelin/vilt-b32-finetuned-vqa", train_vit: bool = False, train_bert: bool = False, train_vilt: bool = True,
                 threshold: float = 0.1, device="cuda") -> None:
        super().__init__()
        
        if pretrained_vit_version is None:
            self.vit = ViTModel(ViTConfig())
        else:
            self.vit = ViTModel.from_pretrained(pretrained_vit_version)
        
        if pretrained_bert_version is None:
            self.bert = BertModel(BertConfig())
        else:
            self.bert = BertModel.from_pretrained(pretrained_bert_version)
        
        self.attn = nn.MultiheadAttention(768, 12, batch_first=True)

        if pretrained_vilt_version is None:
            self.vilt = MultiviewViltForQuestionAnswering(ViltConfig2())
        else:
            self.vilt = MultiviewViltForQuestionAnswering.from_pretrained(pretrained_vilt_version)
        
        if not train_vit:
            for param in self.vit.parameters():
                param.requires_grad = False

        if not train_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        if not train_vilt:
            for param in self.vilt.parameters():
                param.requires_grad = False

        self.threshold = threshold
        self.device = device

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None):

        question = self.bert(input_ids, attention_mask, token_type_ids)
        question_vector = question.pooler_output.unsqueeze(1)
        
        batch_size, set_size = pixel_values.shape[0], pixel_values.shape[1]

        # images = []
        image_vectors = []
        for i in range(set_size):
            # image = self.vit(pixel_values[:, i])
            image_vector = self.vit(pixel_values[:, i]).pooler_output

            # images.append(image.last_hidden_state)
            image_vectors.append(image_vector)

        # images = torch.stack(images, dim=1)
        image_vectors = torch.stack(image_vectors, dim=1)

        _, attn_scores = self.attn(question_vector, image_vectors, image_vectors)
        weights = (attn_scores / attn_scores.max(dim=2)[0].unsqueeze(2)).squeeze()
        print(weights)
        weights = weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        # important_images = (attn_scores > self.threshold).squeeze()
        # important_image_cnt = important_images.sum(dim=1)
        # print(f"\t\t{attn_scores}")

        weighted_pixel_values = weights * pixel_values
        new_pixel_mask = torch.ones_like(weighted_pixel_values[:, :, 0])
                
        return self.vilt(input_ids, attention_mask, token_type_ids, weighted_pixel_values, new_pixel_mask, labels=labels)
    