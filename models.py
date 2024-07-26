import torch

from torch import nn
from typing import Optional
from transformers import ViltForQuestionAnswering, ViltConfig, ViTModel, ViTConfig, BertModel, BertConfig
from modified_transformers import ViltModel as Baseline, ViltConfig as ViltConfig2
    

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
            self.final_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        else:
            self.final_model = ViltForQuestionAnswering(ViltConfig())

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
                                     labels, output_attentions, output_hidden_states, return_dict).last_hidden_state
        
        batch_size = first_output.shape[0]

        # Get the [CLS] tokens of the question and of each image in the image set
        idx = 0
        questions = first_output[:, idx].unsqueeze(1)
        idx += self.question_seq_len

        images = []
        for _ in range(self.set_size):
            images.append(first_output[:, idx])
            idx += self.img_seq_len

        # Concatenate the [CLS] tokens of the images in the image set
        images = torch.stack(images, dim=1)
        
        # Get the attention scores of the question-guided attention on the images. Each score will show how relevant is each image for the question
        _, attn_scores = self.img_attn(questions, images, images)

        # Initialize a tensor that will represent the image set
        image_set = torch.zeros(batch_size, self.img_seq_len, self.emb_dim).to(self.device)

        # Create an embedded representation for the image set that is a weighted average of the images based on their attention score
        idx = self.question_seq_len        
        for i in range(self.set_size):
            image_set += attn_scores[:, :, i].unsqueeze(2) * first_output[:, idx:(idx+self.img_seq_len)]
            idx += self.img_seq_len

        # Pass the question represantetion and the image set representation in a classic ViltForQuestionAnswering
        return self.final_model(inputs_embeds=first_output[:, :self.question_seq_len], image_embeds=image_set, labels=labels)
        # return self.final_model(inputs_embeds=torch.randn(batch_size, 40, 768).to("cuda"), image_embeds=torch.randn(batch_size, 210, 768).to("cuda"), labels=labels)


class ImageSetQuestionAttention(nn.Module):
    def __init__(self, pretrained_vit_version: str = "google/vit-base-patch16-224-in21k", pretrained_bert_version: str = "bert-base-uncased",
                 pretrained_vilt_version: str = "dandelin/vilt-b32-finetuned-vqa", train_vit: bool = False, train_bert: bool = False, train_vilt: bool = True,
                 device="cuda") -> None:
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
            self.vilt = ViltForQuestionAnswering(ViltConfig())
        else:
            self.vilt = ViltForQuestionAnswering.from_pretrained(pretrained_vilt_version)
        
        if not train_vit:
            for param in self.vit.parameters():
                param.requires_grad = False

        if not train_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        if not train_vilt:
            for param in self.vilt.parameters():
                param.requires_grad = False

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

        images = []
        image_vectors = []
        for i in range(set_size):
            image = self.vit(pixel_values[:, i])
            image_vector = image.pooler_output

            images.append(image.last_hidden_state)
            image_vectors.append(image_vector)

        images = torch.stack(images, dim=1)
        image_vectors = torch.stack(image_vectors, dim=1)

        _, attn_scores = self.attn(question_vector, image_vectors, image_vectors)

        image_set = torch.zeros(batch_size, 197, 768).to(self.device)

        # Create an embedded representation for the image set that is a weighted average of the images based on their attention score      
        for i in range(set_size):
            image_set += attn_scores[:, :, i].unsqueeze(2) * images[:, i]
                
        return self.vilt(inputs_embeds=question.last_hidden_state, image_embeds=image_set, labels=labels)
    