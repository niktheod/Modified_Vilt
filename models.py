import torch

from torch import nn
from typing import Optional
from transformers import ViltConfig, ViltModel, ViltForQuestionAnswering, ViTModel, ViTConfig, BertModel, BertConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput
from typing import Tuple, Optional, Union, List


class ViltSetEmbeddings(nn.Module):
    """
    A class based on ViltEmbeddings but it works with pairs of sets of images and question instead of pairs of individual images and question.
    It introduces a set of extra parameters (optional) for positional embedding on the image level (apart from the positional embedding on the patches
    that is already introduced in ViLT), in order to be able to easier say the images apart.
    """
    def __init__(self, set_size: int, seq_len: int, emb_dim: int, pretrained: bool, vqa: bool, img_lvl_pos_embeddings: bool) -> None:
        super().__init__()
        self.set_size = set_size
        self.seq_len = seq_len
        self.emb_dim = emb_dim

        # Initialize the embeddings either with pretrained values (fine-tuned on vqa or not) or wih random values
        if pretrained:
            if vqa:
                self.embeddings = ViltModel.from_pretrained("dandelin/vilt-b32-finetuned-vqa").embeddings
            else:
                self.embeddings = ViltModel.from_pretrained("dandelin/vilt-b32-mlm").embeddings
        else:
            self.embeddings = ViltModel(ViltConfig(hidden_size=emb_dim)).embeddings
        
        # Initialize the image level positional embeddings if needed
        self.img_lvl_pos_embeddings = img_lvl_pos_embeddings
        if img_lvl_pos_embeddings:
            self.img_position_embedding = nn.Parameter(torch.zeros(set_size, seq_len, emb_dim))

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        pixel_values,
        pixel_mask,
        inputs_embeds=None,
        image_embeds=None,
        image_token_type_idx=1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get the text embeddings and add the modality embedding to them
        text_embeds = self.embeddings.text_embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=None
        )

        text_embeds = text_embeds + self.embeddings.token_type_embeddings(
            torch.zeros_like(attention_mask, dtype=torch.long, device=text_embeds.device)
        )

        # Initialize 2 lists to save the embeddings and the mask from each image in the set
        visual_embeds = []
        visual_masks = []
        
        # Go through each image in the set, calculate the embeddings, add the modality embeddings, and save them in the list
        for image, mask in zip(pixel_values, pixel_mask):
            embedding, emb_mask, _ = self.embeddings.visual_embed(image, mask, max_image_length=self.embeddings.config.max_image_length)
            embedding += self.embeddings.token_type_embeddings(torch.ones_like(emb_mask, dtype=torch.long, device=text_embeds.device))

            visual_embeds.append(embedding)
            visual_masks.append(emb_mask)

        # Stack all the visual embeddings together to create the embedding representation of the whole set
        visual_embeds_tensor = torch.stack(visual_embeds)
        if self.img_lvl_pos_embeddings:  # add the image level positional embeddings if needed
            visual_embeds_tensor += self.img_position_embedding

        # Reshape the visual embeddings from shape (batch_size, num_images, seq_length, emb_dimension) to (batch_size, [num_imges * seq_length], emb_dimension)
        # in order for the attention layer to be able to process it, as it can not process 4D tensors.
        visual_embeds_tensor = visual_embeds_tensor.flatten(1, 2)

        # Repeat the same process for the masks (apart from adding the image level positional embeddings)
        visual_masks_tensor = torch.stack(visual_masks)
        visual_masks_tensor = visual_masks_tensor.flatten(1, 2)

        # Concatenate the two modalities together
        embeddings = torch.cat([text_embeds, visual_embeds_tensor], dim=1)
        masks = torch.cat([attention_mask, visual_masks_tensor], dim=1)

        return embeddings, masks


class MultiviewViltModel(nn.Module):
    """
    A class based on ViltModel, but it works with a set of images.
    """

    # Initialize the parameters of the model either with pretrained values (fine-tuned on vqa or not) or wih random values
    def __init__(self, set_size: int, seq_len: int, emb_dim: int, pretrained: bool, vqa: bool, img_lvl_pos_emb: bool) -> None:
        super().__init__()
        if pretrained:
            if vqa:
                self.model = ViltModel.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            else:
                self.model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        else:
            self.model = ViltModel(ViltConfig(hidden_size=emb_dim))
            
        # Replace the embedding of the ViltModel with the ViltSetEmbeddings class, which is the only change we need to make for the model to work with 
        # a set of images
        self.model.embeddings = ViltSetEmbeddings(set_size, seq_len, emb_dim, pretrained, vqa, img_lvl_pos_emb)

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
        image_token_type_idx: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutputWithPooling, Tuple[torch.FloatTensor]]:
        return self.model(input_ids, token_type_ids, attention_mask, pixel_values, pixel_mask, head_mask, inputs_embeds,
                          image_embeds, image_token_type_idx, output_attentions, output_hidden_states, return_dict)


class MultiviewViltForQuestionAnsweringBaseline(nn.Module):
    """
    A baseline based on ViltForQuestionAnswering, but it works with a set of images.
    """
    def __init__(self, set_size: int, seq_len: int, emb_dim: int, pretrained_body: bool, pretrained_head: bool, img_lvl_pos_emb: bool) -> None:
        super().__init__()
        # Initialize the parameters of the model either from pretrained parameters or randomly (choices between pretrained parameters only for the body of the model
        # or both the body and the head or none)
        if pretrained_body:
            if pretrained_head:
                self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            else:
                self.model = ViltForQuestionAnswering(ViltConfig(hidden_size=emb_dim))
            
            self.model.vilt = MultiviewViltModel(set_size, seq_len, emb_dim, True, True, img_lvl_pos_emb)  # Change the body of the model (ViltModel) with the Multiview version of it
        else:
            self.model = ViltForQuestionAnswering(ViltConfig(hidden_size=emb_dim))
            self.model.vilt = MultiviewViltModel(set_size, seq_len, emb_dim, False, False, img_lvl_pos_emb)  # Change the body of the model (ViltModel) with the Multiview version of it

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
    ) -> Union[SequenceClassifierOutput, Tuple[torch.FloatTensor]]:
        return self.model(input_ids, token_type_ids, attention_mask, pixel_values, pixel_mask, head_mask, inputs_embeds,
                          image_embeds, labels, output_attentions, output_hidden_states, return_dict)


class DoubleVilt(nn.Module):
    """
    A class based on ViltForQuestionAnswering, but it works with a set of images.
    """
    def __init__(self, set_size: int, img_seq_len: int, question_seq_len: int, emb_dim: int, pretrained_baseline: bool, pretrained_final_model: bool, pretrained_model_path: str = None,
                 img_lvl_pos_emb: bool = False, device: str = "cuda") -> None:
        super().__init__()
        self.baseline = MultiviewViltModel(set_size, img_seq_len, emb_dim, pretrained_baseline, pretrained_baseline, img_lvl_pos_emb)
        
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
        print(f"\t\t{attn_scores}")

        image_set = torch.zeros(batch_size, 197, 768).to(self.device)

        # Create an embedded representation for the image set that is a weighted average of the images based on their attention score      
        for i in range(set_size):
            image_set += attn_scores[:, :, i].unsqueeze(2) * images[:, i]
                
        return self.vilt(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, image_embeds=image_set, labels=labels)
    