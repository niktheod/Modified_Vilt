import torch

from torch import nn
from typing import Tuple, Optional, Union
from transformers import ViltConfig, ViltModel, ViltForQuestionAnswering
from transformers.modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput


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
        visual_embeds_tensor = visual_embeds_tensor.reshape(visual_embeds_tensor.shape[0], self.set_size*self.seq_len, self.emb_dim)

        # Repeat the same process for the masks (apart from adding the image level positional embeddings)
        visual_masks_tensor = torch.stack(visual_masks)
        visual_masks_tensor = visual_masks_tensor.reshape(visual_masks_tensor.shape[0], self.set_size*self.seq_len)

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
    

class MultiviewViltForQuestionAnswering(nn.Module):
    """
    A class based on ViltForQuestionAnswering, but it works with a set of images.
    """
    def __init__(self, set_size: int, img_seq_len: int, question_seq_len, emb_dim: int, pretrained_body: bool, vqa: bool, pretrained_head: bool, img_lvl_pos_emb: bool) -> None:
        super().__init__()
        self.preprocess = MultiviewViltModel(set_size, img_seq_len, emb_dim, pretrained_body, vqa, img_lvl_pos_emb)
        self.img_attn = nn.MultiheadAttention(emb_dim, 12)
        self.final_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.set_size = set_size
        self.img_seq_len = img_seq_len
        self.question_seq_len = question_seq_len
        self.emb_dim = emb_dim


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
        first_output = self.baseline(input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask, head_mask, inputs_embeds, image_embeds,
                                     labels, output_attentions, output_hidden_states, return_dict)
        
        idx = 0
        
        question = first_output.last_hidden_state[:, idx]
        idx += self.question_seq_len

        images = []
        for _ in range(self.set_size):
            images.append(first_output.last_hidden_state[:, idx])
            idx += self.img_seq_len

        images = torch.stack(images)
        
        _, attn_scores = self.img_attn(question, images, images)

        image_set = torch.zeros(self.img_seq_len, self.emb_dim)
        idx = self.question_seq_len
        
        for i in range(self.set_size):
            image_set += attn_scores[i] * first_output.last_hidden_state[0, idx:(idx+self.img_seq_len)]
            idx += self.img_seq_len

        return self.final_model(inputs_embeds=first_output.last_hidden_state[:, self.question_seq_len], image_embeds=image_set)
    