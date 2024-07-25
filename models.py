import torch

from torch import nn
from typing import Tuple, Optional, Union
from transformers import ViltConfig, ViltModel, ViltForQuestionAnswering
from transformers.models.vilt.modeling_vilt import ViltEmbeddings
from transformers.modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput


class ViltSetEmbeddings(ViltEmbeddings):
    """
    A class based on ViltEmbeddings but it works with pairs of sets of images and question instead of pairs of individual images and question.
    It introduces a set of extra parameters (optional) for positional embedding on the image level (apart from the positional embedding on the patches
    that is already introduced in ViLT), in order to be able to easier say the images apart.
    """
    def __init__(self, set_size: int, seq_len: int, emb_dim: int, pretrained: bool, vqa: bool, img_lvl_pos_embeddings: bool) -> None:
        super().__init__(ViltConfig())
        self.set_size = set_size
        self.seq_len = seq_len
        self.emb_dim = emb_dim

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
        text_embeds = self.text_embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=None
        )

        text_embeds = text_embeds + self.token_type_embeddings(
            torch.zeros_like(attention_mask, dtype=torch.long, device=text_embeds.device)
        )

        # Initialize 2 lists to save the embeddings and the mask from each image in the set
        visual_embeds = []
        visual_masks = []
        
        # Go through each image in the set, calculate the embeddings, add the modality embeddings, and save them in the list
        for image, mask in zip(pixel_values, pixel_mask):
            embedding, emb_mask, _ = self.visual_embed(image, mask, max_image_length=self.config.max_image_length)
            embedding += self.token_type_embeddings(torch.ones_like(emb_mask, dtype=torch.long, device=text_embeds.device))

            visual_embeds.append(embedding)
            visual_masks.append(emb_mask)

        # Stack all the visual embeddings together to create the embedding representation of the whole set
        visual_embeds_tensor = torch.stack(visual_embeds)

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


class MultiviewViltModel(ViltModel):
    """
    A class based on ViltModel, but it works with a set of images.
    """

    # Initialize the parameters of the model either with pretrained values (fine-tuned on vqa or not) or wih random values
    def __init__(self, set_size: int, seq_len: int, emb_dim: int, pretrained: bool, vqa: bool, img_lvl_pos_emb: bool) -> None:
        super().__init__(ViltConfig())
            
        # Replace the embedding of the ViltModel with the ViltSetEmbeddings class, which is the only change we need to make for the model to work with 
        # a set of images
        del self.embeddings
        self.embeddings = ViltSetEmbeddings(set_size, seq_len, emb_dim, pretrained, vqa, img_lvl_pos_emb)

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
        return super().forward(input_ids, token_type_ids, attention_mask, pixel_values, pixel_mask, head_mask, inputs_embeds,
                          image_embeds, image_token_type_idx, output_attentions, output_hidden_states, return_dict)


class MultiviewViltForQuestionAnsweringBaseline(MultiviewViltModel):
    """
    A baseline based on ViltForQuestionAnswering, but it works with a set of images.
    """
    def __init__(self, set_size: int, seq_len: int, emb_dim: int, pretrained_body: bool, pretrained_head: bool, img_lvl_pos_emb: bool, num_answers: int) -> None:
        super().__init__(set_size, seq_len, emb_dim, pretrained_body, pretrained_head, img_lvl_pos_emb)
        self.classifier = nn.Sequential(
            nn.Linear(768, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Linear(1536, num_answers)
        )

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
        x = super().forward(input_ids, token_type_ids, attention_mask, pixel_values, pixel_mask, head_mask, inputs_embeds,
                          image_embeds, labels, output_attentions, output_hidden_states, return_dict)
        
        return self.classifier(x.pooler_output)
    

class MultiviewViltForQuestionAnswering(nn.Module):
    """
    A class based on ViltForQuestionAnswering, but it works with a set of images.
    """
    def __init__(self, set_size: int, img_seq_len: int, question_seq_len: int, emb_dim: int, pretrained_body: bool, vqa: bool, img_lvl_pos_emb: bool, pretrained_model_path: str = None,
                 device: str = "cuda") -> None:
        super().__init__()
        self.preprocess = MultiviewViltModel(set_size, img_seq_len, emb_dim, pretrained_body, vqa, img_lvl_pos_emb)
        
        
        if pretrained_model_path is not None:
            pretrained_dict = torch.load(pretrained_model_path)
            model_dict = self.preprocess.state_dict()
            pretrained_dict = {k[11:]: v for k, v in pretrained_dict.items() if k[11:] in model_dict}
            self.preprocess.load_state_dict(pretrained_dict)

        self.img_attn = nn.MultiheadAttention(emb_dim, 12, batch_first=True)
        self.final_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
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
        first_output = self.preprocess(input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask, head_mask, inputs_embeds, image_embeds,
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
    