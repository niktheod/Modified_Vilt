import torch

from torch import nn
from typing import Optional
from transformers import ViltForQuestionAnswering, ViltConfig
from modified_transformers import ViltModel, ViltConfig as ViltConfig2
    

class DoubleVilt(nn.Module):
    """
    A class based on ViltForQuestionAnswering, but it works with a set of images.
    """
    def __init__(self, set_size: int, img_seq_len: int, question_seq_len: int, emb_dim: int, pretrained_baseline: bool, pretrained_final_model: bool, pretrained_model_path: str = None,
                 device: str = "cuda") -> None:
        super().__init__()
        if pretrained_baseline:
            self.baseline = ViltModel.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        else:
            self.baseline = ViltModel(ViltConfig2())
        
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

        self.cls = nn.Sequential(nn.Linear(768, 1536),
                                 nn.LayerNorm(1536),
                                 nn.GELU(),
                                 nn.Linear(1536, 429))


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


    