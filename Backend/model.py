import torch
import torch.nn as nn
from transformers import AutoModel
class SigLIPDetector(nn.Module):
    def __init__(self, model_name="google/siglip-base-patch16-224"):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)

        hidden_size = self.backbone.config.vision_config.hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2)
        )

    def forward(self, pixel_values):
        outputs = self.backbone.vision_model(pixel_values=pixel_values)
        features = outputs.pooler_output
        logits = self.classifier(features)
        return logits