import torch
import torch.nn as nn
from transformers import AutoModel


# Custom binary image classifier built on top of a pretrained SigLIP vision encoder
class SigLIPDetector(nn.Module):
    def __init__(self, model_name="google/siglip-base-patch16-224"):
        super().__init__()

        # Load the pretrained SigLIP model from Hugging Face.
        # The model contains a vision encoder whose learned representations
        # are reused for AI-generated image detection.
        self.backbone = AutoModel.from_pretrained(model_name)

        # Obtain the dimensionality of the SigLIP vision representation.
        # For SigLIP-Base Patch16-224, the hidden size is 768.
        hidden_size = self.backbone.config.vision_config.hidden_size

        # Custom binary classification head applied to the pooled
        # visual representation produced by the SigLIP vision encoder.
        self.classifier = nn.Sequential(

            # Normalise the pooled SigLIP feature representation.
            nn.LayerNorm(hidden_size),

            # Apply dropout with probability 0.3 for regularisation.
            nn.Dropout(0.3),

            # Project the hidden representation from hidden_size
            # (768 for SigLIP-Base) to 512 features.
            nn.Linear(hidden_size, 512),

            # Apply a non-linear ReLU activation.
            nn.ReLU(),

            # Apply a second dropout layer with probability 0.2.
            nn.Dropout(0.2),

            # Produce two raw class scores (logits):
            # one for the authentic class and one for the AI-generated class.
            nn.Linear(512, 2)
        )

    def forward(self, pixel_values):

        # Pass the input image tensor through the SigLIP vision encoder.
        # Typical input shape:
        # [batch_size, 3, 224, 224]
        outputs = self.backbone.vision_model(pixel_values=pixel_values)

        # Extract the pooled global visual representation produced by SigLIP.
        # Typical output shape for SigLIP-Base:
        # [batch_size, 768]
        features = outputs.pooler_output

        # Pass the pooled visual features through the custom classifier.
        # Output shape:
        # [batch_size, 2]
        logits = self.classifier(features)

        # Return the raw class logits.
        # Softmax can be applied outside the model during inference
        # to obtain class probabilities.
        return logits