import torch.nn as nn
from utils.registry import MODELS, BACKBONES, HEADS

@MODELS.register
class ImageClassifier(nn.Module):
    """
    Standard Image Classifier: Backbone + (Neck) + Head
    """
    def __init__(self, backbone, head, neck=None):
        super().__init__()

        # Instantiate Backbone
        if isinstance(backbone, dict):
            backbone_type = backbone.pop("type")
            backbone_class = BACKBONES.get(backbone_type)
            self.backbone = backbone_class(**backbone)
        else:
            self.backbone = backbone

        # Instantiate Neck (Optional)
        self.neck = None
        if neck is not None:
            # Future implementation
            pass

        # Instantiate Head
        if isinstance(head, dict):
            head_type = head.pop("type")
            head_class = HEADS.get(head_type)
            # Some heads might need input channels from backbone
            # We could automatically infer this if backbone has `output_channels` attribute
            # For now, assume config provides it or head handles it
            self.head = head_class(**head)
        else:
            self.head = head

    def forward(self, x):
        features = self.backbone(x)

        # If backbone returns a tuple/list (multi-scale features),
        # classifier usually takes the last one or pooled one.
        # Assuming our ClassificationHead expects the raw feature tensor.

        if isinstance(features, (list, tuple)):
            x = features[-1]
        else:
            x = features

        out = self.head(x)
        return out

@MODELS.register
class EncoderDecoder(nn.Module):
    """
    Segmentor Architecture: Backbone + DecodeHead
    """
    def __init__(self, backbone, decode_head, neck=None, auxiliary_head=None):
        super().__init__()

        # Instantiate Backbone
        if isinstance(backbone, dict):
            backbone_type = backbone.pop("type")
            backbone_class = BACKBONES.get(backbone_type)
            self.backbone = backbone_class(**backbone)
        else:
            self.backbone = backbone

        # Instantiate Decode Head
        if isinstance(decode_head, dict):
            head_type = decode_head.pop("type")
            head_class = HEADS.get(head_type)
            self.decode_head = head_class(**decode_head)
        else:
            self.decode_head = decode_head

        # Aux Head (TODO)
        self.auxiliary_head = None

    def forward(self, x):
        features = self.backbone(x)
        # Backbone should return a list/tuple of multi-scale features for segmentation

        out = self.decode_head(features)
        return out
