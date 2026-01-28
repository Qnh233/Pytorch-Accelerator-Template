import torch.nn as nn
from utils.registry import MODELS, BACKBONES, HEADS, ADAPTERS, STEMS

@MODELS.register
class ImageClassifier(nn.Module):
    """
    Standard Image Classifier: (Adapter) -> (Stem) -> Backbone -> (Neck) -> Head
    """
    def __init__(self, backbone, head, neck=None, adapter=None, stem=None):
        super().__init__()

        # Instantiate Adapter (Optional)
        self.adapter = None
        if adapter is not None:
            if isinstance(adapter, dict):
                adapter_type = adapter.pop("type")
                adapter_class = ADAPTERS.get(adapter_type)
                self.adapter = adapter_class(**adapter)
            else:
                self.adapter = adapter

        # Instantiate Stem (Optional)
        self.stem = None
        if stem is not None:
            if isinstance(stem, dict):
                stem_type = stem.pop("type")
                stem_class = STEMS.get(stem_type)
                self.stem = stem_class(**stem)
            else:
                self.stem = stem

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
            self.head = head_class(**head)
        else:
            self.head = head

    def forward(self, x):
        # 1. Adapter (Dict/Tensor -> Tensor)
        if self.adapter is not None:
            x = self.adapter(x)

        # 2. Stem (Channel Mapping)
        if self.stem is not None:
            x = self.stem(x)

        # 3. Backbone
        features = self.backbone(x)

        if isinstance(features, (list, tuple)):
            x = features[-1]
        else:
            x = features

        # 4. Head
        out = self.head(x)
        return out

@MODELS.register
class EncoderDecoder(nn.Module):
    """
    Segmentor Architecture: (Adapter) -> (Stem) -> Backbone -> DecodeHead
    """
    def __init__(self, backbone, decode_head, neck=None, auxiliary_head=None, adapter=None, stem=None):
        super().__init__()

        # Instantiate Adapter (Optional)
        self.adapter = None
        if adapter is not None:
            if isinstance(adapter, dict):
                adapter_type = adapter.pop("type")
                adapter_class = ADAPTERS.get(adapter_type)
                self.adapter = adapter_class(**adapter)
            else:
                self.adapter = adapter

        # Instantiate Stem (Optional)
        self.stem = None
        if stem is not None:
            if isinstance(stem, dict):
                stem_type = stem.pop("type")
                stem_class = STEMS.get(stem_type)
                self.stem = stem_class(**stem)
            else:
                self.stem = stem

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
        # 1. Adapter
        if self.adapter is not None:
            x = self.adapter(x)

        # 2. Stem
        if self.stem is not None:
            x = self.stem(x)

        # 3. Backbone
        features = self.backbone(x)

        # 4. Decode Head
        out = self.decode_head(features)
        return out
