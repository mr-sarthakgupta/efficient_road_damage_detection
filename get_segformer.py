import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

def get_segformer():
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-cityscapes-1024-1024", force_download=False)
    return model

def get_segformer_feature_extractor():
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b3-finetuned-cityscapes-1024-1024")
    return feature_extractor