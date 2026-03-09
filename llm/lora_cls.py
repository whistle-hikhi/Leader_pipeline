"""
LoRA helpers for LlamaForMedRec.

The standard peft library doesn't know about the custom cls_head in
LlamaForMedRec, so we add thin wrappers that save/load cls_head weights
alongside the LoRA adapter weights.
"""

import os

import torch
from peft import LoraConfig, PeftModel, get_peft_model

CLS_HEAD_FILE = "cls_head.bin"


def create_lora_model(base_model, lora_rank: int = 8, lora_alpha: float = 32.0,
                      trainable: str = "q_proj,v_proj", lora_dropout: float = 0.1):
    """Wrap base_model with a LoRA adapter (training mode)."""
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=trainable.split(","),
        lora_dropout=lora_dropout,
        bias="none",
    )
    return get_peft_model(base_model, config)


def save_lora_model(model, directory: str):
    """Save LoRA adapter weights + cls_head to directory."""
    os.makedirs(directory, exist_ok=True)
    model.save_pretrained(directory)  # saves LoRA weights via standard peft
    cls_state = {k: v for k, v in model.state_dict().items() if "cls_head" in k}
    torch.save(cls_state, os.path.join(directory, CLS_HEAD_FILE))


def load_lora_model(base_model, directory: str, is_trainable: bool = False):
    """Load LoRA adapter + cls_head onto base_model from directory."""
    model = PeftModel.from_pretrained(base_model, directory, is_trainable=is_trainable)
    cls_path = os.path.join(directory, CLS_HEAD_FILE)
    if os.path.exists(cls_path):
        cls_state = torch.load(cls_path, map_location="cpu")
        model.load_state_dict(cls_state, strict=False)
    return model
