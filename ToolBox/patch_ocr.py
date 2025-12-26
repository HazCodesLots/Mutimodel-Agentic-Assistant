import sys
import os

def patch_file(filepath):
    if not os.path.exists(filepath):
        print(f"Skipping {filepath} (not found)")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Define the mock to be injected (keep the torchvision mock)
    mock_code = """
# --- Auto-generated torchvision Mock ---
import torch
import numpy as np
from PIL import Image
from types import ModuleType

class MockTransforms:
    class InterpolationMode:
        NEAREST = 0
        LANCZOS = 1
        BILINEAR = 2
        BICUBIC = 3
        BOX = 4
        HAMMING = 5
    
    @staticmethod
    def Compose(transforms_list):
        def _apply(img):
            for t in transforms_list:
                img = t(img)
            return img
        return _apply
        
    @staticmethod
    def Resize(size, interpolation=3):
        if isinstance(size, int): size = (size, size)
        return lambda img: img.resize(size[::-1], resample=interpolation)
        
    @staticmethod
    def ToTensor():
        return lambda img: torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0
        
    @staticmethod
    def Normalize(mean, std):
        mean_t = torch.tensor(mean).view(-1, 1, 1)
        std_t = torch.tensor(std).view(-1, 1, 1)
        return lambda tensor: (tensor - mean_t) / std_t

transforms = MockTransforms()
InterpolationMode = MockTransforms.InterpolationMode
# ---------------------------------------
"""

    # Inject mock code after typical imports if not already there
    if "Auto-generated torchvision Mock" not in content:
        if "import torch" in content:
            content = content.replace("import torch", "import torch" + mock_code, 1)
        else:
            content = mock_code + content

    # Fix the brittle Llama attention imports
    target_import = """from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2
)"""
    replacement_import = """try:
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaFlashAttention2,
        LlamaSdpaAttention
    )
except ImportError:
    from transformers.models.llama.modeling_llama import LlamaAttention
    try:
        from transformers.models.llama.modeling_llama import LlamaSdpaAttention
    except ImportError:
        LlamaSdpaAttention = LlamaAttention
    LlamaFlashAttention2 = LlamaAttention"""

    # Also handle alternate formatting (sometimes it's on one line or slightly different)
    new_content = content.replace(target_import, replacement_import)
    
    # Handle single line version if it exists
    new_content = new_content.replace("from transformers.models.llama.modeling_llama import LlamaAttention, LlamaFlashAttention2", replacement_import)
    
    # Handle 3-item version
    new_content = new_content.replace("from transformers.models.llama.modeling_llama import LlamaAttention, LlamaFlashAttention2, LlamaSdpaAttention", replacement_import)

    # Safe Infer Patch: Force inputs to model device
    new_content = new_content.replace(
        "images_list.append(image_transform(global_view).to(torch.bfloat16))",
        "images_list.append(image_transform(global_view).to(self.device, dtype=self.dtype))"
    )
    new_content = new_content.replace(
        "images_list.append(image_transform(local_view).to(torch.bfloat16))",
        "images_list.append(image_transform(local_view).to(self.device, dtype=self.dtype))"
    )
    
    # Ensure input_ids are on the correct device
    new_content = new_content.replace(
        "input_ids = tokenizer(conversation, return_tensors='pt', add_special_tokens=False).input_ids",
        "input_ids = tokenizer(conversation, return_tensors='pt', add_special_tokens=False).input_ids.to(self.device)"
    )

    # REMOVE EXPLICIT CUDA CALLS
    cuda_count = new_content.count(".cuda()")
    if cuda_count > 0:
        print(f"Found {cuda_count} instances of .cuda() in {filepath}")
        new_content = new_content.replace(".cuda()", ".to(self.device)")
        print("Replaced all .cuda() with .to(self.device)")
    else:
        print(f"No .cuda() calls found in {filepath}")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Successfully patched {filepath}")

patch_file(r"C:\Users\HazCodes\Documents\gguf\modeling_deepseekocr.py")
patch_file(r"C:\Users\HazCodes\Documents\gguf\modeling_deepseekv2.py")
patch_file(r"C:\Users\HazCodes\Documents\gguf\modeling_deepseek_v2.py") # Some repos use underscores
