#!/usr/bin/env python3
"""
DeepSeek OCR Module
Modern, Efficient OCR for Images and Documents
Utilizes state-of-the-art vision-language models for accurate text extraction
"""

import torch
import torch.nn.functional as F
from transformers import (
    AutoModel, 
    AutoProcessor, 
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor
)
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import os
import time
from typing import Union, List, Dict, Any, Optional, Tuple
import logging
import subprocess
import re
import gc
from pathlib import Path
import cv2
from spellchecker import SpellChecker
import pypdfium2 as pdfium

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeepSeekOCR")

class DeepSeekOCR:
    """
    Advanced OCR system optimized for modern deep learning models.
    Supports multiple backends: BLIP, CLIP, and custom vision-language models.
    Includes preprocessing, post-processing, and quality optimization.
    """
    
    AVAILABLE_MODELS = {
        'blip-base': 'Salesforce/blip-image-captioning-base',
        'blip-large': 'Salesforce/blip-image-captioning-large',
        'clip-vit': 'openai/clip-vit-large-patch14-336',
        'deepseek-ocr': 'deepseek-ai/DeepSeek-OCR',
        'deepseek-ocr-local': r'C:\Users\HazCodes\Documents\gguf\DeepSeek-OCR',  # Local DeepSeek-OCR model
        'custom-blip': None  # For custom local models
    }
    
    def __init__(
        self,
        model_name: str = 'blip-base',
        device: str = 'auto',
        cache_dir: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torchscript: bool = False,
        optimize_for_inference: bool = True,
        max_memory_mb: int = 4096,
        num_workers: int = 4,
        spell_check: bool = True,
        preprocess_images: bool = True
    ):
        """
        Initialize DeepSeek OCR Engine.
        
        Args:
            model_name: Model identifier from AVAILABLE_MODELS or full HF path
            device: 'auto', 'cuda', 'cpu', or 'cuda:0', 'cuda:1', etc.
            cache_dir: Directory for model cache
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization (for very large models)
            torchscript: Use TorchScript optimization
            optimize_for_inference: Enable inference optimizations
            max_memory_mb: Maximum GPU memory to use (in MB)
            num_workers: Number of parallel processing workers
            spell_check: Enable post-processing spell checking
            preprocess_images: Enable image preprocessing for better accuracy
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.cache_dir = cache_dir
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.torchscript = torchscript
        self.optimize_for_inference = optimize_for_inference
        self.max_memory_mb = max_memory_mb
        self.num_workers = num_workers
        self.spell_check = spell_check
        self.preprocess_images = preprocess_images
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.spell_checker = None
        self.initialized = False
        self.model_type = None
        
        # Performance metrics
        self.total_images_processed = 0
        self.avg_processing_time = 0.0
        self.last_process_time = 0.0
        
        # Setup components
        self._setup_dependencies()
        logger.info(f"DeepSeekOCR initialized with device: {self.device}")
    
    def _setup_device(self, device_str: str) -> torch.device:
        """Configure and validate inference device."""
        if device_str == 'auto':
            if torch.cuda.is_available():
                gpu_info = self._get_gpu_info()
                if gpu_info['free_memory_gb'] >= 4:  # Need at least 4GB GPU memory
                    device = torch.device('cuda')
                    logger.info(f"Selected GPU: {gpu_info['name']} ({gpu_info['free_memory_gb']:.1f}GB free)")
                else:
                    logger.warning(f"GPU memory insufficient ({gpu_info['free_memory_gb']:.1f}GB). Using CPU.")
                    device = torch.device('cpu')
            else:
                logger.info("No CUDA devices available. Using CPU.")
                device = torch.device('cpu')
        else:
            # Force CPU if explicitly requested
            if device_str.lower() == 'cpu':
                logger.info("Forcing CPU usage for DeepSeek-OCR.")
                device = torch.device('cpu')
            else:
                device = torch.device(device_str)

        return device
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information using nvidia-smi."""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=name,memory.total,memory.free,driver_version',
                '--format=csv,noheader'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                return {"name": "Unknown", "total_memory_gb": 0, "free_memory_gb": 0}
            
            match = re.search(
                r'([^,]+),\s*(\d+)\s*MiB,\s*(\d+)\s*MiB,\s*([0-9.]+)',
                result.stdout.strip()
            )
            
            if match:
                return {
                    "name": match.group(1).strip(),
                    "total_memory_gb": int(match.group(2)) / 1024,
                    "free_memory_gb": int(match.group(3)) / 1024,
                    "driver": match.group(4).strip()
                }
            
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
        
        return {"name": "Unknown", "total_memory_gb": 0, "free_memory_gb": 0}
    
    def _setup_dependencies(self):
        """Initialize spell checker and other dependencies."""
        if self.spell_check:
            try:
                self.spell_checker = SpellChecker()
                logger.info("Spell checker initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize spell checker: {e}")
                self.spell_checker = None
    
    def initialize(
        self,
        hf_access_token: Optional[str] = None,
        force_download: bool = False
    ) -> Dict[str, Any]:
        """
        Initialize the OCR models with lazy loading.
        
        Args:
            hf_access_token: HuggingFace access token for private/gated models
            force_download: Force re-download of model files
        
        Returns:
            Dictionary with initialization details
        """
        if self.initialized:
            return {
                "status": "already_initialized",
                "model": self.model_name,
                "device": str(self.device)
            }
        
        try:
            init_start = time.time()
            logger.info(f"Initializing OCR model: {self.model_name}")
            
            # Resolve model name
            full_model_name = self.AVAILABLE_MODELS.get(
                self.model_name, 
                self.model_name
            )
            
            if not full_model_name:
                raise ValueError(f"No model configuration found for '{self.model_name}'")
            
            # Determine model type
            if 'blip' in self.model_name.lower():
                self.model_type = 'blip'
                self._setup_blip_model(full_model_name, hf_access_token, force_download)
            elif 'clip' in self.model_name.lower():
                self.model_type = 'clip'
                self._setup_clip_model(full_model_name, hf_access_token, force_download)
            elif 'deepseek' in self.model_name.lower():
                self.model_type = 'deepseek'
                self._setup_deepseek_model(full_model_name, hf_access_token, force_download)
            else:
                raise ValueError(f"Unsupported model type: {self.model_name}")
            
            # Apply optimizations
            self._apply_inference_optimizations()
            
            # Memory management
            self._optimize_memory_usage()
            
            self.initialized = True
            init_time = time.time() - init_start
            
            result = {
                "status": "success",
                "model": full_model_name,
                "device": str(self.device),
                "type": self.model_type,
                "initialization_time": init_time
            }
            
            logger.info(f"✅ OCR Model initialized successfully in {init_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"❌ Model initialization failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _setup_blip_model(self, model_name: str, hf_token: Optional[str], force_download: bool):
        """Initialize BLIP model for image captioning."""
        logger.info(f"Loading BLIP model: {model_name}")
        
        # Load processor
        self.processor = BlipProcessor.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            use_auth_token=hf_token,
            force_download=force_download
        )
        
        # Prepare model loading kwargs
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "use_auth_token": hf_token,
            "force_download": force_download
        }
        
        # Add quantization support
        if self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        else:
            # Use half precision on GPU
            model_kwargs["torch_dtype"] = torch.float16 if self.device.type == 'cuda' else torch.float32
        
        # BLIP models don't support device_map, so skip memory optimization for them
        if 'blip' in self.model_name.lower():
            # BLIP models load entirely on single device
            pass
        elif self.device.type == 'cuda' and not (self.load_in_8bit or self.load_in_4bit):
            # Device map for large models (not BLIP)
            if 'large' in model_name:
                model_kwargs["device_map"] = "balanced"
                max_memory = {0: f"{self.max_memory_mb}MB"}
                model_kwargs["max_memory"] = max_memory
        
        # Load the BLIP model
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if "device_map" not in model_kwargs:
            self.model = self.model.to(self.device)
        
        # Set evaluation mode
        self.model.eval()
        
        logger.info(f"BLIP model '{model_name}' loaded successfully")
    
    def _setup_clip_model(self, model_name: str, hf_token: Optional[str], force_download: bool):
        """Initialize CLIP model for visual understanding."""
        logger.info(f"Loading CLIP model: {model_name}")
        
        # Load processor and model
        try:
            self.processor = CLIPProcessor.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                use_auth_token=hf_token,
                force_download=force_download
            )
            
            self.model = CLIPModel.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                use_auth_token=hf_token,
                force_download=force_download,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            ).to(self.device).eval()
            
            logger.info(f"CLIP model '{model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def _setup_deepseek_model(self, model_name: str, hf_token: Optional[str], force_download: bool):
        """Initialize official DeepSeek-OCR model."""
        logger.info(f"Loading official DeepSeek-OCR model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            use_auth_token=hf_token
        )

        # FIX: Set pad token to avoid warnings
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,  # REQUIRED for custom model class
            "cache_dir": self.cache_dir,
            "use_auth_token": hf_token,
            "attn_implementation": "eager"
        }

        # FIX: Device-specific settings WITHOUT device_map conflict
        if self.device.type == 'cuda':
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            # For CPU: use float32, NO device_map parameter
            model_kwargs["torch_dtype"] = torch.float32
            model_kwargs["low_cpu_mem_usage"] = True

        # Load with AutoModel (NOT AutoModelForCausalLM) - Required for custom DeepseekOCRForCausalLM
        self.model = AutoModel.from_pretrained(
            model_name,
            **model_kwargs
        )

        # Load with AutoModel (NOT AutoModelForCausalLM) - Required for custom DeepseekOCRForCausalLM
        self.model = AutoModel.from_pretrained(
            model_name,
            **model_kwargs
        ).to(self.device).eval()

        # FIX: Aggressively override generation config
        from transformers import GenerationConfig

        gen_config = GenerationConfig(
            max_new_tokens=2048,
            do_sample=False,              # Greedy decoding
            num_beams=1,
            repetition_penalty=1.2,        # Still apply penalty in greedy mode
            no_repeat_ngram_size=3,        # Block 3-gram repetitions
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            forced_eos_token_id=self.tokenizer.eos_token_id
        )

        # Force the config into the model
        self.model.generation_config = gen_config

        # Also try to set on internal language model if it exists
        if hasattr(self.model, 'language_model'):
            self.model.language_model.generation_config = gen_config

        logger.info(f"DeepSeek-OCR model '{model_name}' loaded successfully")



    def _apply_inference_optimizations(self):
        """Apply various optimizations for faster inference."""
        if not self.optimize_for_inference:
            return
        
        logger.info("Applying inference optimizations...")
        
        if self.device.type == 'cuda' and hasattr(torch.backends.cuda, 'sdp_kernel'):
            # Enable flash attention for efficiency
            torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=True
            )
        
        if self.torchscript and self.device.type == 'cpu':
            # Convert to TorchScript for CPU optimization
            try:
                logger.info("Converting to TorchScript...")
                self.model = torch.jit.trace(
                    self.model,
                    example_inputs=self._get_example_inputs(),
                    strict=False
                )
                logger.info("TorchScript optimization applied")
            except Exception as e:
                logger.warning(f"TorchScript conversion failed: {e}")
        
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
    
    def _get_example_inputs(self):
        """Generate example inputs for TorchScript tracing."""
        # Create dummy inputs matching expected format
        pixel_values = torch.randn(1, 3, 224, 224)
        if self.model_type == 'blip':
            input_ids = torch.randint(0, 1000, (1, 10))
            attention_mask = torch.ones(1, 10)
            return {
                "pixel_values": pixel_values.to(self.device),
                "input_ids": input_ids.to(self.device),
                "attention_mask": attention_mask.to(self.device)
            }
        return {"pixel_values": pixel_values.to(self.device)}
    
    def _optimize_memory_usage(self):
        """Apply memory optimizations based on available resources."""
        if self.device.type == 'cuda':
            # Set memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.85)
            
            # Clear unused memory
            torch.cuda.empty_cache()
            
            # Enable gradient checkpointing if available
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                try:
                    self.model.gradient_checkpointing_enable()
                    logger.info("Gradient checkpointing enabled")
                except:
                    pass
        
        logger.info("Memory optimization completed")
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Preprocess image for optimal OCR performance.
        
        Args:
            image: Image path, PIL Image, or numpy array
        
        Returns:
            Preprocessed PIL Image and enhancement report
        """
        # Load image
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            img = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            img = image.convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        if not self.preprocess_images:
            return img, {"preprocessing_applied": False}
        
        start_time = time.time()
        enhancements = {"original_size": img.size, "preprocessing_applied": True}
        
        # Resize while maintaining aspect ratio
        max_size = (640, 640)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        enhancements["resized"] = img.size
        
        # Auto enhance contrast and sharpness
        try:
            # Convert to grayscale for analysis
            gray = img.convert('L')
            hist = gray.histogram()
            
            # Check if image needs contrast enhancement
            if max(hist) / sum(hist) > 0.3:  # If histogram is too concentrated
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.2)
                enhancements["contrast_enhanced"] = True
            
            # Improve sharpness
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.1)
            enhancements["sharpness_enhanced"] = True
            
            # Apply gentle noise reduction
            img = img.filter(ImageFilter.MedianFilter(size=3))
            enhancements["noise_reduction"] = True
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
        
        enhancements["processing_time"] = time.time() - start_time
        return img, enhancements
    
    def _filter_repetitive_words(self, text: str) -> str:
        """Filter out words that repeat more than 3 times in sequence."""
        if not text or len(text.strip()) == 0:
            return text

        try:
            # Split text into words, keeping punctuation attached to words
            words = re.findall(r'\S+', text)
            filtered_words = []
            prev_word = None
            repeat_count = 0

            for word in words:
                # Clean word for comparison (remove punctuation for checking)
                clean_word = re.sub(r'[^\w\s]', '', word).lower()

                if clean_word == prev_word:
                    repeat_count += 1
                    # If this word repeats more than 3 times, skip it
                    if repeat_count >= 3:
                        continue
                else:
                    repeat_count = 0
                    prev_word = clean_word

                filtered_words.append(word)

            # Join back into text
            filtered_text = ' '.join(filtered_words)

            # Log if we filtered anything
            if len(filtered_words) < len(words):
                removed_count = len(words) - len(filtered_words)
                logger.info(f"Filtered out {removed_count} repetitive words")

            return filtered_text

        except Exception as e:
            logger.warning(f"Repetitive word filtering failed: {e}")
            return text

    def spell_check_text(self, text: str) -> str:
        """Apply spell checking to extracted text."""
        if not self.spell_check or not self.spell_checker:
            return text

        try:
            words = text.split()
            corrected_words = []

            for word in words:
                # Skip if it's alphanumeric mixed or contains numbers
                if any(char.isdigit() for char in word):
                    corrected_words.append(word)
                else:
                    # Find candidate corrections
                    candidates = self.spell_checker.candidates(word)
                    if candidates and word not in candidates:
                        # Choose the most likely correction
                        corrected = max(candidates, key=lambda x: 1)
                        corrected_words.append(corrected)
                    else:
                        corrected_words.append(word)

            return ' '.join(corrected_words)

        except Exception as e:
            logger.warning(f"Spell checking failed: {e}")
            return text
    
    def process_image(
        self,
        image: Union[str, Image.Image, np.ndarray],
        prompt: Optional[str] = None,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_beams: int = 3,
        no_repeat_ngram_size: int = 2,
        return_full_results: bool = False
    ) -> Dict[str, Any]:
        """
        Process an image and extract comprehensive information.
        
        Args:
            image: Image path, PIL Image, or numpy array
            prompt: Optional text prompt for conditional generation
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            num_beams: Number of beams for beam search
            no_repeat_ngram_size: Prevent n-gram repetition
            return_full_results: Return detailed results and metadata
        
        Returns:
            Dictionary containing extracted text and metadata
        """
        if not self.initialized:
            self.initialize()
        
        start_time = time.time()
        result = {
            "success": False,
            "text": "",
            "confidence": 0.0,
            "processing_time": 0.0
        }
        
        try:
            # Preprocess image
            preprocessed_img, preprocessing_info = self.preprocess_image(image)
            result["preprocessing"] = preprocessing_info
            
            # Process based on model type
            if self.model_type == 'blip':
                extracted_text = self._process_with_blip(
                    preprocessed_img, prompt, max_new_tokens,
                    temperature, top_p, num_beams, no_repeat_ngram_size
                )
            elif self.model_type == 'clip':
                extracted_text = self._process_with_clip(preprocessed_img)
            elif self.model_type == 'deepseek':
                extracted_text = self._process_with_deepseek(preprocessed_img, prompt)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Apply spell checking
            corrected_text = self.spell_check_text(extracted_text)

            # Filter out repetitive words (more than 3 repetitions)
            corrected_text = self._filter_repetitive_words(corrected_text)

            # Fill result details
            result.update({
                "success": True,
                "text": corrected_text,
                "original_text": extracted_text,
                "confidence": self._calculate_confidence(corrected_text, extracted_text),
                "processing_time": time.time() - start_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Add full results if requested
            if return_full_results:
                result.update({
                    "model_used": self.model_name,
                    "device": str(self.device),
                    "model_type": self.model_type,
                    "generation_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_beams": num_beams,
                        "no_repeat_ngram_size": no_repeat_ngram_size
                    }
                })
            
            # Update statistics
            self._update_statistics(result["processing_time"])
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Image processing failed: {e}")
            self.last_process_time = 0.0
        
        return result
    
    def _process_with_blip(
        self,
        image: Image.Image,
        prompt: Optional[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        num_beams: int,
        no_repeat_ngram_size: int
    ) -> str:
        """Process image using BLIP model."""
        # Prepare inputs
        if prompt:
            inputs = self.processor(images=image, text=[prompt], return_tensors="pt")
        else:
            inputs = self.processor(images=image, return_tensors="pt")
        
        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Generate with optimized parameters
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
                use_cache=True
            )
            
            # Decode output
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0].strip()
        
        # Cleanup
        del inputs, generated_ids
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return generated_text

    def _process_with_deepseek(
        self,
        image: Image.Image,
        prompt: Optional[str] = None
    ) -> str:
        """Process image using official DeepSeek-OCR model."""
        if not prompt:
            prompt = "<|grounding|>Convert the document to markdown."

        # Save temporary image for DeepSeek's custom .infer() which expects a file path
        import tempfile
        temp_dir = tempfile.gettempdir()
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(suffix=".jpg", dir=temp_dir)

        try:
            os.close(fd)
            image.save(tmp_path, format="JPEG")

            # Use the official .infer() method - it uses generation_config internally
            res = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=tmp_path,
                output_path=temp_dir,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False
            )

            return res

        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {tmp_path}: {e}")

    def _process_with_clip(self, image: Image.Image) -> str:
        """Process image using CLIP model."""
        # Placeholder for CLIP-based extraction
        # This would typically involve using CLIP embeddings with similarity search
        
        # Get image features
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # For now return a basic description
        return "CLIP model processing - visual features extracted"


    
    def _calculate_confidence(self, corrected_text: str, original_text: str) -> float:
        """Calculate confidence score based on several factors."""
        if not original_text or len(original_text.strip()) == 0:
            return 0.0
        
        confidence_factors = []
        
        # Text length factor (longer is generally better)
        length_score = min(len(original_text) / 100, 1.0)
        confidence_factors.append(length_score * 0.3)
        
        # Character variety factor
        char_variety = len(set(original_text)) / len(original_text) if len(original_text) > 0 else 0
        confidence_factors.append(char_variety * 0.2)
        
        # Word count factor
        word_count = len(original_text.split())
        word_score = min(word_count / 20, 1.0)
        confidence_factors.append(word_score * 0.3)
        
        # Correction factor (less correction is better)
        if corrected_text != original_text:
            corrections = sum(1 for a, b in zip(corrected_text.split(), original_text.split()) if a != b)
            correction_score = max(0, 1 - (corrections / max(len(original_text.split()), 1)))
            confidence_factors.append(correction_score * 0.2)
        else:
            confidence_factors.append(0.2)
        
        return float(np.mean(confidence_factors))
    
    def process_batch(
        self,
        image_paths: List[str],
        batch_size: int = 4,
        show_progress: bool = True,
        **process_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images in batches.
        
        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process simultaneously
            show_progress: Show progress information
            **process_kwargs: Additional arguments passed to process_image
        
        Returns:
            List of processing results for each image
        """
        if not image_paths:
            logger.warning("No image paths provided for batch processing")
            return []
        
        total_images = len(image_paths)
        results = []
        processed = 0
        
        logger.info(f"Starting batch processing of {total_images} images, batch size: {batch_size}")
        batch_start_time = time.time()
        
        for i in range(0, total_images, batch_size):
            # Prepare batch
            batch_paths = image_paths[i:i + batch_size]
            batch_results = []
            
            batch_process_start = time.time()
            
            # Process each image in the batch
            for image_path in batch_paths:
                try:
                    result = self.process_image(image_path, **process_kwargs)
                    
                    # Add filename information
                    result["filename"] = os.path.basename(image_path)
                    result["filepath"] = image_path
                    
                    batch_results.append(result)
                    processed += 1
                    
                except Exception as e:
                    batch_results.append({
                        "success": False,
                        "error": str(e),
                        "filename": os.path.basename(image_path) if image_path else "unknown",
                        "filepath": image_path,
                        "processing_time": 0.0
                    })
            
            results.extend(batch_results)
            
            batch_time = time.time() - batch_process_start
            
            # Progress logging
            if show_progress and processed % batch_size == 0:
                remaining = total_images - processed
                avg_time_per_image = (time.time() - batch_start_time) / processed
                estimated_time = remaining * avg_time_per_image
                
                logger.info(
                    f"Progress: {processed}/{total_images} "
                    f"({processed/total_images*100:.1f}%) - "
                    f"Avg: {avg_time_per_image:.2f}s/image, "
                    f"ETA: {estimated_time/60:.1f}min"
                )
        
        total_time = time.time() - batch_start_time
        success_count = sum(1 for r in results if r.get("success", False))
        
        logger.info(
            f"✅ Batch processing completed: "
            f"{success_count}/{total_images} successful, "
            f"{total_time:.1f}s total ({total_time/total_images:.2f}s per image)"
        )
        
        return results

    def process_pdf(
        self,
        pdf_path: str,
        scale: int = 2,
        **process_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process a PDF file by converting its pages to images and performing OCR.
        
        Args:
            pdf_path: Path to the PDF file
            scale: Scaling factor for rendering PDF pages (higher = better quality)
            **process_kwargs: Additional arguments for process_image
            
        Returns:
            List of OCR results for each page
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        logger.info(f"📄 Processing PDF: {pdf_path}")
        results = []
        
        try:
            pdf = pdfium.PdfDocument(pdf_path)
            num_pages = len(pdf)
            logger.info(f"Total pages: {num_pages}")
            
            for i in range(num_pages):
                page = pdf[i]
                # Render page to bitmap
                bitmap = page.render(
                    scale=scale,
                    rotation=0,
                )
                # Convert to PIL Image
                pil_image = bitmap.to_pil()
                
                logger.info(f"Processing page {i+1}/{num_pages}...")
                result = self.process_image(pil_image, **process_kwargs)
                result["page_number"] = i + 1
                result["filename"] = f"{os.path.basename(pdf_path)}_page_{i+1}"
                
                results.append(result)
                
                # Cleanup to save memory
                del pil_image
                bitmap.close()
            
            pdf.close()
            logger.info(f"✅ Finished processing PDF: {pdf_path}")
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            results.append({
                "success": False,
                "error": str(e),
                "filename": os.path.basename(pdf_path),
                "processing_time": 0.0
            })
            
        return results

    def _update_statistics(self, processing_time: float):
        """Update performance statistics."""
        self.last_process_time = processing_time
        self.total_images_processed += 1
        
        # Update running average
        if self.avg_processing_time == 0:
            self.avg_processing_time = processing_time
        else:
            # Exponential moving average
            self.avg_processing_time = 0.9 * self.avg_processing_time + 0.1 * processing_time
    
    def export_results(self, results: List[Dict[str, Any]], format: str = "json", **kwargs):
        """
        Export OCR results in various formats.
        
        Args:
            results: List of processing results
            format: Export format ('json', 'text', 'html', 'pdf')
            **kwargs: Export-specific parameters
        
        Returns:
            Exported content or file path
        """
        if not results:
            logger.warning("No results to export")
            return None
        
        try:
            if format.lower() == "json":
                import json
                return json.dumps(results, indent=2, ensure_ascii=False)
            
            elif format.lower() == "text":
                text_content = ""
                for i, result in enumerate(results, 1):
                    if result.get("success"):
                        text_content += f"Image {i}: {result['filename']}\n"
                        text_content += f"Text: {result['text']}\n"
                        text_content += f"Confidence: {result['confidence']:.2f}\n"
                        text_content += f"Time: {result['processing_time']:.2f}s\n\n"
                    else:
                        text_content += f"Image {i}: Failed - {result.get('error', 'Unknown error')}\n\n"
                return text_content
            
            elif format.lower() == "html":
                html_content = """
                <html>
                <head><title>OCR Results</title></head>
                <body>
                <h1>DeepSeek OCR Results</h1>
                """
                for i, result in enumerate(results, 1):
                    if result.get("success"):
                        html_content += f"""
                        <div class='result'>
                            <h2>Image {i}: {result['filename']}</h2>
                            <p class='text'>{result['text']}</p>
                            <p class='metadata'>Confidence: {result['confidence']:.2f} | Time: {result['processing_time']:.2f}s</p>
                        </div>
                        """
                    else:
                        html_content += f"""
                        <div class='result error'>
                            <h2>Image {i}: Processing Failed</h2>
                            <p class='error'>{result.get('error', 'Unknown error')}</p>
                        </div>
                        """
                html_content += "</body></html>"
                return html_content
            
            elif format.lower() == "pdf":
                logger.warning("PDF export not implemented yet")
                return None
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
        
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about OCR processing."""
        return {
            "total_images_processed": self.total_images_processed,
            "average_processing_time": self.avg_processing_time,
            "last_processing_time": self.last_process_time,
            "device": str(self.device),
            "model_loaded": self.initialized,
            "model_name": self.model_name,
            "model_type": self.model_type
        }
    
    def reset_statistics(self):
        """Reset performance statistics."""
        self.total_images_processed = 0
        self.avg_processing_time = 0.0
        self.last_process_time = 0.0
        logger.info("Statistics reset")
    
    def __repr__(self):
        """String representation of the OCR engine."""
        status = "Active" if self.initialized else "Not Initialized"
        device_info = f"{self.device.type}"
        if self.device.type == 'cuda':
            device_info += f":{self.device.index}"
        
        return (
            f"DeepSeekOCR(model='{self.model_name}', device={device_info}, "
            f"status={status}, processed={self.total_images_processed})"
        )
    
    def __enter__(self):
        """Context manager entry."""
        if not self.initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        # Perform cleanup
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        logger.info("DeepSeekOCR context manager cleanup completed")

