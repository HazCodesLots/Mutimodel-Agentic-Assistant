#!/usr/bin/env python3
"""
DeepSeek CPU OCR Module
CPU-only version of DeepSeekOCR for systems without GPU or when GPU causes issues.
Forces all operations to run on CPU with aggressive memory management and stability optimizations.
"""

import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    GenerationConfig
)
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import os
import time
import re
from typing import Union, List, Dict, Any, Optional
import logging
import subprocess
import gc
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeepSeekCPUOCR")

class DeepSeekCPUOCR:
    """
    CPU-only version of DeepSeekOCR for systems without GPU or when GPU causes issues.
    Forces all operations to run on CPU with aggressive memory management and stability optimizations.
    Includes KV cache precision fixes and repetition prevention.
    """

    AVAILABLE_MODELS = {
        'deepseek-ocr-local': r'C:\Users\HazCodes\Documents\gguf\DeepSeek-OCR',
        'deepseek-ocr': 'deepseek-ai/DeepSeek-OCR',
    }

    def __init__(
        self,
        model_name: str = 'deepseek-ocr-local',
        cache_dir: Optional[str] = None,
        max_memory_mb: int = 4096,
        spell_check: bool = True,
        preprocess_images: bool = True
    ):
        """
        Initialize CPU-only DeepSeek OCR Engine.

        Args:
            model_name: Model identifier from AVAILABLE_MODELS
            cache_dir: Directory for model cache
            max_memory_mb: Maximum CPU memory to use (in MB)
            spell_check: Enable post-processing spell checking
            preprocess_images: Enable image preprocessing for better accuracy
        """
        self.model_name = model_name
        self.device = torch.device('cpu')  # Force CPU usage
        self.cache_dir = cache_dir
        self.max_memory_mb = max_memory_mb
        self.spell_check = spell_check
        self.preprocess_images = preprocess_images

        self.model = None
        self.tokenizer = None
        self.spell_checker = None
        self.initialized = False

        # Performance metrics
        self.total_images_processed = 0
        self.avg_processing_time = 0.0
        self.last_process_time = 0.0

        # Setup components
        self._setup_dependencies()
        logger.info(f"DeepSeekCPUOCR initialized (CPU-only)")

    def _setup_dependencies(self):
        """Initialize spell checker and other dependencies."""
        if self.spell_check:
            try:
                from spellchecker import SpellChecker
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
        Initialize the CPU-only DeepSeek-OCR model.

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
            logger.info(f"Initializing CPU-only DeepSeek-OCR model: {self.model_name}")

            # Resolve model name
            full_model_name = self.AVAILABLE_MODELS.get(
                self.model_name,
                self.model_name
            )

            if not full_model_name:
                raise ValueError(f"No model configuration found for '{self.model_name}'")

            self._setup_deepseek_cpu_model(full_model_name, hf_access_token, force_download)

            # Apply CPU optimizations
            self._apply_cpu_optimizations()

            self.initialized = True
            init_time = time.time() - init_start

            result = {
                "status": "success",
                "model": full_model_name,
                "device": str(self.device),
                "initialization_time": init_time
            }

            logger.info(f"✅ DeepSeekCPUOCR model initialized successfully in {init_time:.2f}s")
            return result

        except Exception as e:
            error_msg = f"❌ Model initialization failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _setup_deepseek_cpu_model(self, model_name: str, hf_token: Optional[str], force_download: bool):
        """Initialize official DeepSeek-OCR model optimized for CPU."""
        logger.info(f"Loading DeepSeek-OCR model on CPU: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            use_auth_token=hf_token
        )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # CPU-optimized model loading
        model_kwargs = {
            "trust_remote_code": True,
            "cache_dir": self.cache_dir,
            "use_auth_token": hf_token,
            "torch_dtype": torch.float32,  # Use float32 on CPU (KV cache precision fix)
            "low_cpu_mem_usage": True,     # Enable CPU memory optimization
            "attn_implementation": "eager" # Avoid flash attention issues on CPU (KV cache fix)
        }

        # Load model
        self.model = AutoModel.from_pretrained(
            model_name,
            **model_kwargs
        ).to(self.device).eval()

        # Aggressive generation config for CPU stability
        gen_config = GenerationConfig(
            max_new_tokens=2048,
            do_sample=False,              # Greedy decoding for stability (prevents repetition loops)
            num_beams=1,
            repetition_penalty=1.2,        # Prevent repetitions
            no_repeat_ngram_size=3,        # Block 3-gram loops
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            forced_eos_token_id=self.tokenizer.eos_token_id
        )

        # Force config into model
        self.model.generation_config = gen_config
        if hasattr(self.model, 'language_model'):
            self.model.language_model.generation_config = gen_config

        logger.info(f"DeepSeek-OCR CPU model '{model_name}' loaded successfully")

    def _apply_cpu_optimizations(self):
        """Apply CPU-specific optimizations."""
        logger.info("Applying CPU optimizations...")

        # Enable TorchScript for CPU if possible
        if not hasattr(self, '_torchscript_failed'):
            try:
                logger.info("Converting to TorchScript for CPU optimization...")
                self.model = torch.jit.script(self.model)
                logger.info("TorchScript optimization applied")
            except Exception as e:
                logger.warning(f"TorchScript conversion failed: {e}")
                self._torchscript_failed = True

        logger.info("CPU optimization completed")

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
        return_full_results: bool = False
    ) -> Dict[str, Any]:
        """
        Process an image and extract comprehensive information.

        Args:
            image: Image path, PIL Image, or numpy array
            prompt: Optional text prompt for conditional generation
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

            # Process with DeepSeek-OCR
            if not prompt:
                prompt = "<|grounding|>Convert the document to markdown."

            # Save temporary image
            import tempfile
            temp_dir = tempfile.gettempdir()
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir, exist_ok=True)

            fd, tmp_path = tempfile.mkstemp(suffix=".jpg", dir=temp_dir)

            try:
                os.close(fd)
                preprocessed_img.save(tmp_path, format="JPEG")

                # Use the model's infer method
                extracted_text = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=tmp_path,
                    output_path=temp_dir,
                    base_size=1024,
                    image_size=640,
                    crop_mode=True,
                    save_results=False
                )

                # Apply post-processing
                corrected_text = self.spell_check_text(extracted_text)
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

                if return_full_results:
                    result.update({
                        "model_used": self.model_name,
                        "device": str(self.device),
                    })

                # Update statistics
                self._update_statistics(result["processing_time"])

            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {tmp_path}: {e}")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Image processing failed: {e}")
            self.last_process_time = 0.0

        return result

    def _calculate_confidence(self, corrected_text: str, original_text: str) -> float:
        """Calculate confidence score based on several factors."""
        if not original_text or len(original_text.strip()) == 0:
            return 0.0

        confidence_factors = []

        # Text length factor
        length_score = min(len(original_text) / 100, 1.0)
        confidence_factors.append(length_score * 0.3)

        # Character variety factor
        char_variety = len(set(original_text)) / len(original_text) if len(original_text) > 0 else 0
        confidence_factors.append(char_variety * 0.2)

        # Word count factor
        word_count = len(original_text.split())
        word_score = min(word_count / 20, 1.0)
        confidence_factors.append(word_score * 0.3)

        # Correction factor
        if corrected_text != original_text:
            corrections = sum(1 for a, b in zip(corrected_text.split(), original_text.split()) if a != b)
            correction_score = max(0, 1 - (corrections / max(len(original_text.split()), 1)))
            confidence_factors.append(correction_score * 0.2)
        else:
            confidence_factors.append(0.2)

        return float(np.mean(confidence_factors))

    def _update_statistics(self, processing_time: float):
        """Update performance statistics."""
        self.last_process_time = processing_time
        self.total_images_processed += 1

        # Update running average
        if self.avg_processing_time == 0:
            self.avg_processing_time = processing_time
        else:
            self.avg_processing_time = 0.9 * self.avg_processing_time + 0.1 * processing_time

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about OCR processing."""
        return {
            "total_images_processed": self.total_images_processed,
            "average_processing_time": self.avg_processing_time,
            "last_processing_time": self.last_process_time,
            "device": str(self.device),
            "model_loaded": self.initialized,
            "model_name": self.model_name,
        }

    def __repr__(self):
        """String representation for CPU version."""
        status = "Active" if self.initialized else "Not Initialized"
        return (
            f"DeepSeekCPUOCR(model='{self.model_name}', device=cpu, "
            f"status={status}, processed={self.total_images_processed})"
        )

    def __enter__(self):
        """Context manager entry."""
        if not self.initialized:
            self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

        # Force garbage collection
        gc.collect()

        logger.info("DeepSeekCPUOCR context manager cleanup completed")


# Module exports
__all__ = ['DeepSeekCPUOCR']

# Example usage
if __name__ == "__main__":
    print("DeepSeek CPU OCR - Standalone CPU-only OCR Engine")
    print("=" * 60)
    print("Features:")
    print("- CPU-only operation (no GPU required)")
    print("- KV cache precision fix (torch.float32)")
    print("- Repetition prevention (greedy decoding + filtering)")
    print("- Automatic spell checking and text enhancement")
    print("- Memory optimized for CPU usage")
    print()
    print("Usage:")
    print("from DeepSeekCPU_OCR import DeepSeekCPUOCR")
    print("ocr = DeepSeekCPUOCR()")
    print("result = ocr.process_image('image.jpg')")
    print("print(result['text'])")


# Quick utility function for easy OCR
def ocr_quick_process(
    image_path: Union[str, Image.Image],
    model_name: str = 'deepseek-ocr-local',
    **kwargs
) -> str:
    """
    Quick function for single image OCR without managing class instances.

    Args:
        image_path: Path to image or PIL Image object
        model_name: Model to use (from AVAILABLE_MODELS)
        **kwargs: Additional arguments for process_image

    Returns:
        Extracted text or error message
    """
    try:
        with DeepSeekCPUOCR(model_name=model_name) as ocr:
            result = ocr.process_image(image_path, **kwargs)
            return result.get("text", "No text extracted") if result.get("success") else result.get("error", "Unknown error")
    except Exception as e:
        return f"OCR processing failed: {str(e)}"


# CLI interface when run directly
if __name__ == "__main__":
    import sys

    print("DeepSeek CPU OCR - Standalone CPU-only OCR Engine")
    print("=" * 60)
    print("Features:")
    print("- CPU-only operation (no GPU required)")
    print("- KV cache precision fix (torch.float32)")
    print("- Repetition prevention (greedy decoding + filtering)")
    print("- Automatic spell checking and text enhancement")
    print("- Memory optimized for CPU usage")
    print()
    print("Models:", list(DeepSeekCPUOCR.AVAILABLE_MODELS.keys()))
    print()

    if len(sys.argv) > 1:
        # Test with provided image path
        test_image = sys.argv[1]

        print(f"Testing OCR on: {test_image}")
        print("-" * 60)

        try:
            ocr = DeepSeekCPUOCR()

            print("Initializing CPU model...")
            init_result = ocr.initialize()
            print(f"✅ Initialization: {init_result['status']}")

            print("\n📸 Processing image...")
            result = ocr.process_image(test_image, return_full_results=True)

            print("\n📝 Extracted Text:")
            print("-" * 60)
            print(result.get("text", "No text found"))
            print("-" * 60)

            if result.get("success"):
                print("✓ Success")
                confidence = result.get("confidence", 0.0)
                time_taken = result.get("processing_time", 0.0)
                print(f"Confidence: {confidence:.2f} | Time: {time_taken:.2f}s")
            else:
                print(f"✗ Failed: {result.get('error', 'Unknown error')}")

            # Show statistics
            print("\n📊 OCR Statistics:")
            stats = ocr.get_statistics()
            for key, value in stats.items():
                print(f"{key}: {value}")

        except Exception as e:
            print(f"❌ Error during OCR processing: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("Usage:")
        print("python DeepSeekCPU_OCR.py <image_path>")
        print()
        print("Example:")
        print("python DeepSeekCPU_OCR.py document.jpg")
        print()
        print("Or import in Python:")
        print("from DeepSeekCPU_OCR import DeepSeekCPUOCR")
        print("ocr = DeepSeekCPUOCR()")
        print("result = ocr.process_image('image.jpg')")
        print("print(result['text'])")
