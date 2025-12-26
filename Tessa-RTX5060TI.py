#!/usr/bin/env python3
"""
Tessa-GGUF.py
Modern Multimodal AI Assistant using GGUF models for LLM inference
Features:
- GPU Memory Management (14GB VRAM limit for large models)
- Input length-based routing (short to aya-23-8B, medium to DeepSeek-R1, long to qwen2.5-coder)
- Simplified user interface with Gradio
- Statistics and performance monitoring
"""

import os
import json
from datetime import datetime
import gradio as gr
import subprocess
import re
import torch
from typing import Tuple, Optional, List
from ToolBox.DeepSeekOCR import DeepSeekOCR

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Tessa")

# GGUF model paths
GGUF_MODELS = {
    "aya-23-8b": r"C:\Users\HazCodes\Documents\gguf\aya-23-8B-Q4_K_M.gguf",
    "deepseek-r1": r"C:\Users\HazCodes\Documents\gguf\DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
    "qwen2.5-coder": r"C:\Users\HazCodes\Documents\gguf\qwen2.5-coder-32b-instruct-q4_k_m.gguf"
}

# Model memory configurations for hardware limitations
#RTX 5060TI 16GB
MODEL_CONFIGS = {
    "aya-23-8b": {
        "name": "Aya-23-8B",
        "description": "General purpose multilingual model",
        "max_layers": 20,
        "cpu_layers": 0,
        "gpu_layers": 20,
        "memory_usage": "4.7GB"
    },
    "deepseek-r1": {
        "name": "DeepSeek-R1",
        "description": "Research and analysis model",
        "max_layers": 64,
        "gpu_layers": 50,
        "memory_usage": "11GB"
    },
    "qwen2.5-coder": {
        "name": "Qwen2.5-Coder",
        "description": "Code generation and programming",
        "max_layers": 64,
        "gpu_layers": 50,
        "memory_usage": "11GB"
    }
}


# Hardware optimization settings
HARDWARE_CONFIG = {
    "cpu_cores": 6,
    "cpu_threads": 12,
    "total_ram_gb": 32,
    "gpu_vram_gb": 14,
    "cpu_ram_available_gb": 18
}

HISTORY_DIR = r"D:\Work\Wraps\Tessa\History"
os.makedirs(HISTORY_DIR, exist_ok=True)


class AgentContext:
    """Manages conversation context and message history."""
    def __init__(self):
        self.history = []
        self.latest_input = ""
        self.latest_output = ""
        self.image_context = ""

    def add_message(self, source: str, content: str):
        source = source.lower()
        self.history.append({"source": source, "content": content})
        
        if source == "user":
            self.latest_input = content
        elif source in ["gguf-aya", "gguf-deepseek", "gguf-qwen"]:
            self.latest_output = content

    def get_conversation(self):
        return [msg["content"] for msg in self.history]

ctx = AgentContext()

ocr_engine = DeepSeekOCR(model_name='deepseek-ocr-local', device='cuda')

model_cache = {}


def check_gpu_memory() -> dict:
    """Check available GPU memory and adjust layer allocation accordingly."""
    try:
        if torch.cuda.is_available():
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=memory.total,memory.free',
                '--format=csv,noheader'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                match = re.search(r'(\d+).*?(\d+)', result.stdout)
                if match:
                    total_mem = int(match.group(1))
                    free_mem = int(match.group(2))
                    
                    total_gb = total_mem / 1024
                    free_gb = free_mem / 1024
                    
                    print(f"GPU Memory: Total={total_gb:.1f}GB, Free={free_gb:.1f}GB")
                    
                    available_vram_gb = min(free_gb, HARDWARE_CONFIG['gpu_vram_gb'])
                    estimated_layers = int((available_vram_gb * 1024) / 400)
                    if available_vram_gb >= 14:
                        estimated_layers = int((14 * 1024) / 400)
                        print(f"Maximizing GPU usage: {estimated_layers} layers for 14GB VRAM")
                    else:
                        print(f"Using {estimated_layers} layers for {available_vram_gb:.1f}GB VRAM")
                    
                    return {
                        'total_gb': total_gb,
                        'free_gb': free_gb,
                        'available_vram_gb': available_vram_gb,
                        'estimated_layers': estimated_layers
                    }
    except Exception as e:
        print(f"⚠️ Error checking GPU memory: {e}. Using default layer allocation.")
    return None

def load_gguf_model(model_name: str) -> object:
    """Load a GGUF model using llama-cpp-python with hardware-optimized settings."""
    try:
        from llama_cpp import Llama
        import torch
        
        model_path = GGUF_MODELS[model_name]
        config = MODEL_CONFIGS[model_name]
        
        print(f"Loading {config['name']} ({config['description']})")
        print(f"Memory allocation: {config['memory_usage']} total")
        
        gpu_info = check_gpu_memory()
        if gpu_info and gpu_info['available_vram_gb'] >= 14:
            n_gpu_layers = min(config['max_layers'], config['gpu_layers'])
            n_cpu_layers = config['max_layers'] - n_gpu_layers
            
            print(f"Hardware Optimized Offloading:")
            print(f"  GPU Layers: {n_gpu_layers} / {config['max_layers']}")
            print(f"  CPU Layers: {n_cpu_layers}")
        else:
            # Fallback to balanced configuration
            n_gpu_layers = config['gpu_layers']
            n_cpu_layers = config['max_layers'] - n_gpu_layers
            print(f"Using standard offload: GPU {n_gpu_layers}, CPU {n_cpu_layers}")
        
        # Load model with hardware-optimized settings
        model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=4096,  # Context length
            n_batch=512,  # Batch size
            verbose=True,
            seed=42,
            # Hardware optimization settings
            use_mmap=False,  # Disable memory mapping for better GPU utilization
            use_mlock=False,  # Allow memory management
            n_threads=HARDWARE_CONFIG["cpu_threads"],  # Use all available CPU threads
            main_gpu=0,  # Primary GPU device
            # Memory management for 32GB RAM + RTX 5060 Ti 16GB
            # GPU: 14GB limit, CPU: 18GB available
            flash_attn=False,  # Explicitly disable to avoid backend compatibility issues
        )
        
        return {"model": model, "tokenizer": None, "config": config}
        
    except ImportError:
        print("⚠️ llama-cpp-python not installed. Please install it with: pip install llama-cpp_python")
        return None
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return None


def build_prompt(context: AgentContext) -> list:
    """Build messages list for GGUF model."""
    messages = []
    for message in context.history:
        source = message["source"].lower()
        if source == "user":
            messages.append({"role": "user", "content": message['content']})
        elif source == "system":
            messages.append({"role": "system", "content": message['content']})
        elif source.startswith("gguf-"):
            messages.append({"role": "assistant", "content": message['content']})
    return messages

def get_model_for_input(user_input: str) -> str:
    """Determine which model to use based on input length."""
    input_length = len(user_input.strip())
    
    if input_length <= 100:  # Short input
        return "aya-23-8b"
    elif input_length <= 500:  # Medium input
        return "deepseek-r1"
    else:  # Long input
        return "qwen2.5-coder"


def gguf_respond(user_input: str, context: AgentContext, model_name: str, document_context: str = None) -> str:
    """Generate response using GGUF model with llama-cpp-python."""
    context.add_message("user", user_input)

    # Get or load model
    if model_name not in model_cache:
        model_cache[model_name] = load_gguf_model(model_name)
    
    model_data = model_cache[model_name]
    if not model_data:
        return "[Error: Failed to load model]"
    
    model = model_data["model"]

    messages = build_prompt(context)

    # Check for existing system messages in prompt
    has_system = any(msg["role"] == "system" for msg in messages)
    
    if not has_system:
        # If no system message found in history, use default
        system_prompt = """IMPORTANT: You MUST respond in the user's language.
When users speak in Spanish, Italian, or German, please respond in the SAME language.

Language Matching Rules:
- User speaks Spanish → You respond in Spanish
- User speaks Italian → You respond in Italian
- User speaks German → You respond in German
- User speaks English → You respond in English

Always match the user's language for your responses."""
        
        if document_context:
            system_prompt += f"\n\nUse the following document context to answer questions:\n\n{document_context.strip()}"

        messages.insert(0, {
            "role": "system",
            "content": system_prompt
        })

    try:
        # Generate response using native chat completion which uses model's internal template
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            stream=False
        )
        
        # Extract the generated text from chat completion format
        result = response['choices'][0]['message']['content'].strip()
        
        context.add_message(f"gguf-{model_name}", result)
        return result
        
    except Exception as e:
        error_msg = f"[Error generating response: {str(e)}]"
        context.add_message(f"gguf-{model_name}", error_msg)
        return error_msg


def route_to_model(prompt: str, context: AgentContext, force_model: str = None) -> str:
    """Route user input to appropriate GGUF model based on input length or force model."""
    if force_model:
        model_name = force_model
        print(f"Using forced model: {MODEL_CONFIGS[model_name]['name']}")
    else:
        model_name = get_model_for_input(prompt)
        print(f"Routing to {MODEL_CONFIGS[model_name]['name']} model")
    
    return gguf_respond(prompt, context, model_name)


def save_chat() -> str:
    """Save conversation history to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(HISTORY_DIR, f"chat_{timestamp}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(ctx.history, f, indent=2)
    return f"Chat saved to {path}"


def format_pairs(history: list) -> list:
    """Convert chat history to Gradio's chat format."""
    formatted = []
    for msg in history:
        # Skip system messages to prevent them from being displayed
        if msg["source"] == "system":
            continue
        role = "user" if msg["source"] == "user" else "assistant"
        content = msg["content"]
        formatted.append({"role": role, "content": content})
    return formatted


def load_chat(file) -> list:
    """Load conversation history from JSON file."""
    with open(file.name, 'r', encoding='utf-8') as f:
        ctx.history = json.load(f)
    return gr.update(value=format_pairs(ctx.history))

def cleanup_models():
    """Clean up loaded models when switching modes or shutting down."""
    print("🧹 Cleaning up model cache...")
    for model_name, model_data in model_cache.items():
        if model_data and "model" in model_data:
            try:
                del model_data["model"]
                print(f"✅ Unloaded {model_name}")
            except Exception as e:
                print(f"⚠️ Error unloading {model_name}: {e}")
    model_cache.clear()
    # Force garbage collection
    import gc
    gc.collect()
    print("🧹 Model cache cleared")


def chat_query(user_input: str, file_upload = None, force_model: str = None):
    """Handle chat queries with optional file upload and model forcing."""
    if not user_input.strip() and not file_upload:
        return format_pairs(ctx.history), "", "Please enter a message or upload a file."
    
    try:
        status_msg = ""
        document_context = ""
        
        if file_upload:
            file_path = file_upload.name
            filename = os.path.basename(file_path).lower()
            status_msg = f"Processing '{filename}' with OCR..."
            print(f"OCR: Processing {file_path}")
            
            ocr_results = []
            if filename.endswith(".pdf"):
                ocr_results = ocr_engine.process_pdf(file_path)
            elif filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                res = ocr_engine.process_image(file_path)
                ocr_results = [res]
            
            # Combine extracted text
            extracted_texts = []
            for res in ocr_results:
                if res.get("success"):
                    text = res.get("text", "").strip()
                    if text:
                        extracted_texts.append(text)
            
            if extracted_texts:
                document_context = "\n---\n".join(extracted_texts)
                # Add to internal context for model to "see" it
                ctx.add_message("system", f"[Extracted from {filename} via OCR]:\n{document_context}")
                status_msg = f"OCR complete for '{filename}'! Extracted {len(extracted_texts)} page(s)/image(s)."
            else:
                status_msg = f"OCR failed or no text found in '{filename}'."
            
            # If no user input, just return file acknowledgement in status
            if not user_input.strip():
                return format_pairs(ctx.history), "", status_msg
        
        if user_input.strip():
            # Pass document context to response if available
            response = gguf_respond(user_input, ctx, force_model or get_model_for_input(user_input), document_context=document_context)
            return format_pairs(ctx.history), "", status_msg or "Ready"
        
        return format_pairs(ctx.history), "", status_msg or "Ready"
        
    except Exception as e:
        error_msg = f"Error in chat query: {str(e)}"
        logger.error(error_msg)
        return format_pairs(ctx.history), error_msg


def main():
    """Main function to start the Tessa assistant."""
    print("Starting Tessa - Multimodal Assistant")
    print(f"Models: {', '.join(GGUF_MODELS.keys())}")
    print(f"GPU Memory Limit: {HARDWARE_CONFIG['gpu_vram_gb']}GB")
    print("Initializing...")
    
    ctx.add_message("system", "Please respond in English")
    
    try:
        APP_CSS = """
            /* Dark theme */
            .gradio-container, .gradio-container > div {
                background-color: #0b0f19 !important;
            }

            /* Title spacing */
            .tessa-title, .tessa-title * {
                margin-top: 0 !important;
                padding-top: 0 !important;
                margin-bottom: 8px !important;
            }
            .tessa-header-row {
                margin-top: 0 !important;
                padding-top: 0 !important;
            }

            /* Dark textbox - always grey */
            .input-row textarea, 
            .input-row textarea:focus,
            .input-row .border-none,
            .input-row .border-none:focus {
                background-color: #1a1a1a !important;
                color: #ffffff !important;
                border: 1px solid #444 !important;
                font-size: 16px !important;
                box-shadow: none !important;
                height: 56px !important;
                min-height: 56px !important;
                max-height: 56px !important;
                padding: 12px 16px !important;
                box-sizing: border-box !important;
            }
            
            /* Target Gradio's internal container for the textbox */
            .input-row .gr-box, 
            .input-row .gr-input, 
            .input-row .gr-form,
            .input-row [data-testid="textbox"] {
                background-color: #1a1a1a !important;
                border-color: #444 !important;
            }
            
            /* Ensure textbox container has dark theme */
            .input-row .gr-textbox {
                background-color: #1a1a1a !important;
            }
            
            /* Fix placeholder text color */
            .input-row textarea::placeholder {
                color: #888 !important;
            }
            
            /* Force dark theme for all textbox states */
            .input-row .gr-textbox input {
                background-color: #1a1a1a !important;
                color: #ffffff !important;
            }
            
            /* Ensure dark theme when not focused */
            .input-row .gr-textbox:not(:focus-within) {
                background-color: #1a1a1a !important;
            }

            /* Hide textarea scrollbar completely */
            textarea {
                resize: none !important;
                overflow: hidden !important;
                scrollbar-width: none !important;
                -ms-overflow-style: none !important;
            }
            textarea::-webkit-scrollbar {
                display: none !important;
                width: 0 !important;
                height: 0 !important;
            }
            
            /* Force all buttons in input row to same height */
            .input-row button {
                min-height: 56px !important;
                max-height: 56px !important;
                height: 56px !important;
                font-size: 16px !important;
                font-weight: bold !important;
            }
            
            /* Target upload button specifically */
            .input-row label[for] {
                min-height: 56px !important;
                max-height: 56px !important;
                height: 56px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                font-size: 16px !important;
                font-weight: bold !important;
            }
            
            /* Fix theme mismatch for textbox */
            .input-row textarea,
            .input-row .border-none {
                background-color: #1a1a1a !important;
                color: #ffffff !important;
                border: 1px solid #444 !important;
                font-size: 16px !important;
            }
            
            /* Ensure consistent styling for all input elements */
            .input-row .gr-form,
            .input-row .border-none {
                background-color: #1a1a1a !important;
            }
            
            /* Target by ID - highest specificity */
            #chat-input,
            #chat-input textarea,
            #chat-input .border-none {
                background-color: #1a1a1a !important;
                color: #ffffff !important;
            }
        """

        
        with gr.Blocks() as demo:
            gr.Markdown("# 🤖 Tessa - Multimodal Assistant", elem_classes=["tessa-title"])

            with gr.Row(elem_classes=["tessa-header-row"]):
                with gr.Column(scale=1, min_width=250):
                    with gr.Row():
                        lang_english = gr.Button("🇬🇧 English", variant="secondary")
                        lang_spanish = gr.Button("🇪🇸 Español", variant="secondary")
                    with gr.Row():
                        lang_italian = gr.Button("🇮🇹 Italian", variant="secondary")
                        lang_german = gr.Button("🇩🇪 German", variant="secondary")
                    
                    current_language = gr.State("English")
                    coding_mode = gr.State(False)
                    research_mode = gr.State(False)
                    multilingual_mode = gr.State(False)
                    
                    def set_english():
                        return "English", True, False, False
                    def set_spanish():
                        return "Spanish", True, False, False
                    def set_italian():
                        return "Italian", True, False, False
                    def set_german():
                        return "German", True, False, False
                    
                    lang_english.click(set_english, outputs=[current_language, multilingual_mode, coding_mode, research_mode])
                    lang_spanish.click(set_spanish, outputs=[current_language, multilingual_mode, coding_mode, research_mode])
                    lang_italian.click(set_italian, outputs=[current_language, multilingual_mode, coding_mode, research_mode])
                    lang_german.click(set_german, outputs=[current_language, multilingual_mode, coding_mode, research_mode])
            
            with gr.Row():
                with gr.Column(scale=3):
                    chat_history = gr.Chatbot(label="Chat History", height=600)
                    
                    with gr.Row(elem_classes=["input-row"]):
                        user_input = gr.Textbox(
                            placeholder="Type your message here...",
                            show_label=False,
                            scale=4,
                            lines=1,
                            max_lines=1,
                            elem_id="chat-input"
                        )
                        upload_btn = gr.UploadButton("🔍 OCR", file_types=[".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".webp"], file_count="single")
                        send_btn = gr.Button("📤 Send")
                    
                    with gr.Row():
                        general_btn = gr.Button("💬 General", variant="secondary")
                        coding_btn = gr.Button("💻 Coding", variant="secondary")
                        research_btn = gr.Button("🔬 Research", variant="secondary")
                    
                    
                    def set_general_mode():
                        return False, False, False
                    def set_coding_mode():
                        return True, False, False
                    def set_research_mode():
                        return False, True, False
                    
                    general_btn.click(set_general_mode, outputs=[coding_mode, research_mode, multilingual_mode])
                    coding_btn.click(set_coding_mode, outputs=[coding_mode, research_mode, multilingual_mode])
                    research_btn.click(set_research_mode, outputs=[coding_mode, research_mode, multilingual_mode])
                    
                    status_output = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Row():
                        clear_btn = gr.Button("✏️ Clear Chat")
                        save_btn = gr.Button("💾 Save Chat")
                        load_btn = gr.UploadButton("📂 Load Chat", file_types=[".json"])
            
            def on_send(user_text, files, language, coding, research, multilingual):
                prompt_suffix = ""
                if multilingual:
                    prompt_suffix = "\nIMPORTANT: Provide your main response in the requested language, then on a NEW LINE, provide the English translation in brackets like this:\nText in language\n(English translation)"

                new_sys_content = ""
                if language == "Spanish":
                    new_sys_content = f"Por favor responde en español.{prompt_suffix}"
                elif language == "Italian":
                    new_sys_content = f"Per favore rispondi in italiano.{prompt_suffix}"
                elif language == "German":
                    new_sys_content = f"Bitte antworte auf Deutsch.{prompt_suffix}"
                else:
                    new_sys_content = f"Please respond in English.{prompt_suffix}"
                
                # Check if last message was identical system message to avoid spam
                last_system = next((msg["content"] for msg in reversed(ctx.history) if msg["source"] == "system"), None)
                if last_system != new_sys_content:
                    ctx.add_message("system", new_sys_content)
                
                # Priority routing: High-level modes override input length-based routing
                force_model = None
                if multilingual:
                    force_model = "aya-23-8b"
                elif research:
                    force_model = "deepseek-r1"
                elif coding:
                    force_model = "qwen2.5-coder"
                else:
                    # Default model is now Qwen-coder as requested
                    force_model = "qwen2.5-coder"
                
                return chat_query(user_text, files, force_model)
            
            send_btn.click(
                fn=on_send,
                inputs=[user_input, upload_btn, current_language, coding_mode, research_mode, multilingual_mode],
                outputs=[chat_history, user_input, status_output]
            )
            
            user_input.submit(
                fn=on_send,
                inputs=[user_input, upload_btn, current_language, coding_mode, research_mode, multilingual_mode],
                outputs=[chat_history, user_input, status_output]
            )
            
            upload_btn.upload(
                fn=lambda file, lang, coding, research, multilingual: chat_query("", file, None),
                inputs=[upload_btn, current_language, coding_mode, research_mode, multilingual_mode],
                outputs=[chat_history, user_input, status_output]
            )
            
            clear_btn.click(fn=lambda: (None, ""), outputs=[chat_history, user_input])
            save_btn.click(fn=save_chat, outputs=status_output)
            load_btn.upload(fn=load_chat, inputs=load_btn, outputs=chat_history)
        
        demo.launch(share=False, css=APP_CSS)
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Tessa...")
        cleanup_models()
    except Exception as e:
        print(f"❌ Error starting Tessa: {e}")
        cleanup_models()


if __name__ == "__main__":
    main()
