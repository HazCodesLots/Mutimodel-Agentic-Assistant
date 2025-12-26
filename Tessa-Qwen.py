import ollama
import os
import json
from datetime import datetime
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import easyocr
from pdf2image import convert_from_path

# Ollama model names
OLLAMA_CHAT_MODEL = "deepseek-r1-distill-qwen-14b-q4_k_m"
OLLAMA_CODE_MODEL = "qwen2.5-coder-32b-instruct-q4_k_m"

HISTORY_DIR = r"D:\Work\Wraps\Tessa\History"
os.makedirs(HISTORY_DIR, exist_ok=True)

class AgentContext:
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
        elif source in ["ollamachat", "ollamacode"]:
            self.latest_output = content

    def get_conversation(self):
        return [msg["content"] for msg in self.history]

ctx = AgentContext()

def build_prompt(context: AgentContext) -> list:
    messages = []
    for message in context.history:
        role = message["source"].lower()
        if role == "user":
            messages.append({"role": "user", "content": message['content']})
        else:
            messages.append({"role": "assistant", "content": message['content']})
    return messages

def ollama_chat_respond(user_input: str, context: AgentContext, document_context: str = None):
    context.add_message("user", user_input)

    # Prepare messages for Ollama
    messages = build_prompt(context)

    # Add system prompt if document context is provided
    if document_context:
        messages.insert(0, {
            "role": "system",
            "content": f"You are a helpful assistant. Use the following document context to answer questions:\n\n{document_context.strip()}"
        })

    response = ollama.chat(
        model=OLLAMA_CHAT_MODEL,
        messages=messages,
        options={
            "temperature": 0.7,
            "max_tokens": 1024
        }
    )

    result = response['message']['content'].strip() if response and 'message' in response else "[Error: no response]"
    context.add_message("ollamachat", result)
    return result

# Initialize BLIP model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip_model.to(device)

def blip_respond(image_path: str, prompt: str = "", context: AgentContext = None) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = blip_model.generate(**inputs, max_new_tokens=100)

    caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    if context is not None:
        context.add_message("tool", caption)

    return caption

def blip_to_ollama(image_path: str, blip_prompt: str = "", ollama_question: str = "What do you observe?", context: AgentContext = None) -> str:
    caption = blip_respond(image_path, prompt=blip_prompt, context=None)
    if context is not None:
        context.image_context = caption
    return ollama_chat_respond(ollama_question, context, document_context=caption)

# Initialize embedding and OCR models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
ocr_reader = easyocr.Reader(['en'])
faiss_texts = []
dimension = embedding_model.get_sentence_embedding_dimension()
faiss_index = faiss.IndexFlatL2(dimension)

def pdf_to_images(pdf_path: str, dpi: int = 300):
    return convert_from_path(pdf_path, dpi=dpi)

def retrieve_similar_text(query, top_k=1):
    if faiss_index.ntotal == 0:
        return ["(No data indexed yet. Please process a PDF first.)"]
    query_embedding = embedding_model.encode([query])
    D, I = faiss_index.search(np.array(query_embedding, dtype=np.float32), top_k)
    return [faiss_texts[i] for i in I[0] if 0 <= i < len(faiss_texts)]

def process_pdf_with_ocr(pdf_path: str):
    images = pdf_to_images(pdf_path)
    all_text = []

    for page_num, img in enumerate(images):
        text_lines = ocr_reader.readtext(np.array(img), detail=0)
        page_text = " ".join(text_lines).strip()
        
        if page_text:
            embedding = embedding_model.encode([page_text])
            faiss_index.add(np.array(embedding, dtype=np.float32))
            faiss_texts.append(page_text)
            all_text.append((page_num, page_text))

    return all_text

def query_pdf_ocr(query: str, top_k: int = 3):
    if faiss_index.ntotal == 0:
        return ["(No data indexed yet. Please process a PDF first.)"]

    query_embedding = embedding_model.encode([query])
    D, I = faiss_index.search(np.array(query_embedding, dtype=np.float32), top_k)
    return [faiss_texts[i] for i in I[0] if 0 <= i < len(faiss_texts)]

def ask_pdf_ocr(pdf_path: str, query: str, context: AgentContext):
    process_pdf_with_ocr(pdf_path)
    results = query_pdf_ocr(query, top_k=1)
    document_context = results[0]
    return route_to_model(query, context=context)

def ollama_code_respond(user_input: str, context: AgentContext, document_context: str = None):
    context.add_message("user", user_input)

    # Prepare messages for Ollama
    messages = build_prompt(context)

    # Add system prompt with optional document context
    system_prompt = "You are an expert coding assistant. Provide clear, accurate code solutions and explanations."
    if document_context:
        system_prompt += f"\n\nUse the following document context to inform your coding decisions:\n{document_context.strip()}"
    
    messages.insert(0, {"role": "system", "content": system_prompt})

    response = ollama.chat(
        model=OLLAMA_CODE_MODEL,
        messages=messages,
        options={
            "temperature": 0.2,
            "max_tokens": 3200
        }
    )

    result = response['message']['content'].strip() if response and 'message' in response else "[Error: no response]"
    context.add_message("ollamacode", result)
    return result

def route_to_model(prompt: str, context: AgentContext) -> str:
    code_keywords = [
        "function", "class", "python", "java", "code", "script", "loop", "algorithm",
        "regex", "compile", "bug", "error", "fix", "sort", "data structure", "pandas",
        "API", "decorator", "recursion", "print", "for loop", "if statement",
        "train", "model", "neural network", "transformer", "architecture", "mlp",
        "loss", "dataset", "optimizer", "backpropagation", "torch", "tensorflow"
    ]

    prompt_lower = prompt.lower()
    is_code_related = any(keyword in prompt_lower for keyword in code_keywords)

    retrieved_chunks = retrieve_similar_text(prompt, top_k=1)
    document_context = retrieved_chunks[0] if retrieved_chunks else ""

    if is_code_related:
        return ollama_code_respond(prompt, context, document_context=document_context)
    else:
        return ollama_chat_respond(prompt, context, document_context=document_context)

def save_chat():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(HISTORY_DIR, f"chat_{timestamp}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(ctx.history, f, indent=2)
    return f"Chat saved to {path}"

def format_pairs(history):
    pairs = []
    i = 0
    while i < len(history):
        if history[i]["source"] == "user":
            user_msg = history[i]["content"]
            i += 1
            if i < len(history) and history[i]["source"] in ["tool", "ollamachat", "ollamacode", "blip"]:
                tool_msg = history[i]["content"]
                pairs.append([user_msg, tool_msg])
                i += 1
            else:
                pairs.append([user_msg, ""])
        elif history[i]["source"] in ["tool", "ollamachat", "ollamacode", "blip"]:
            tool_msg = history[i]["content"]
            pairs.append([None, tool_msg])
            i += 1
        else:
            i += 1
    return pairs

def load_chat(file):
    with open(file.name, 'r', encoding='utf-8') as f:
        ctx.history = json.load(f)
    return gr.update(value=format_pairs(ctx.history))

def chat_query(user_input):
    _ = route_to_model(user_input, ctx)
    return gr.update(value=format_pairs(ctx.history)), ""

def handle_pdf(pdf_file):
    text = process_pdf_with_ocr(pdf_file.name)
    return f"Processed {len(text)} pages via OCR. Now searchable via questions."

def handle_image(image_path):
    if image_path is None:
        return gr.update(value=format_pairs(ctx.history)),
    caption = blip_respond(image_path, prompt="Describe this image", context=ctx)
    return gr.update(value=format_pairs(ctx.history)), "Image processed"

def main():
    with gr.Blocks() as demo:
        gr.HTML("""
        <style>
            .top-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                width: 100%;
            }
            .top-row h1 {
                margin: 0;
                font-size: 1.8em;
            }
            .top-buttons {
                display: flex;
                gap: 10px;
            }
        </style>
        """)

        with gr.Row():
            with gr.Column(scale=6):
                gr.Markdown("## 🤖 Tessa - Multimodal Assistant")
            with gr.Column(scale=6, min_width=300):
                with gr.Row():
                    with gr.Column(scale=1, min_width=150):
                        image_upload = gr.UploadButton(label="🖼️ Upload Image", file_types=[".png", ".jpg", ".jpeg"])
                    with gr.Column(scale=1, min_width=150):
                        pdf_upload = gr.UploadButton(label="📄 Upload PDF", file_types=[".pdf"])
                    with gr.Column(scale=1, min_width=100):
                        load_btn = gr.UploadButton(label="📂 Load Chat", file_types=[".json"])
                    with gr.Column(scale=1, min_width=100):
                        save_btn = gr.Button("💾 Save Chat")

        with gr.Row():
            chatbox = gr.Chatbot(label="Tessa", show_label=False, bubble_full_width=True, value=[])

        with gr.Row():
            query = gr.Textbox(show_label=False, placeholder="Type your message here...")
            send_btn = gr.Button("Send")

        with gr.Row():
            status_bar = gr.Textbox(label="Status", interactive=False)

        send_btn.click(fn=chat_query, inputs=query, outputs=[chatbox, query])
        query.submit(fn=chat_query, inputs=query, outputs=[chatbox, query])
        save_btn.click(fn=save_chat, inputs=[], outputs=status_bar)
        load_btn.upload(fn=load_chat, inputs=load_btn, outputs=chatbox)
        pdf_upload.upload(fn=handle_pdf, inputs=pdf_upload, outputs=status_bar)
        image_upload.upload(fn=handle_image, inputs=image_upload, outputs=[chatbox, status_bar])

    demo.launch()

if __name__ == "__main__":
    main()
