# Tessa - Multimodel Agentic Assitant
TESSA is a multimodal multilingual language teacher and research assistant with **Optical Character Recognition(OCR)** and **Retrieval-Augmented Generation(RAG)**, offering optimized LLM configurations for on-device deployment<br>


text
## 📂 Repository Structure

├── 📁 Colab-demo/  
│ └────────── 💻 TESSA-Phi+DeepSeek.ipynb → GoogleColab Demo  
├── 📁 Local-Notebook/  
│ └────────── 💻 Tessa-Qwen.ipynb → Local Jupyter notebook (4GB VRAM Optimized)  
├── 📁 ToolBox/  
│ ├────────── 💻 BLIP.ipynb → BLIP image captioning notebook  
│ ├────────── 🔧 DeepSeekCPU-OCR.py → CPU-optimized DeepSeek OCR  
│ ├────────── 🔧 DeepSeekOCR.py → GPU-optimized DeepSeek OCR  
│ ├────────── 💻 FAISS-RAG.ipynb → FAISS vector database RAG demo  
│ ├────────── 💻 PineConeRAG.ipynb → Pinecone cloud RAG implementation  
│ └────────── 🔧 patch_ocr.py → OCR patching utilities (removes cuda calls to shift to CPU)  
├── 💻 Tessa-Qwen.py → TESSA standalone (Qwen-optimized)  
├── 🖥️ Tessa-RTX5060TI.py → Main application (14GB VRAM Optimized)  
└── 📖 README.md → Project documentation  

### 14GB VRAM Configuration
[Qwen2.5-Coder-32B-Instruct (Q4_K_M)](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF) - Default routed LLM | context window - 128k | [paper](https://arxiv.org/abs/2409.12186)<br>
[DeepSeek-R1-Distill-Qwen-14B Q4_K_M)](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) - Research mode LLM | context window - 128k | [paper](https://arxiv.org/abs/2501.12948)<br>
[Aya-23-8B (Q4_K_M)](https://huggngface.co/CohereLabs/aya-23-8B) - MultiLingual LLM | context window - 8k | [paper](https://arxiv.org/abs/2405.15032)<br>


### 4GB VRAM Configuration
Qwen1.5-1.8B-Chat - https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GGUF/tree/main<br>
OpenHermes-2.5-mistral-7b - https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/tree/main<br>
Phi-2 - https://huggingface.co/microsoft/phi-2/tree/main<br>

Qwen2.5-coder-1.5B - https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/tree/main<br>
Deepseek-coder-6.7b-instruct - https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct/tree/main<br>
Deepseek-coder-1.3b-instruct - https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct/tree/main<br>
</details>

📷 Vision: Image Captioning
BLIP: Quickly generate captions and descriptions for images using Salesforce’s BLIP image captioning model <br>

<ul>
  <li>llama_cpp: Fast local inference of models in GGUF format.</li>
  <li>BLIP: image captioning.</li>
  <li>sentence-transformers, For semantic document retrieval and vector search.</li>
  <li>EasyOCR: Accurate OCR extraction from PDFs.</li>
  <li>Gradio: Responsive, modern web UI with drag-and-drop uploads.</li>
  <li>PIL, pdf2image: Image processing utilities.</li>
  <li>PyTorch: GPU-accelerated deep learning on compatible hardware.</li>
</ul>

</ul>

Simple User Interface <br>

Type your message: questions, code or follow-ups.

🖼️ Upload Image: Get a natural-language description of any image.

📄 Upload PDF: Instantly OCR and embed your PDF for semantic search.

💾 Save/📂 Load Chat: Export and import chat history in JSON format easily.
</ul>
