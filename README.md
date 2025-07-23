# Tessa - Multimodel Agentic Assitant
Tessa is a context-aware multimodal assitant powered by three different LLM combinations paired with OCR and BLIP<br>

Qwen1.5-1.8B-Chat - https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GGUF/tree/main<br>
OpenHermes-2.5-mistral-7b - https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/tree/main<br>
Phi-2 - https://huggingface.co/microsoft/phi-2/tree/main<br>

Qwen2.5-coder-1.5B - https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/tree/main<br>
Deepseek-coder-6.7b-instruct - https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct/tree/main<br>
Deepseek-coder-1.3b-instruct - https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct/tree/main<br>
 
🗨️ Natural Language & Code Chat <br>
<br>
Chat: Dialogue and general-purpose question answering, powered by qwen1.5-1.8B-Chat, openhermes-2.5-mistral-7b.Q4_K_M.gguf, phi-2.Q4_K_M.gguf <br>

Code: Code understanding, generation and debugging via qwen2.5-Coder-1.5B-Instruct, deepseek-coder-6.7b-instruct-q4_k_m.gguf, deepseek-coder-1.3b-instruct.Q4_K_M.gguf<br>

Automatic Routing: User prompts are intelligently routed to the conversational or code model based on detected intent (keywords relevant to code or programming)<br>

📄 OCR PDF Search & QA
PDF OCR: Upload PDFs; pages are converted to images, processed with EasyOCR and embedded with SentenceTransformers. This is featured through Retrieval augmented generation using FAISS and Pinecone <br>

📷 Vision: Image Captioning
BLIP: Quickly generate captions and descriptions for images using Salesforce’s BLIP image captioning model <br>

<ul>
  <li>llama_cpp: Fast local inference of models in GGUF format.</li>
  <li>BLIP: image captioning.</li>
  <li>sentence-transformers, For semantic document retrieval and vector search.</li>
  <li>EasyOCR: Accurate OCR extraction from PDFs.</li>
  <li>Tradio: Responsive, modern web UI with drag-and-drop uploads.</li>
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
