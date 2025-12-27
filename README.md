# Tessa - Multimodal Research Assistant
TESSA is a multimodal and multilingual research assistant with OCR (Optical Character Recognition) and RAG (Retrieval Augmented Generation) capabilities. Optimized for on-device deployment, TESSA specializes in breaking down research papers and generating contextual code based on the input research papers.

- **DeepSeekOCR** ([paper](https://arxiv.org/abs/2510.18234)) - High-compression OCR using DeepEncoder (vision encoder) and DeepSeek3B-MoE-A570M (decoder)  | (pending PyTorch CUDA 12.4 stable release for sm_120 support)  
- **FAISS-RAG** ([paper](https://arxiv.org/abs/2005.11401)) - Retrieval-augmented generation with EasyOCR extraction, Sentence-BERT embeddings, and FAISS L2 similarity search for PDF question answering. Pinecone RAG added for 4GB VRAM module.
- **BLIP** ([paper](https://arxiv.org/abs/2201.12086)) -  Vision-language model for image captioning using patch embedding vision encoder (768-dim) and cross-attention text decoder with conditional generation.

<img width="1909" height="1123" alt="image" src="https://github.com/user-attachments/assets/39264e3f-8e19-42ae-b415-177161314c33" />

### 14GB VRAM Configuration
[Qwen2.5-Coder-32B-Instruct (Q4_K_M)](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF) - Default routed LLM | context window - 128k | [paper](https://arxiv.org/abs/2409.12186)  
[DeepSeek-R1-Distill-Qwen-14B (Q4_K_M)](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) - Research mode LLM | context window - 128k | [paper](https://arxiv.org/abs/2501.12948)  
[Aya-23-8B (Q4_K_M)](https://huggingface.co/CohereLabs/aya-23-8B) - MultiLingual LLM | context window - 8k | [paper](https://arxiv.org/abs/2405.15032)  

<img width="1885" height="114" alt="Screenshot 2025-12-24 142434" src="https://github.com/user-attachments/assets/8a92f70d-416f-41d1-a988-0eec5918ea9a" />

### Google Colab T4
[Phi-2](https://huggingface.co/microsoft/phi-2/tree/main) - Default routed LLM | Google Colab T4 |
[Deepseek-coder-6.7b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct/tree/main) - Coding Mode LLM | Google Colab T4 |

### 4GB VRAM Configuration
[Qwen1.5-1.8B-Chat](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GGUF/tree/main) - Default routed LLM | 4GB VRAM |
[OpenHermes-2.5-mistral-7b](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/tree/main) - Default routed LLM | 4GB VRAM |

[Qwen2.5-coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/tree/main) - Coding Mode LLM | 4GB VRAM |
[Deepseek-coder-1.3b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct/tree/main) - Coding Mode LLM | 4GB VRAM |

#### Config
<img width="1284" height="1255" alt="image" src="https://github.com/user-attachments/assets/5f0f337f-88ac-423a-a37e-d1f71bd3ac46" />

#### Simple User Interface

Type your message: questions, code or follow-ups.

🖼️ Upload Image: Get a natural-language description of any image.  
📄 Upload PDF: Instantly OCR and embed your PDF for semantic search.  
💾 Save/📂 Load Chat: Export and import chat history in JSON format easily.  
