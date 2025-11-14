import os
import glob
import time
import pickle
import numpy as np
import torch
import faiss
from openai import OpenAI
import gradio as gr
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# Model
MERGED_MODEL_PATH = " "

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# RAG Data
RAG_DATA_DIR = " "

VECTOR_DIR = " "
INDEX_PATH = os.path.join(VECTOR_DIR, "faiss_index.bin")
META_PATH = os.path.join(VECTOR_DIR, "docs_meta.pkl")

# 阿里百炼API接口
client = OpenAI(api_key = " ", base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBED_MODEL = "text-embedding-v4"
EMBED_DIM = 1024  # text-embedding-v4 输出维度

def get_aliyun_embeddings(texts):
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [d.embedding for d in resp.data]


def build_vector_store():
    os.makedirs(VECTOR_DIR, exist_ok=True)
    docs = []
    for path in glob.glob(os.path.join(RAG_DATA_DIR, "*.md")):
        loader = TextLoader(path, encoding="utf-8")
        loaded_docs = loader.load()
        for doc in loaded_docs:
            docs.append(doc)

    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    print(f"加载 {len(split_docs)} 段文本，开始生成向量……")

    all_embeddings = []
    batch_size = 10
    for i in range(0, len(split_docs), batch_size):
        batch_texts = [d.page_content for d in split_docs[i:i+batch_size]]
        batch_emb = get_aliyun_embeddings(batch_texts)
        all_embeddings.extend(batch_emb)
        time.sleep(0.2)

    embeddings_np = np.array(all_embeddings, dtype="float32")
    faiss.normalize_L2(embeddings_np)

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings_np)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(split_docs, f)

    print("Finish!")


def load_vector_store():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        split_docs = pickle.load(f)
    return index, split_docs

if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
    build_vector_store()

index, split_docs = load_vector_store()


def retrieve_docs(query, k=3):
    q_emb = np.array(get_aliyun_embeddings([query])[0], dtype="float32")
    faiss.normalize_L2(q_emb.reshape(1, -1))
    D, I = index.search(q_emb.reshape(1, -1), k)
    return [split_docs[i] for i in I[0]]

print("Loading...")

tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

print("Done!")

def generate_answer(query: str):
    docs = retrieve_docs(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # prompt
    prompt = f"""以下是从知识库检索到的内容：
                {context}

                请根据以上内容回答用户问题：
                {query}

                如果知识库中没有相关内容，请按照你自己的想法输出经过你深度思考后的内容，但是请不要重复回答问题，加油！"""

    # input
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )

    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    for token in streamer:
        yield token

# Web UI
LOGO_LEFT = " "          
LOGO_RIGHT = " "     

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        with gr.Column(scale=0.5, min_width=80):
            gr.Image(LOGO_LEFT, scale=0.4, elem_id="logo1", show_download_button=False, container=False, show_fullscreen_button=False)
        with gr.Column(scale=1.2, min_width=80):
            gr.Markdown('<center><font size="45">麓言湘语AI智能体</font></center>', elem_id="title")
        with gr.Column(scale=0.65, min_width=80):
            gr.Image(LOGO_RIGHT, scale=0.5, elem_id="logo2", show_download_button=False, container=False,  show_fullscreen_button=False)

    chatbot = gr.Chatbot(
        value=[],
        elem_id="chatbot_area",
        height=700,
        show_label=False,
        avatar_images=[" ", " "]
    )

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="你好，我是小麓，请问有什么可以帮助你的吗？",
            show_label=False,
            lines=2,
            elem_id="user_input"
        )

    with gr.Row():
        with gr.Column(scale=1, min_width=120):
            send_btn = gr.Button("📡 发送(Submit)", variant="primary", elem_id="send-btn")
        with gr.Column(scale=1, min_width=120):
            clear_btn = gr.Button("👋 清空对话(Clear)", elem_id="clear-btn")
            
    def chat_fn(message, history):
        history = history or []

        user_msg = message.strip()
        if not user_msg:
            return history

        history.append((user_msg, ""))
        yield history, gr.update(value="") 

        partial = ""

        for token in generate_answer(user_msg):
            partial += token
            history[-1] = (user_msg, partial)
            yield history, None  

        history[-1] = (user_msg, partial)
        yield history, None

    send_btn.click(fn=chat_fn, inputs=[user_input, chatbot], outputs=[chatbot, user_input])

    user_input.submit(fn=chat_fn, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    clear_btn.click(lambda: [], None, chatbot)
demo.queue().launch(server_name="0.0.0.0", server_port=7860)