import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from dashscope import TextEmbedding
import dashscope

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="my_knowledge")

st.title("📚 我的知识库问答助手")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 接收用户输入
if prompt := st.chat_input("请输入你的问题"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 检索
    resp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v3,
        input=prompt
    )
    query_vector = resp.output["embeddings"][0]["embedding"]
    results = collection.query(query_embeddings=[query_vector], n_results=8)
    context = "\n".join(results["documents"][0])

    # 生成回答
    full_prompt = f"""请严格根据以下资料回答问题，资料中没有的信息不要编造。

资料：
{context}

问题：{prompt}
"""
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": full_prompt}]
    )
    answer = completion.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)