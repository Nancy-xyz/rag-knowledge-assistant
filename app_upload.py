import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from dashscope import TextEmbedding
import dashscope
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="my_knowledge")

st.title("📚 我的知识库问答助手")

# ========== 侧边栏：上传文档 ==========
with st.sidebar:
    st.header("上传文档")
    uploaded_file = st.file_uploader("选择一个 PDF 文件", type=["pdf"])

    if uploaded_file is not None:
        if st.button("开始构建知识库"):
            with st.spinner("正在处理文档，请稍候..."):
                # 1. 保存上传的 PDF 到本地
                os.makedirs("documents", exist_ok=True)
                pdf_path = os.path.join("documents", uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 2. 加载 PDF
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                # 3. 切块
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150,
                    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
                )
                chunks = splitter.split_documents(documents)

                # 4. 清空旧数据，重新构建
                existing = collection.get()
                if existing["ids"]:
                    collection.delete(ids=existing["ids"])

                # 5. 向量化并存入
                texts = [chunk.page_content for chunk in chunks]
                for i, text in enumerate(texts):
                    resp = TextEmbedding.call(
                        model=TextEmbedding.Models.text_embedding_v3,
                        input=text
                    )
                    embedding = resp.output["embeddings"][0]["embedding"]
                    collection.add(
                        documents=[text],
                        embeddings=[embedding],
                        ids=[f"chunk_{i}"]
                    )

                st.success(f"知识库构建完成！共处理 {len(texts)} 个文本块。")

# ========== 主区域：聊天 ==========
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

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