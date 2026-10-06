import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

load_dotenv()

# 1. 初始化大模型客户端（用于后面的对话）
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 2. 加载文档
loader = PyPDFLoader("documents/2026面试题.pdf")
documents = loader.load()
print(f"加载了 {len(documents)} 页")

# 3. 切块：中文建议 300-500 字一块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
chunks = splitter.split_documents(documents)
print(f"切成了 {len(chunks)} 个文本块")

# 4. 用 Embedding 模型把文本转成向量
# 这里我们用阿里云百炼的 text-embedding-v3 模型
from dashscope import TextEmbedding

texts = [chunk.page_content for chunk in chunks]
embeddings = []
for text in texts:
    resp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v3,
        input=text
    )
    embeddings.append(resp.output["embeddings"][0]["embedding"])

print(f"生成了 {len(embeddings)} 个向量")

# 5. 存入 Chroma 向量数据库
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="my_knowledge")

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(texts))]
)

print("向量已存入数据库，路径：./chroma_db")