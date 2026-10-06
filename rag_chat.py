import os
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

# 连接之前建好的向量数据库
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="my_knowledge")

def ask(question):
    # 1. 把用户问题转成向量
    resp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v3,
        input=question
    )
    query_vector = resp.output["embeddings"][0]["embedding"]

    # 2. 在数据库里检索最相似的 3 个片段
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=8
    )
    context = "\n".join(results["documents"][0])
    print(f"\n【检索到的资料】\n{context}\n")

    # 3. 把资料和问题拼成提示词，发给模型
    prompt = f"""请严格根据以下资料回答问题，资料中没有的信息不要编造。

资料：
{context}

问题：{question}
"""
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

# 测试
if __name__ == "__main__":
    while True:
        q = input("你的问题（输入退出结束）：")
        if q == "退出":
            break
        print("\nAI回答：" + ask(q) + "\n")