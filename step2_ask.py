"""
第 2 步：检索 + 生成（RAG 的核心）
"""
import io
import json
import os
import time

import joblib
import numpy as np
import requests
from sklearn.preprocessing import normalize

INDEX_DIR = "index"
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen2.5-3b"
TOP_K = 4  # 检索几块资料
TEMPERATURE = 0.3

# 强制直连本地服务。
# requests 默认会读系统里的 HTTP_PROXY / HTTPS_PROXY 环境变量。
# 在某些环境下（公司网络、代理软件），连 127.0.0.1 的请求也会被送去代理，导致连接失败，
# 表现就是 ProxyError 或「目标计算机积极拒绝」。这里显式关掉代理，保证本地调用直连。
LOCAL_ONLY = {"http": None, "https": None}

SYSTEM_PROMPT = (
    "你是学习助手。请依据下面提供的资料回答问题——"
    "只要资料和问题沾边，就要基于资料给出回答，不要轻易说没有。"
    "只有当资料和问题完全无关时，才回答「资料中没有相关内容」。"
)


def check_ollama(model=MODEL):
    """开工前先自检：Ollama 服务在不在、模型有没有。给出人话提示，不甩报错。"""
    try:
        resp = requests.get(OLLAMA_URL + "/api/tags", proxies=LOCAL_ONLY, timeout=5)
        resp.raise_for_status()
    except Exception:
        print("=" * 64)
        print("连不上本地 Ollama 服务（%s）" % OLLAMA_URL)
        print()
        print("原因：Ollama 服务没有在运行。先启动它，再重新运行本脚本。")
        print()
        print("启动方法（Windows）：")
        print("  1. 按 Win 键，在搜索框输入  ollama")
        print("  2. 在搜索结果里点开「Ollama」")
        print("  3. 右下角任务栏出现小羊驼图标，说明服务已就绪（首次启动等 5～10 秒）")
        print()
        print("其他系统：在终端里执行  ollama serve  即可。")
        print("=" * 64)
        return False

    names = [m.get("name", "") for m in resp.json().get("models", [])]
    if not any(n.startswith(model) for n in names):
        print("=" * 64)
        print("Ollama 在跑，但没找到模型 %s" % model)
        print("本地现有模型：")
        for n in names:
            print("   -", n)
        print("=" * 64)
        return False

    print("Ollama 就绪，模型 %s 已加载入口\n" % model)
    return True


def load_index(index_dir=INDEX_DIR):
    """把第 1 步存好的索引读回来"""
    with io.open(os.path.join(index_dir, "chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    vectors = np.load(os.path.join(index_dir, "vectors.npy"))
    tfidf = joblib.load(os.path.join(index_dir, "tfidf.joblib"))
    svd = joblib.load(os.path.join(index_dir, "svd.joblib"))
    return chunks, vectors, tfidf, svd


def retrieve(question, chunks, vectors, tfidf, svd, k=TOP_K):
    """把问题也变成 256 维向量，和所有块算余弦相似度，取最高的 k 个"""
    q_vec = normalize(svd.transform(tfidf.transform([question])))[0]
    sims = vectors @ q_vec  # 一次矩阵乘法算完所有相似度
    idx = np.argsort(-sims)[:k]
    return [(float(sims[i]), chunks[i]["section"], chunks[i]["text"]) for i in idx]


def generate(question, hits, model=MODEL):
    """把检索到的资料拼进提示词，交给本地大模型生成答案"""
    context = "\n\n".join(
        "[资料%d]（%s）\n%s" % (i + 1, sec or "无章节", text)
        for i, (_, sec, text) in enumerate(hits)
    )
    user_prompt = "资料：\n%s\n\n问题：%s\n\n请依据上面的资料回答。" % (context, question)

    t0 = time.time()
    resp = requests.post(
        OLLAMA_URL + "/api/chat",
        proxies=LOCAL_ONLY,
        timeout=300,
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": TEMPERATURE, "num_predict": 400},
        },
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip(), time.time() - t0


def answer(question, index, k=TOP_K):
    """一条龙：检索 → 生成"""
    chunks, vectors, tfidf, svd = index
    hits = retrieve(question, chunks, vectors, tfidf, svd, k)
    text, cost = generate(question, hits)
    return {"answer": text, "hits": hits, "seconds": cost}


if __name__ == "__main__":
    if not check_ollama():
        raise SystemExit(1)

    print("载入索引 …")
    t0 = time.time()
    index = load_index()
    print("  共 %d 块，耗时 %.1f 秒" % (len(index[0]), time.time() - t0))

    print("\n输入问题回车即可（直接回车退出）\n")
    while True:
        q = input("问：").strip()
        if not q:
            break
        result = answer(q, index)
        print("\n答：%s" % result["answer"])
        print("\n来源（相似度 / 章节）：")
        for score, sec, _ in result["hits"]:
            print("  %.3f  %s" % (score, sec or "无章节"))
        print("耗时 %.1f 秒\n" % result["seconds"])
