"""
第 3 步：做成网页界面（streamlit）
"""
import streamlit as st

import step2_ask as R

st.set_page_config(page_title="学习笔记问答助手", page_icon="📚", layout="wide")

st.title("📚 学习笔记问答助手")
st.caption(
    "知识库：《AI大模型开发（Python）阶段 1–7 学习笔记》（50 万字，切分 1260 块）｜ "
    "检索：TF-IDF + SVD 降维成 256 维向量，余弦相似度排序 ｜ "
    "生成：本地 Qwen2.5-3B（Ollama）"
)


@st.cache_resource
def get_index():
    """索引只加载一次，不要每次刷新都重读"""
    return R.load_index()


try:
    index = get_index()
except Exception as e:
    st.error("载入索引失败：%s\n\n请先运行 python step1_build_index.py 生成 index/ 目录。" % e)
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("设置")
    top_k = st.slider("检索块数 Top-K", min_value=1, max_value=8, value=R.TOP_K,
                      help="每次喂给大模型几块资料。少了容易漏，多了会引入噪声")
    show_source = st.checkbox("显示检索到的原文", value=True)
    if st.button("清空对话", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.caption("知识库统计")
    st.write("总块数：**%d**" % len(index[0]))
    st.write("向量维度：**%d**" % index[1].shape[1])
    st.caption("试试这些问题：")
    for q in ["装饰器的原理是什么？", "深浅拷贝有什么区别？",
              "MySQL 为什么用 B+ 树做索引？", "RAG 的完整链路是怎样的？",
              "明天天气怎么样？（应该拒答）"]:
        st.caption("· " + q)

for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["q"])
    with st.chat_message("assistant"):
        st.write(item["answer"])
        if show_source:
            with st.expander("检索到的 %d 块资料（相似度从高到低）" % len(item["hits"])):
                for score, sec, text in item["hits"]:
                    st.markdown("**%.3f** ｜ %s" % (score, sec or "无章节"))
                    st.caption(text[:400] + ("…" if len(text) > 400 else ""))
                    st.divider()
        st.caption("耗时 %.1f 秒" % item["seconds"])

question = st.chat_input("问一个关于课程内容的问题…")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("检索中…"):
            result = R.answer(question, index, k=top_k)
        st.write(result["answer"])
        if show_source:
            with st.expander("检索到的 %d 块资料（相似度从高到低）" % len(result["hits"])):
                for score, sec, text in result["hits"]:
                    st.markdown("**%.3f** ｜ %s" % (score, sec or "无章节"))
                    st.caption(text[:400] + ("…" if len(text) > 400 else ""))
                    st.divider()
        st.caption("耗时 %.1f 秒" % result["seconds"])
    st.session_state.history.append({"q": question, **result})
