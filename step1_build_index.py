"""
第 1 步：把文档切块 → 算成向量 → 存成索引文件
"""
import io
import json
import os
import re
import sys
import time

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DOCS_DIR = "docs"  # 待问答的文档放这里（.md 或 .txt）
LOCAL_SOURCE = "local_source.txt"  # 本机私有文档路径（已写进 .gitignore，不会上传）
OUT_DIR = "index"  # 索引输出目录（已写进 .gitignore，不会上传）
MAX_LEN = 600  # 每块最多多少字
MIN_LEN = 40  # 太短的块合并掉
SVD_DIM = 256  # 向量维度

# 关键：1~4 级标题都算标题
HEAD = re.compile(r"^(#{1,4})\s+(.*)")


def _read_local_source():
    """读 local_source.txt：取第一个非空、且不以 # 开头的行，当作文档路径。

    没有这个文件就返回 None（说明没做本地固定）。这个文件是本地私有的，
    已写进 .gitignore，不会跟着仓库走。
    """
    if not os.path.isfile(LOCAL_SOURCE):
        return None
    with io.open(LOCAL_SOURCE, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                return ln
    return None


def resolve_source():
    """决定这次读哪份文档。

    三级回退，越靠前优先级越高：
      ① 命令行参数：python step1_build_index.py 你的文档.md
      ② local_source.txt 里写的那一行（可选，用来在本机固定一份私有文档）
      ③ docs/ 目录里唯一的 .md / .txt
    所以换台电脑、换份文档都不用改代码。
    """
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if not os.path.isfile(path):
            raise SystemExit("找不到文档：%s" % path)
        return path

    local = _read_local_source()
    if local:
        if os.path.isfile(local):
            return local
        raise SystemExit(
            "%s 里写的路径不存在：\n  %s\n"
            "改掉它，或者删掉这个文件（删掉之后就回到 docs/ 目录里取文档）。"
            % (LOCAL_SOURCE, local)
        )

    if not os.path.isdir(DOCS_DIR):
        raise SystemExit(
            "找不到 %s/ 目录。\n"
            "请新建一个 docs/ 文件夹，把要问答的文档（.md 或 .txt）放进去；\n"
            "或者直接指定文件：python step1_build_index.py 你的文档.md" % DOCS_DIR
        )

    files = sorted(f for f in os.listdir(DOCS_DIR)
                   if f.lower().endswith((".md", ".txt")))
    if not files:
        raise SystemExit("%s/ 里没有 .md 或 .txt 文件，放一份进去再运行。" % DOCS_DIR)
    if len(files) > 1:
        raise SystemExit(
            "%s/ 里有多个文档，请指定用哪一个：\n"
            "  python step1_build_index.py %s\n"
            "当前目录下：%s" % (DOCS_DIR, os.path.join(DOCS_DIR, files[0]), "、".join(files))
        )
    return os.path.join(DOCS_DIR, files[0])


def split_chunks(text, max_len=MAX_LEN, min_len=MIN_LEN):
    """切块。返回 [(章节路径, 向量用的文本, 展示用的正文)]"""
    lines = text.split("\n")
    path = {1: "", 2: "", 3: "", 4: ""}
    sections = []
    buf, head, in_code = [], "", False

    def flush():
        nonlocal head
        body = "\n".join(buf).strip()
        buf.clear()
        if body or head:
            sec = " > ".join(p for p in (path[1], path[2], path[3]) if p)
            if body:
                sections.append((sec, head, body))
        head = ""

    for ln in lines:
        if ln.startswith("```"):
            in_code = not in_code
        if not in_code:
            m = HEAD.match(ln)
            if m:
                flush()
                lvl, title = len(m.group(1)), m.group(2).strip()
                path[lvl] = title
                for k in range(lvl + 1, 5):
                    path[k] = ""
                head = title
                continue
        buf.append(ln)
    flush()

    def hard_split(body):
        out, cur = [], ""
        for ln in body.split("\n"):
            if len(cur) + len(ln) + 1 <= max_len:
                cur = (cur + "\n" + ln) if cur else ln
            else:
                if cur:
                    out.append(cur)
                cur = ln
        if cur:
            out.append(cur)
        return out

    chunks = []
    for sec, head, body in sections:
        prefix = (sec + " > " + head) if head else sec
        pieces = []
        if len(body) <= max_len:
            pieces = [body]
        else:
            cur = ""
            for para in [p for p in body.split("\n\n") if p.strip()]:
                if len(para) > max_len:
                    if cur:
                        pieces.append(cur)
                        cur = ""
                    pieces.extend(hard_split(para))
                elif cur and len(cur) + len(para) + 2 > max_len:
                    pieces.append(cur)
                    cur = para
                else:
                    cur = (cur + "\n\n" + para) if cur else para
            if cur:
                pieces.append(cur)
        pieces = [p for p in pieces if len(p.strip()) >= min_len] or pieces

        for pc in pieces:
            text_for_embed = (prefix + "\n" + pc).strip() if prefix else pc.strip()
            chunks.append((sec, text_for_embed, pc.strip()))
    return chunks


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    src = resolve_source()
    print("[1/4] 读取文档:", src)
    t0 = time.time()
    text = io.open(src, encoding="utf-8", errors="replace").read()
    print("      字数 %.1f 万，耗时 %.1f 秒" % (len(text) / 10000, time.time() - t0))

    print("[2/4] 切块 …")
    t0 = time.time()
    chunks = split_chunks(text)
    lens = [len(c[1]) for c in chunks]
    print("      切出 %d 块 | 最短 %d 平均 %d 最长 %d 字 | 耗时 %.1f 秒"
          % (len(chunks), min(lens), sum(lens) / len(lens), max(lens), time.time() - t0))

    print("[3/4] 算向量（TF-IDF → SVD 降维）…")
    t0 = time.time()
    docs = [c[1] for c in chunks]
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2),
                            min_df=2, max_features=60000)
    X = tfidf.fit_transform(docs)
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=42)
    vectors = normalize(svd.fit_transform(X))
    print("      TF-IDF 矩阵 %s | 稠密向量 %s | 信息保留 %.0f%% | 耗时 %.1f 秒"
          % (X.shape, vectors.shape, svd.explained_variance_ratio_.sum() * 100, time.time() - t0))

    print("[4/4] 存索引 …")
    with io.open(os.path.join(OUT_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump([{"section": s, "text": t} for s, _, t in chunks], f, ensure_ascii=False)
    np.save(os.path.join(OUT_DIR, "vectors.npy"), vectors)
    joblib.dump(tfidf, os.path.join(OUT_DIR, "tfidf.joblib"))
    joblib.dump(svd, os.path.join(OUT_DIR, "svd.joblib"))
    size = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR))
    print("      完成，索引共 %.1f MB" % (size / 1024 / 1024))
    print("      下一步：python step2_ask.py")


if __name__ == "__main__":
    main()
