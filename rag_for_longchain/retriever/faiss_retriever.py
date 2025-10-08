import numpy as np
import faiss
from sentence_transformers import util

# 加载数据
def load_npz_database(npz_file_path):
    """
    从 NPZ 文件加载辩论数据库。
    数据库包含嵌入向量、立场、辩手、辩论文本和标签。
    :param npz_file_path: NPZ 文件路径
    :return: embeddings, stances, debaters, utterances, labels
    """
    data = np.load(npz_file_path, allow_pickle=True)
    embeddings = data['embeddings']
    stances = data['stances']
    debaters = data['debaters']
    utterances = data['utterances']
    labels = data['labels']
    return embeddings, stances, debaters, utterances, labels

# 初始化数据和模型
npz_file_path = "D:\converstional_rag/rag_for_longchain\data\processed_data/allnpz/all.npz"
embeddings, stances, debaters, utterances, labels = load_npz_database(npz_file_path)

# 初始化 FAISS 索引
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)  # 内积相似度索引
faiss.normalize_L2(embeddings)  # 归一化嵌入向量
faiss_index.add(embeddings)

def retrieve_candidates(input_embeddings, top_k=5):
    """
    检索与辩论历史最相关的候选文本（粗筛和细筛）。
    :param input_embeddings: 辩论历史的嵌入向量，形状为 (32, 1024)
    :param top_k: 粗筛阶段的候选文本数量
    :return: 前 top_k 个匹配的文本及其相关信息，以及每个 chunk 的得分
    """
    # 转换数据类型
    input_embeddings = input_embeddings.astype(np.float32)  # 确保输入为 float32

    # 初始化结果容器
    candidate_scores = {}  # 存储候选文本的累计得分
    candidate_hits = {}  # 存储候选文本被命中的次数

    # 粗筛：利用 FAISS 最近邻搜索
    input_embeddings = input_embeddings / np.linalg.norm(input_embeddings, axis=1, keepdims=True)  # 归一化输入
    for chunk in input_embeddings:
        distances, indices = faiss_index.search(chunk.reshape(1, -1), top_k)
        for idx, dist in zip(indices[0], distances[0]):
            if idx not in candidate_scores:
                candidate_scores[idx] = 0
                candidate_hits[idx] = 0
            candidate_scores[idx] += dist  # 累加得分
            candidate_hits[idx] += 1      # 命中计数

    # 累计得分归一化
    for idx in candidate_scores:
        candidate_scores[idx] /= candidate_hits[idx]

    # 按得分排序，取前 top_k
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # 细筛：基于语义相似度和余弦相似度
    refined_scores = []
    for idx, score in sorted_candidates:
        # 语义相似度计算
        semantic_similarity = np.dot(input_embeddings.mean(axis=0), embeddings[idx])
        semantic_similarity /= (np.linalg.norm(input_embeddings.mean(axis=0)) * np.linalg.norm(embeddings[idx]))

        # 余弦相似度计算（确保 dtype 一致）
        cosine_similarity = util.pytorch_cos_sim(input_embeddings.mean(axis=0), embeddings[idx]).item()

        # 综合得分（加权计算）
        final_score = 0.6 * semantic_similarity + 0.4 * cosine_similarity
        refined_scores.append((idx, final_score))

    # 按综合得分排序
    refined_scores = sorted(refined_scores, key=lambda x: x[1], reverse=True)[:top_k]

    # 返回前 top_k 个匹配结果
    results = []
    for idx, final_score in refined_scores:
        results.append({
            "text": utterances[idx],
            "labels": labels[idx],
            "score": final_score
        })

    # 返回每个 chunk 的得分
    chunk_scores = {}
    for i, chunk in enumerate(input_embeddings):
        chunk_scores[i] = []
        for idx in range(len(embeddings)):
            semantic_similarity = np.dot(chunk, embeddings[idx]) / (
                np.linalg.norm(chunk) * np.linalg.norm(embeddings[idx]))
            cosine_similarity = util.pytorch_cos_sim(chunk, embeddings[idx]).item()
            combined_score = 0.6 * semantic_similarity + 0.4 * cosine_similarity
            chunk_scores[i].append(combined_score)

    return results, chunk_scores

# 主函数
if __name__ == "__main__":
    # 示例输入
    input_embeddings = np.random.rand(32, 1024)  # 示例：32个chunk的随机嵌入向量

    # 调用检索函数
    results, chunk_scores = retrieve_candidates(input_embeddings, top_k=5)

    # 打印结果
    print("前 5 个最相关的结果:")
    for i, result in enumerate(results):
        print(f"Rank {i + 1}:")
        print(f"Text: {result['text']}")
        print(f"Labels: {result['labels']}")
        print(f"Score: {result['score']}\n")

    print("每个 chunk 的得分:")
    for chunk_idx, scores in chunk_scores.items():
        print(f"Chunk {chunk_idx} Scores:", scores[:10], "...")  # 打印前10个得分
