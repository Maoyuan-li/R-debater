# -*- coding: utf-8 -*-
# @Time    : 2024/11/25 17:05
# @Author  : Maoyuan Li
# @File    : similarity_experiments.py.py
# @Software: PyCharm
# experiments/similarity_experiments.py
from rag_for_longchain.utils.evaluation import (
    calculate_cosine_similarity,
    calculate_semantic_similarity,
    calculate_bleu_score,
    calculate_rouge_score,
    calculate_bert_score
)

import jieba

def run_similarity_experiments(query, result, generator):
    # 实验1：计算问题与检索文档的余弦相似度
    question_embedding = generator.generate_embedding(query)
    print("\n问题与检索出来的文档的余弦相似度：")
    for i, doc in enumerate(result['source_documents']):
        doc_embedding = generator.generate_embedding(doc.page_content[:200])
        similarity = calculate_cosine_similarity(question_embedding, doc_embedding)
        print(f"文档 {i + 1} 相似度: {similarity:.4f}")

    # 实验2：计算语义相似度
    print("\n问题与检索出来的文档的语义相似度：")
    for i, doc in enumerate(result['source_documents']):
        doc_text = doc.page_content[:200]
        similarity = calculate_semantic_similarity(query, doc_text)
        print(f"文档 {i + 1} 相似度: {similarity:.4f}")

    # 实验3：计算回答内容与检索文档的语义相似度
    answer_text = result['answer']
    print("\n回答内容与检索出来的文档的语义相似度：")
    for i, doc in enumerate(result['source_documents']):
        doc_text = doc.page_content[:200]
        similarity = calculate_semantic_similarity(answer_text, doc_text)
        print(f"文档 {i + 1} 相似度: {similarity:.4f}")

    # 实验4：计算生成的回答与真实回答的 BLEU 分数
    generated_answer = result['answer']
    real_answer = input("请输入真实的参考答案：")
    reference = ''.join(jieba.cut(real_answer))
    candidate = ''.join(jieba.cut(generated_answer))
    bleu_score = calculate_bleu_score(candidate, reference)
    print(f"\n生成回答与真实回答的 BLEU 分数: {bleu_score:.4f}")

    # 实验5：计算 ROUGE 分数
    rouge_scores = calculate_rouge_score(generated_answer, real_answer)
    print("\nROUGE 分数：")
    print("ROUGE-1: ", rouge_scores['rouge-1'])
    print("ROUGE-2: ", rouge_scores['rouge-2'])
    print("ROUGE-L: ", rouge_scores['rouge-l'])

    # 实验6：计算 BERTScore
    bert_scores = calculate_bert_score(generated_answer, real_answer)
    print("\nBERTScore 分数：")
    print(f"Precision: {bert_scores['Precision']:.4f}, Recall: {bert_scores['Recall']:.4f}, F1: {bert_scores['F1']:.4f}")
