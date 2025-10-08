# -*- coding: utf-8 -*-
# @Time    : 2024/11/25 17:05
# @Author  : Maoyuan Li
# @File    : evaluation.py.py
# @Software: PyCharm
# utils/evaluation.py
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score

def calculate_cosine_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]

def calculate_semantic_similarity(text1, text2, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2)
    return similarity.item()

def calculate_bleu_score(candidate, reference):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([list(reference)], list(candidate), smoothing_function=smoothie)

def calculate_rouge_score(candidate, reference):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference, avg=True)
    return scores

def calculate_bert_score(candidate, reference):
    P, R, F1 = score([candidate], [reference], lang="zh")
    return {'Precision': P.mean().item(), 'Recall': R.mean().item(), 'F1': F1.mean().item()}
