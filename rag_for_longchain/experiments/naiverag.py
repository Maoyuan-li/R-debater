import os
import json
import yaml
import re
import requests
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# 解决 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 配置文件路径
CONFIG_FILE = r"D:/converstional_rag/rag_for_longchain/config/config.yaml"

def load_config(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except:
        return {}

config = load_config(CONFIG_FILE)
api_key         = config.get('openai', {}).get('api_key')
api_base        = config.get('openai', {}).get('api_base')
model_name_used = config.get('openai', {}).get('llm_model')

# 文本提取工具
def extract_utterances(data):
    utterances = []
    if isinstance(data, str):
        utterances.extend([line.strip() for line in data.splitlines() if line.strip()])
    elif isinstance(data, dict):
        for k, v in data.items():
            if k.lower() in ['utterance', 'utterances']:
                utterances.extend(v if isinstance(v, list) else [v])
            else:
                utterances.extend(extract_utterances(v))
    elif isinstance(data, list):
        for item in data:
            utterances.extend(extract_utterances(item))
    return utterances

# 准备查询内容
def prepare_query(path_or_obj):
    if isinstance(path_or_obj, str) and path_or_obj.lower().endswith('.json') and os.path.isfile(path_or_obj):
        with open(path_or_obj, 'r', encoding='utf-8') as f:
            obj = json.load(f)
    else:
        obj = path_or_obj
    return "\n".join(extract_utterances(obj))

# 检索模型配置
DB_PATH    = r"D:\converstional_rag\rag_for_longchain\experiments\structured_embeddings_for_naive_rag.npz"
MODEL_NAME = "hfl/chinese-roberta-wwm-ext-large"
MAX_TOKENS = 512

# 加载Tokenizer与Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_TOKENS)
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

class Retriever:
    def __init__(self):
        data = np.load(DB_PATH, allow_pickle=True)
        files = set(data.files)
        # 加载数据库原utterances
        self.utterances = data['utterances'] if 'utterances' in files else data['utterance']
        # 加载数据库embeddings
        emb_key = 'embeddings' if 'embeddings' in files else 'embedding'
        self.embeddings = data[emb_key]
        dim = model.config.hidden_size
        if self.embeddings.shape[1] != dim:
            raise ValueError(f"DB embeddings dim {self.embeddings.shape[1]} != model dim {dim}")
        # 归一化
        self.embeddings /= np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10

    def retrieve(self, json_path, top_k=5):
        # 读取并拼接用户查询文本
        text = prepare_query(json_path)
        tokens = tokenizer.tokenize(text)
        # 切分成chunks，不影响检索库
        chunks = [text] if len(tokens) <= MAX_TOKENS else []
        if not chunks:
            for line in text.split("\n"):
                line_tokens = tokenizer.tokenize(line)
                if len(line_tokens) <= MAX_TOKENS:
                    chunks.append(line)
                else:
                    for i in range(0, len(line), MAX_TOKENS):
                        chunks.append(line[i:i+MAX_TOKENS])
        # 查询embedding
        embs = np.vstack([embed_text(c) for c in chunks])
        qemb = embs.mean(axis=0)
        qemb /= np.linalg.norm(qemb) + 1e-10
        # 余弦检索
        sims = self.embeddings.dot(qemb)
        idxs = np.argsort(sims)[-top_k:][::-1]
        # 返回数据库中的utterances而不是chunks
        return [self.utterances[i] for i in idxs]

# 输出文件路径
OUTPUT_FILE = r"D:\converstional_rag\23acldata\output_data\rag实验\debate_speeches_output_claude.json"

# 调用LLM并提取content

def generate_debate_via_api(title, texts, retrieved_utts):
    if not api_key or not api_base:
        return {'error': 'API未配置'}
    url = api_base.rstrip('/') + '/chat/completions'
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    # 构建prompt
    meta = json.dumps({'title': title}, ensure_ascii=False, indent=2)
    context = '\n'.join([f"Chunk {i+1}: {u}" for i, u in enumerate(retrieved_utts)])
    prompt = f"""
以下给你一场辩论的元数据（JSON 格式）：
{meta}
用户输入：
{texts}
检索材料：
{context}
请产生输出，严格按照下面格式：
辩论标题（Debate Title）
使用的 LLM 模型名称（Model Used）
正方陈词（Pro Speech），一段话
反方陈词（Con Speech），一段话
每一方陈词的字数不超过1100字
用 JSON 对象返回，键为 title, model, pro_speech, con_speech。
"""
    payload = {
        'model': model_name_used,
        'messages': [
            {'role': 'system', 'content': 'You are a debate assistant.'},
            {'role': 'user',   'content': prompt}
        ],
        'temperature': 0.7,
        'max_tokens': 1200,
        'stream': False
    }
    resp = requests.post(url, headers=headers, json=payload)
    content = resp.json()['choices'][0]['message']['content'].strip()
    if content.startswith('```'):
        content = re.sub(r'^```[\w]*\n', '', content)
        content = re.sub(r'\n```$', '', content)
    try:
        out = json.loads(content)
        out['model'] = model_name_used
    except:
        out = {'title': title, 'model': model_name_used, 'pro_speech': content, 'con_speech': ''}
    return out

# 主流程：写死路径，无需输入
if __name__ == '__main__':
    folder = r"D:\converstional_rag\23acldata\input_data\processed_input"
    top_k = 5
    retr = Retriever()
    results = []
    for sub in os.listdir(folder):
        subdir = os.path.join(folder, sub)
        if os.path.isdir(subdir):
            for fn in os.listdir(subdir):
                if fn.lower().endswith('.json') and fn != 'last_two.json':
                    path = os.path.join(subdir, fn)
                    title = os.path.splitext(fn)[0]
                    retrieved_utts = retr.retrieve(path, top_k)
                    texts = prepare_query(path)
                    res = generate_debate_via_api(title, texts, retrieved_utts)
                    results.append(res)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

# 依赖: pip install transformers torch numpy pyyaml requests
