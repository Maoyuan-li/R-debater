import os
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings

def generate_and_save_embeddings_for_all(files_chunks, output_directory, api_key, model="text-embedding-ada-002"):
    """
    为多个 JSON 文件的分块生成嵌入，并将结果保存到指定目录。

    Args:
        files_chunks (dict): 文件名为键，分块后的文本列表为值。
        output_directory (str): 嵌入文件保存目录。
        api_key (str): OpenAI API 密钥。
        model (str): 使用的嵌入模型名称。
    """
    os.makedirs(output_directory, exist_ok=True)
    embedding_model = OpenAIEmbeddings(
        openai_api_key=api_key,
        model=model
    )

    for filename, chunks in files_chunks.items():
        embeddings = [embedding_model.embed_query(chunk) for chunk in chunks]
        embeddings = np.array(embeddings)

        output_file = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_embeddings.npy")
        np.save(output_file, embeddings)
        print(f"Embeddings saved for {filename} -> {output_file}")
