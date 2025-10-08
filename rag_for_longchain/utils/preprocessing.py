cd import json
import os
import re

class JSONChunkSplitter:
    def __init__(self, directory, max_words=512):
        """
        初始化类，设置文件目录和最大单词限制。
        :param directory: 包含 JSON 文件的目录路径
        :param max_words: 每个 chunk 的最大单词数
        """
        self.directory = directory
        self.max_words = max_words

    def split_utterance(self, utterance):
        """
        将单个发言切分为多个块，每个块的最大单词数为 max_words。
        如果某个发言超过最大单词数限制，将最后两句话作为下一块的开头。
        :param utterance: 要切分的发言字符串
        :return: 切分后的块列表
        """
        words = utterance.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.max_words, len(words))
            chunk = " ".join(words[start:end])
            if end < len(words):  # 处理非最后一块
                sentences = re.split(r'(?<=[.!?。！？])', chunk)
                if len(sentences) > 2:
                    carry_over = "".join(sentences[-2:])
                    chunk = "".join(sentences[:-2])
                else:
                    carry_over = chunk
                chunks.append(chunk.strip())
                start += len(chunk.split())
                words = carry_over.split() + words[end:]
            else:  # 处理最后一块
                chunks.append(chunk.strip())
                break
        return chunks

    def process_file(self, file_path):
        """
        处理单个 JSON 文件，将所有发言切分为块。
        :param file_path: JSON 文件路径
        :return: 包含所有块的列表
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        chunks = []
        for debate in data["debate"]:
            stance = debate["stance"]
            debater = debate["debater"]
            utterance = debate["utterance"]
            utterance_chunks = self.split_utterance(utterance)
            for chunk in utterance_chunks:
                chunks.append({
                    "stance": stance,
                    "debater": debater,
                    "utterance": chunk
                })
        return chunks

    def process_directory(self):
        """
        处理目录中的所有 JSON 文件，将所有发言切分为块。
        :return: 包含所有文件中块的列表
        """
        all_chunks = []
        for file_name in os.listdir(self.directory):
            if file_name.endswith('.json'):  # 仅处理 JSON 文件
                file_path = os.path.join(self.directory, file_name)
                print(f"正在处理文件: {file_name}")
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
        return all_chunks


# 示例用法
if __name__ == "__main__":

    # 设置文件目录路径
    directory = r"D:\converstional_rag\rag_for_longchain\data\ACL_raw_data"  # 替换为你的 JSON 文件目录路径

    # 创建 JSONChunkSplitter 实例
    splitter = JSONChunkSplitter(directory)

    # 处理目录中的所有 JSON 文件
    all_chunks = splitter.process_directory()

    # 打印处理后的结果
    print("切分后的内容：")
    print(json.dumps(all_chunks, ensure_ascii=False, indent=4))  # 格式化输出为 JSON 字符串

