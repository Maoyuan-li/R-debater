# -*- coding: utf-8 -*-
# @Time    : 2025/10/16 12:50
# @Author  : Maoyuan Li
# @File    : test.py
# @Software: PyCharm
with open("../config/config.yaml", "r", encoding="utf-8") as f:
    content = f.read()
    print("✅ 文件内容如下：")
    print(content)
