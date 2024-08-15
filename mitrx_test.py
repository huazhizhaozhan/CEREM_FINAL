from transformers import AutoTokenizer

# 加载预训练模型的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("pretrain/chinese-roberta-wwm-ext")

# 中文文本
text = "你好，今天天气怎么样？"

# 使用 tokenizer 进行分词
tokens = tokenizer.tokenize(text)

# 输出分词结果
print(tokens)