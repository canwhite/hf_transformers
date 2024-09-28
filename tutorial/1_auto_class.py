'''
使用AutoClass加载预训练实例
from_pretrained()方法允许您快速加载任何架构的预训练模型，因此您不必花费时间和精力从头开始训练模型。

在这个教程中，学习如何：
加载预训练的分词器（tokenizer）
加载预训练的图像处理器(image processor)
加载预训练的特征提取器(feature extractor)
加载预训练的处理器(processor)
加载预训练的模型。
'''

from transformers import pipeline



'''===================AutoTokenizer:分词===================='''
from transformers import AutoTokenizer
'''
# 设置按句分
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_basic_tokenize=False)
tokenizer.add_special_tokens({'sep_token': '[SEP]'})
tokenizer.add_special_tokens({'cls_token': '[CLS]'})
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.add_special_tokens({'unk_token': '[UNK]'})
tokenizer.add_special_tokens({'mask_token': '[MASK]'})

def tokenize_by_sentence(text):
    sentences = text.split('. ')
    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    return tokenized_sentences

sequence = "In a hole in the ground there lived a hobbit."
tokenized_sequence = tokenize_by_sentence(sequence)
print(tokenized_sequence)
'''

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

sequence = "In a hole in the ground there lived a hobbit."
print(tokenizer(sequence))



'''===================AutoImageProcessor:视觉任务==================='''
from transformers import AutoImageProcessor
'''
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
'''


'''===================AutoFeatureExtractor:音频任务================='''
from transformers import AutoFeatureExtractor
'''
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
'''

'''==========================AutoProcessor:多模态========================'''
# 多模态任务需要一种processor，将两种类型的预处理工具结合起来。
# 例如，LayoutLMV2模型需要一个image processo来处理图像和一个tokenizer来处理文本；processor将两者结合起来
from transformers import AutoProcessor

'''
# 加载图像处理器
image_processor = AutoImageProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")

# 结合图像处理器和分词器
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")

# 示例：处理图像和文本
image = "path_to_image.jpg"
text = "这是一个示例文本"

# 使用processor处理图像和文本
processed_inputs = processor(images=image, text=text, return_tensors="pt")

# 打印处理后的输入
print(processed_inputs)
'''


'''==========================AutoModel========================'''
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification #加载用于序列分类的模型
from transformers import AutoModelForTokenClassification #加载用于标记的模型
'''
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased")
'''

# 上边这些有些是为了推理，有些是为了训练，我们这里选为了推理，so是可以灵活应用的，点题了
pt_model = AutoModelForSequenceClassification.from_pretrained("/Users/zack/Desktop/hf_transformers/pt_save_pretrained")

classifier = pipeline("sentiment-analysis", model=pt_model, tokenizer=tokenizer)
result =  classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")
print(result)