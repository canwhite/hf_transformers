import torch
from jssyntax import List,Set,Map
from torch import nn
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from transformers import TFAutoModelForSequenceClassification

''' ================section1: pipeline==============='''

''' 1)基础使用'''
#pipeline() 会下载并缓存一个用于情感分析的默认的预训练模型和分词器。现在你可以在目标文本上使用 classifier 了：
classifier = pipeline("sentiment-analysis",device = 0)
#如果不止一个输入，可以将输入作为列表传入
result =  classifier("We are very happy to show you the 🤗 Transformers library.")



''' 2)制定模型和分词器 '''
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
# 加载模型
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
# 获取分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)


classifier = pipeline("sentiment-analysis", model=pt_model, tokenizer=tokenizer)
result =  classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")




''' ================section2: autoClass 以及情感分析过程==============='''

''' 1)分词器'''
encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(encoding)
# input_ids：用数字表示的 token。
# attention_mask：应该关注哪些 token 的指示。
# {'input_ids': [101, 11312, 10320, 12495, 19308, 10114, 11391, 10855, 10103, 100, 58263, 13299, 119, 102], 
# 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

#分词器也可以接受列表作为输入，
#并填充和截断文本，
#返回具有统一长度的批次：
pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
print(pt_batch)



''' 2)model
# 将分词后的输入批次传递给模型进行推理。
模型输出 logits，通过 softmax 函数转换为概率。
'''

pt_outputs = pt_model(**pt_batch)
# 模型在 logits 属性输出最终的激活结果. 在 logits 上应用 softmax 函数来查询概率:
pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
print(pt_predictions)


''' 3)保存模型
下次可以直接从本地加载模型和分词器，避免重复下载。
'''
pt_save_directory = "./pt_save_pretrained"
tokenizer.save_pretrained(pt_save_directory)
pt_model.save_pretrained(pt_save_directory)


# 下一次就从本地获取信息了
pt_model = AutoModelForSequenceClassification.from_pretrained("./pt_save_pretrained")


'''==================section3: 自定义模型构建===================='''
''' 可以配置模型，以适应自己的任务。'''
from transformers import AutoModel
from transformers import AutoConfig

my_config = AutoConfig.from_pretrained(pt_save_directory, n_heads=12)
my_model = AutoModel.from_config(my_config)




'''===================section4: Trainer - PyTorch优化训练循环 '''
from transformers import TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import Trainer

''' 1）获取model'''

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")


''' 2）分词器 '''
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")



''' 3）TrainingArguments 含有你可以修改的模型超参数'''
#比如学习率，
#批次大小
#训练时的迭代次数。
#如果你没有指定训练参数，那么它会使用默认值：

training_args = TrainingArguments(
    output_dir="path/to/save/folder/", # 这里存的是训练过程中模型的检查点、日志和其他输出文件
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
)

'''4）数据集
每次都下载一次的原因是因为默认情况下，`load_dataset` 函数会从 Hugging Face Hub 下载数据集。
为了避免每次都下载，可以将数据集缓存到本地。
-使用 `cache_dir` 参数指定缓存目录
'''

dataset = load_dataset("rotten_tomatoes", cache_dir="./datasets_cache")

'''整体数据集，分词,返回值还是dataset'''
def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])
# map方法需要两个参数：第一个参数是一个函数，第二个参数是可选的batched参数，默认为False。
# 函数参数用于对数据集中的每个元素进行转换。
# batched参数如果设置为True，则会对整个批次的数据进行转换，而不是逐个元素进行转换。
dataset = dataset.map(tokenize_dataset, batched=True)





'''5）数据批次处理器'''
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

''' 6）训练器 '''
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)  # doctest: +SKIP
trainer.train()