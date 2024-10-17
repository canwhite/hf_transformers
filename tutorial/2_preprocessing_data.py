'''
预处理数据
在您可以在数据集上训练模型之前，数据需要被预处理为期望的模型输入格式。
无论您的数据是文本、图像还是音频，它们都需要被转换并组合成批量的张量。
🤗 Transformers 提供了一组预处理类来帮助准备数据以供模型使用。在本教程中，您将了解以下内容：
1）对于文本，使用分词器(Tokenizer)将文本转换为一系列标记(tokens)，并创建tokens的数字表示，将它们组合成张量。
2) 对于语音和音频，使用特征提取器(Feature extractor)从音频波形中提取顺序特征并将其转换为张量。
3) 图像输入使用图像处理器(ImageProcessor)将图像转换为张量。
4) 多模态输入，使用处理器(Processor)结合了Tokenizer和ImageProcessor或Processor。
'''

'''
==============================================1.文本=======================================================
对于文本，您可以使用Tokenizer类将文本转换为一系列标记(tokens)，并创建tokens的数字表示，将它们组合成张量。
'''

'''1)基本使用'''
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
print(encoded_input)
'''
input_ids 是与句子中每个token对应的索引。
attention_mask 指示是否应该关注一个toekn。
token_type_ids 在存在多个序列时标识一个token属于哪个序列。
'''
# {'input_ids': [101, 2079, 2025, 19960, 10362, 1999, 1996, 3821, 1997, 16657, 1010, 2005, 2027, 2024, 11259, 1998, 4248, 2000, 4963, 1012, 102],
#  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

input_ids = encoded_input["input_ids"]
#也通过解码 input_ids 来返回您的输入：
decoderInfo =  tokenizer.decode(input_ids)
print("----",decoderInfo)



'''
2）如果有多个句子需要预处理，将它们作为列表传递给tokenizer：
'''

batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_inputs = tokenizer(batch_sentences)
print(encoded_inputs)

'''
3）填充、截断和构建张量：
3-1）填充
句子的长度并不总是相同，这可能会成为一个问题，因为模型输入的张量需要具有统一的形状。
填充是一种策略，通过在较短的句子中添加一个特殊的padding token，以确保张量是矩形的。

3-2）截断
另一方面，有时候一个序列可能对模型来说太长了。
在这种情况下，您需要将序列截断为更短的长度。
将 truncation 参数设置为 True，以将序列截断为模型接受的最大长度：

3-3)构建张量
将 return_tensors 参数设置为 pt（对于PyTorch）或 tf（对于TensorFlow）：

'''

batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_input = tokenizer(batch_sentences, padding=True ,truncation=True,return_tensors="pt")
print(encoded_input)



'''
==============================================2.音频=======================================================
对于文本，您可以使用Tokenizer类将文本转换为一系列标记(tokens)，并创建tokens的数字表示，将它们组合成张量。
'''

from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor

# 1）加载数据
dataset = load_dataset(
    "PolyAI/minds14", 
    name="en-US", 
    split="train", 
    cache_dir="/Users/zack/Desktop/hf_transformers/data",
    use_auth_token=True)

print(dataset[0]["audio"])
# array 是加载的语音信号 - 并在必要时重新采为1D array。
# path 指向音频文件的位置。
# sampling_rate 是每秒测量的语音信号数据点数量。

# 2）设置采样率， 提高采样率到16kHz
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
#在看一下输出
print(dataset[0]["audio"])

# 3)对数据进行标准化和填充，需要一个特性提取器-这东西类似
# 当填充文本数据时，会为较短的序列添加 0。相同的理念适用于音频数据。
# feature extractor添加 0 - 被解释为静音
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# 4）将音频 array 传递给feature extractor。
audio_input = [dataset[0]["audio"]["array"]]
feature_extractor(audio_input, sampling_rate=16000)







'''
==============================================2.视觉=======================================================

'''