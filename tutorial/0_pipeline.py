''' 
pipeline() 让使用Hub上的任何模型进行任何语言、计算机视觉、语音以及多模态任务的推理变得非常简单。
pipeline可以让调用模型变得很简单
'''
from transformers import pipeline

'''参数讲解：
1）task 和 model 的区别
# task 参数会自动下载并缓存一个用于指定任务的默认预训练模型和分词器。
# 例如，如果 task 是 "sentiment-analysis"，则会自动下载并缓存一个用于情感分析的默认模型。
# model 参数允许你指定一个特定的预训练模型，而不是使用默认的模型。
# 你可以使用 model 参数来加载你自己的模型或 Hugging Face Hub 上的特定模型。
# 例如，你可以使用 model 参数来加载一个特定的情感分析模型，而不是使用默认的模型。
# 以下是一个使用 model 参数的示例：
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
classifier = pipeline(task="sentiment-analysis", model=model_name)
result = classifier("We are very happy to show you the 🤗 Transformers library.")
print(result)

2）device参数
# device参数用于指定模型运行的设备。它可以是CPU或GPU。
# 如果device=0，则表示使用第一个GPU设备。
# 如果device=-1，则表示使用CPU设备。
# 例如，以下代码将使用第一个GPU设备进行情感分析：
# classifier = pipeline(task="sentiment-analysis", device=0)

PS：
如果安装了accelerate，那么device参数会自动被忽略，因为accelerate会自动选择最佳设备。
so,可以使用
device_map="auto"


3）批量处理
batch_size=2


4) 任务特定参数
每个任务都有许多可用的参数，因此请查看每个任务的API参考，以了解您可以进行哪些调整！
例如，AutomaticSpeechRecognitionPipeline 具有 chunk_length_s 参数，
对于处理非常长的音频文件（例如，为整部电影或长达一小时的视频配字幕）非常有帮助，这通常是模型无法单独处理的
transformers.AutomaticSpeechRecognitionPipeline.call() 方法具有一个 return_timestamps 参数，对于字幕视频似乎很有帮助：

'''

transcriber = pipeline(task="automatic-speech-recognition",device=1, batch_size=4,chunk_length_s=30, return_timestamps='char')

#语音识别
result =  transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print(result)

