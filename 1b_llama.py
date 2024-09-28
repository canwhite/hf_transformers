
from transformers import AutoTokenizer, AutoModelForCausalLM , AutoModelForSequenceClassification
from transformers import pipeline


# 模型存储地址
llama_1b_save_pretrained = "./llama_1b_save_pretrained"
'''
## 第一次保存模型,方便下次使用
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B")
tokenizer.save_pretrained(llama_1b_save_pretrained)
model.save_pretrained(llama_1b_save_pretrained)
print("save completely")
'''

print("====0")
# model = AutoModelForSequenceClassification.from_pretrained(llama_1b_save_pretrained)
model = AutoModelForCausalLM.from_pretrained(llama_1b_save_pretrained)
tokenizer = AutoTokenizer.from_pretrained(llama_1b_save_pretrained)
print("====1")
# 设置最大 token 数目
# max_length = 128000
max_length = 64



pipe = pipeline(
    "text-generation", 
    model= model, 
    tokenizer=tokenizer,
    # max_new_tokens = max_length,
    max_length = max_length,
    truncation=True,
    device = -1
)
print("====2")

result = pipe("The key to life is")
print(result)
print(result[0]['generated_text'])