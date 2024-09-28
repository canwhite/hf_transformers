from transformers import AutoModelForCausalLM, AutoTokenizer


# 模型存储地址
llama_1b_save_pretrained = "./llama_1b_save_pretrained"


'''
model_name = "Vikhrmodels/Vikhr-Llama-3.2-1B-instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(llama_1b_save_pretrained)
model.save_pretrained(llama_1b_save_pretrained)
'''

model_name = "Vikhrmodels/Vikhr-Llama-3.2-1B-instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = AutoModelForCausalLM.from_pretrained(llama_1b_save_pretrained)
# tokenizer = AutoTokenizer.from_pretrained(llama_1b_save_pretrained)


# Подготовка входного текста
input_text = "The key to life is"

# Токенизация и генерация текста
input_ids = tokenizer.encode(input_text, return_tensors="pt")


output = model.generate(
  input_ids,
  max_length=1512,
  temperature=1,
  num_return_sequences=1,
  no_repeat_ngram_size=2,
  top_k=50,
  top_p=0.95,
)

# Декодирование и вывод результата
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)