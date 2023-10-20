#Using Meta's blenderbot LM a self-improving model
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

#Creating the tokenizing object
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

#input text before tokenizing
input_text = "How old is he?"

#tokenizing the text
inputs = tokenizer(input_text, return_tensors="pt")

#prompting the input to the model
outputs = model.generate(**inputs)

#decoding the models response
response = tokenizer.decode(outputs[0], skip_special_tokens = True).strip()

print("Human: " + input_text)
print("Bot: " + response)
