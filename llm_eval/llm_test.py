import ollama

response = ollama.chat(model='qwen2.5:3b', messages=[
    {'role': 'system', 'content': 'Отвечай полно и подробно'},
    {'role': 'user', 'content': 'как думаешь сколько ресурсов ты у меня занимаешь?'},
])
print(response['message']['content'])