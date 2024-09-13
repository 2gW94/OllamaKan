import json
import time
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Класс для работы с API Ollama
class OllamaAPI:
    def __init__(self, url):
        self.URL = url

    def improve_prompt(self, prompt):
        # Системный промпт
        system_prompt = "You are a helpful assistant. Improve the following prompt for generating an image:  Only output the improved prompt, nothing else:"

        # Объединяем системный и пользовательский промпт
        combined_prompt = f"{system_prompt}\n\n{prompt}"

        # Ограничиваем длину до 1000 символов
        if len(combined_prompt) > 1000:
            combined_prompt = combined_prompt[:1000]

        data = {
            "model": "llama3.1:8b",  # Используем модель LLaMA 3.1
            "prompt": combined_prompt,
            "stream": False  # Отключаем стриминг для получения полного ответа сразу
        }
        response = requests.post(self.URL, json=data)
        response_data = response.json()
        return response_data['response'].strip()

class Text2ImageAPI:
    def __init__(self, url, api_key, secret_key):
        self.URL = url
        self.AUTH_HEADERS = {
            'X-Key': f'Key {api_key}',
            'X-Secret': f'Secret {secret_key}',
        }

    def get_model(self):
        response = requests.get(self.URL + 'key/api/v1/models', headers=self.AUTH_HEADERS)
        data = response.json()
        return data[0]['id']

    def generate(self, prompt, model, images=1, width=1024, height=1024):
        params = {
            "type": "GENERATE",
            "numImages": images,
            "width": width,
            "height": height,
            "generateParams": {
                "query": f"{prompt}"
            }
        }

        data = {
            'model_id': (None, model),
            'params': (None, json.dumps(params), 'application/json')
        }
        response = requests.post(self.URL + 'key/api/v1/text2image/run', headers=self.AUTH_HEADERS, files=data)
        data = response.json()
        return data['uuid']

    def check_generation(self, request_id, attempts=10, delay=10):
        while attempts > 0:
            response = requests.get(self.URL + 'key/api/v1/text2image/status/' + request_id, headers=self.AUTH_HEADERS)
            data = response.json()
            if data['status'] == 'DONE':
                return data['images']

            attempts -= 1
            time.sleep(delay)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    initial_prompt = request.form['prompt']

    # Инициализация API Ollama
    ollama_api = OllamaAPI('http://localhost:11434/api/generate')

    # Улучшение промпта через Ollama
    improved_prompt = ollama_api.improve_prompt(initial_prompt)

    # Вывод улучшенного промпта в терминал
    print(f"Improved Prompt: {improved_prompt}")

    # Инициализация API Kandinsky
    kandinsky_api = Text2ImageAPI('https://api-key.fusionbrain.ai/', "ВАШ_API_KEY", "ВАШ_SECRET_KEY")
    model_id = kandinsky_api.get_model()

    # Генерация изображения через Kandinsky
    uuid = kandinsky_api.generate(improved_prompt, model_id)
    images = kandinsky_api.check_generation(uuid)

    # Возвращаем изображение в формате Base64
    return jsonify({"image": images[0]})


if __name__ == '__main__':
    app.run(debug=True)
