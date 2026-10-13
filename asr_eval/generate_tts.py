"""Генерация тестовых аудио через Silero TTS (русский)."""
import torch
import soundfile as sf

torch.set_num_threads(4)
model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language='ru', speaker='v4_ru')

texts = [
    ("clean_001", "Сегодня хорошая погода, и я решил пойти в парк."),
    ("clean_002", "Мой заказ номер двести сорок пять должен прийти завтра."),
    ("clean_003", "Подскажите, пожалуйста, как оформить возврат товара?"),
]

for name, text in texts:
    audio = model.apply_tts(text=text, speaker='baya', sample_rate=24000)
    sf.write(f"audio/{name}.wav", audio.numpy(), 24000)
    print(f"Создан: audio/{name}.wav")