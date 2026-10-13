"""Добавление шума к аудио для создания стратифицированного тестового набора."""
import numpy as np
import soundfile as sf
from pathlib import Path

def add_noise(audio_path: Path, output_path: Path, noise_level: float = 0.1):
    """Добавляет белый шум к аудиофайлу."""
    data, samplerate = sf.read(str(audio_path))
    
    # Генерируем белый шум той же длины
    noise = np.random.normal(0, noise_level, data.shape)
    
    # Смешиваем
    noisy = data + noise
    
    # Нормализуем, чтобы не было клиппинга
    max_val = np.max(np.abs(noisy))
    if max_val > 1.0:
        noisy = noisy / max_val
    
    sf.write(str(output_path), noisy, samplerate)

# Создаем шумные версии
Path("audio/noisy").mkdir(exist_ok=True)

for wav in Path("audio").glob("clean_*.wav"):
    noisy_path = Path(f"audio/noisy/{wav.stem}_noisy.wav")
    add_noise(wav, noisy_path, noise_level=0.15)  # 15% шума
    print(f"Создан: {noisy_path}")