import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import re
import time

FILENAME = "26.wav"  # Фиксированное имя файла

def validate_filename(filename):
    """Проверка имени файла с помощью регулярного выражения"""
    pattern = r'^[0-9]+\.wav$'
    if not re.match(pattern, filename):
        raise ValueError("Имя файла должно быть в формате 'номер.wav', например '26.wav'")
    return True

def read_wav(filename):
    """Чтение wav-файла и возврат частоты дискретизации и данных"""
    try:
        fs, data = wavfile.read(filename)
        if len(data.shape) > 1:
            data = data[:, 0]
        return fs, data.astype(np.float64)
    except FileNotFoundError:
        raise RuntimeError(f"Файл {filename} не найден!")
    except Exception as e:
        raise RuntimeError(f"Ошибка чтения файла {filename}: {e}")

def compute_dst_vectorized(data):
    """Векторизованная версия DST"""
    N = len(data)
    n = np.arange(N)
    k = np.arange(N).reshape(-1, 1)
    sin_values = np.sin(np.pi * k * (n + 0.5) / N)
    dst_result = np.dot(sin_values, data)
    return dst_result

def compute_dst_spectrum(data):
    """Вычисление DST"""
    spectrum = compute_dst_vectorized(data)
    return spectrum

# НАЧАЛО ПРОГРАММЫ
start_time = time.time()  # Начало отсчета времени

try:
    validate_filename(FILENAME)
    fs, data = read_wav(FILENAME)
    print(f"Файл успешно загружен")
    print(f"Частота дискретизации: {fs} Гц")
    print(f"Длительность сигнала: {len(data)/fs:.2f} сек")
    print(f"Количество отсчетов: {len(data)}")
except (ValueError, RuntimeError) as e:
    print(f"Ошибка: {e}")
    exit()

# Ввод количества отсчетов
while True:
    try:
        num_samples = int(input(f"\nВведите количество отсчетов для визуализации (макс. {len(data)}): "))
        if num_samples <= 0:
            print("Число должно быть положительным.")
            continue
        if num_samples > len(data):
            print("Число должно быть меньше или равно максимальному.")
            continue
        break
    except ValueError:
        print("Ошибка: введите целое число.")
    
# Подготовка данных для графиков
samples = data[:num_samples]
time_axis_samples = np.arange(num_samples) / fs
time_axis_full = np.arange(len(data)) / fs

# Вычисление спектра
spectrum = compute_dst_spectrum(data)
n = len(spectrum)
freqs = np.linspace(0, fs/2, n)
half_n = n // 2
freqs = freqs[:half_n]
spectrum_magnitude = np.abs(spectrum[:half_n])

elapsed_time = time.time() - start_time  # Время работы программы
print(f"\nВремя выполнения программы: {elapsed_time:.4f} секунд")
    
# График 1: Визуализация дискретных отсчетов (с закрашенной областью)
plt.figure(1, figsize=(12, 6))
plt.plot(time_axis_samples, samples, 'b-', linewidth=0.8, label='Сигнал')
plt.fill_between(time_axis_samples, samples, alpha=0.3, color='blue')
plt.title(f'Визуализация первых {num_samples} отсчетов сигнала', fontsize=14)
plt.xlabel('Время (с)', fontsize=12)
plt.ylabel('Амплитуда', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# График 2: Осциллограмма всего сигнала
plt.figure(2, figsize=(12, 6))
plt.plot(time_axis_full, data, 'g-', linewidth=0.6)
plt.title('Осциллограмма речевого сигнала', fontsize=14)
plt.xlabel('Время (с)', fontsize=12)
plt.ylabel('Амплитуда', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# График 3: Спектр сигнала (DST)
plt.figure(3, figsize=(12, 6))
plt.plot(freqs, spectrum_magnitude, 'r-', linewidth=0.8)
plt.title('Спектр сигнала (дискретное синусное преобразование, DST)', fontsize=14)
plt.xlabel('Частота (Гц)', fontsize=12)
plt.ylabel('Амплитуда спектра |DST|', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# График 4: Гистограмма отсчетов
plt.figure(4, figsize=(12, 6))
plt.hist(data, bins=50, color='purple', alpha=0.7, edgecolor='black')
plt.title('Гистограмма амплитуд отсчетов сигнала', fontsize=14)
plt.xlabel('Амплитуда', fontsize=12)
plt.ylabel('Частота попаданий', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()