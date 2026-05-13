from PIL import Image
import os


def load_keys(keys_path):
    keys = []
    with open(keys_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                line = line.replace("(", "").replace(")", "")
                x, y = map(int, line.split(","))
                keys.append((x, y))
    return keys


def decode_full_blue(image_path, keys_path, title="Декодирование"):
    
    # Проверка существования файлов
    if not os.path.exists(image_path):
        print(f"Ошибка: файл {image_path} не найден!")
        return ""
    
    if not os.path.exists(keys_path):
        print(f"Ошибка: файл {keys_path} не найден!")
        return ""
    
    # Загружаем ключи
    keys = load_keys(keys_path)
    print(f"Загружено ключей: {len(keys)}")
    
    # Загружаем изображение (конвертируем в RGB, чтобы убрать альфа-канал)
    image = Image.open(image_path).convert("RGB")
    pixels = image.load()
    
    # Извлекаем байты напрямую из синего канала
    bytes_data = []
    
    # Показываем первые 10 пикселей для отладки
    print("\nПервые 10 пикселей (синий канал)")
    for idx, (x, y) in enumerate(keys[:10]):
        r, g, b = pixels[x, y]
        bytes_data.append(b)
        # Показываем символ, если он печатный
        char_display = chr(b) if 32 <= b <= 126 else '■'
        print(f"Пиксель {idx+1}: ({x},{y}) B={b:3d} (0x{b:02x}) -> '{char_display}'")
    
    # Добавляем остальные пиксели
    for idx, (x, y) in enumerate(keys[10:], start=10):
        r, g, b = pixels[x, y]
        bytes_data.append(b)
    
    print(f"\nВсего байт: {len(bytes_data)}")
    
    # Ищем нулевой байт (конец строки)
    result_bytes = []
    for b in bytes_data:
        if b == 0:
            break
        result_bytes.append(b)
    
    # Декодируем текст
    try:
        text = bytes(result_bytes).decode('utf-8')
        print(f"Результат: {text}") 
        return text
    except UnicodeDecodeError as e:
        print(f"\nОшибка декодирования UTF-8: {e}")
        # Пробуем другие кодировки
        for encoding in ['latin-1', 'cp1251', 'ascii']:
            try:
                text = bytes(result_bytes).decode(encoding)
                print(f"\nДекодировано как {encoding}:")
                print(text)
                return text
            except:
                continue
        
        # Если не декодируется, показываем байты
        print(f"\nБайты: {bytes_data[:30]}...")
        print(f"Первые 20 байт: {result_bytes[:20]}")
        print(f"Как символы: {[chr(b) if 32 <= b <= 126 else f'[{b}]' for b in result_bytes[:20]]}")
        return ""


def encode_full_blue(image_path, output_path, keys_path, text):
 
    # Загружаем ключи
    keys = load_keys(keys_path)
    print(f"Загружено ключей для кодирования: {len(keys)}")
    
    # Подготовка текста (добавляем нулевой байт в конец)
    text_bytes = text.encode('utf-8') + b'\x00'
    
    # Проверка длины
    if len(text_bytes) > len(keys):
        print(f"Ошибка: текст слишком длинный!")
        print(f"Максимум символов: {len(keys)} (включая терминатор)")
        print(f"Ваш текст: {len(text_bytes)} символов")
        print(f"Текст для кодирования: {len(text)} символов")
        return False
    
    print(f"\nТекст для кодирования: {text}")
    print(f"Длина текста: {len(text)} символов")
    print(f"Всего байт для записи: {len(text_bytes)} (включая терминатор)")
    
    # Отладка: биты первого символа
    if text:
        first_char = text[0]
        first_char_code = ord(first_char)
        print(f"Отладка: ")
        print(f"а) Биты первого символа '{first_char}':")
        print(f"   Код символа: {first_char_code}")
        print(f"   Биты (b7 b6 b5 b4 b3 b2 b1 b0): {format(first_char_code, '08b')}")
    
    # Загружаем изображение (конвертируем в RGB, чтобы убрать альфа-канал)
    img = Image.open(image_path).convert("RGB")
    pixels = img.load()
    
    print("\nб) Исходные и изменённые значения пикселей (первые 4):")
    
    # Кодируем: записываем байты напрямую в синий канал
    for idx, ((x, y), byte_val) in enumerate(zip(keys, text_bytes)):
        r, g, b = pixels[x, y]
        old_b = b
        
        # Записываем новый байт в синий канал
        new_b = byte_val
        pixels[x, y] = (r, g, new_b)
        
        if idx < 4:
            # Показываем символ, если он печатный
            old_char = chr(old_b) if 32 <= old_b <= 126 else '■'
            new_char = chr(new_b) if 32 <= new_b <= 126 else '■'
            print(f"\nПиксель {idx+1}: ({x},{y})")
            print(f"  Было:   R={r:3d}, G={g:3d}, B={old_b:3d} (0x{old_b:02x}) -> '{old_char}'")
            print(f"  Стало:  R={r:3d}, G={g:3d}, B={new_b:3d} (0x{new_b:02x}) -> '{new_char}'")
    
    print("\nв) Кодирование завершено")
    
    # Сохраняем изображение
    img.save(output_path)
    print(f"\nИзображение сохранено: {output_path}")
    print(f"Закодировано пикселей: {len(text_bytes)}")

    return True


def main():
    
    # Файлы
    encoded_img = "new26.png"           # Закодированное изображение из облака
    keys_file = "keys26.txt"            # Ключи для варианта 26
    source_img = "test.png"             # Исходное изображение для кодирования
    output_img = "test_encoded.png"     # Результат кодирования
    output_keys = "keys_test.txt"       # Ключи для закодированного изображения
    
    # Проверка файлов
    files_status = {}
    for f in [encoded_img, keys_file, source_img]:
        if os.path.exists(f):
            files_status[f] = True
        else:
            print(f"{f} (отсутствует)")
            files_status[f] = False
            if f == source_img:
                print(f"Создайте файл {source_img} для кодирования")
    
    # Декодирование new26.png
    if files_status.get(encoded_img, False) and files_status.get(keys_file, False):
        print("Декодирование изображения new26.png")
        decoded_text = decode_full_blue(encoded_img, keys_file, "Декодирование new26.png")
    
    # Кодирование текста в test.png
    user_input = input("Введите текст для кодирования: ").strip()
    
    if user_input and files_status.get(source_img, False):
        max_len = len(load_keys(keys_file))
        if len(user_input) > max_len - 1:  # -1 для нулевого байта
            print(f"Текст слишком длинный! Максимум {max_len - 1} символов")
        else:
            # Кодируем
            encode_full_blue(source_img, output_img, keys_file, user_input)
            
            # Сохраняем ключи для закодированного изображения (те же координаты)
            with open(output_keys, "w", encoding="utf-8") as f:
                keys = load_keys(keys_file)
                for x, y in keys:
                    f.write(f"({x},{y})\n")
            print(f"Ключи сохранены: {output_keys}")
            
            # Декодирование созданного файла test_encoded.png
            print("Декодирование созданного файла test_encoded.png")
            decoded_own = decode_full_blue(output_img, output_keys, "Декодирование test_encoded.png")
            
            # Проверяем, совпадает ли декодированный текст с исходным
            if decoded_own == user_input:
                print(f"\nТекст совпадает с исходным.")
            elif decoded_own:
                print(f"\nДекодировано '{decoded_own}', исходный '{user_input}'.")
            else:
                print(f"\nНе удалось декодировать текст.")
    
    elif not user_input:
        print("Текст не введён. Кодирование пропущено.")
    elif not files_status.get(source_img, False):
        print(f"Файл {source_img} не найден. Кодирование пропущено.")


if __name__ == "__main__":
    main()