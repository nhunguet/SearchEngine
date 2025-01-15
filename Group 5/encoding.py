import sys
sys.stdout.reconfigure(encoding='utf-8')

# Demo: Mã hóa và chuyển đổi lại ký tự ASCII và CJK
def encode_character(character):
    """
    Mã hóa ký tự thành chuỗi byte UTF-8.
    """
    utf8_encoded = character.encode('utf-8')  # Mã hóa UTF-8
    utf8_hex = ' '.join(f"{byte:02X}" for byte in utf8_encoded)  # Chuyển sang hex
    return utf8_hex

def decode_character(utf8_hex):
    """
    Giải mã chuỗi byte UTF-8 thành ký tự.
    """
    bytes_array = bytes(int(byte, 16) for byte in utf8_hex.split())  # Chuyển hex sang byte
    character = bytes_array.decode('utf-8')  # Giải mã UTF-8
    return character

# Danh sách ký tự demo (ASCII và CJK)
characters = [
    'B',        # ASCII (U+0041)
    '!',        # ASCII (U+0021)
    '\u4E2D',   # CJK: Trung Quốc "中" (U+4E2D)
    '\u3042',   # CJK: Nhật Bản "あ" (U+3042)
    '\uAC00'    # CJK: Hàn Quốc "가" (U+AC00)
]

print("=== Mã hóa ký tự ===")
for char in characters:
    utf8_hex = encode_character(char)
    print(f"Ký tự: '{char}' -> UTF-8 Hex: {utf8_hex}")

print("\n=== Giải mã UTF-8 Hex ===")
for char in characters:
    utf8_hex = encode_character(char)
    decoded_char = decode_character(utf8_hex)
    print(f"UTF-8 Hex: {utf8_hex} -> Ký tự: '{decoded_char}'")
