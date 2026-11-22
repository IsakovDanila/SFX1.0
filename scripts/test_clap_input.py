"""Регрессия: вход в laion_clap должен быть списком дорожек.

Дефект, который этот тест ловит (был в audio_vector_file):

    self.model.get_audio_embedding_from_data(np.asarray(y), use_tensor=False)

`laion_clap/hook.py` делает `for audio_waveform in x`, то есть ждёт
БАТЧ дорожек формы (N, T). Для 1-D массива он итерирует по отдельным
сэмплам, и каждый numpy.float32 (0 измерений) падает:

    TypeError: object of type 'numpy.float32' has no len()

Следствие: при сканировании падал каждый файл, библиотека оставалась
пустой, поиск по референсу не работал вообще.

Run:  python3 scripts/test_clap_input.py
"""
import ast
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.join(HERE, "..", "backend", "app", "embeddings.py")


# --- 1. исходник передаёт СПИСОК, а не массив -------------------------
src = open(EMB, encoding="utf-8").read()
tree = ast.parse(src)
call_arg = None
for node in ast.walk(tree):
    if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_audio_embedding_from_data"):
        call_arg = node.args[0]
        break
assert call_arg is not None, "вызов get_audio_embedding_from_data не найден"
assert isinstance(call_arg, ast.List), (
    "вход должен быть списком [np.asarray(y)], а не "
    f"{type(call_arg).__name__} — иначе laion_clap обходит сэмплы")
print("1. вход в CLAP — список дорожек OK")


# --- 2. имитация hook.py: батч из одного файла проходит ----------------
def hook_iterates(x):
    """Строка `for audio_waveform in x` из laion_clap/hook.py:174."""
    for w in x:
        try:
            len(w)              # data.py:417 -> len(audio_data)
        except TypeError as e:
            return f"TypeError: {e}"
    return "ok"


y = np.zeros(48000, dtype=np.float32)
assert hook_iterates([np.asarray(y, dtype=np.float32)]) == "ok"
# старый вариант должен падать — иначе тест бессмысленный
assert hook_iterates(np.asarray(y, dtype=np.float32)).startswith("TypeError")
print("2. батч проходит, 1-D массив падает (как и было) OK")


# --- 3. пустой звук отсекается до вызова модели -----------------------
assert "нет звука после обрезки тишины" in src, (
    "нет защиты от пустого сигнала после _trim_silence: "
    "repeatpad сделает int(480000/0) -> ZeroDivisionError")
print("3. защита от пустого сигнала OK")

print("\nВСЕ ТЕСТЫ ВХОДА В CLAP ПРОЙДЕНЫ")
