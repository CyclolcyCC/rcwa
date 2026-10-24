"""
Точка входа для запуска графического интерфейса RCWA GIXRF (s vs p).
Запуск из корня проекта:
    python gui.py
"""

import sys
import os

# Добавление корневой директории в sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.gui import main

if __name__ == "__main__":
    main()
