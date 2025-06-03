import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
import tensorflow as tf
from Photo_Detection.app import App

tf.config.run_functions_eagerly(True)
tf.keras.backend.clear_session()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    root.title("Детектор фейкових зображень")

    window_width = 800
    window_height = 600
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    root.geometry(f'{window_width}x{window_height}+{x}+{y}')

    app = App(root)
    root.mainloop()