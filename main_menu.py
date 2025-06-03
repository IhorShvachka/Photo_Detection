import tkinter as tk
from tkinter import ttk

class MainMenu:
    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.frame = ttk.Frame(root)
        self.frame.pack(pady=50, padx=50)

        ttk.Label(self.frame, text="Детектор фейкових зображень", font=('Arial', 16)).pack(pady=20)

        ttk.Button(self.frame, text="Навчання моделі",
                   command=self.app.show_training_window).pack(pady=10, fill='x')
        ttk.Button(self.frame, text="Розпізнавання зображень",
                   command=self.app.show_recognition_window).pack(pady=10, fill='x')
        ttk.Button(self.frame, text="Вихід",
                   command=self.root.quit).pack(pady=10, fill='x')