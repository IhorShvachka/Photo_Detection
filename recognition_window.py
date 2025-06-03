import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk

class RecognitionWindow:
    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.frame = ttk.Frame(root)
        self.create_widgets()
        self.image_tk = None

    def create_widgets(self):
        self.image_frame = ttk.LabelFrame(self.frame, text="Зображення", padding=10)
        self.image_frame.pack(pady=10, padx=10, fill='both', expand=True)

        self.canvas = tk.Canvas(self.image_frame)
        self.scroll_y = ttk.Scrollbar(self.image_frame, orient="vertical", command=self.canvas.yview)
        self.scroll_x = ttk.Scrollbar(self.image_frame, orient="horizontal", command=self.canvas.xview)

        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)

        self.scroll_y.pack(side="right", fill="y")
        self.scroll_x.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        result_frame = ttk.LabelFrame(self.frame, text="Результат", padding=10)
        result_frame.pack(pady=10, padx=10, fill='x')

        self.result_label = ttk.Label(result_frame, text="Оберіть зображення для аналізу",
                                      font=('Arial', 12))
        self.result_label.pack()

        button_frame = ttk.Frame(self.frame)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Вибрати зображення",
                   command=self.select_image).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Повернутись",
                   command=self.app.show_main_menu).pack(side='left', padx=5)

    def select_image(self):
        if not hasattr(self.app, 'model_trained') or not self.app.model_trained:
            messagebox.showwarning("Попередження", "Спочатку навчіть або завантажте модель!")
            return

        filepath = filedialog.askopenfilename(
            filetypes=[("Зображення", "*.png *.jpg *.jpeg *.bmp")])

        if filepath:
            try:
                image = cv2.imread(filepath)
                if image is None:
                    raise ValueError("Не вдалося завантажити зображення")

                self.display_image(image)

                prediction = self.app.model.predict(image)
                result = "Фейкове" if prediction > 0.5 else "Справжнє"
                confidence = max(prediction, 1 - prediction) * 100

                self.result_label.config(
                    text=f"Результат: {result}\nВпевненість: {confidence:.2f}%")
            except Exception as e:
                messagebox.showerror("Помилка", f"Помилка аналізу: {str(e)}")

    def display_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        self.original_image = image

        max_width = 800
        max_height = 600
        width, height = pil_image.size

        if width > max_width or height > max_height:
            ratio = min(max_width / width, max_height / height)
            new_size = (int(width * ratio), int(height * ratio))
            pil_image = pil_image.resize(new_size, Image.LANCZOS)

        self.image_tk = ImageTk.PhotoImage(pil_image)

        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, pil_image.width, pil_image.height))
        self.canvas.create_image(0, 0, anchor='nw', image=self.image_tk)

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))