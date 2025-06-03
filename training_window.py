import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import os
import threading
import queue
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from Photo_Detection.progress_callback import ProgressCallback

class TrainingWindow:
    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.frame = ttk.Frame(root)
        self.progress_queue = queue.Queue()
        self.create_widgets()
        self.check_progress_queue()
        self.start_time = None
        self.current_epoch = 0

    def create_widgets(self):
        dir_frame = ttk.LabelFrame(self.frame, text="Папки з даними", padding=10)
        dir_frame.pack(pady=10, fill='x', padx=10)

        ttk.Label(dir_frame, text="Папка REAL:").grid(row=0, column=0, sticky='w')
        self.real_dir = ttk.Entry(dir_frame, width=40)
        self.real_dir.grid(row=0, column=1, padx=5)
        ttk.Button(dir_frame, text="Огляд...",
                   command=lambda: self.browse_directory(self.real_dir)).grid(row=0, column=2)

        ttk.Label(dir_frame, text="Папка FAKE:").grid(row=1, column=0, sticky='w')
        self.fake_dir = ttk.Entry(dir_frame, width=40)
        self.fake_dir.grid(row=1, column=1, padx=5)
        ttk.Button(dir_frame, text="Огляд...",
                   command=lambda: self.browse_directory(self.fake_dir)).grid(row=1, column=2)

        control_frame = ttk.LabelFrame(self.frame, text="Налаштування навчання", padding=10)
        control_frame.pack(pady=10, fill='x', padx=10)

        ttk.Label(control_frame, text="Епохи:").grid(row=0, column=0, padx=5)
        self.epochs = ttk.Combobox(control_frame, values=[20, 30, 50, 100], width=5)
        self.epochs.current(1)
        self.epochs.grid(row=0, column=1, padx=5, sticky='w')

        ttk.Label(control_frame, text="Batch size:").grid(row=0, column=2, padx=5)
        self.batch_size = ttk.Combobox(control_frame, values=[16, 32, 64], width=5)
        self.batch_size.current(1)
        self.batch_size.grid(row=0, column=3, padx=5, sticky='w')

        ttk.Label(control_frame, text="Макс. зображень (на клас):").grid(row=1, column=0, padx=5)
        self.max_images = ttk.Entry(control_frame, width=10)
        self.max_images.insert(0, "0")
        self.max_images.grid(row=1, column=1, padx=5, sticky='w')

        self.train_progress_frame = ttk.LabelFrame(self.frame, text="Прогрес навчання", padding=10)
        self.train_progress_frame.pack(pady=10, fill='x', padx=10)

        self.epoch_label = ttk.Label(self.train_progress_frame, text="Епоха: 0/0")
        self.epoch_label.pack(anchor='w')

        self.accuracy_label = ttk.Label(self.train_progress_frame, text="Точність: -")
        self.accuracy_label.pack(anchor='w')

        self.loss_label = ttk.Label(self.train_progress_frame, text="Втрати: -")
        self.loss_label.pack(anchor='w')

        self.train_progress = ttk.Progressbar(self.train_progress_frame, orient='horizontal', mode='determinate')
        self.train_progress.pack(fill='x', pady=5)

        self.time_remaining_label = ttk.Label(self.train_progress_frame, text="Залишилось: -")
        self.time_remaining_label.pack(anchor='w')

        button_frame = ttk.Frame(self.frame)
        button_frame.pack(pady=10)

        self.load_btn = ttk.Button(button_frame, text="Завантажити дані",
                                   command=self.start_loading_data)
        self.load_btn.pack(side='left', padx=5)

        self.train_btn = ttk.Button(button_frame, text="Почати навчання",
                                    command=self.start_training)
        self.train_btn.pack(side='left', padx=5)
        self.train_btn['state'] = 'disabled'

        ttk.Button(button_frame, text="Завантажити модель",
                   command=self.load_model).pack(side='left', padx=5)

        ttk.Button(button_frame, text="Повернутись",
                   command=self.app.show_main_menu).pack(side='left', padx=5)

    def browse_directory(self, entry_widget):
        directory = filedialog.askdirectory()
        if directory:
            entry_widget.delete(0, 'end')
            entry_widget.insert(0, directory)

    def load_model(self):
        filepath = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5")])
        if filepath:
            if self.app.model.load_model(filepath):
                self.app.model_trained = True
                messagebox.showinfo("Успіх", "Модель успішно завантажена!")
            else:
                messagebox.showerror("Помилка", "Не вдалося завантажити модель")

    def start_loading_data(self):
        real_folder = self.real_dir.get()
        fake_folder = self.fake_dir.get()

        if not real_folder or not fake_folder:
            messagebox.showwarning("Попередження", "Виберіть обидві папки з даними!")
            return

        if not os.path.isdir(real_folder) or not os.path.isdir(fake_folder):
            messagebox.showerror("Помилка", "Одна з папок не існує!")
            return

        try:
            real_files = [f for f in os.listdir(real_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            fake_files = [f for f in os.listdir(fake_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            max_images = int(self.max_images.get())
            if max_images > 0:
                real_files = real_files[:max_images]
                fake_files = fake_files[:max_images]

            total_files = len(real_files) + len(fake_files)
            if total_files == 0:
                messagebox.showwarning("Попередження", "Не знайдено зображень у вказаних папках!")
                return

            estimated_memory = total_files * self.app.model.input_shape[0] * self.app.model.input_shape[1] * 3 * 4 / (
                    1024 ** 2)
            if estimated_memory > 1000:
                if not messagebox.askyesno("Попередження",
                                           f"Завантаження {total_files} зображень може вимагати {estimated_memory:.1f}MB пам'яті. Продовжити?"):
                    return

            self.load_btn['state'] = 'disabled'
            self.train_btn['state'] = 'disabled'
            self.status_label = ttk.Label(self.frame, text="Завантаження даних...")
            self.status_label.pack(pady=5)
            self.root.update()

            threading.Thread(
                target=self.load_data_thread,
                args=(real_folder, fake_folder, max_images),
                daemon=True
            ).start()
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при перевірці файлів: {str(e)}")

    def load_data_thread(self, real_folder, fake_folder, max_images=0):
        try:
            real_files = [f for f in os.listdir(real_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            fake_files = [f for f in os.listdir(fake_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if max_images > 0:
                real_files = real_files[:max_images]
                fake_files = fake_files[:max_images]

            total_files = len(real_files) + len(fake_files)
            if total_files == 0:
                raise ValueError("Не знайдено зображень у вказаних папках")

            X = np.zeros((total_files, *self.app.model.input_shape), dtype=np.float32)
            y = np.zeros((total_files, 1), dtype=np.float32)

            for i, filename in enumerate(real_files):
                img_path = os.path.join(real_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    X[i] = cv2.resize(img, (self.app.model.input_shape[0], self.app.model.input_shape[1]))
                    y[i] = 0

                progress = int(((i + 1) / len(real_files)) * 50)
                self.progress_queue.put(('load_progress', progress, f"Завантаження REAL: {i + 1}/{len(real_files)}"))

            for i, filename in enumerate(fake_files):
                img_path = os.path.join(fake_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    X[len(real_files) + i] = cv2.resize(img,
                                                        (self.app.model.input_shape[0], self.app.model.input_shape[1]))
                    y[len(real_files) + i] = 1

                progress = 50 + int(((i + 1) / len(fake_files)) * 50)
                self.progress_queue.put(('load_progress', progress, f"Завантаження FAKE: {i + 1}/{len(fake_files)}"))

            X = X / 255.0

            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            self.app.X_train = X_train
            self.app.X_val = X_val
            self.app.y_train = y_train
            self.app.y_val = y_val
            self.progress_queue.put(('load_complete', len(real_files), len(fake_files)))

        except MemoryError:
            self.progress_queue.put(('load_error',
                                     "Недостатньо пам'яті для завантаження всіх зображень. Спробуйте зменшити кількість зображень."))
        except Exception as e:
            self.progress_queue.put(('load_error', str(e)))

    def check_progress_queue(self):
        try:
            while True:
                msg_type, *args = self.progress_queue.get_nowait()

                if msg_type == 'load_progress':
                    progress, message = args
                    if hasattr(self, 'status_label'):
                        self.status_label.config(text=message)

                elif msg_type == 'load_complete':
                    real_count, fake_count = args
                    if hasattr(self, 'status_label'):
                        self.status_label.pack_forget()
                    self.load_btn['state'] = 'normal'
                    self.train_btn['state'] = 'normal'
                    messagebox.showinfo("Успіх",
                                        f"Дані успішно завантажено!\nРеальні зображення: {real_count}\nФейкові зображення: {fake_count}")

                elif msg_type == 'load_error':
                    error_msg = args[0]
                    if hasattr(self, 'status_label'):
                        self.status_label.pack_forget()
                    self.load_btn['state'] = 'normal'
                    messagebox.showerror("Помилка", f"Не вдалося завантажити дані: {error_msg}")

                elif msg_type == 'epoch_progress':
                    epoch, logs, progress = args
                    self.update_training_progress(epoch, logs)

                elif msg_type == 'training_complete':
                    history = args[0]
                    self.training_complete(history)

                elif msg_type == 'training_error':
                    error_msg = args[0]
                    self.training_error(error_msg)

                self.root.update()

        except queue.Empty:
            pass

        self.root.after(100, self.check_progress_queue)

    def start_training(self):
        if not hasattr(self.app, 'X_train'):
            messagebox.showwarning("Попередження", "Спочатку завантажте дані!")
            return

        epochs = int(self.epochs.get())
        batch_size = int(self.batch_size.get())

        self.train_btn['state'] = 'disabled'
        self.load_btn['state'] = 'disabled'
        self.train_progress['value'] = 0
        self.train_progress['maximum'] = epochs
        self.epoch_label.config(text=f"Епоха: 0/{epochs}")
        self.accuracy_label.config(text="Точність: -")
        self.loss_label.config(text="Втрати: -")
        self.time_remaining_label.config(text="Залишилось: -")
        self.start_time = time.time()
        self.current_epoch = 0
        self.root.update()

        progress_callback = ProgressCallback(self.progress_queue, epochs)

        threading.Thread(
            target=self.train_model_thread,
            args=(epochs, batch_size, progress_callback),
            daemon=True
        ).start()

    def train_model_thread(self, epochs, batch_size, progress_callback):
        try:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                min_delta=0.001,
                restore_best_weights=True,
                verbose=1
            )

            model_checkpoint = ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            )

            history = self.app.model.train(
                self.app.X_train, self.app.y_train,
                self.app.X_val, self.app.y_val,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[
                    progress_callback,
                    early_stopping,
                    model_checkpoint
                ]
            )
            self.progress_queue.put(('training_complete', history))
        except Exception as e:
            self.progress_queue.put(('training_error', str(e)))

    def update_training_progress(self, epoch, logs):
        self.current_epoch = epoch
        epochs_total = int(self.epochs.get())

        self.epoch_label.config(text=f"Епоха: {epoch}/{epochs_total}")
        self.train_progress['value'] = epoch

        if logs:
            acc = logs.get('accuracy', 0)
            val_acc = logs.get('val_accuracy', 0)
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)

            self.accuracy_label.config(text=f"Точність: {acc:.4f} (Val: {val_acc:.4f})")
            self.loss_label.config(text=f"Втрати: {loss:.4f} (Val: {val_loss:.4f})")

        if epoch > 1 and self.start_time:
            elapsed = time.time() - self.start_time
            time_per_epoch = elapsed / epoch
            remaining = (epochs_total - epoch) * time_per_epoch
            mins, secs = divmod(remaining, 60)
            self.time_remaining_label.config(text=f"Залишилось: {int(mins)} хв {int(secs)} сек")

    def training_complete(self, history):
        self.train_btn['state'] = 'normal'
        self.load_btn['state'] = 'normal'
        self.app.model_trained = True

        if not self.app.model.save_model():
            messagebox.showerror("Помилка", "Не вдалося зберегти модель")
            return

        epochs_total = int(self.epochs.get())
        self.epoch_label.config(text=f"Навчання завершено! Епох: {epochs_total}/{epochs_total}")
        self.time_remaining_label.config(text="Готово!")

        self.plot_training_results(history)
        messagebox.showinfo("Успіх", "Модель успішно навчено та збережено!")

    def plot_training_results(self, history):
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.plot(history.history['accuracy'], label='Тренувальна')
        plt.plot(history.history['val_accuracy'], label='Валідаційна')
        plt.title('Динаміка точності')
        plt.ylabel('Точність')
        plt.xlabel('Епоха')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(history.history['loss'], label='Тренувальні')
        plt.plot(history.history['val_loss'], label='Валідаційні')
        plt.title('Динаміка втрат')
        plt.ylabel('Втрати')
        plt.xlabel('Епоха')
        plt.legend()

        plt.subplot(2, 2, 3)
        y_pred = (self.app.model.model.predict(self.app.X_val) > 0.5).astype(int)
        y_true = self.app.y_val

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Реальні', 'Фейкові'],
                    yticklabels=['Реальні', 'Фейкові'])
        plt.title('Матриця сплутаності')
        plt.ylabel('Справжні')
        plt.xlabel('Прогнозовані')

        plt.subplot(2, 2, 4)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        metrics = ['Точність', 'Precision', 'Recall', 'F1-score']
        values = [accuracy, precision, recall, f1]

        plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
        plt.title('Метрики якості')
        plt.ylim(0, 1)

        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center')

        plt.tight_layout()
        plt.show()

    def training_error(self, error_msg):
        self.train_btn['state'] = 'normal'
        self.load_btn['state'] = 'normal'

        self.epoch_label.config(text="Навчання перервано через помилку")
        self.time_remaining_label.config(text="-")

        messagebox.showerror("Помилка навчання", f"Сталася помилка: {error_msg}")