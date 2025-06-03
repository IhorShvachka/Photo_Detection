from Photo_Detection.main_menu import MainMenu
from Photo_Detection.training_window import TrainingWindow
from Photo_Detection.recognition_window import RecognitionWindow
from Photo_Detection.cnn_model import CNNModel

class App:
    def __init__(self, root):
        self.root = root
        self.model = CNNModel()
        self.model_trained = self.model.load_model()

        self.main_menu = MainMenu(root, self)
        self.training_window = TrainingWindow(root, self)
        self.recognition_window = RecognitionWindow(root, self)

        self.show_main_menu()

    def show_main_menu(self):
        self.hide_all_windows()
        self.main_menu.frame.pack(fill='both', expand=True)
        self.root.title("Детектор фейкових зображень - Головне меню")

    def show_training_window(self):
        self.hide_all_windows()
        self.training_window.frame.pack(fill='both', expand=True, padx=20, pady=20)
        self.root.title("Детектор фейкових зображень - Навчання моделі")

    def show_recognition_window(self):
        self.hide_all_windows()
        self.recognition_window.frame.pack(fill='both', expand=True, padx=20, pady=20)
        self.root.title("Детектор фейкових зображень - Розпізнавання")

    def hide_all_windows(self):
        for window in [self.main_menu, self.training_window, self.recognition_window]:
            window.frame.pack_forget()