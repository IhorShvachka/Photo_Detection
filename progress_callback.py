from tensorflow.keras.callbacks import Callback

class ProgressCallback(Callback):
    def __init__(self, progress_queue, total_epochs):
        super().__init__()
        self.progress_queue = progress_queue
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.total_epochs * 100
        self.progress_queue.put(('epoch_progress', epoch + 1, logs, progress))