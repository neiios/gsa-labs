import sys
import numpy as np
import matplotlib
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QLineEdit,
    QMessageBox,
)
from PySide6.QtCore import QUrl, Qt, QThread, Signal, QObject
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtGui import QMovie

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from scipy.io import wavfile

import tempfile
import os


class AudioData:
    def __init__(self, file_path: str):
        self.sample_rate, data = wavfile.read(file_path)

        if data.ndim == 1:
            self.n_channels = 1
            self.samples = data
        else:
            raise ValueError(
                f"Only mono audio is supported. Audio has {data.ndim} channels"
            )

        self.n_samples = self.samples.shape[0]
        self.sample_width = self.samples.dtype.itemsize

        if self.samples.dtype == np.int32:
            self.samples = (self.samples >> 16).astype(np.int16)
        elif self.samples.dtype == np.int8:
            self.samples = self.samples.astype(np.int16) << 8
        elif self.samples.dtype == np.float32 or self.samples.dtype == np.float64:
            self.samples = np.clip(self.samples, -1.0, 1.0)
            self.samples = (self.samples * 32767).astype(np.int16)
        elif self.samples.dtype != np.int16:
            raise ValueError(f"Unsupported audio format: {self.samples.dtype}")

    def get_time_axis(self):
        return np.linspace(0, self.n_samples / self.sample_rate, num=self.n_samples)

    @property
    def duration(self):
        return self.n_samples / self.sample_rate


class Worker(QObject):
    finished = Signal()
    result = Signal(object)

    def __init__(self, audio_data, delay_ms):
        super().__init__()
        self.audio_data = audio_data
        self.delay_ms = delay_ms

    def process(self):
        delay_samples = int(self.audio_data.sample_rate * self.delay_ms / 1000)
        mono_samples = self.audio_data.samples

        zero_padding = np.zeros(delay_samples, dtype=mono_samples.dtype)
        left_channel = np.concatenate((mono_samples, zero_padding))
        right_channel = np.concatenate((zero_padding, mono_samples))

        min_length = min(len(left_channel), len(right_channel))
        left_channel = left_channel[:min_length]
        right_channel = right_channel[:min_length]

        processed_data = np.vstack((left_channel, right_channel)).T

        self.result.emit(processed_data)
        self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("Pseudo Stereo Effect")
        self.resize(1000, 800)

        self.current_file = None
        self.processed_file = None

        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.5)

        self.main_widget = QWidget()
        self.layout = QVBoxLayout(self.main_widget)
        self.setCentralWidget(self.main_widget)

        self.button_layout = QHBoxLayout()

        self.select_button = QPushButton("Select WAV File")
        self.select_button.clicked.connect(self.select_file)
        self.button_layout.addWidget(self.select_button)

        self.play_original_button = QPushButton("Play Original")
        self.play_original_button.clicked.connect(self.play_original_audio)
        self.play_original_button.setEnabled(False)
        self.button_layout.addWidget(self.play_original_button)

        self.play_processed_button = QPushButton("Play Processed")
        self.play_processed_button.clicked.connect(self.play_processed_audio)
        self.play_processed_button.setEnabled(False)
        self.button_layout.addWidget(self.play_processed_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_audio)
        self.stop_button.setEnabled(False)
        self.button_layout.addWidget(self.stop_button)

        self.layout.addLayout(self.button_layout)

        self.delay_layout = QHBoxLayout()
        self.delay_label = QLabel("Delay (ms):")
        self.delay_input = QLineEdit()
        self.delay_input.setText("3")
        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process_audio)

        self.delay_layout.addWidget(self.delay_label)
        self.delay_layout.addWidget(self.delay_input)
        self.delay_layout.addWidget(self.process_button)
        self.layout.addLayout(self.delay_layout)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.spinner_label = QLabel()
        self.spinner_movie = QMovie("spinner.gif")
        self.spinner_label.setMovie(self.spinner_movie)
        self.spinner_label.setAlignment(Qt.AlignCenter)

        self.spinner_label.hide()
        self.layout.addWidget(self.spinner_label)

        self.player.playbackStateChanged.connect(self.playback_state_changed)

        self.audio_data = None
        self.processed_data = None

        self.thread = None

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select a WAV file", "", "WAV files (*.wav)"
        )
        if file_path:
            self.current_file = file_path
            self.play_original_button.setEnabled(True)
            self.load_audio_data(file_path)
            self.process_audio()

    def load_audio_data(self, file_path):
        self.audio_data = AudioData(file_path)
        if self.audio_data.n_channels != 1:
            QMessageBox.warning(self, "Error", "Please select a mono audio file.")
            self.audio_data = None
            return
        self.plot_waveform(self.audio_data)

    def process_audio(self):
        if self.audio_data is None:
            return

        try:
            delay_ms = float(self.delay_input.text())
            if not (1 <= delay_ms <= 1000):
                raise ValueError
        except ValueError:
            QMessageBox.warning(
                self, "Error", "Please enter a valid delay between 1 and 1000 ms."
            )
            return

        self.canvas.hide()
        self.spinner_label.show()
        self.spinner_movie.start()

        self.thread = QThread()
        self.worker = Worker(self.audio_data, delay_ms)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.process)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.result.connect(self.processing_finished)

        self.thread.start()

    def processing_finished(self, processed_data):
        self.processed_data = processed_data

        temp_dir = tempfile.gettempdir()
        self.processed_file = os.path.join(temp_dir, "processed_audio.wav")
        wavfile.write(
            self.processed_file, self.audio_data.sample_rate, self.processed_data
        )

        self.play_processed_button.setEnabled(True)
        self.plot_waveform_processed()

        self.spinner_movie.stop()
        self.spinner_label.hide()
        self.canvas.show()

    def play_original_audio(self):
        if self.current_file:
            self.stop_audio()
            self.player.setSource(QUrl.fromLocalFile(self.current_file))
            self.player.play()
            self.stop_button.setEnabled(True)

    def play_processed_audio(self):
        if self.processed_file:
            self.stop_audio()
            self.player.setSource(QUrl.fromLocalFile(self.processed_file))
            self.player.play()
            self.stop_button.setEnabled(True)

    def stop_audio(self):
        self.player.stop()
        self.stop_button.setEnabled(False)

    def playback_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self.stop_button.setEnabled(False)
        elif state == QMediaPlayer.PlaybackState.PlayingState:
            self.stop_button.setEnabled(True)

    def plot_waveform(self, audio_data):
        time_axis = audio_data.get_time_axis()
        samples = audio_data.samples

        self.figure.clear()

        ax1 = self.figure.add_subplot(211)
        ax1.plot(
            time_axis, samples, label="Original Mono Audio", color="blue", alpha=0.7
        )
        ax1.set_ylabel("Amplitude")
        ax1.legend(loc="upper right")
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)
        ax1.set_title("Original Audio Waveform")

        self.canvas.draw()

    def plot_waveform_processed(self):
        if self.processed_data is None:
            return

        time_axis = np.linspace(
            0,
            len(self.processed_data) / self.audio_data.sample_rate,
            num=len(self.processed_data),
        )
        samples = self.processed_data

        # Remove the old processed waveform
        self.figure.subplots_adjust(hspace=0.5)
        if len(self.figure.axes) > 1:
            self.figure.axes[1].remove()

        ax2 = self.figure.add_subplot(212)
        ax2.plot(time_axis, samples[:, 0], label="Left Channel", color="red", alpha=0.7)
        ax2.plot(
            time_axis, samples[:, 1], label="Right Channel", color="green", alpha=0.7
        )
        ax2.set_ylabel("Amplitude")
        ax2.set_xlabel("Time [s]")
        ax2.legend(loc="upper right")
        ax2.grid(True, which="both", linestyle="--", alpha=0.5)
        ax2.set_title("Processed Audio Waveform")

        self.canvas.draw()

    def closeEvent(self, event):
        self.player.stop()
        if self.processed_file and os.path.exists(self.processed_file):
            os.remove(self.processed_file)
        super().closeEvent(event)


if __name__ == "__main__":
    matplotlib.use("QtAgg")
    app = QApplication(sys.argv)

    w = MainWindow()
    w.show()

    sys.exit(app.exec())
