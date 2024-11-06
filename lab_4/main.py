import sys
import numpy as np
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
    QComboBox,
)
from PySide6.QtCore import QThread, Signal, QObject
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.io import wavfile
from scipy.signal import get_window
from scipy.fft import fft
import matplotlib
from matplotlib.ticker import ScalarFormatter


class Worker(QObject):
    finished = Signal()
    result = Signal(object, object)  # fft_data, freq_axis

    def __init__(self, data, sample_rate):
        super().__init__()
        self.data = data
        self.sample_rate = sample_rate

    def process(self):
        N = len(self.data)
        # Perform FFT
        yf = fft(self.data)
        # a) Calculate complex modulus
        yf = np.abs(yf)

        # b) Cut off the results
        if N % 2 == 0:
            cutoff = N // 2 + 1
        else:
            cutoff = (N + 1) // 2
        yf = yf[:cutoff]

        # c) Apply scaling factor
        if N % 2 == 0:
            yf[1:-1] *= 2
        else:
            yf[1:] *= 2

        # Normalize by N
        yf = yf / N

        # d) Generate frequency values
        xf = np.linspace(0.0, self.sample_rate / 2, cutoff)

        self.result.emit(yf, xf)
        self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fourier Lab 4")
        self.resize(1000, 800)

        self.audio_data = None
        self.sample_rate = None
        self.time_axis = None
        self.selected_data = None
        self.selected_time_axis = None
        self.windowed_data = None
        self.fft_data = None
        self.freq_axis = None

        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.setCentralWidget(self.main_widget)

        self.file_layout = QHBoxLayout()
        self.select_button = QPushButton("Select WAV File")
        self.select_button.clicked.connect(self.select_file)
        self.file_layout.addWidget(self.select_button)
        self.main_layout.addLayout(self.file_layout)

        self.selection_layout = QHBoxLayout()
        self.start_time_label = QLabel("Start Time (s):")
        self.start_time_input = QLineEdit()
        self.duration_label = QLabel("Duration (s):")
        self.duration_input = QLineEdit()
        self.window_label = QLabel("Window Function:")
        self.window_combo = QComboBox()
        self.window_combo.addItems(["hamming", "hann"])
        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process_audio)
        self.process_button.setEnabled(False)

        self.selection_layout.addWidget(self.start_time_label)
        self.selection_layout.addWidget(self.start_time_input)
        self.selection_layout.addWidget(self.duration_label)
        self.selection_layout.addWidget(self.duration_input)
        self.selection_layout.addWidget(self.window_label)
        self.selection_layout.addWidget(self.window_combo)
        self.selection_layout.addWidget(self.process_button)
        self.main_layout.addLayout(self.selection_layout)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.main_layout.addWidget(self.canvas)

        self.processing_thread = None

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a WAV file", "", "WAV files (*.wav)")
        if file_path:
            self.load_audio_data(file_path)
            self.process_button.setEnabled(True)

    def load_audio_data(self, file_path):
        try:
            self.sample_rate, data = wavfile.read(file_path)
            if data.ndim == 2:
                data = data[:, 0]
            self.audio_data = data

            self.time_axis = np.linspace(0, len(self.audio_data) / self.sample_rate, num=len(self.audio_data))
            self.plot_waveform()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not load audio file: {e}")

    def plot_waveform(self):
        self.figure.clear()
        self.ax1 = self.figure.add_subplot(311)
        self.ax1.plot(self.time_axis, self.audio_data, color="blue", alpha=0.7)
        self.ax1.set_title("Entire Signal")
        self.ax1.set_xlabel("Time [s]")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.grid(True)
        self.canvas.draw()

    def process_audio(self):
        if self.audio_data is None:
            return

        start_time = float(self.start_time_input.text())
        duration = float(self.duration_input.text())
        start_sample = int(start_time * self.sample_rate)
        end_sample = start_sample + int(duration * self.sample_rate)
        if end_sample > len(self.audio_data):
            end_sample = len(self.audio_data)
        self.selected_data = self.audio_data[start_sample:end_sample]
        self.selected_time_axis = self.time_axis[start_sample:end_sample]

        if len(self.selected_data) == 0:
            QMessageBox.warning(self, "Error", "Selected duration is too short.")
            return

        window_type = self.window_combo.currentText().lower()
        window = get_window(window_type, len(self.selected_data))
        self.windowed_data = self.selected_data * window

        self.figure.clear()
        self.start_processing_thread()

    def start_processing_thread(self):
        self.processing_thread = QThread()
        self.worker = Worker(self.windowed_data, self.sample_rate)
        self.worker.moveToThread(self.processing_thread)
        self.processing_thread.started.connect(self.worker.process)
        self.worker.finished.connect(self.processing_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.processing_thread.finished.connect(self.processing_thread.deleteLater)
        self.worker.result.connect(self.processing_finished)
        self.processing_thread.start()

    def processing_finished(self, fft_data, freq_axis):
        self.fft_data = fft_data
        self.freq_axis = freq_axis
        self.plot_selected_waveform()
        self.plot_spectrum()
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_selected_waveform(self):
        self.ax1 = self.figure.add_subplot(311)
        self.ax1.plot(self.time_axis, self.audio_data, color="blue", alpha=0.7)
        self.ax1.set_title("Entire Signal")
        self.ax1.set_xlabel("Time [s]")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.grid(True)

        self.ax2 = self.figure.add_subplot(312)
        self.ax2.plot(self.selected_time_axis, self.selected_data, color="green", alpha=0.7)
        self.ax2.set_title("Selected Signal Portion")
        self.ax2.set_xlabel("Time [s]")
        self.ax2.set_ylabel("Amplitude")
        self.ax2.grid(True)

    def plot_spectrum(self):
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        self.ax3 = self.figure.add_subplot(313)
        self.ax3.plot(self.freq_axis, self.fft_data, color="red", alpha=0.7)
        self.ax3.set_title("Fourier Transform")
        self.ax3.set_xlabel("Frequency [Hz]")
        self.ax3.set_ylabel("Magnitude")
        self.ax3.set_xscale("log")
        self.ax3.xaxis.set_major_formatter(formatter)
        self.ax3.grid(True)

    def closeEvent(self, event):
        super().closeEvent(event)


if __name__ == "__main__":
    matplotlib.use("QtAgg")
    app = QApplication(sys.argv)

    w = MainWindow()
    w.show()

    sys.exit(app.exec())
