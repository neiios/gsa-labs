from dataclasses import dataclass
from pathlib import Path
import sys
from typing import cast, Literal, TypeAlias
import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile
from scipy import signal
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QLineEdit,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.axes import Axes

WindowType = Literal["hann", "hamming"]
AudioArray: TypeAlias = NDArray[np.int16]
FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]


@dataclass
class AnalysisParams:
    start_time: float
    duration_ms: float
    n_fft: int
    window_type: WindowType


class SpectralAnalyzer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Short-time Spectrum Analysis")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize data attributes
        self.signal_data: AudioArray | None = None
        self.sample_rate: int | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Controls
        controls_layout = QHBoxLayout()

        # Create and configure widgets
        self.load_button = QPushButton("Load WAV File")
        self.load_button.clicked.connect(self.load_file)

        self.start_time_input = QLineEdit("0.0")
        self.duration_input = QLineEdit("20")
        self.padding_input = QLineEdit("4096")
        self.window_input = QLineEdit("hann")
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.clicked.connect(self.analyze_signal)

        # Add widgets to layout with labels
        widgets = [
            (self.load_button, None),
            (self.start_time_input, "Start Time (s):"),
            (self.duration_input, "Duration (ms):"),
            (self.padding_input, "FFT Size:"),
            (self.window_input, "Window:"),
            (self.analyze_button, None),
        ]

        for widget, label_text in widgets:
            if label_text:
                controls_layout.addWidget(QLabel(label_text))
            controls_layout.addWidget(widget)

        layout.addLayout(controls_layout)

        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def load_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open WAV File", "", "WAV Files (*.wav)"
        )
        if not file_path:
            return

        try:
            sample_rate, data = wavfile.read(Path(file_path))
            self.sample_rate = sample_rate

            # Ensure the data is in the correct format
            if isinstance(data, np.ndarray):
                if data.ndim > 1:
                    self.signal_data = data[:, 0].astype(np.int16)
                else:
                    self.signal_data = data.astype(np.int16)
                self.plot_signal()
        except Exception as e:
            print(f"Error loading file: {e}")

    def get_analysis_params(self) -> AnalysisParams:
        return AnalysisParams(
            start_time=float(self.start_time_input.text()),
            duration_ms=float(self.duration_input.text()),
            n_fft=int(self.padding_input.text()),
            window_type=cast(WindowType, self.window_input.text().lower()),
        )

    def plot_signal(self) -> None:
        if self.signal_data is None or self.sample_rate is None:
            return

        self.figure.clear()
        ax1: Axes = self.figure.add_subplot(211)
        time = np.arange(len(self.signal_data), dtype=np.float64) / float(
            self.sample_rate
        )
        ax1.plot(time, self.signal_data)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.set_title("Time Domain Signal")
        ax1.grid(True)
        self.canvas.draw()

    def apply_window(
        self, signal_segment: AudioArray, window_type: WindowType
    ) -> FloatArray:
        window_funcs = {
            "hann": signal.windows.hann,
            "hamming": signal.windows.hamming,
        }
        window = window_funcs[window_type](len(signal_segment))
        return signal_segment.astype(np.float64) * window

    def process_fft(
        self, fft_result: ComplexArray, n_fft: int
    ) -> tuple[ComplexArray, int]:
        if n_fft % 2 == 0:
            num_points = n_fft // 2 + 1
            result = fft_result[:num_points].copy()
            result[1:-1] *= 2
        else:
            num_points = (n_fft + 1) // 2
            result = fft_result[:num_points].copy()
            result[1:] *= 2
        return result, num_points

    def analyze_signal(self) -> None:
        if self.signal_data is None or self.sample_rate is None:
            return

        try:
            params = self.get_analysis_params()
            duration_s = params.duration_ms / 1000.0

            # Extract signal segment
            start_sample = int(params.start_time * self.sample_rate)
            segment_length = int(duration_s * self.sample_rate)
            signal_segment = self.signal_data[
                start_sample : start_sample + segment_length
            ]

            # Process signal
            windowed_segment = self.apply_window(signal_segment, params.window_type)
            fft_result = np.fft.fft(windowed_segment, n=params.n_fft)
            fft_result, num_points = self.process_fft(fft_result, params.n_fft)

            # Calculate frequency axis and magnitude spectrum
            freq = np.linspace(
                0, float(self.sample_rate) / 2, num=len(fft_result), dtype=np.float64
            )
            magnitude_db = 20 * np.log10(np.abs(fft_result) + 1e-10).astype(np.float64)

            self.plot_analysis_results(
                signal_segment,
                params.start_time,
                params.duration_ms,
                freq,
                magnitude_db,
                params.window_type,
                params.n_fft,
            )

        except ValueError as e:
            print(f"Error during analysis: {e}")

    def plot_analysis_results(
        self,
        signal_segment: AudioArray,
        start_time: float,
        duration_ms: float,
        freq: FloatArray,
        magnitude_db: FloatArray,
        window_type: str,
        n_fft: int,
    ) -> None:
        self.figure.clear()

        # Time domain plot
        ax1: Axes = self.figure.add_subplot(211)
        if self.sample_rate is not None:
            sample_rate_float = float(self.sample_rate)
            time_segment = (
                np.arange(len(signal_segment), dtype=np.float64) / sample_rate_float
                + start_time
            )
            ax1.plot(time_segment, signal_segment)
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Amplitude")
            ax1.set_title(f"Signal Segment ({duration_ms} ms)")
            ax1.grid(True)

            # Frequency domain plot
            ax2: Axes = self.figure.add_subplot(212)
            ax2.plot(freq, magnitude_db, label="Magnitude (dB)")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Magnitude (dB)")
            ax2.set_title(
                f"Amplitude Spectrum (Window: {window_type}, FFT size: {n_fft})"
            )
            ax2.grid(True)
            ax2.set_xscale("log")

            # Set y-axis limits
            max_db = float(np.max(magnitude_db))
            ax2.set_ylim(max_db - 80, max_db + 5)

            # Add frequency markers
            notable_freqs = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
            ax2.set_xticks(notable_freqs)
            ax2.set_xticklabels([f"{f}" for f in notable_freqs])

            ax2.legend()
            self.figure.tight_layout()
            self.canvas.draw()


def main() -> None:
    app = QApplication(sys.argv)
    window = SpectralAnalyzer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
