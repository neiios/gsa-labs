import sys
import wave
import numpy as np
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QDialog, QVBoxLayout,
    QLabel, QPushButton, QInputDialog
)
from PyQt6.QtCore import Qt

def get_samples_with_conversion_24(n_samples, n_channels, audio_data):
    total_samples = n_samples * n_channels
    raw_bytes = np.frombuffer(audio_data, dtype=np.uint8)
    raw_bytes = raw_bytes.reshape(-1, 3)
    shifted = np.left_shift(raw_bytes[:, 2].astype(np.int32), 16) | \
            np.left_shift(raw_bytes[:, 1].astype(np.int32), 8) | \
            raw_bytes[:, 0].astype(np.int32)
    samples = shifted
    mask = (1 << 23)
    samples = (samples ^ mask) - mask
    samples = samples.reshape(-1, n_channels)
    return samples

def select_time_scale():
    scale_options = ['milliseconds', 'seconds', 'minutes', 'hours']
    selected_time_units = 'seconds'

    class TimeScaleDialog(QDialog):
        def __init__(self):
            super().__init__()
            self.selected_time_units = None
            self.init_ui()

        def init_ui(self):
            self.setWindowTitle("Select Time Scale")
            self.setFixedWidth(300)

            layout = QVBoxLayout()
            label = QLabel("Select Time Scale:")
            layout.addWidget(label)

            for option in scale_options:
                btn = QPushButton(option.capitalize())
                btn.clicked.connect(lambda checked, opt=option: self.set_time_scale(opt))
                layout.addWidget(btn)

            layout.setSpacing(10)
            layout.setContentsMargins(20, 20, 20, 20)
            self.setLayout(layout)

        def set_time_scale(self, value):
            self.selected_time_units = value
            self.accept()

    dialog = TimeScaleDialog()
    if dialog.exec():
        return dialog.selected_time_units
    else:
        return selected_time_units

def main():
    app = QApplication(sys.argv)

    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select a WAV file",
        "",
        "WAV files (*.wav)"
    )

    if not file_path:
        print("No file selected.")
        sys.exit()

    with wave.open(file_path, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()  # in bytes
        sample_rate = wav_file.getframerate()
        n_samples = wav_file.getnframes()
        audio_data = wav_file.readframes(n_samples)

    if sample_width == 1:
        dtype = np.uint8
    elif sample_width == 2:
        dtype = np.int16
    elif sample_width == 3:
        dtype = np.int32  # 24-bit audio will be handled as 32-bit
    elif sample_width == 4:
        dtype = np.int32
    else:
        print(f"Sample width {sample_width} bytes not supported.")
        sys.exit()

    if sample_width == 3:
        # 24 bit data is a bit special
        # numpy doesn't have a 24-bit data type
        # so we need to apply some magic from stack overflow to convert it to 32-bit
        samples = get_samples_with_conversion_24(n_samples, n_channels, audio_data)
    else:
        samples = np.frombuffer(audio_data, dtype=dtype)
        samples = samples.reshape(-1, n_channels)

    duration = n_samples / sample_rate

    time_units = select_time_scale()

    if time_units == 'milliseconds':
        time_scale = 1000
    elif time_units == 'seconds':
        time_scale = 1
    elif time_units == 'minutes':
        time_scale = 1 / 60
    elif time_units == 'hours':
        time_scale = 1 / 3600
    else:
        time_units = 'seconds'
        time_scale = 1

    time_axis = np.linspace(0, duration * time_scale, num=n_samples)

    fig, axs = plt.subplots(n_channels, 1, figsize=(12, 6), sharex=True)

    if n_channels == 1:
        axs = [axs]

    for ch in range(n_channels):
        axs[ch].plot(time_axis, samples[:, ch], label=f'Channel {ch + 1}', alpha=0.7)
        axs[ch].set_ylabel('Amplitude')
        axs[ch].legend(loc='upper right')
        axs[ch].grid(True, which='both', linestyle='--', alpha=0.5)

    axs[-1].set_xlabel(f'Time ({time_units})')

    fig.suptitle(
        f'File: {file_path}\n'
        f'Channels: {n_channels}, Sample Rate: {sample_rate} Hz, '
        f'Quantization Depth: {sample_width * 8} bits',
        fontsize=10
    )

    plt.tight_layout()

    max_time = duration * time_scale
    marker_time, ok = QInputDialog.getDouble(
        None,
        "Input Marker Time",
        f"Enter a time to place a marker on the plot (in {time_units}, 0 to {max_time:.2f}):",
        decimals=2,
        min=0,
        max=max_time
    )

    if not ok:
        print("No marker time entered.")
        sys.exit()

    if 0 <= marker_time <= max_time:
        for ax in axs:
            ax.axvline(x=marker_time, color='r', linestyle='--', label='Marker')
            ax.legend(loc="upper right")
    else:
        print("Marker time is out of range.")

    plt.savefig('plot.png')
    plt.show()

if __name__ == '__main__':
    main()
