import sys
import wave
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

def select_frame_length() -> int:
    frame_length_options = ['10', '20', '30']
    selected_frame_length = '20'  # default

    class FrameLengthDialog(QDialog):
        def __init__(self):
            super().__init__()
            self.selected_frame_length = 20
            self.init_ui()

        def init_ui(self):
            self.setWindowTitle("Select Frame Length (ms)")
            self.setFixedWidth(300)

            layout = QVBoxLayout()
            label = QLabel("Select Frame Length (ms):")
            layout.addWidget(label)

            for option in frame_length_options:
                btn = QPushButton(option + " ms")
                btn.clicked.connect(lambda checked, opt=option: self.set_frame_length(opt))
                layout.addWidget(btn)

            layout.setSpacing(10)
            layout.setContentsMargins(20, 20, 20, 20)
            self.setLayout(layout)

        def set_frame_length(self, value):
            self.selected_frame_length = int(value)
            self.accept()

    dialog = FrameLengthDialog()
    if dialog.exec():
        return dialog.selected_frame_length
    else:
        return int(selected_frame_length)

def calculate_energy(samples, frame_length, frame_shift, sample_rate):
    n_samples = len(samples)
    num_frames = int(np.ceil((n_samples - frame_length) / frame_shift)) + 1
    energies = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_length
        frame = samples[start:end]
        frame = frame * np.hamming(len(frame))  # Apply windowing if desired
        energies[i] = np.sum(frame ** 2)
    time_axis = (np.arange(num_frames) * frame_shift) / sample_rate
    return energies, time_axis

def calculate_zcr(samples, frame_length, frame_shift, sample_rate):
    n_samples = len(samples)
    num_frames = int(np.ceil((n_samples - frame_length) / frame_shift)) + 1
    zcrs = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_length
        frame = samples[start:end]
        signs = np.sign(frame)
        signs[signs == 0] = 1  # Treat zero as positive
        zcr = np.sum(np.abs(np.diff(signs))) / (2 * len(frame))
        zcrs[i] = zcr
    time_axis = (np.arange(num_frames) * frame_shift) / sample_rate
    return zcrs, time_axis

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
        samples = get_samples_with_conversion_24(n_samples, n_channels, audio_data)
    else:
        samples = np.frombuffer(audio_data, dtype=dtype)
        samples = samples.reshape(-1, n_channels)

    duration = n_samples / sample_rate

    # Time units are in seconds
    time_units = 'seconds'
    time_scale = 1

    frame_length_ms = select_frame_length()
    frame_length = int(frame_length_ms * sample_rate / 1000)
    frame_shift = frame_length // 2  # 50% overlap

    # Use only the first channel for energy and ZCR calculations
    mono_samples = samples[:, 0]

    energies, energy_time_axis = calculate_energy(mono_samples, frame_length, frame_shift, sample_rate)
    zcrs, zcr_time_axis = calculate_zcr(mono_samples, frame_length, frame_shift, sample_rate)

    # Prepare to plot
    num_plots = n_channels + 2
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, num_plots * 2), sharex=True)

    time_axis = np.linspace(0, duration, num=n_samples)

    # Plot time-domain waveform
    for ch in range(n_channels):
        axs[ch].plot(time_axis, samples[:, ch], label=f'Channel {ch + 1}', alpha=0.7)
        axs[ch].set_ylabel('Amplitude')
        axs[ch].legend(loc='upper right')
        axs[ch].grid(True, which='both', linestyle='--', alpha=0.5)

    # Plot energy
    axs[n_channels].plot(energy_time_axis, energies, label='Energy', color='g')
    axs[n_channels].set_ylabel('Energy')
    axs[n_channels].legend(loc='upper right')
    axs[n_channels].grid(True, which='both', linestyle='--', alpha=0.5)

    # Plot ZCR
    axs[n_channels + 1].plot(zcr_time_axis, zcrs, label='ZCR', color='m')
    axs[n_channels + 1].set_ylabel('ZCR')
    axs[n_channels + 1].legend(loc='upper right')
    axs[n_channels + 1].grid(True, which='both', linestyle='--', alpha=0.5)

    axs[-1].set_xlabel('Time (seconds)')

    fig.suptitle(
        f'File: {Path(file_path).name}\n'
        f'Channels: {n_channels}, Sample Rate: {sample_rate} Hz, '
        f'Quantization Depth: {sample_width * 8} bits\n'
        f'Frame Length: {frame_length_ms} ms',
        fontsize=10
    )

    plt.tight_layout()

    # Prompt for energy threshold
    energy_threshold, ok = QInputDialog.getDouble(
        None,
        "Energy Threshold",
        "Enter energy threshold value:",
        value=np.mean(energies),
        min=0,
        max=np.max(energies)
    )

    if ok:
        # Plot energy threshold line
        axs[n_channels].axhline(y=energy_threshold, color='r', linestyle='--', label='Threshold')
        axs[n_channels].legend(loc='upper right')

        # Detect segments where energy exceeds threshold
        signal_segments = energies > energy_threshold

        # Highlight detected segments in waveform plot
        for ch in range(n_channels):
            for i in range(len(signal_segments)):
                if signal_segments[i]:
                    start_time = energy_time_axis[i]
                    end_time = start_time + (frame_length / sample_rate)
                    axs[ch].axvspan(start_time, end_time, color='yellow', alpha=0.3)
    else:
        print("No energy threshold entered.")

    # Prompt for ZCR threshold (optional)
    zcr_threshold, ok = QInputDialog.getDouble(
        None,
        "ZCR Threshold",
        "Enter ZCR threshold value:",
        value=np.mean(zcrs),
        min=0,
        max=np.max(zcrs)
    )

    if ok:
        # Plot ZCR threshold line
        axs[n_channels + 1].axhline(y=zcr_threshold, color='r', linestyle='--', label='Threshold')
        axs[n_channels + 1].legend(loc='upper right')

        # Detect segments where ZCR exceeds threshold
        zcr_segments = zcrs > zcr_threshold

        # Highlight detected segments in waveform plot
        for ch in range(n_channels):
            for i in range(len(zcr_segments)):
                if zcr_segments[i]:
                    start_time = zcr_time_axis[i]
                    end_time = start_time + (frame_length / sample_rate)
                    axs[ch].axvspan(start_time, end_time, color='cyan', alpha=0.2)
    else:
        print("No ZCR threshold entered.")

    plt.savefig('plot.png')
    plt.show()

if __name__ == '__main__':
    main()
