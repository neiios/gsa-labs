import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft, rfftfreq
from scipy.io import wavfile

file_path = "best_part.wav"
sample_rate, audio_data = wavfile.read(file_path)

if audio_data.ndim > 1:
    audio_data = audio_data[:, 0]

t = np.arange(len(audio_data)) / sample_rate

fft_values_original = rfft(audio_data)
fft_freqs = rfftfreq(len(audio_data), 1 / sample_rate)

cutoff_frequency = 1000

fft_values_filtered = np.copy(fft_values_original)
fft_values_filtered[fft_freqs > cutoff_frequency] = 0

filtered_signal = irfft(fft_values_filtered)

output_file_path = "filtered_best_part.wav"
wavfile.write(output_file_path, sample_rate, filtered_signal.astype(np.int16))

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(t, audio_data, color="blue", label="Original Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Original Signal (Time Domain)")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(fft_freqs, np.abs(fft_values_original), color="blue", label="Original Signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.title("Original Signal (Frequency Domain)")
plt.xscale("log")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t, filtered_signal, color="orange", label="Filtered Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title(f"Filtered Signal (Time Domain, cutoff={cutoff_frequency} Hz)")
plt.legend()

fft_values_filtered_signal = rfft(filtered_signal)

plt.subplot(2, 2, 4)
plt.plot(fft_freqs, np.abs(fft_values_filtered_signal), color="orange", label="Filtered Signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.title(f"Filtered Signal (Frequency Domain, cutoff={cutoff_frequency} Hz)")
plt.xscale("log")
plt.legend()

plt.tight_layout()
plt.show()
