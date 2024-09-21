SOURCE = song_16_48.wav
OUTPUTS = song_16_48_mono.wav song_8_48.wav song_24_48.wav song_32_48.wav song_16_96.wav song_16_48_shortest.wav song_16_48_shorter.wav

all: $(OUTPUTS)

song_16_48_mono.wav: $(SOURCE)
	ffmpeg -i $< -acodec pcm_s16le -ac 1 $@

song_8_48.wav: $(SOURCE)
	ffmpeg -i $< -acodec pcm_u8 $@

song_24_48.wav: $(SOURCE)
	ffmpeg -i $< -acodec pcm_s24le $@

song_32_48.wav: $(SOURCE)
	ffmpeg -i $< -acodec pcm_s32le $@

song_16_96.wav: $(SOURCE)
	ffmpeg -i $< -acodec pcm_s16le -ar 96000 $@

song_16_48_shortest.wav: $(SOURCE)
	ffmpeg -i $< -acodec pcm_s16le -t 00:00:01 $@

song_16_48_shorter.wav: $(SOURCE)
	ffmpeg -i $< -acodec pcm_s16le -t 00:01:00 $@

clean:
	rm -f $(OUTPUTS)

.PHONY: all clean
