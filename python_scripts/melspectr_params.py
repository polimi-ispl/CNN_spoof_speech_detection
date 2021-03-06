SAMPLE_RATE = 16000
STFT_WINDOW_LENGTH = 512
NUM_MEL_BINS = 256
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
EXAMPLE_HOP_SECONDS = 0.96
EMBEDDING_SIZE = 128