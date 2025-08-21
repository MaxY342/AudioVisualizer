import pyaudio
import pyglet
import numpy as np
from pyglet import shapes

#Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024

p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev['name'] == 'Stereo Mix (Realtek(R) Audio)' and dev['hostApi'] == 0:
        devIndex = dev['index']

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
    input_device_index=devIndex
)

# pyglet setup
window = pyglet.window.Window(width=800, height=600, caption='Audio Visualizer')
batch = pyglet.graphics.Batch()

# How many log-spaced bars
NUM_BARS = 60  

# Frequency mapping for each bar
freqs = np.fft.rfftfreq(CHUNK, 1.0 / RATE)  # FFT bin frequencies (ranging from 0 to RATE/2)
# Define log-spaced targets freqencies (bar centers)
log_freqs = np.logspace(np.log10(40), np.log10(RATE/2), NUM_BARS)

# FFT data storage
fft_bars = []  # Stores Rectangle objects for FFT bars
smooth_fft = np.zeros(NUM_BARS)

# Create FFT bars (pre-initialize)
bar_width = 800 / NUM_BARS
for i in range(NUM_BARS):
    bar = shapes.Rectangle(
        x=i * bar_width,
        y=0,
        width=bar_width - 1,
        height=0,
        color=(160, 0, 0),
        batch=batch
    )
    fft_bars.append(bar)

def update(dt):
    global smooth_fft

    data = stream.read(CHUNK)
    data = np.frombuffer(data, dtype=np.int16)
    left = data[::2]
    right = data[1::2]
    mono_audio_data = ((left + right) / 2).astype(np.float32) / 32768.0

    # Apply Hanning window
    window = np.hanning(len(mono_audio_data))
    mono_audio_data *= window

    # Compute the FFT
    fft = np.abs(np.fft.rfft(mono_audio_data)) / CHUNK

    # Interpolate FFT magnitudes at log-spaced frequencies
    log_fft = np.interp(log_freqs, freqs, fft)

    # Normalize the FFT data
    log_fft = np.clip(log_fft*50, 0, 1)

    # Smooth the FFT data
    smooth_fft = 0.7 * smooth_fft + 0.3 * log_fft

    # Update FFT bars
    for i, mag in enumerate(smooth_fft):
        fft_bars[i].height = mag * 600  # Scale height to window

@window.event
def on_draw():
    window.clear()
    batch.draw()

# Start Pyglet's update loop
pyglet.clock.schedule_interval(update, 1/60.0)  # 60 FPS
pyglet.app.run()

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()