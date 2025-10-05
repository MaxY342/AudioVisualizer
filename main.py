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
window_length = 400
window_height = 200
window = pyglet.window.Window(width=window_length, height=window_height, caption='Audio Visualizer')
batch = pyglet.graphics.Batch()

# How many log-spaced bars
NUM_BARS = 32  

# Frequency mapping for each bar
freqs = np.fft.rfftfreq(CHUNK, 1.0 / RATE)  # x-axis, linear FFT bin frequencies (ranging from 0 to RATE/2)
# new x-axis, Define log-spaced targets freqencies (bar centers)
start_freq = 150  # Start frequency for visualization
end_freq = 10000  # End frequency for visualization
log_freqs = np.logspace(np.log10(start_freq), np.log10(end_freq), NUM_BARS)
print("Log frequencies:", log_freqs)

# FFT data storage
fft_bars = []  # Stores Rectangle objects for FFT bars
smooth_fft = np.zeros(NUM_BARS)

# Create FFT bars (pre-initialize)
bar_width = window_length / NUM_BARS
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
    fft = np.abs(np.fft.rfft(mono_audio_data)) / CHUNK # y-axis, FFT magnitudes

    # Interpolate FFT magnitudes (linear spaced y-axis values) from linear freq bins (linear spaced x-axis values) to log-spaced freq bins (log spaced x-axis values)
    log_fft = np.interp(log_freqs, freqs, fft)

    # Balance lower and higher frequencies
    log_fft *= np.logspace(0, 2.5, NUM_BARS, base=2)

    # Normalize the FFT data
    log_fft = np.clip(log_fft*40, 0, 1)

    # Smooth the FFT data
    smooth_fft = 0.8 * smooth_fft + 0.2 * log_fft

    # Update FFT bars
    for i, mag in enumerate(smooth_fft):
        fft_bars[i].height = mag * window_height  # Scale height to window

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