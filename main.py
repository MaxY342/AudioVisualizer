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

# FFT data storage
fft_bars = []  # Stores Rectangle objects for FFT bars
smooth_fft = np.zeros(CHUNK // 2)

# Create FFT bars (pre-initialize)
bar_width = 800 / (CHUNK // 2)
for i in range(CHUNK // 2):
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
    mono_audio_data_padded = np.zeros(CHUNK, dtype=np.float32)
    mono_audio_data_padded[:len(mono_audio_data)] = mono_audio_data

    # apply Hanning window
    window = np.hanning(len(mono_audio_data_padded))
    mono_audio_data_padded *= window

    # compute the FFT
    fft = np.abs(np.fft.rfft(mono_audio_data_padded)[:CHUNK//2]) / CHUNK
    fft *= np.logspace(-1.5, 1.5, CHUNK//2, base=2)
    fft_magnitude = np.clip(fft * 50, 0, 1)

    smooth_fft = 0.7 * smooth_fft + 0.3 * fft_magnitude
    fft_magnitude = smooth_fft

    # Update FFT bars
    for i, mag in enumerate(fft_magnitude):
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