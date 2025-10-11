import asyncio
import os
from datetime import datetime
import numpy as np
import sounddevice as sd
import httpx
import json
import base64
from collections import deque
from threading import Thread, Lock, Event
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import sys

# -----------------------
# Audio player + buffer
# -----------------------
class AudioStreamPlayer:
    def __init__(self, sample_rate=48000, channels=2, chunk_size=1024, prebuffer_chunks=20, max_buffer_chunks=100):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.prebuffer_chunks = prebuffer_chunks  # Increased to 20 chunks (~427ms buffer)
        self.max_buffer_chunks = max_buffer_chunks  # Limit buffer to prevent memory issues

        self.audio_buffer = deque()
        self.buffer_lock = Lock()

        self.stream = None
        self.is_playing = False
        self.playback_started = Event()
        self.current_sample_position = 0

        # Chunk tracking
        self.received_chunks = {}
        self.total_chunks = 0
        self.played_chunks = 0
        
        # Buffer state
        self.chunks_received_count = 0
        self.underruns = 0
        self.dropped_chunks = 0  # Track chunks dropped due to buffer overflow

    def audio_callback(self, outdata, frames, time_info, status):
        """Called by sounddevice for audio playback."""
        try:
            if status:
                if status.output_underflow:
                    self.underruns += 1
                    if self.underruns <= 5:  # Only log first few underruns
                        print(f"⚠ Audio underrun #{self.underruns} - buffer was empty!")
                # Log any other status flags
                if status.output_overflow:
                    print(f"⚠ Audio output overflow detected")

            with self.buffer_lock:
                # Don't start playing until we have minimum buffer safety margin
                if not self.playback_started.is_set() and len(self.audio_buffer) < self.prebuffer_chunks // 2:
                    outdata.fill(0)
                    return
                    
                if len(self.audio_buffer) > 0:
                    chunk = self.audio_buffer.popleft()

                    if not isinstance(chunk, np.ndarray):
                        chunk = np.array(chunk, dtype=np.float32)
                    else:
                        chunk = chunk.astype(np.float32, copy=False)

                    if chunk.ndim == 1:
                        if chunk.size % self.channels == 0:
                            chunk = chunk.reshape(-1, self.channels)
                        else:
                            outdata.fill(0)
                            return

                    frames_available = chunk.shape[0]
                    to_copy = min(frames, frames_available)

                    outdata[:to_copy, :] = chunk[:to_copy, :]

                    if to_copy < frames:
                        outdata[to_copy:frames, :].fill(0)

                    if frames_available > to_copy:
                        remaining = chunk[to_copy:, :]
                        self.audio_buffer.appendleft(remaining)

                    self.current_sample_position += to_copy
                    self.played_chunks += 1
                else:
                    outdata.fill(0)
        except Exception as e:
            print(f"❌ Audio callback error: {e}")
            import traceback
            traceback.print_exc()
            outdata.fill(0)
            raise  # Re-raise to stop stream

    def finished_callback(self):
        """Called when stream stops unexpectedly."""
        print("⚠ Audio stream finished/stopped unexpectedly")
        self.is_playing = False

    def start_playback(self):
        """Start the audio stream."""
        if not self.is_playing:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.chunk_size,
                callback=self.audio_callback,
                finished_callback=self.finished_callback,
                dtype='float32'
            )
            self.stream.start()
            self.is_playing = True
            self.playback_started.set()
            print("🔊 Playback started")

    def stop_playback(self):
        """Stop the audio stream."""
        if self.is_playing and self.stream:
            self.stream.stop()
            self.stream.close()
            self.is_playing = False
            self.playback_started.clear()
            print("⏹ Playback stopped")

    def reset(self):
        """Reset player state for new stream"""
        with self.buffer_lock:
            self.audio_buffer.clear()
            self.received_chunks.clear()
            self.total_chunks = 0
            self.played_chunks = 0
            self.current_sample_position = 0
            self.chunks_received_count = 0
            self.underruns = 0
            self.dropped_chunks = 0
            self.playback_started.clear()

    def add_chunk(self, chunk_index, audio_data):
        """Add an audio chunk to the buffer."""
        with self.buffer_lock:
            self.received_chunks[int(chunk_index)] = True
            self.chunks_received_count += 1

            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.float32)
            else:
                audio_data = audio_data.astype(np.float32, copy=False)

            if audio_data.ndim == 1:
                if audio_data.size % self.channels == 0:
                    audio_data = audio_data.reshape(-1, self.channels)
                else:
                    print("⚠ Received chunk with size not divisible by channels; dropping")
                    return

            # With real-time server pacing, we don't need aggressive buffer limits
            # Just add the chunk - server generates at playback speed
            self.audio_buffer.append(audio_data)
            
            # Optional: warn if buffer grows unexpectedly large
            if len(self.audio_buffer) > 200:
                if self.dropped_chunks == 0:  # Only warn once
                    print(f"⚠ Buffer unusually large: {len(self.audio_buffer)} chunks (may indicate network latency)")
                    self.dropped_chunks = 1  # Use as flag to suppress repeated warnings
            
            # Auto-start playback after prebuffering with safety margin
            if not self.is_playing and len(self.audio_buffer) >= self.prebuffer_chunks:
                buffer_duration_ms = (len(self.audio_buffer) * self.chunk_size / self.sample_rate) * 1000
                print(f"✓ Prebuffer complete: {len(self.audio_buffer)} chunks ({buffer_duration_ms:.1f}ms), starting playback...")
                self.start_playback()

    def get_buffer_status(self):
        """Get current buffer and playback status."""
        with self.buffer_lock:
            return {
                "buffered_chunks": len(self.audio_buffer),
                "received_chunks": len(self.received_chunks),
                "total_chunks": self.total_chunks,
                "played_chunks": self.played_chunks,
                "current_position_seconds": self.current_sample_position / self.sample_rate,
                "underruns": self.underruns,
                "dropped_chunks": self.dropped_chunks
            }

# -----------------------
# Visualization (PyQtGraph)
# -----------------------
class AudioVisualizerWindow(QtWidgets.QMainWindow):
    def __init__(self, player: AudioStreamPlayer):
        super().__init__()
        self.player = player
        self.is_streaming = False
        self.current_stream_task = None

        self.setWindowTitle("Audio Stream Visualizer & Player")
        self.setGeometry(100, 100, 1200, 700)

        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Control panel
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QHBoxLayout(control_panel)
        
        # Server URL
        control_layout.addWidget(QtWidgets.QLabel("Server:"))
        self.server_input = QtWidgets.QLineEdit("http://localhost:8080")        
        self.server_input.setMinimumWidth(200)
        control_layout.addWidget(self.server_input)
        
        # MIDI path
        control_layout.addWidget(QtWidgets.QLabel("MIDI:"))
        self.midi_input = QtWidgets.QLineEdit("C:\\src\\forge\\testing\\midi\\2.mid")
        self.midi_input.setMinimumWidth(250)
        control_layout.addWidget(self.midi_input)
        
        # Browse button
        self.browse_btn = QtWidgets.QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_midi_file)
        control_layout.addWidget(self.browse_btn)
        
        # Preset path
        control_layout.addWidget(QtWidgets.QLabel("Preset:"))
        self.preset_input = QtWidgets.QLineEdit("C:\\src\\forge\\testing\\presets\\glassgrandpiano.vstpreset")
        self.preset_input.setMinimumWidth(250)
        control_layout.addWidget(self.preset_input)
        
        # Plugin path
        control_layout.addWidget(QtWidgets.QLabel("Plugin:"))
        self.plugin_input = QtWidgets.QLineEdit("C:/Program Files/Common Files/VST3/Spitfire Audio/LABS.vst3")
        self.plugin_input.setMinimumWidth(300)
        control_layout.addWidget(self.plugin_input)
        
        control_layout.addStretch()
        layout.addWidget(control_panel)

        # Start/Stop button
        button_panel = QtWidgets.QWidget()
        button_layout = QtWidgets.QHBoxLayout(button_panel)
        
        self.start_btn = QtWidgets.QPushButton("▶ Start Generation (Space)")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setStyleSheet("font-size: 14pt; font-weight: bold;")
        self.start_btn.clicked.connect(self.toggle_streaming)
        self.start_btn.setShortcut(QtCore.Qt.Key_Space)
        button_layout.addWidget(self.start_btn)
        
        layout.addWidget(button_panel)

        # Status label
        self.status_label = QtWidgets.QLabel("Status: Ready")
        self.status_label.setStyleSheet("font-size: 11pt; padding: 5px;")
        layout.addWidget(self.status_label)

        # Progress bar for chunks
        self.progress_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.progress_widget)

        self.progress_plot = self.progress_widget.addPlot(title="Chunk Loading Progress (Gray = Received, Red Line = Playing)")
        self.progress_plot.setYRange(0, 1)
        self.progress_plot.setLabel('left', 'Status')
        self.progress_plot.setLabel('bottom', 'Chunk Index')

        self.received_bars = pg.BarGraphItem(x=[], height=[], width=0.8, brush='g')
        self.playback_position = pg.InfiniteLine(angle=90, pen='r', movable=False)

        self.progress_plot.addItem(self.received_bars)
        self.progress_plot.addItem(self.playback_position)

        # Waveform display
        self.waveform_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.waveform_widget)

        self.waveform_plot = self.waveform_widget.addPlot(title="Live Waveform")
        self.waveform_plot.setLabel('left', 'Amplitude')
        self.waveform_plot.setLabel('bottom', 'Samples')
        self.waveform_curve = self.waveform_plot.plot(pen='y')

        self.waveform_buffer = deque(maxlen=4096)

        # Update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start(50)

    def browse_midi_file(self):
        """Open file dialog to browse for MIDI file"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select MIDI File",
            "",
            "MIDI Files (*.mid *.midi);;All Files (*.*)"
        )
        if file_path:
            self.midi_input.setText(file_path)

    def toggle_streaming(self):
        """Start or stop streaming"""
        if not self.is_streaming:
            self.start_streaming()
        else:
            # Confirm stop if playback is active
            reply = QtWidgets.QMessageBox.question(
                self,
                "Stop Playback?",
                "Are you sure you want to stop the current playback?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                self.stop_streaming()

    def start_streaming(self):
        """Start the streaming process"""
        server_url = self.server_input.text().strip()
        midi_path = self.midi_input.text().strip()
        preset_path = self.preset_input.text().strip()
        plugin_path = self.plugin_input.text().strip()

        if not server_url or not midi_path:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please provide server URL and MIDI path")
            return

        self.is_streaming = True
        self.start_btn.setText("⏹ Stop (Space or Esc)")
        self.start_btn.setStyleSheet("font-size: 14pt; font-weight: bold; background-color: #ff6b6b;")
        self.status_label.setText("Status: Starting stream...")
        
        # Disable inputs during streaming
        self.server_input.setEnabled(False)
        self.midi_input.setEnabled(False)
        self.preset_input.setEnabled(False)
        self.plugin_input.setEnabled(False)
        self.browse_btn.setEnabled(False)

        # Reset player
        self.player.reset()
        self.waveform_buffer.clear()

        # Start streaming in background thread
        self.stream_thread = Thread(
            target=self.run_async_stream,
            args=(server_url, midi_path, preset_path, plugin_path),
            daemon=True
        )
        self.stream_thread.start()

    @QtCore.pyqtSlot()
    def stop_streaming(self):
        """Stop the streaming process"""
        self.is_streaming = False
        self.player.stop_playback()
        self.start_btn.setText("▶ Start Generation (Space)")
        self.start_btn.setStyleSheet("font-size: 14pt; font-weight: bold;")
        self.status_label.setText("Status: Stopped")
        
        # Re-enable inputs
        self.server_input.setEnabled(True)
        self.midi_input.setEnabled(True)
        self.preset_input.setEnabled(True)
        self.plugin_input.setEnabled(True)
        self.browse_btn.setEnabled(True)

    def run_async_stream(self, server_url, midi_path, preset_path, plugin_path):
        """Run async streaming in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                self.stream_audio_from_server(server_url, midi_path, preset_path, plugin_path)
            )
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            import traceback
            traceback.print_exc()
            self.status_label.setText(f"Status: Error - {e}")
        finally:
            loop.close()
            if self.is_streaming:
                QtCore.QMetaObject.invokeMethod(
                    self, "stop_streaming", QtCore.Qt.QueuedConnection
                )

    async def stream_audio_from_server(self, server_url, midi_path, preset_path, plugin_path):
        """Stream NDJSON audio chunks from Drogon server"""
        url = f"{server_url}/generate"
        
        # Compute default WAV output path under project output directory
        # Try to place it in <workspace>/output/<basename>_<timestamp>.wav
        try:
            base = os.path.splitext(os.path.basename(midi_path))[0] or "stream"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
            out_dir = os.path.join(workspace_root, "output")
            os.makedirs(out_dir, exist_ok=True)
            wav_output_path = os.path.join(out_dir, f"{base}_{ts}.wav")
        except Exception:
            wav_output_path = ""  # let server decide

        print(f"\n📡 Connecting to: {url}")
        print(f"🎹 MIDI: {midi_path}")
        print(f"🎚 Preset: {preset_path}")
        print(f"🔌 Plugin: {plugin_path}")
        print(f"💾 WAV output: {wav_output_path or '<server-default>'}")
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            try:
                async with client.stream(
                    "POST",
                    url,
                    json={
                        "midi_path": midi_path,
                        "preset_path": preset_path,
                        "plugin_path": plugin_path,
                        # Ask server to write WAV in real-time
                        "save_wav": True,
                        "wav_output_path": wav_output_path,
                    },
                    headers={"Accept": "application/x-ndjson"}
                ) as response:
                    print(f"✓ Response status: {response.status_code}")
                    
                    response.raise_for_status()
                    first_chunk = True
                    line_count = 0
                    
                    async for line in response.aiter_lines():
                        line_count += 1
                        
                        if not line.strip():
                            continue
                            
                        if not self.is_streaming:
                            print("⏹ Stopping stream (user requested)")
                            break
                            
                        try:
                            chunk_data = json.loads(line)
                        except json.JSONDecodeError as e:
                            print(f"⚠ Failed to parse JSON line {line_count}: {e}")
                            print(f"Line content: {line[:100]}...")
                            continue

                        if "error" in chunk_data:
                            print(f"❌ Server error: {chunk_data['error']}")
                            self.status_label.setText(f"Status: Server Error - {chunk_data['error']}")
                            break

                        # On first chunk, set total
                        if first_chunk:
                            self.player.total_chunks = int(chunk_data.get("total_chunks", 0))
                            print(f"📊 Total chunks expected: {self.player.total_chunks}")
                            first_chunk = False
                            self.status_label.setText(f"Status: Buffering... (need {self.player.prebuffer_chunks} chunks)")

                        # Decode base64 -> float32
                        try:
                            audio_bytes = base64.b64decode(chunk_data["audio_data"])
                            audio_floats = np.frombuffer(audio_bytes, dtype=np.float32)
                        except Exception as e:
                            print(f"⚠ Failed to decode audio data: {e}")
                            continue

                        channels = int(chunk_data.get("channels", self.player.channels))

                        # Reshape to (frames, channels)
                        if channels > 1:
                            if audio_floats.size % channels == 0:
                                audio_array = audio_floats.reshape(-1, channels)
                            else:
                                print(f"⚠ Chunk size not divisible by channels, skipping chunk {chunk_data.get('chunk_index')}")
                                continue
                        else:
                            audio_array = audio_floats.reshape(-1, 1)

                        # Add to player
                        chunk_idx = int(chunk_data["chunk_index"])
                        self.player.add_chunk(chunk_idx, audio_array)

                        if chunk_idx % 50 == 0:
                            print(f"📦 Received chunk {chunk_idx + 1}/{int(chunk_data['total_chunks'])}")
                        
                        # Update status after playback starts
                        if self.player.is_playing and line_count % 10 == 0:
                            self.status_label.setText("Status: Streaming & Playing...")
                        
                        if chunk_data.get("is_last"):
                            print("✓ All chunks received!")
                            self.status_label.setText("Status: Playback finishing...")
                            break

                    print(f"✓ Stream ended. Total lines received: {line_count}")
                    
                    # Ensure playback starts even if we didn't hit prebuffer threshold
                    if not self.player.is_playing and self.player.chunks_received_count > 0:
                        print("⚡ Starting playback with available buffer...")
                        self.player.start_playback()
                    
                    # Wait for playback to finish
                    wait_count = 0
                    while self.player.get_buffer_status()["buffered_chunks"] > 0:
                        await asyncio.sleep(0.1)
                        wait_count += 1
                        if wait_count % 20 == 0:
                            status = self.player.get_buffer_status()
                            print(f"⏳ Waiting for playback to complete... {status['buffered_chunks']} chunks remaining")
                    
                    self.status_label.setText("Status: Complete ✓")
                    print("🎉 Playback complete!")
                    
            except httpx.ConnectError as e:
                error_msg = f"Connection failed: {e}"
                print(f"❌ {error_msg}")
                self.status_label.setText(f"Status: {error_msg}")
            except httpx.TimeoutException as e:
                error_msg = f"Connection timeout: {e}"
                print(f"❌ {error_msg}")
                self.status_label.setText(f"Status: {error_msg}")
            except Exception as e:
                error_msg = f"Streaming error: {e}"
                print(f"❌ {error_msg}")
                import traceback
                traceback.print_exc()
                self.status_label.setText(f"Status: {error_msg}")

    def update_visualization(self):
        """Update the visualization"""
        status = self.player.get_buffer_status()

        # Update status label with stats
        if self.is_streaming or status['buffered_chunks'] > 0:
            underrun_text = f" | ⚠ Underruns: {status['underruns']}" if status['underruns'] > 0 else ""
            self.status_label.setText(
                f"Status: Received {status['received_chunks']}/{status['total_chunks']} | "
                f"Buffered {status['buffered_chunks']} | "
                f"Played {status['played_chunks']} | "
                f"Position {status['current_position_seconds']:.2f}s{underrun_text}"
            )

        # Update progress bars
        if status['total_chunks'] > 0:
            x = list(range(status['total_chunks']))
            height = [1 if i in self.player.received_chunks else 0
                      for i in range(status['total_chunks'])]
            self.received_bars.setOpts(x=x, height=height, width=0.8)
            self.progress_plot.setXRange(0, max(1, status['total_chunks']))

            # Update playback position
            self.playback_position.setValue(status['played_chunks'])

        # Update waveform
        with self.player.buffer_lock:
            if len(self.player.audio_buffer) > 0:
                latest_chunk = self.player.audio_buffer[-1]
                if isinstance(latest_chunk, np.ndarray):
                    if latest_chunk.ndim == 2 and latest_chunk.shape[1] >= 1:
                        mono_data = latest_chunk[:, 0][:2048]
                        self.waveform_buffer.extend(mono_data)

        if len(self.waveform_buffer) > 0:
            waveform_array = np.array(self.waveform_buffer)
            self.waveform_curve.setData(waveform_array)

    def closeEvent(self, event):
        """Handle window close"""
        if self.is_streaming:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Playback in Progress",
                "Audio is currently playing. Stop playback and exit?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.No:
                event.ignore()
                return
        
        self.is_streaming = False
        self.player.stop_playback()
        event.accept()

# -----------------------
# Main
# -----------------------
def main():
    # Create player with 20-chunk prebuffering (~427ms) for glitch-free startup
    player = AudioStreamPlayer(sample_rate=48000, channels=2, chunk_size=1024, prebuffer_chunks=20, max_buffer_chunks=100)

    # Create Qt application
    app = QtWidgets.QApplication(sys.argv)
    window = AudioVisualizerWindow(player)
    window.show()

    # Run Qt event loop
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass
    finally:
        player.stop_playback()

if __name__ == "__main__":
    main()