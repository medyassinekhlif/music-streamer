# Music Streamer – VST Streaming Processor

A C++ project that automates the real-time rendering of MIDI files through VST3 plugins using the JUCE framework. Designed for streaming audio processing with optional plugin presets, delivering chunked audio data over HTTP.

---

## Features

- Loads VST3 plugins dynamically.
- Reads and processes MIDI files.
- Supports plugin preset loading.
- Handles stereo audio output with customizable sample rate, block size, and bit depth.
- Processes audio in fixed-size chunks (e.g., 1024 samples) for efficient streaming.
- Applies fade-out to tail samples for clean rendering.
- Real-time pacing of chunk generation to match playback speed (e.g., ~21ms per chunk at 48kHz).
- HTTP-based communication: Exposes a POST `/generate` endpoint that accepts JSON requests with MIDI, preset, and plugin paths; streams responses as NDJSON (newline-delimited JSON) containing base64-encoded audio chunks.
- Optional real-time WAV file writing during processing.
- Command-line progress updates during rendering.

---

## Requirements

- **C++17 or newer**
- **JUCE 7+** (added as a Git submodule)
- **Drogon** (HTTP framework for the server)
- Compatible VST3 plugin(s)
- Windows (paths in example code are Windows-specific)
- CMake for building the project
