# Music Streamer – Basic VST Batch Processor

A C++ project that automates the rendering of MIDI files through VST3 plugins using the JUCE framework. Designed for batch processing of audio with optional plugin presets, producing WAV outputs.

---

## Features

- Loads VST3 plugins dynamically.
- Reads and processes MIDI files.
- Supports plugin preset loading.
- Handles stereo audio output with customizable sample rate, block size, and bit depth.
- Applies fade-out to tail samples for clean rendering.
- Command-line progress updates during rendering.

---

## Requirements

- **C++17 or newer**
- **JUCE 7+** (added as a Git submodule)
- Compatible VST3 plugin(s)
- Windows (paths in example code are Windows-specific)
- CMake for building the project
