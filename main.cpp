#include <drogon/drogon.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <iostream>
#include <cmath>
#include <condition_variable>
#include <sstream>
#include <thread>
#include <queue>
#include <mutex>

static size_t secondsToSamples(double sec, double sr)
{
    return static_cast<size_t>(sec * sr);
}

class StreamingProcessor
{
public:
    StreamingProcessor(const juce::String &pluginPathIn, double sr = 48000.0,
                       int blockSizeIn = 1024, int bitDepthIn = 24)
        : pluginPath(pluginPathIn),
          sampleRate(sr),
          blockSize(blockSizeIn),
          bitDepth(bitDepthIn),
          isProcessing(false)
    {
        audioFormatManager.registerBasicFormats();
        audioPluginFormatManager.addDefaultFormats();
    }

    struct AudioChunk
    {
        std::vector<float> data;
        size_t chunkIndex;
        size_t totalChunks;
        bool isLastChunk;
        size_t samplesPerChannel; // number of valid samples in this chunk (may be < blockSize on last chunk)
    };

    using ChunkCallback = std::function<void(const AudioChunk &)>;

    bool processStreaming(const juce::File &midiFile, const juce::File &presetFile,
                          ChunkCallback callback)
    {
        std::cout << "Starting streaming processing: " << midiFile.getFileName() << std::endl;
        std::cout << "Creating plugin instance..." << std::endl;

        auto plugin = createPluginInstance();
        if (!plugin)
        {
            std::cerr << "Error: plugin instance creation failed\n";
            return false;
        }

        std::cout << "✓ Plugin loaded: " << plugin->getName() << std::endl;

        std::cout << "Reading MIDI file..." << std::endl;
        size_t totalInputLength = 0;
        auto midiFileData = readMidiFile(midiFile, sampleRate, totalInputLength);
        if (midiFileData.getNumTracks() == 0)
        {
            std::cerr << "Error: MIDI contains no tracks\n";
            return false;
        }

        std::cout << "✓ MIDI tracks: " << midiFileData.getNumTracks() << std::endl;
        std::cout << "✓ MIDI length: " << totalInputLength << " samples" << std::endl;

        // Bus setup
        juce::AudioPluginInstance::BusesLayout layout = plugin->getBusesLayout();
        if (layout.outputBuses.size() == 0)
            layout.outputBuses.add(juce::AudioChannelSet::stereo());
        else
            layout.outputBuses.getReference(0) = juce::AudioChannelSet::stereo();
        plugin->setBusesLayout(layout);

        unsigned int totalNumOutputChannels = static_cast<unsigned int>(plugin->getTotalNumOutputChannels());
        std::cout << "Output channels: " << totalNumOutputChannels << std::endl;

        if (presetFile.existsAsFile())
        {
            loadPreset(*plugin, presetFile);
        }

        // Prepare plugin BEFORE getting latency
        plugin->prepareToPlay(sampleRate, blockSize);
        plugin->setNonRealtime(true);  // Optimize for offline processing
        
        int latency = plugin->getLatencySamples();
        std::cout << "Plugin latency: " << latency << " samples" << std::endl;
        
        // Add extra pre-roll time to ensure plugin state is initialized
        // Many synths need time to load samples and initialize effects
        size_t preRollSamples = std::max<size_t>(static_cast<size_t>(latency), secondsToSamples(0.5, sampleRate));
        size_t tailSamples = secondsToSamples(6.0, sampleRate);

        // Start processing BEFORE time zero to capture all notes from the beginning
        size_t totalSamples = totalInputLength + tailSamples;
        size_t totalChunks = (totalSamples + preRollSamples + blockSize - 1) / blockSize;

        std::cout << "Pre-roll samples: " << preRollSamples << " (" << (preRollSamples / sampleRate) << "s)" << std::endl;
        std::cout << "Total chunks to process: " << totalChunks << std::endl;

        juce::AudioBuffer<float> buffer(static_cast<int>(totalNumOutputChannels), blockSize);
        juce::MidiBuffer midiBuffer;
        
        // Start from negative offset to give plugin warm-up time
        int64_t sampleIndex = -static_cast<int64_t>(preRollSamples);
        // Track absolute MIDI timeline position (always starts at 0, independent of pre-roll)
        int64_t midiTimelineSample = 0;
        size_t chunkIndex = 0;

        isProcessing = true;

        std::cout << "Starting processing loop (including pre-roll)..." << std::endl;
        
        while (sampleIndex < static_cast<int64_t>(totalSamples) && isProcessing)
        {
            buffer.clear();
            midiBuffer.clear();

            // Add MIDI events for this block
            // CRITICAL: MIDI timeline is independent of pre-roll!
            // We process MIDI from time 0, but output audio starts after pre-roll
            int64_t blockStartTime = midiTimelineSample;
            int64_t blockEndTime = midiTimelineSample + static_cast<int64_t>(blockSize);
            
            int midiEventsThisBlock = 0;
            for (int t = 0; t < midiFileData.getNumTracks(); ++t)
            {
                auto *track = midiFileData.getTrack(t);
                for (auto &meh : *track)
                {
                    // MIDI timestamps are absolute from the start of the file (sample 0)
                    auto msgTsSamples = static_cast<int64_t>(secondsToSamples(meh->message.getTimeStamp(), sampleRate));
                    
                    // Check if this MIDI event falls within the current processing block
                    if (msgTsSamples >= blockStartTime && msgTsSamples < blockEndTime)
                    {
                        // Calculate offset within the block
                        int offsetInBlock = static_cast<int>(msgTsSamples - blockStartTime);
                        if (offsetInBlock >= 0 && offsetInBlock < blockSize)
                        {
                            midiBuffer.addEvent(meh->message, offsetInBlock);
                            midiEventsThisBlock++;
                            
                            // Enhanced debugging - show all note events in first 100 chunks
                            if (chunkIndex < 100 && (meh->message.isNoteOn() || meh->message.isNoteOff()))
                            {
                                std::cout << "  MIDI " << (meh->message.isNoteOn() ? "ON " : "OFF")
                                         << " at abs_sample=" << msgTsSamples 
                                         << " (chunk=" << chunkIndex 
                                         << ", offset=" << offsetInBlock 
                                         << ", note=" << meh->message.getNoteNumber()
                                         << ", vel=" << meh->message.getVelocity() << ")" << std::endl;
                            }
                        }
                    }
                }
            }
            
            // Warn if no MIDI events in early chunks (indicates timing problem)
            if (chunkIndex < 50 && midiEventsThisBlock > 0)
            {
                std::cout << "  → " << midiEventsThisBlock << " MIDI event(s) sent to plugin in chunk " << chunkIndex << std::endl;
            }

            plugin->processBlock(buffer, midiBuffer);

            // Monitor audio levels to detect silent output (debugging)
            if (chunkIndex < 100 || chunkIndex % 100 == 0)
            {
                float maxLevel = 0.0f;
                for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
                {
                    for (int s = 0; s < buffer.getNumSamples(); ++s)
                    {
                        maxLevel = std::max(maxLevel, std::abs(buffer.getSample(ch, s)));
                    }
                }
                if (maxLevel > 0.0001f)
                {
                    std::cout << "  Audio level chunk " << chunkIndex << ": " << maxLevel << std::endl;
                }
                else if (chunkIndex > 20)  // After initial silence
                {
                    std::cout << "  ⚠ SILENT chunk " << chunkIndex << " (no audio output from plugin!)" << std::endl;
                }
            }

            // Only send audio chunks for positive sample positions (skip pre-roll output)
            if (sampleIndex >= 0)
            {
                // Apply fade out near end
                const int fadeLen = std::min(512, blockSize);
                if (static_cast<size_t>(sampleIndex) + static_cast<size_t>(blockSize) >= totalSamples - static_cast<size_t>(blockSize))
                {
                    int start = std::max(0, blockSize - fadeLen);
                    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
                        buffer.applyGainRamp(ch, start, fadeLen, 1.0f, 0.0f);
                }

                // Prepare chunk
                AudioChunk chunk;
                chunk.chunkIndex = chunkIndex;
                chunk.totalChunks = totalChunks;
                chunk.isLastChunk = (static_cast<size_t>(sampleIndex) + blockSize >= totalSamples);
                // Determine valid samples for this chunk
                size_t samplesThisChunk = blockSize;
                if (chunk.isLastChunk)
                {
                    auto remaining = (static_cast<int64_t>(totalSamples) - sampleIndex);
                    if (remaining > 0)
                        samplesThisChunk = static_cast<size_t>(std::min<int64_t>(remaining, blockSize));
                    else
                        samplesThisChunk = 0;
                }
                chunk.samplesPerChannel = samplesThisChunk;

                // Interleave audio data
                size_t numSamples = samplesThisChunk * totalNumOutputChannels;
                chunk.data.resize(numSamples);

                for (size_t sample = 0; sample < samplesThisChunk; ++sample)
                {
                    for (unsigned int ch = 0; ch < totalNumOutputChannels; ++ch)
                    {
                        chunk.data[sample * totalNumOutputChannels + ch] =
                            buffer.getSample(static_cast<int>(ch), static_cast<int>(sample));
                    }
                }

                // Send chunk via callback
                callback(chunk);

                if (chunkIndex % 50 == 0) {
                    std::cout << "Processed chunk " << chunkIndex << "/" << totalChunks << std::endl;
                }
                
                chunkIndex++;
            }

            // Advance both timelines
            sampleIndex += static_cast<int64_t>(blockSize);
            midiTimelineSample += static_cast<int64_t>(blockSize);
        }

        std::cout << "Processing complete. Total chunks: " << chunkIndex << std::endl;

        plugin->releaseResources();
        plugin.reset();  // Explicitly release plugin
        
        isProcessing = false;
        return true;
    }

    void stopProcessing()
    {
        isProcessing = false;
    }

private:
    juce::String pluginPath;
    double sampleRate;
    int blockSize;
    int bitDepth;
    std::atomic<bool> isProcessing;
    juce::AudioFormatManager audioFormatManager;
    juce::AudioPluginFormatManager audioPluginFormatManager;

    std::unique_ptr<juce::AudioPluginInstance> createPluginInstance()
    {
        std::cout << "  Searching for plugins at: " << pluginPath << std::endl;
        juce::OwnedArray<juce::PluginDescription> foundPlugins;
        audioPluginFormatManager.addDefaultFormats();
        auto* format = audioPluginFormatManager.getFormat(0); // Assume VST3 is first, or search for VST3 by name if needed
        if (!format)
        {
            std::cerr << "  ✗ No plugin formats available" << std::endl;
            return nullptr;
        }
        format->findAllTypesForFile(foundPlugins, pluginPath);

        if (foundPlugins.isEmpty())
        {
            std::cerr << "  ✗ No plugin found at: " << pluginPath << std::endl;
            return nullptr;
        }

        std::cout << "  ✓ Found " << foundPlugins.size() << " plugin(s)" << std::endl;
        juce::PluginDescription desc = *foundPlugins[0];
        std::cout << "  Plugin name: " << desc.name << std::endl;
        std::cout << "  Manufacturer: " << desc.manufacturerName << std::endl;

        // Create plugin on message thread with proper blocking
        std::unique_ptr<juce::AudioPluginInstance> pluginInstance;
        juce::String errorMessage;

        std::cout << "  Creating plugin instance on message thread..." << std::endl;

        // Use mutex and condition variable for proper synchronization
        std::mutex mtx;
        std::condition_variable cv;
        bool done = false;

        // Capture desc by value to ensure it stays valid
        juce::MessageManager::callAsync([desc, &pluginInstance, &errorMessage, &mtx, &cv, &done, this, format]() {
            std::cout << "  -> Executing on message thread..." << std::endl;
            juce::String err;
            pluginInstance = format->createInstanceFromDescription(desc, sampleRate, blockSize, err);
            errorMessage = err;

            std::unique_lock<std::mutex> lock(mtx);
            done = true;
            cv.notify_one();
        });

        // Wait for completion with timeout
        std::unique_lock<std::mutex> lock(mtx);
        bool success = cv.wait_for(lock, std::chrono::seconds(30), [&]{ return done; });

        if (!success)
        {
            std::cerr << "  ✗ Plugin creation timed out after 30 seconds" << std::endl;
            return nullptr;
        }

        if (!pluginInstance)
        {
            std::cerr << "  ✗ Plugin creation failed: " << errorMessage << std::endl;
        }
        else
        {
            std::cout << "  ✓ Plugin instance created successfully" << std::endl;
        }

        return pluginInstance;
    }

    bool loadPreset(juce::AudioPluginInstance &plugin, const juce::File &presetFile)
    {
        juce::MemoryBlock presetData;
        auto in = presetFile.createInputStream();
        if (!in || !in->readIntoMemoryBlock(presetData))
            return false;
        plugin.setStateInformation(presetData.getData(), static_cast<int>(presetData.getSize()));
        std::cout << "Preset loaded" << std::endl;
        return true;
    }

    juce::MidiFile readMidiFile(const juce::File &file, double sr, size_t &lengthOut)
    {
        juce::MidiFile midi;
        lengthOut = 0;
        auto in = file.createInputStream();
        if (!in || !midi.readFrom(*in, true))
            return midi;
        midi.convertTimestampTicksToSeconds();
        for (int t = 0; t < midi.getNumTracks(); ++t)
        {
            auto *track = midi.getTrack(t);
            for (auto &meh : *track)
            {
                const size_t ts = secondsToSamples(meh->message.getTimeStamp(), sr);
                if (ts > lengthOut)
                    lengthOut = ts;
            }
        }
        return midi;
    }
};

// Thread-safe queue for streaming chunks
class ChunkQueue
{
public:
    void push(const std::string& chunk)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(chunk);
        cv_.notify_one();
    }

    bool pop(std::string& chunk, int timeout_ms = 5000)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                        [this] { return !queue_.empty() || finished_; }))
        {
            if (!queue_.empty())
            {
                chunk = queue_.front();
                queue_.pop();
                return true;
            }
        }
        return false;
    }

    void setFinished()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        cv_.notify_all();
    }

    bool isFinished() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return finished_ && queue_.empty();
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    std::queue<std::string> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool finished_ = false;
};

class JuceInitializer
{
public:
    JuceInitializer()
    {
        juce::initialiseJuce_GUI();
        
        // Start JUCE message thread for plugin operations
        messageThread = std::make_unique<std::thread>([]() {
            std::cout << "JUCE message thread started" << std::endl;
            
            // Initialize message manager for this thread
            auto* mm = juce::MessageManager::getInstance();
            mm->setCurrentThreadAsMessageThread();
            
            // Create a timer to periodically check shouldStop flag
            class StopChecker : public juce::Timer
            {
            public:
                void timerCallback() override
                {
                    if (shouldStop)
                    {
                        juce::MessageManager::getInstance()->stopDispatchLoop();
                    }
                }
            };
            
            StopChecker checker;
            checker.startTimer(100); // Check every 100ms
            
            // Run the dispatch loop (blocks until stopDispatchLoop is called)
            std::cout << "Starting message dispatch loop..." << std::endl;
            mm->runDispatchLoop();
            
            std::cout << "JUCE message thread stopped" << std::endl;
        });
        
        // Give it time to initialize
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::cout << "JUCE initialization complete" << std::endl;
    }
    
    ~JuceInitializer()
    {
        shouldStop = true;
        
        // Give some time for the stopDispatchLoop to be called
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        
        if (messageThread && messageThread->joinable())
        {
            messageThread->join();
        }
        juce::shutdownJuce_GUI();
    }

private:
    static inline std::atomic<bool> shouldStop{false};
    std::unique_ptr<std::thread> messageThread;
};

int main()
{
    JuceInitializer juceInit;

    // Configure Drogon BEFORE registering handlers
    drogon::app()
        .addListener("0.0.0.0", 8080)
        .setThreadNum(4)
        .setLogLevel(trantor::Logger::kDebug);

    // Register handler
    drogon::app().registerHandler(
          "/generate",
          [](const drogon::HttpRequestPtr &req,
           std::function<void(const drogon::HttpResponsePtr &)> &&callback)
        {
            std::cout << "\n=== Received /generate request ===" << std::endl;
            
            // Parse JSON request
            auto jsonPtr = req->getJsonObject();
            if (!jsonPtr)
            {
                std::cout << "Error: Invalid JSON" << std::endl;
                auto resp = drogon::HttpResponse::newHttpResponse();
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setBody("Invalid JSON");
                callback(resp);
                return;
            }

            const Json::Value &json = *jsonPtr;
            std::string midiPath = json.get("midi_path", "").asString();
            std::string presetPath = json.get("preset_path", "").asString();
          std::string pluginPath = json.get("plugin_path",
                                     "C:/Program Files/Common Files/VST3/Spitfire Audio/LABS.vst3")
                                 .asString();
          bool saveWav = json.get("save_wav", false).asBool();
          std::string wavOutputPath = json.get("wav_output_path", "").asString();

            std::cout << "MIDI: " << midiPath << std::endl;
            std::cout << "Preset: " << presetPath << std::endl;
            std::cout << "Plugin: " << pluginPath << std::endl;
            if (saveWav)
                std::cout << "WAV output: " << (wavOutputPath.empty() ? std::string("<auto>") : wavOutputPath) << std::endl;

            juce::File midiFile(midiPath);
            juce::File presetFile(presetPath);

            if (!midiFile.existsAsFile())
            {
                std::cout << "Error: MIDI file not found" << std::endl;
                auto resp = drogon::HttpResponse::newHttpResponse();
                resp->setStatusCode(drogon::k400BadRequest);
                resp->setBody("MIDI file not found");
                callback(resp);
                return;
            }

            // Create shared chunk queue and buffer state
            auto chunkQueue = std::make_shared<ChunkQueue>();
            auto bufferState = std::make_shared<std::string>();
            auto bufferOffset = std::make_shared<size_t>(0);

            // Start processing in background thread
            std::thread processingThread([midiFile, presetFile, pluginPath, chunkQueue, saveWav, wavOutputPath]() {
                std::cout << "==> Processing thread started" << std::endl;
                
                try {
                    // Optional WAV writer setup
                    std::unique_ptr<juce::AudioFormatWriter> wavWriter;
                    std::unique_ptr<juce::FileOutputStream> wavStream;
                    if (saveWav)
                    {
                        juce::File outFile(wavOutputPath);
                        if (wavOutputPath.empty())
                        {
                            // Auto-generate a filename in current working directory "output"
                            juce::File outDir = juce::File::getCurrentWorkingDirectory().getChildFile("output");
                            outDir.createDirectory();
                            auto ts = juce::Time::getCurrentTime().formatted("%Y%m%d_%H%M%S");
                            outFile = outDir.getChildFile("stream_" + ts + ".wav");
                        }
                        else
                        {
                            outFile.getParentDirectory().createDirectory();
                        }

                        juce::WavAudioFormat wavFormat;
                        wavStream = outFile.createOutputStream();
                        if (wavStream)
                        {
                            // 32-bit float WAV for fidelity using AudioFormatWriterOptions builder pattern
                            auto wavStreamPtr = std::unique_ptr<juce::OutputStream>(wavStream.release());
                            auto opts = juce::AudioFormatWriterOptions()
                                .withSampleRate(48000.0)
                                .withNumChannels(2)
                                .withBitsPerSample(32)
                                .withSampleFormat(juce::AudioFormatWriterOptions::SampleFormat::floatingPoint);
                            wavWriter = wavFormat.createWriterFor(wavStreamPtr, opts);
                            if (wavWriter)
                                std::cout << "==> Writing WAV in real-time to: " << outFile.getFullPathName() << std::endl;
                            else
                                std::cerr << "==> Failed to create WAV writer for: " << outFile.getFullPathName() << std::endl;
                        }
                        else
                        {
                            std::cerr << "==> Failed to open WAV output stream: " << outFile.getFullPathName() << std::endl;
                        }
                    }

                    StreamingProcessor processor(pluginPath, 48000.0, 1024, 24);
                    
                    // Calculate real-time pacing: 1024 samples at 48kHz = 21.33ms per chunk
                    const auto chunkDuration = std::chrono::microseconds(static_cast<int64_t>((1024.0 / 48000.0) * 1000000.0));
                    auto lastChunkTime = std::chrono::steady_clock::now();
                    
                    std::cout << "==> Starting processStreaming..." << std::endl;
                    std::cout << "==> Real-time mode: pacing chunks at " << chunkDuration.count() << " microseconds each" << std::endl;
                    
                    bool success = processor.processStreaming(midiFile, presetFile,
                        [chunkQueue, &wavWriter, chunkDuration, &lastChunkTime](const StreamingProcessor::AudioChunk& chunk) {
                            Json::Value chunkJson;
                            chunkJson["chunk_index"] = static_cast<Json::UInt64>(chunk.chunkIndex);
                            chunkJson["total_chunks"] = static_cast<Json::UInt64>(chunk.totalChunks);
                            chunkJson["is_last"] = chunk.isLastChunk;
                            chunkJson["sample_rate"] = 48000;
                            chunkJson["channels"] = 2;
                            chunkJson["samples_per_channel"] = static_cast<Json::UInt64>(chunk.samplesPerChannel);

                            const unsigned char* dataPtr = reinterpret_cast<const unsigned char*>(chunk.data.data());
                            size_t dataSize = chunk.data.size() * sizeof(float);
                            std::string base64Data = drogon::utils::base64Encode(dataPtr, dataSize);
                            chunkJson["audio_data"] = base64Data;

                            Json::StreamWriterBuilder builder;
                            builder["indentation"] = "";
                            std::string jsonStr = Json::writeString(builder, chunkJson) + "\n";
                            
                            // Real-time pacing: wait until it's time to send the next chunk
                            auto now = std::chrono::steady_clock::now();
                            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - lastChunkTime);
                            
                            if (elapsed < chunkDuration)
                            {
                                auto sleepTime = chunkDuration - elapsed;
                                std::this_thread::sleep_for(sleepTime);
                            }
                            lastChunkTime = std::chrono::steady_clock::now();
                            
                            chunkQueue->push(jsonStr);
                            
                            // Write to WAV in real-time if enabled
                            if (wavWriter)
                            {
                                const int channels = 2;
                                const int samples = static_cast<int>(chunk.samplesPerChannel);
                                if (samples > 0)
                                {
                                    juce::AudioBuffer<float> tmp(channels, samples);
                                    for (int s = 0; s < samples; ++s)
                                    {
                                        for (int ch = 0; ch < channels; ++ch)
                                        {
                                            tmp.setSample(ch, s, chunk.data[static_cast<size_t>(s) * channels + static_cast<size_t>(ch)]);
                                        }
                                    }
                                    wavWriter->writeFromAudioSampleBuffer(tmp, 0, samples);
                                }
                            }

                            if (chunk.chunkIndex % 50 == 0) {
                                std::cout << "==> Queued chunk " << chunk.chunkIndex << " (queue size: " << chunkQueue->size() << ")" << std::endl;
                            }
                        }
                    );

                    std::cout << "==> Processing finished. Success: " << success << std::endl;
                }
                catch (const std::exception& e) {
                    std::cerr << "==> Exception in processing thread: " << e.what() << std::endl;
                }
                catch (...) {
                    std::cerr << "==> Unknown exception in processing thread" << std::endl;
                }
                
                chunkQueue->setFinished();
                std::cout << "==> Queue marked as finished" << std::endl;
            });
            processingThread.detach();

            // Create streaming response with callback
            auto resp = drogon::HttpResponse::newStreamResponse(
                [chunkQueue, bufferState, bufferOffset](char* buffer, size_t size) -> size_t {
                    // If we have leftover data from previous call
                    if (*bufferOffset < bufferState->size())
                    {
                        size_t remaining = bufferState->size() - *bufferOffset;
                        size_t copySize = std::min(remaining, size);
                        std::memcpy(buffer, bufferState->data() + *bufferOffset, copySize);
                        *bufferOffset += copySize;
                        
                        if (*bufferOffset >= bufferState->size())
                        {
                            bufferState->clear();
                            *bufferOffset = 0;
                        }
                        
                        return copySize;
                    }
                    
                    // Try to get new chunk - block until available or finished
                    while (true)
                    {
                        std::string chunk;
                        if (chunkQueue->pop(chunk, 100))  // 100ms timeout for each attempt
                        {
                            *bufferState = std::move(chunk);
                            *bufferOffset = 0;
                            
                            size_t copySize = std::min(bufferState->size(), size);
                            std::memcpy(buffer, bufferState->data(), copySize);
                            *bufferOffset = copySize;
                            
                            if (copySize >= bufferState->size())
                            {
                                bufferState->clear();
                                *bufferOffset = 0;
                            }
                            
                            return copySize;
                        }
                        
                        // Check if finished
                        if (chunkQueue->isFinished())
                        {
                            std::cout << "Stream finished, closing" << std::endl;
                            return 0;  // Signal end of stream
                        }
                        
                        // Continue waiting loop
                    }
                },
                "application/x-ndjson"
            );
            
            resp->addHeader("Cache-Control", "no-cache");
            resp->addHeader("X-Accel-Buffering", "no");
            
            callback(resp);
            std::cout << "Response streaming started\n" << std::endl;
        },
        {drogon::Post});

    std::cout << "\n========================================" << std::endl;
    std::cout << "Drogon audio streaming server running" << std::endl;
    std::cout << "Listening on: http://0.0.0.0:8080" << std::endl;
    std::cout << "Endpoint: POST /generate" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    drogon::app().run();

    return 0;
}