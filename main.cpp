#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_events/juce_events.h>
#include <juce_core/juce_core.h>
#include <iostream>
#include <cmath>
#include <condition_variable>

static size_t secondsToSamples(double sec, double sr) {
    return static_cast<size_t>(sec * sr);
}

class SingleJobProcessor {
public:
    SingleJobProcessor(const juce::String& pluginPathIn, double sr = 48000.0,
                       int blockSizeIn = 1024, int bitDepthIn = 24)
        : pluginPath(pluginPathIn),
          sampleRate(sr),
          blockSize(blockSizeIn),
          bitDepth(bitDepthIn)
    {
        audioFormatManager.registerBasicFormats();
        audioPluginFormatManager.addDefaultFormats();
    }

    bool process(const juce::File& midiFile, const juce::File& presetFile, const juce::File& outputFile)
    {
        std::cout << "Processing: " << midiFile.getFileName() << " with " << presetFile.getFileName() << std::endl;

        auto plugin = createPluginInstance();
        if (!plugin) {
            std::cerr << "Error: plugin instance creation failed\n";
            return false;
        }

        std::cout << "Plugin loaded: " << plugin->getName() << std::endl;

        size_t totalInputLength = 0;
        auto midiFileData = readMidiFile(midiFile, sampleRate, totalInputLength);
        if (midiFileData.getNumTracks() == 0) {
            std::cerr << "Error: MIDI contains no tracks\n";
            return false;
        }

        std::cout << "MIDI tracks: " << midiFileData.getNumTracks() << std::endl;
        std::cout << "Total MIDI length: " << (totalInputLength / sampleRate) << " seconds" << std::endl;

        // --- Bus setup ---
        {
            juce::AudioPluginInstance::BusesLayout layout = plugin->getBusesLayout();
            if (layout.outputBuses.size() == 0)
                layout.outputBuses.add(juce::AudioChannelSet::stereo());
            else
                layout.outputBuses.getReference(0) = juce::AudioChannelSet::stereo();

            if (!plugin->setBusesLayout(layout)) {
                std::cerr << "Warning: could not set stereo bus layout, proceeding with default layout\n";
            }
        }

        unsigned int totalNumOutputChannels = static_cast<unsigned int>(plugin->getTotalNumOutputChannels());
        std::cout << "Output channels: " << totalNumOutputChannels << std::endl;

        if (presetFile.existsAsFile()) {
            if (!loadPreset(*plugin, presetFile)) {
                std::cerr << "Warning: preset load failed or skipped\n";
            }
        }

        plugin->prepareToPlay(sampleRate, blockSize);
        int latency = plugin->getLatencySamples();
        std::cout << "Plugin latency: " << latency << " samples" << std::endl;

        size_t tailSamples = secondsToSamples(6.0, sampleRate);

        if (outputFile.exists())
            outputFile.deleteFile();

        auto outputStream = outputFile.createOutputStream();
        if (!outputStream) {
            std::cerr << "Error creating output stream\n";
            return false;
        }

        juce::WavAudioFormat wavFormat;

        std::unique_ptr<juce::AudioFormatWriter> writer(
            wavFormat.createWriterFor(outputStream.release(),
                                      sampleRate,
                                      totalNumOutputChannels,
                                      bitDepth,
                                      juce::StringPairArray(),
                                      0));

        if (!writer) {
            std::cerr << "Error creating writer\n";
            return false;
        }

        juce::AudioBuffer<float> buffer(static_cast<int>(totalNumOutputChannels), blockSize);
        juce::MidiBuffer midiBuffer;
        size_t sampleIndex = 0;
        int midiEventCount = 0;

        std::cout << "Starting render...\n";

        while (sampleIndex < totalInputLength + static_cast<size_t>(latency) + tailSamples) {
            buffer.clear();
            midiBuffer.clear();

            for (int t = 0; t < midiFileData.getNumTracks(); ++t) {
                auto* track = midiFileData.getTrack(t);
                for (auto& meh : *track) {
                    auto msgTsSamples = secondsToSamples(meh->message.getTimeStamp(), sampleRate);
                    if (msgTsSamples >= sampleIndex && msgTsSamples < sampleIndex + static_cast<size_t>(blockSize)) {
                        int offsetInBlock = static_cast<int>(msgTsSamples - sampleIndex);
                        midiBuffer.addEvent(meh->message, offsetInBlock);
                        ++midiEventCount;
                    }
                }
            }

            plugin->processBlock(buffer, midiBuffer);

            const int fadeLen = std::min(512, blockSize);
            if (sampleIndex + static_cast<size_t>(blockSize) >= totalInputLength + static_cast<size_t>(latency) + tailSamples - static_cast<size_t>(blockSize)) {
                int start = std::max(0, blockSize - fadeLen);
                for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
                    buffer.applyGainRamp(ch, start, fadeLen, 1.0f, 0.0f);
            }

            if (!writer->writeFromAudioSampleBuffer(buffer, 0, blockSize)) {
                std::cerr << "Error writing audio block\n";
                return false;
            }

            sampleIndex += static_cast<size_t>(blockSize);

            if ((sampleIndex / static_cast<size_t>(sampleRate * 5)) != ((sampleIndex - blockSize) / static_cast<size_t>(sampleRate * 5))) {
                std::cout << "Progress: " << (sampleIndex / sampleRate) << "s\n";
            }
        }

        std::cout << "MIDI events processed: " << midiEventCount << std::endl;
        std::cout << "Output saved to: " << outputFile.getFullPathName() << std::endl;

        plugin->releaseResources();
        return true;
    }

private:
    juce::String pluginPath;
    double sampleRate;
    int blockSize;
    int bitDepth;
    juce::AudioFormatManager audioFormatManager;
    juce::AudioPluginFormatManager audioPluginFormatManager;

    std::unique_ptr<juce::AudioPluginInstance> createPluginInstance()
    {
        juce::PluginDescription desc;
        desc.pluginFormatName = "VST3";
        desc.fileOrIdentifier = pluginPath;
        desc.name = juce::File(pluginPath).getFileNameWithoutExtension();

        std::unique_ptr<juce::AudioPluginInstance> pluginInstance;
        std::mutex mtx;
        std::condition_variable cv;
        bool done = false;
        juce::String createError;

        audioPluginFormatManager.createPluginInstanceAsync(
            desc,
            sampleRate,
            blockSize,
            [&pluginInstance, &mtx, &cv, &done, &createError]
            (std::unique_ptr<juce::AudioPluginInstance> instance, const juce::String& error)
            {
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    if (instance)
                        pluginInstance = std::move(instance);
                    else
                        createError = error;
                    done = true;
                }
                cv.notify_one();
            }
        );

        std::unique_lock<std::mutex> lk(mtx);
        if (!cv.wait_for(lk, std::chrono::seconds(10), [&done]() { return done; })) {
            std::cerr << "Timeout creating plugin instance\n";
            return nullptr;
        }

        if (!pluginInstance) {
            std::cerr << "Failed to create plugin instance: " << createError << std::endl;
            return nullptr;
        }

        return pluginInstance;
    }

    bool loadPreset(juce::AudioPluginInstance& plugin, const juce::File& presetFile)
    {
        juce::MemoryBlock presetData;
        auto in = presetFile.createInputStream();
        if (!in || !in->readIntoMemoryBlock(presetData)) {
            std::cerr << "Error reading preset\n";
            return false;
        }

        plugin.setStateInformation(presetData.getData(), static_cast<int>(presetData.getSize()));
        std::cout << "Preset loaded successfully\n";
        return true;
    }

    juce::MidiFile readMidiFile(const juce::File& file, double sr, size_t& lengthOut)
    {
        juce::MidiFile midi;
        lengthOut = 0;

        auto in = file.createInputStream();
        if (!in || !midi.readFrom(*in, true)) {
            std::cerr << "Error reading MIDI file\n";
            return midi;
        }

        midi.convertTimestampTicksToSeconds();

        for (int t = 0; t < midi.getNumTracks(); ++t) {
            auto* track = midi.getTrack(t);
            for (auto& meh : *track) {
                const size_t ts = secondsToSamples(meh->message.getTimeStamp(), sr);
                if (ts > lengthOut) lengthOut = ts;
            }
        }

        return midi;
    }
};

// ---------- JUCE Application ----------
class BatchProcessorApp : public juce::JUCEApplicationBase {
public:
    const juce::String getApplicationName() override { return "main"; }
    const juce::String getApplicationVersion() override { return "1.2"; }
    bool moreThanOneInstanceAllowed() override { return true; }

    void initialise(const juce::String&) override {
        juce::String pluginPath = "C:\\Program Files\\Common Files\\VST3\\Spitfire Audio\\LABS.vst3";
        juce::File midiFile("C:\\src\\forge\\midi\\2.mid");
        juce::File presetFile("C:\\src\\forge\\presets\\glassgrandpiano.vstpreset");
        juce::File outputFile("C:\\src\\forge\\output\\2_softpiano_final.wav");

        outputFile.getParentDirectory().createDirectory();
        SingleJobProcessor processor(pluginPath, 48000.0, 1024, 24);
        bool success = processor.process(midiFile, presetFile, outputFile);
        setApplicationReturnValue(success ? 0 : 1);
        quit();
    }

    void shutdown() override {}
    void anotherInstanceStarted(const juce::String&) override {}
    void suspended() override {}
    void resumed() override {}
    void systemRequestedQuit() override { quit(); }
    void unhandledException(const std::exception*, const juce::String&, int) override {}
};

START_JUCE_APPLICATION(BatchProcessorApp)
