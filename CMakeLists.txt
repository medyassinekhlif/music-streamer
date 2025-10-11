cmake_minimum_required(VERSION 3.15)

project(audio_streaming_server VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find vcpkg packages
find_package(Drogon CONFIG REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)

# Add JUCE from submodule
add_subdirectory(JUCE)

# Create the executable
add_executable(audio_streaming_server
    main.cpp
)

# Compiler definitions
target_compile_definitions(audio_streaming_server
    PRIVATE
    JUCE_APPLICATION_NAME_STRING="AudioStreamingServer"
    JUCE_APPLICATION_VERSION_STRING="1.0.0"
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_PLUGINHOST_VST3=1
    JUCE_PLUGINHOST_AU=0
    JUCE_PLUGINHOST_VST=0
    JUCE_STANDALONE_APPLICATION=1
)

# Link libraries
target_link_libraries(audio_streaming_server
    PRIVATE
        Drogon::Drogon
        jsoncpp_lib
        juce::juce_core
        juce::juce_audio_basics
        juce::juce_audio_formats
        juce::juce_audio_processors
        juce::juce_audio_devices
        juce::juce_audio_utils
        juce::juce_events
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# Define binary output directory
set_target_properties(audio_streaming_server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

message(STATUS "✅ Drogon + JUCE Audio Streaming Server configured successfully.")