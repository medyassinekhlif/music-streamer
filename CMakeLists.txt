cmake_minimum_required(VERSION 3.15)

project(main VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add JUCE from submodule
add_subdirectory(JUCE)

# Create the console app
juce_add_console_app(main
    COMPANY_NAME "YourCompany"
    PRODUCT_NAME "main"
)

# Add main source file
target_sources(main
    PRIVATE
    main.cpp
)

# Compiler definitions
target_compile_definitions(main
    PRIVATE
    JUCE_APPLICATION_NAME_STRING="$<TARGET_PROPERTY:main,JUCE_PRODUCT_NAME>"
    JUCE_APPLICATION_VERSION_STRING="$<TARGET_PROPERTY:main,JUCE_VERSION>"
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_PLUGINHOST_VST3=1   # Enable VST3 hosting
    JUCE_PLUGINHOST_AU=0     # Disable AU
    JUCE_PLUGINHOST_VST=0    # Disable VST2 (deprecated)
)

# Link JUCE modules
target_link_libraries(main
    PRIVATE
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
set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

message(STATUS "✅ JUCE VST3 Batch MIDI-to-WAV Project configured successfully.")
