Madeira 
======

A GStreamer plugin for segmenting everything using Segment-Anything v2. Choose between rust or python, each utilize ONNX runtime.


## Models

See [jquadrino/segment-anything-v2-onnx](https://huggingface.co/jquadrino/segment-anything-v2-onnx)


## Installation

### Prerequisites

- Rust or Python
- GStreamer 1.20+
- ONNX Runtime
- SAMv2 ONNX model files


## Usage


```bash
export GST_PLUGIN_PATH=$(pwd)/madeira/plugins

# or

cargo build --release
export GST_PLUGIN_PATH=$(pwd)/madeira/target/release

# and

gst-launch-1.0 \
    filesrc location=car-detection.mp4 ! \
    decodebin ! \
    videoconvert ! \
    video/x-raw,format=BGRx ! \
    videorate ! \
    video/x-raw,framerate=1/1 ! \
    madeira mask-threshold=0.4 grid-stride=64 ! \
    videoconvert ! \
    x264enc tune=zerolatency ! \
    mp4mux ! filesink location=output.mp4
```
