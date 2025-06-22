Madeira 
======

A GStreamer plugin for video intelligence using Segment-Anything v2 with ONNX runtime in python.


## Models

See [jquadrino/segment-anything-v2-onnx](https://huggingface.co/jquadrino/segment-anything-v2-onnx)


## Installation

### Prerequisites

- Python 3.7+
- GStreamer 1.20+
- ONNX Runtime
- SAMv2 ONNX model file


## Usage


```bash
export GST_PLUGIN_PATH=$(pwd)/madeira/plugins

gst-launch-1.0 \
    filesrc location=car-detection.mp4 !
    decodebin !
    videoconvert !
    video/x-raw,format=BGRx !
    videorate !
    video/x-raw,framerate=1/1 !
    madeira mask-threshold=0.4 grid-stride=100 !
    videoconvert !
    x264enc tune=zerolatency ! 
    mp4mux ! filesink location=output.mp4
```
