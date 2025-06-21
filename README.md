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


### Install

```bash
pip install -r requirements.txt

# download and place ONNX model here

export GST_PLUGIN_PATH=$PWD:$GST_PLUGIN_PATH
```

## Usage


```bash
gst-launch-1.0 \
    videotestsrc ! \
    videoconvert ! \
    madeira model-path=/path/to/samv2.onnx ! \
    videoconvert ! \
    autovideosink
```


```bash
gst-launch-1.0 \
    filesrc location=input.mp4 ! \
    decodebin ! \
    videoconvert ! \
    madeira model-path=/path/to/samv2.onnx name=segmenter ! \
    videoconvert ! \
    x264enc ! \
    mp4mux ! \
    filesink location=output.mp4 \
    \
    segmenter.detection ! \
    filesink location=detections.json
```
