import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GstBase, GObject, GLib

import onnxruntime as ort
import numpy as np
from PIL import Image

import os
import traceback
Gst.init(None)

class GstMadeira(GstBase.BaseTransform):
    __gstmetadata__ = ('madeira',
                       'SAM2 Image Segmentation Element', 
                       'Filter/Effect/Video',
                       'Performs image segmentation using SAM2 ONNX models')

    __gst_plugin_name__ = "madeira"
    __gst_plugin_description__ = "Performs image segmentation using SAM2 ONNX models"
    __gst_plugin_version__ = "0.1"
    __gst_plugin_license__ = "MIT"

    __gproperties__ = {
        "encoder-path": (GObject.TYPE_STRING, "Encoder ONNX Path",
                         "Path to the SAM2 encoder ONNX model file.",
                         "sam2_hiera_tiny.encoder.onnx",
                         GObject.PARAM_READWRITE | GObject.PARAM_STATIC_STRINGS),
        "decoder-path": (GObject.TYPE_STRING, "Decoder ONNX Path",
                         "Path to the SAM2 decoder ONNX model file.",
                         "sam2_hiera_tiny.decoder.onnx",
                         GObject.PARAM_READWRITE | GObject.PARAM_STATIC_STRINGS),
        "target-size": (GObject.TYPE_INT, "Target Image Size",
                        "The square dimension (e.g., 1024) to which the image will be "
                        "resized before being fed into the SAM2 models.",
                        128, 4096, 1024,
                        GObject.PARAM_READWRITE | GObject.PARAM_STATIC_STRINGS),
        "mask-threshold": (GObject.TYPE_FLOAT, "Mask Threshold",
                           "Threshold (0.0-1.0) applied to the decoder's output logits "
                           "to binarize the segmentation mask. Higher values result in "
                           "stricter masks.",
                           0.0, 1.0, 0.3,
                           GObject.PARAM_READWRITE | GObject.PARAM_STATIC_STRINGS),
        "grid-stride": (GObject.TYPE_INT, "Grid Stride",
                        "Stride (in pixels relative to original image size) for "
                        "generating the grid of prompt points. Smaller strides "
                        "generate more points and potentially more detailed segmentation "
                        "but are slower.",
                        1, 100, 32,
                        GObject.PARAM_READWRITE | GObject.PARAM_STATIC_STRINGS),
    }

    _caps_str = "video/x-raw, format=(string){RGB, RGBA, BGRx, BGRA}, width=(int)[1, MAX], height=(int)[1, MAX], framerate=(fraction)[0/1, MAX]"
    _caps = Gst.Caps.from_string(_caps_str)

    _sink_template = Gst.PadTemplate.new(
        "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, _caps
    )
    _src_template = Gst.PadTemplate.new(
        "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, _caps
    )

    __gsttemplates__ = (_sink_template, _src_template)

    def __init__(self):
        super().__init__()
        self.encoder_path = "sam2_hiera_tiny.encoder.onnx"
        self.decoder_path = "sam2_hiera_tiny.decoder.onnx"
        self.target_size = 1024
        self.mask_threshold = 0.3
        self.grid_stride = 32

        self.enc = None
        self.dec = None

        self.image_width = 0
        self.image_height = 0
        self.input_format = None
        self.bytes_per_pixel = 0

        self.set_in_place(True)

    def do_get_property(self, prop):
        if prop.name == "encoder-path":
            return self.encoder_path
        elif prop.name == "decoder-path":
            return self.decoder_path
        elif prop.name == "target-size":
            return self.target_size
        elif prop.name == "mask-threshold":
            return self.mask_threshold
        elif prop.name == "grid-stride":
            return self.grid_stride
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_set_property(self, prop, value):
        if prop.name == "encoder-path":
            self.encoder_path = value
        elif prop.name == "decoder-path":
            self.decoder_path = value
        elif prop.name == "target-size":
            self.target_size = value
        elif prop.name == "mask-threshold":
            self.mask_threshold = value
        elif prop.name == "grid-stride":
            self.grid_stride = value
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_start(self):
        Gst.debug(f"Starting GstMadeira: Encoder={self.encoder_path}, Decoder={self.decoder_path}")
        try:
            if not os.path.exists(self.encoder_path):
                Gst.error(f"Encoder model not found: {self.encoder_path}")
                return False
            if not os.path.exists(self.decoder_path):
                Gst.error(f"Decoder model not found: {self.decoder_path}")
                return False

            self.enc = ort.InferenceSession(self.encoder_path, providers=["CPUExecutionProvider"])
            self.dec = ort.InferenceSession(self.decoder_path, providers=["CPUExecutionProvider"])
            Gst.info("SAM2 ONNX models loaded successfully.")
            return True
        except Exception as e:
            Gst.error(f"Failed to load ONNX models: {e}")
            traceback.print_exc()
            return False

    def do_stop(self):
        self.enc = None
        self.dec = None
        Gst.info("SAM2 ONNX models unloaded.")
        return True

    def do_set_caps(self, incaps, outcaps):
        Gst.debug(f"Setting caps: Input={incaps}, Output={outcaps}")
        structure = incaps.get_structure(0)
        self.image_width = structure.get_value("width")
        self.image_height = structure.get_value("height")
        self.input_format = structure.get_value("format")

        if self.input_format in ["RGB", "BGRx"]:
            self.bytes_per_pixel = 3
        elif self.input_format in ["RGBA", "BGRA"]:
            self.bytes_per_pixel = 4
        else:
            Gst.error(f"Unsupported input format: {self.input_format}")
            return False

        Gst.info(f"Image dimensions: {self.image_width}x{self.image_height}, "
                 f"Format: {self.input_format}, Bytes per pixel: {self.bytes_per_pixel}")
        return True

    def _preprocess_image(self, np_image, target_size):
        processed_image_np = np_image
        if self.input_format == "BGRx":
            processed_image_np = np_image[:, :, :3][:, :, ::-1]
        elif self.input_format == "BGRA":
            processed_image_np = np_image[:, :, :3][:, :, ::-1]
        elif self.input_format == "RGBA":
            processed_image_np = np_image[:, :, :3]
        elif self.input_format == "RGB":
            pass

        img_pil = Image.fromarray(processed_image_np.astype(np.uint8))
        img_resized_pil = img_pil.resize((target_size, target_size), Image.BILINEAR)
        
        im_arr = np.array(img_resized_pil).astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
        image_np_normalized = (im_arr - mean) / std
        
        image_tensor = image_np_normalized.transpose(2, 0, 1)[None]
        
        return image_tensor.astype(np.float32), img_resized_pil, (self.image_width, self.image_height)

    def _make_grid(self, h_feat, w_feat, stride):
        ys = np.arange(stride // 2, h_feat * 4, stride)
        xs = np.arange(stride // 2, w_feat * 4, stride)
        
        grid = np.stack(np.meshgrid(xs, ys, indexing='xy'), axis=-1).reshape(-1, 2)
        grid = grid / np.array([w_feat * 4, h_feat * 4]) 
        return grid

    def _overlay_segmentation_mask(self, original_resized_pil_image, combined_mask_np):
        mask_img_pil = Image.fromarray((combined_mask_np * 255).astype(np.uint8))
        mask_img_pil = mask_img_pil.resize(original_resized_pil_image.size, Image.NEAREST)

        overlay_pil = Image.new("RGBA", original_resized_pil_image.size)
        overlay_pil.paste((255, 0, 0, 128), mask=mask_img_pil)

        result_img_pil = original_resized_pil_image.convert("RGBA")
        result_img_pil.alpha_composite(overlay_pil)
        return result_img_pil

    def do_transform_ip(self, buffer):
        if not self.enc or not self.dec:
            Gst.error("ONNX models not loaded. Cannot process buffer. Please check plugin initialization.")
            return Gst.FlowReturn.ERROR

        try:
            result, mapinfo = buffer.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE)
            if not result:
                Gst.error("Failed to map buffer. Skipping frame.")
                return Gst.FlowReturn.ERROR

            np_image = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(
                self.image_height, self.image_width, self.bytes_per_pixel
            )

            image_tensor, resized_img_pil, (orig_w, orig_h) = self._preprocess_image(np_image, self.target_size)
            
            enc_out = self.enc.run(None, {"image": image_tensor})
            h0, h1, image_embeddings = enc_out
            Hc, Wc = image_embeddings.shape[2], image_embeddings.shape[3]
            grid = self._make_grid(Hc, Wc, self.grid_stride)

            all_masks = []
            for (x, y) in grid:
                dec_inputs = {
                    "image_embed": image_embeddings,
                    "high_res_feats_0": h0,
                    "high_res_feats_1": h1,
                    "point_coords": np.array([[[x, y]]], dtype=np.float32),
                    "point_labels": np.array([[1]], dtype=np.float32),
                    "has_mask_input": np.array([0], dtype=np.float32),
                    "mask_input": np.zeros((1, 1, Hc * 4, Wc * 4), dtype=np.float32),
                }
                dec_out = self.dec.run(None, dec_inputs)
                mask_logits = dec_out[0]
                mask = (mask_logits[0, 0] > self.mask_threshold)
                all_masks.append(mask)

            if all_masks:
                combined_mask = np.any(np.stack(all_masks, axis=0), axis=0)
            else:
                combined_mask = np.zeros((self.target_size, self.target_size), dtype=bool)

            res_img_pil = self._overlay_segmentation_mask(resized_img_pil, combined_mask)
            result_np_image = np.array(res_img_pil)
            result_img_pil = Image.fromarray(result_np_image)
            result_img_pil = result_img_pil.resize((self.image_width, self.image_height), Image.BILINEAR)
            result_np_image = np.array(result_img_pil)

            if self.input_format == "BGRx":
                if result_np_image.shape[2] == 3:
                    result_np_image = result_np_image[:, :, ::-1]
                    alpha_channel = np.full((self.image_height, self.image_width, 1), 255, dtype=np.uint8)
                    result_np_image = np.concatenate((result_np_image, alpha_channel), axis=2)
                elif result_np_image.shape[2] == 4:
                    result_np_image = result_np_image[:, :, [2,1,0,3]]
            elif self.input_format == "BGRA":
                if result_np_image.shape[2] == 3:
                    alpha_channel = np.full((self.image_height, self.image_width, 1), 255, dtype=np.uint8)
                    result_np_image = np.concatenate((result_np_image[:, :, ::-1], alpha_channel), axis=2)
                elif result_np_image.shape[2] == 4:
                    result_np_image = result_np_image[:, :, [2,1,0,3]]
            elif self.input_format == "RGB":
                if result_np_image.shape[2] == 4:
                    result_np_image = result_np_image[:, :, :3]
            elif self.input_format == "RGBA":
                pass

            expected_buffer_size = self.image_width * self.image_height * self.bytes_per_pixel
            if result_np_image.nbytes != expected_buffer_size:
                Gst.warning(f"Output image byte size ({result_np_image.nbytes}) does not match expected buffer size ({expected_buffer_size}). "
                            f"This could lead to buffer issues. Input format was {self.input_format}, output shape {result_np_image.shape}")

            mapinfo.data[:] = result_np_image.tobytes()

            buffer.unmap(mapinfo)
            return Gst.FlowReturn.OK

        except Exception as e:
            Gst.error(f"Error during SAM2 transformation: {e}")
            traceback.print_exc()
            return Gst.FlowReturn.ERROR

GObject.type_register(GstMadeira)

__gstelementfactory__ = ("madeira", Gst.Rank.NONE, GstMadeira)
