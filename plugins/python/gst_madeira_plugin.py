import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GstBase, GObject, GLib

import numpy as np
from PIL import Image
import torch

import os
import traceback

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

Gst.init(None)

class GstMadeira(GstBase.BaseTransform):
    __gstmetadata__ = ('madeira',
                       'SAM2 Image Segmentation Element', 
                       'Filter/Effect/Video',
                       'Performs image segmentation using SAM2 models')

    __gst_plugin_name__ = "madeira"
    __gst_plugin_description__ = "Performs image segmentation using SAM2 models"
    __gst_plugin_version__ = "0.2"
    __gst_plugin_license__ = "MIT"

    __gproperties__ = {
        "checkpoint-path": (GObject.TYPE_STRING, "SAM2 Checkpoint Path",
                           "Path to the SAM2 checkpoint file (.pt).",
                           "sam2.1_hiera_large.pt",
                           GObject.PARAM_READWRITE | GObject.PARAM_STATIC_STRINGS),
        "config-path": (GObject.TYPE_STRING, "SAM2 Config Path",
                       "Path to the SAM2 config file (.yaml).",
                       "configs/sam2.1/sam2.1_hiera_l.yaml",
                       GObject.PARAM_READWRITE | GObject.PARAM_STATIC_STRINGS),
        "device": (GObject.TYPE_STRING, "Device",
                  "Device to run inference on (cpu, cuda, mps).",
                  "cpu",
                  GObject.PARAM_READWRITE | GObject.PARAM_STATIC_STRINGS),
        "points-per-side": (GObject.TYPE_INT, "Points Per Side",
                           "Number of points to sample along each side of the image.",
                           1, 64, 32,
                           GObject.PARAM_READWRITE | GObject.PARAM_STATIC_STRINGS),
        "points-per-batch": (GObject.TYPE_INT, "Points Per Batch",
                            "Number of points to process in each batch.",
                            1, 1024, 64,
                            GObject.PARAM_READWRITE | GObject.PARAM_STATIC_STRINGS),
        "pred-iou-thresh": (GObject.TYPE_FLOAT, "Predicted IoU Threshold",
                           "Threshold for predicted IoU to filter masks.",
                           0.0, 1.0, 0.8,
                           GObject.PARAM_READWRITE | GObject.PARAM_STATIC_STRINGS),
        "stability-score-thresh": (GObject.TYPE_FLOAT, "Stability Score Threshold",
                                  "Threshold for stability score to filter masks.",
                                  0.0, 1.0, 0.95,
                                  GObject.PARAM_READWRITE | GObject.PARAM_STATIC_STRINGS),
        "box-nms-thresh": (GObject.TYPE_FLOAT, "Box NMS Threshold",
                          "IoU threshold for box-based NMS filtering.",
                          0.0, 1.0, 0.7,
                          GObject.PARAM_READWRITE | GObject.PARAM_STATIC_STRINGS),
        "crop-n-layers": (GObject.TYPE_INT, "Crop N Layers",
                         "Number of layers for crop-based processing.",
                         0, 10, 0,
                         GObject.PARAM_READWRITE | GObject.PARAM_STATIC_STRINGS),
        "crop-nms-thresh": (GObject.TYPE_FLOAT, "Crop NMS Threshold",
                           "IoU threshold for crop-based NMS filtering.",
                           0.0, 1.0, 0.7,
                           GObject.PARAM_READWRITE | GObject.PARAM_STATIC_STRINGS),
        "crop-overlap-ratio": (GObject.TYPE_FLOAT, "Crop Overlap Ratio",
                              "Overlap ratio for crop-based processing.",
                              0.0, 1.0, 512.0/1500.0,
                              GObject.PARAM_READWRITE | GObject.PARAM_STATIC_STRINGS),
        "crop-n-points-downscale-factor": (GObject.TYPE_INT, "Crop Points Downscale Factor",
                                          "Factor to downscale points in crops.",
                                          1, 10, 1,
                                          GObject.PARAM_READWRITE | GObject.PARAM_STATIC_STRINGS),
        "min-mask-region-area": (GObject.TYPE_INT, "Min Mask Region Area",
                                "Minimum area (in pixels) for mask regions.",
                                0, 100000, 0,
                                GObject.PARAM_READWRITE | GObject.PARAM_STATIC_STRINGS),
        "use-m2m": (GObject.TYPE_BOOLEAN, "Use Mask-to-Mask",
                   "Whether to use mask-to-mask processing.",
                   False,
                   GObject.PARAM_READWRITE | GObject.PARAM_STATIC_STRINGS),
    }

    _GstMadeira__gst_caps = Gst.Caps.from_string(
        "video/x-raw, format=(string){RGB, RGBA, BGRx, BGRA}, width=(int)[1, MAX], height=(int)[1, MAX], framerate=(fraction)[0/1, MAX]"
    )
    _GstMadeira__gst_sink_template = Gst.PadTemplate.new(
        "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, _GstMadeira__gst_caps
    )
    _GstMadeira__gst_src_template = Gst.PadTemplate.new(
        "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, _GstMadeira__gst_caps
    )

    __gst_pad_templates__ = (_GstMadeira__gst_sink_template,
                             _GstMadeira__gst_src_template,)

    def __init__(self):
        super().__init__()
        self.checkpoint_path = "sam2.1_hiera_large.pt"
        self.config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.device = "cpu"
        
        self.points_per_side = 32
        self.points_per_batch = 64
        self.pred_iou_thresh = 0.8
        self.stability_score_thresh = 0.95
        self.box_nms_thresh = 0.7
        self.crop_n_layers = 0
        self.crop_nms_thresh = 0.7
        self.crop_overlap_ratio = 512.0/1500.0
        self.crop_n_points_downscale_factor = 1
        self.min_mask_region_area = 0
        self.use_m2m = False

        self.sam2_model = None
        self.mask_generator = None

        self.image_width = 0
        self.image_height = 0
        self.input_format = None
        self.bytes_per_pixel = 0

        self.set_in_place(True)

    def do_get_property(self, prop):
        prop_map = {
            "checkpoint-path": self.checkpoint_path,
            "config-path": self.config_path,
            "device": self.device,
            "points-per-side": self.points_per_side,
            "points-per-batch": self.points_per_batch,
            "pred-iou-thresh": self.pred_iou_thresh,
            "stability-score-thresh": self.stability_score_thresh,
            "box-nms-thresh": self.box_nms_thresh,
            "crop-n-layers": self.crop_n_layers,
            "crop-nms-thresh": self.crop_nms_thresh,
            "crop-overlap-ratio": self.crop_overlap_ratio,
            "crop-n-points-downscale-factor": self.crop_n_points_downscale_factor,
            "min-mask-region-area": self.min_mask_region_area,
            "use-m2m": self.use_m2m,
        }
        
        if prop.name in prop_map:
            return prop_map[prop.name]
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_set_property(self, prop, value):
        if prop.name == "checkpoint-path":
            self.checkpoint_path = value
        elif prop.name == "config-path":
            self.config_path = value
        elif prop.name == "device":
            self.device = value
        elif prop.name == "points-per-side":
            self.points_per_side = value
        elif prop.name == "points-per-batch":
            self.points_per_batch = value
        elif prop.name == "pred-iou-thresh":
            self.pred_iou_thresh = value
        elif prop.name == "stability-score-thresh":
            self.stability_score_thresh = value
        elif prop.name == "box-nms-thresh":
            self.box_nms_thresh = value
        elif prop.name == "crop-n-layers":
            self.crop_n_layers = value
        elif prop.name == "crop-nms-thresh":
            self.crop_nms_thresh = value
        elif prop.name == "crop-overlap-ratio":
            self.crop_overlap_ratio = value
        elif prop.name == "crop-n-points-downscale-factor":
            self.crop_n_points_downscale_factor = value
        elif prop.name == "min-mask-region-area":
            self.min_mask_region_area = value
        elif prop.name == "use-m2m":
            self.use_m2m = value
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_start(self):
        Gst.debug(f"Starting GstMadeira: Checkpoint={self.checkpoint_path}, Config={self.config_path}")
        try:
            if not os.path.exists(self.checkpoint_path):
                Gst.error(f"Checkpoint file not found: {self.checkpoint_path}")
                return False
            if not os.path.exists(self.config_path):
                Gst.error(f"Config file not found: {self.config_path}")
                return False

            self.sam2_model = build_sam2(
                self.config_path, 
                self.checkpoint_path, 
                device=self.device, 
                apply_postprocessing=False
            )
            
            self.mask_generator = SAM2AutomaticMaskGenerator(
                self.sam2_model,
                points_per_side=self.points_per_side,
                points_per_batch=self.points_per_batch,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                box_nms_thresh=self.box_nms_thresh,
                crop_n_layers=self.crop_n_layers,
                crop_nms_thresh=self.crop_nms_thresh,
                crop_overlap_ratio=self.crop_overlap_ratio,
                crop_n_points_downscale_factor=self.crop_n_points_downscale_factor,
                min_mask_region_area=self.min_mask_region_area,
                use_m2m=self.use_m2m,
            )
            
            Gst.info("SAM2 model loaded successfully.")
            return True
        except Exception as e:
            Gst.error(f"Failed to load SAM2 model: {e}")
            traceback.print_exc()
            return False

    def do_stop(self):
        self.sam2_model = None
        self.mask_generator = None
        Gst.info("SAM2 model unloaded.")
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

    def _preprocess_image(self, np_image):
        processed_image_np = np_image
        
        if self.input_format == "BGRx":
            processed_image_np = np_image[:, :, :3][:, :, ::-1]
        elif self.input_format == "BGRA":
            processed_image_np = np_image[:, :, :3][:, :, ::-1]
        elif self.input_format == "RGBA":
            processed_image_np = np_image[:, :, :3]
        elif self.input_format == "RGB":
            pass
        
        return processed_image_np.astype(np.uint8)

    def _create_combined_mask(self, masks):
        if not masks:
            return np.zeros((self.image_height, self.image_width), dtype=bool)
        
        combined_mask = masks[0]['segmentation'].astype(bool)
        for mask_data in masks[1:]:
            combined_mask = np.logical_or(combined_mask, mask_data['segmentation'])
        
        return combined_mask

    def _overlay_segmentation_mask(self, original_image, combined_mask):
        overlay = np.zeros((*combined_mask.shape, 4), dtype=np.uint8)
        overlay[combined_mask] = [255, 0, 0, 128]
        
        if original_image.shape[2] == 3:
            original_rgba = np.concatenate([
                original_image, 
                np.full((original_image.shape[0], original_image.shape[1], 1), 255, dtype=np.uint8)
            ], axis=2)
        else:
            original_rgba = original_image.copy()
        
        mask_alpha = overlay[:, :, 3:4] / 255.0
        result = original_rgba.astype(np.float32)
        result[:, :, :3] = (1 - mask_alpha) * result[:, :, :3] + mask_alpha * overlay[:, :, :3]
        
        return result.astype(np.uint8)

    def _convert_output_format(self, result_image):
        if self.input_format == "BGRx":
            if result_image.shape[2] == 4:
                result_image = result_image[:, :, [2, 1, 0, 3]]
            else:
                result_image = result_image[:, :, ::-1]
                alpha_channel = np.full((self.image_height, self.image_width, 1), 255, dtype=np.uint8)
                result_image = np.concatenate((result_image, alpha_channel), axis=2)
        elif self.input_format == "BGRA":
            if result_image.shape[2] == 4:
                result_image = result_image[:, :, [2, 1, 0, 3]]
            else:
                result_image = result_image[:, :, ::-1]
                alpha_channel = np.full((self.image_height, self.image_width, 1), 255, dtype=np.uint8)
                result_image = np.concatenate((result_image, alpha_channel), axis=2)
        elif self.input_format == "RGB":
            if result_image.shape[2] == 4:
                result_image = result_image[:, :, :3]
        elif self.input_format == "RGBA":
            pass
        
        return result_image

    def do_transform_ip(self, buffer):
        if not self.mask_generator:
            Gst.error("SAM2 model not loaded. Cannot process buffer.")
            return Gst.FlowReturn.ERROR

        try:
            result, mapinfo = buffer.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE)
            if not result:
                Gst.error("Failed to map buffer. Skipping frame.")
                return Gst.FlowReturn.ERROR

            np_image = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(
                self.image_height, self.image_width, self.bytes_per_pixel
            )

            rgb_image = self._preprocess_image(np_image)
            masks = self.mask_generator.generate(rgb_image)
            combined_mask = self._create_combined_mask(masks)
            result_image = self._overlay_segmentation_mask(rgb_image, combined_mask)
            result_image = self._convert_output_format(result_image)
            expected_buffer_size = self.image_width * self.image_height * self.bytes_per_pixel
            if result_image.nbytes != expected_buffer_size:
                Gst.warning(f"Output image byte size ({result_image.nbytes}) does not match expected buffer size ({expected_buffer_size}). "
                           f"Input format: {self.input_format}, output shape: {result_image.shape}")

            mapinfo.data[:] = result_image.tobytes()
            buffer.unmap(mapinfo)
            
            return Gst.FlowReturn.OK

        except Exception as e:
            Gst.error(f"Error during SAM2 transformation: {e}")
            traceback.print_exc()
            if 'mapinfo' in locals():
                buffer.unmap(mapinfo)
            return Gst.FlowReturn.ERROR

GObject.type_register(GstMadeira)

__gstelementfactory__ = ("madeira", Gst.Rank.NONE, GstMadeira)
