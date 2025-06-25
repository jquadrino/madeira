use gstreamer as gst;
use gstreamer_base as gst_base;
use gstreamer_video as gst_video;

use gst::glib;
use gst_video::prelude::*;
use ndarray::{Array, ArrayD, Axis, arr1, arr2, arr3};
use image::{DynamicImage, RgbaImage, ImageBuffer, Rgba};

use std::sync::Arc;
use ort::value::{Value as OrtValue, Tensor};
use ort::session::Session;
use ort::session::builder::SessionBuilder;
use ort::session::builder::GraphOptimizationLevel;
use ort::execution_providers::CPUExecutionProvider;

glib::wrapper! {
    pub struct Madeira(ObjectSubclass<imp::Madeira>) @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "madeira", 
        gst::Rank::MARGINAL,
        Madeira::static_type(), 
    )?;
    Ok(())
}

gst::plugin_define!(
    madeira,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("CARGO_PKG_VERSION_PRE")),
    "MIT/X11",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    "2024-01-01"
);

fn make_grid(h_feat: usize, w_feat: usize, stride: u32) -> Vec<(f32, f32)> {
    let mut grid = Vec::new();
    let stride_f = stride as f32;
    let h_img = (h_feat * 4) as f32;
    let w_img = (w_feat * 4) as f32;
    let mut y = stride_f / 2.0;
    while y < h_img {
        let mut x = stride_f / 2.0;
        while x < w_img {
            grid.push((x / w_img, y / h_img));
            x += stride_f;
        }
        y += stride_f;
    }
    grid
}

mod imp {
    use super::*;
    use gstreamer::subclass::prelude::*;
    use gstreamer_base::subclass::base_transform::BaseTransformMode;
    use gstreamer_base::subclass::prelude::*;
    use gstreamer::FlowSuccess;
    use gstreamer_video::{VideoCapsBuilder, VideoFrameRef, VideoInfo};
    use once_cell::sync::Lazy;
    use std::sync::Mutex;

    pub static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
        gst::DebugCategory::new(
            "madeira",
            gst::DebugColorFlags::empty(),
            Some("Rust segment everything filter"),
        )
    });

    const DEFAULT_ENCODER_PATH: &str = "/tmp/sam2_hiera_tiny.encoder.onnx";
    const DEFAULT_DECODER_PATH: &str = "/tmp/sam2_hiera_tiny.decoder.onnx";
    const DEFAULT_MASK_THRESHOLD: f32 = 0.3;
    const DEFAULT_GRID_STRIDE: u32 = 64;

    #[derive(Debug, Clone, Copy)]
    #[repr(u32)]
    enum Property {
        EncoderPath = 1,
        DecoderPath,
        MaskThreshold,
        GridStride,
    }

    impl From<u32> for Property {
        fn from(id: u32) -> Self {
            match id {
                1 => Self::EncoderPath,
                2 => Self::DecoderPath,
                3 => Self::MaskThreshold,
                4 => Self::GridStride,
                _ => panic!("Invalid property id: {}", id),
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct Settings {
        pub encoder_path: String,
        pub decoder_path: String,
        pub mask_threshold: f32,
        pub grid_stride: u32,
    }

    impl Default for Settings {
        fn default() -> Self {
            Self {
                encoder_path: DEFAULT_ENCODER_PATH.to_string(),
                decoder_path: DEFAULT_DECODER_PATH.to_string(),
                mask_threshold: DEFAULT_MASK_THRESHOLD,
                grid_stride: DEFAULT_GRID_STRIDE,
            }
        }
    }

    pub struct State {
        encoder: Arc<Mutex<Session>>,
        decoder: Arc<Mutex<Session>>,
    }

    #[derive(Default)]
    pub struct Madeira {
        video_info: Mutex<Option<VideoInfo>>,
        state: Mutex<Option<State>>,
        settings: Mutex<Settings>,
    }

    #[glib::object_subclass]
    impl ObjectSubclass for Madeira {
        const NAME: &'static str = "GstMadeira";
        type Type = super::Madeira;
        type ParentType = gst_base::BaseTransform;
        type Interfaces = ();

        fn with_class(_klass: &Self::Class) -> Self {
            Self {
                video_info: Mutex::new(None),
                state: Mutex::new(None),
                settings: Mutex::new(Settings::default()),
            }
        }
    }

    impl ObjectImpl for Madeira {
        fn properties() -> &'static [glib::ParamSpec] {
            static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
                vec![
                    glib::ParamSpecString::builder("encoder-path")
                        .nick("Encoder ONNX Path")
                        .blurb("Path to the SAM2 encoder ONNX model file.")
                        .default_value(Some(DEFAULT_ENCODER_PATH))
                        .flags(glib::ParamFlags::READWRITE)
                        .build(),
                    glib::ParamSpecString::builder("decoder-path")
                        .nick("Decoder ONNX Path")
                        .blurb("Path to the SAM2 decoder ONNX model file.")
                        .default_value(Some(DEFAULT_DECODER_PATH))
                        .flags(glib::ParamFlags::READWRITE)
                        .build(),
                    glib::ParamSpecFloat::builder("mask-threshold")
                        .nick("Mask Threshold")
                        .blurb("Threshold (0.0-1.0) applied to the decoder's output logits to binarize the segmentation mask. Higher values result in stricter masks.")
                        .minimum(0.0)
                        .maximum(1.0)
                        .default_value(DEFAULT_MASK_THRESHOLD)
                        .flags(glib::ParamFlags::READWRITE)
                        .build(),
                    glib::ParamSpecUInt::builder("grid-stride")
                        .nick("Grid Stride")
                        .blurb("Stride (in pixels relative to original image size) for generating the grid of prompt points. Smaller strides generate more points and potentially more detailed segmentation but are slower.")
                        .minimum(1)
                        .maximum(110)
                        .default_value(DEFAULT_GRID_STRIDE)
                        .flags(glib::ParamFlags::READWRITE)
                        .build(),
                ]
            });
            PROPERTIES.as_ref()
        }

        fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
            let prop_id = Property::from(pspec.name().parse::<u32>().unwrap_or_else(|_| {
                match pspec.name() {
                    "encoder-path" => 1,
                    "decoder-path" => 2,
                    "mask-threshold" => 3,
                    "grid-stride" => 4,
                    _ => panic!("Invalid property name: {}", pspec.name()),
                }
            }));

            match prop_id {
                Property::EncoderPath => {
                    let mut settings = self.settings.lock().unwrap();
                    let encoder_path = value.get::<String>().expect("type checked upstream");
                    gst::info!(CAT, imp: self, "Changing encoder-path from {} to {}", settings.encoder_path, encoder_path);
                    settings.encoder_path = encoder_path;
                }
                Property::DecoderPath => {
                    let mut settings = self.settings.lock().unwrap();
                    let decoder_path = value.get::<String>().expect("type checked upstream");
                    gst::info!(CAT, imp: self, "Changing decoder-path from {} to {}", settings.decoder_path, decoder_path);
                    settings.decoder_path = decoder_path;
                }
                Property::MaskThreshold => {
                    let mut settings = self.settings.lock().unwrap();
                    let mask_threshold = value.get::<f32>().expect("type checked upstream");
                    gst::info!(CAT, imp: self, "Changing mask-threshold from {} to {}", settings.mask_threshold, mask_threshold);
                    settings.mask_threshold = mask_threshold;
                }
                Property::GridStride => {
                    let mut settings = self.settings.lock().unwrap();
                    let grid_stride = value.get::<u32>().expect("type checked upstream");
                    gst::info!(CAT, imp: self, "Changing grid-stride from {} to {}", settings.grid_stride, grid_stride);
                    settings.grid_stride = grid_stride;
                }
            }
        }

        fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
            let prop_id = Property::from(pspec.name().parse::<u32>().unwrap_or_else(|_| {
                match pspec.name() {
                    "encoder-path" => 1,
                    "decoder-path" => 2,
                    "mask-threshold" => 3,
                    "grid-stride" => 4,
                    _ => panic!("Invalid property name: {}", pspec.name()),
                }
            }));

            let settings = self.settings.lock().unwrap();
            match prop_id {
                Property::EncoderPath => settings.encoder_path.to_value(),
                Property::DecoderPath => settings.decoder_path.to_value(),
                Property::MaskThreshold => settings.mask_threshold.to_value(),
                Property::GridStride => settings.grid_stride.to_value(),
            }
        }

        fn constructed(&self) {
            self.parent_constructed();

            let obj = self.obj();
            obj.set_in_place(true);
            obj.set_passthrough(false);
        }
    }
    
    impl GstObjectImpl for Madeira {}

    impl ElementImpl for Madeira {
        fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
            static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
                gst::subclass::ElementMetadata::new(
                    "Madeira SAM v2 Filter",
                    "Filter/Effect/Video",
                    "Prompt-based video intelligence / segment everything",
                    "Joseph Quadrino <joseph.a.quadrino@gmail.com>",
                )
            });
            Some(&*ELEMENT_METADATA)
        }

        fn pad_templates() -> &'static [gst::PadTemplate] {
            static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
                let caps = VideoCapsBuilder::new()
                    .format_list([
                        gstreamer_video::VideoFormat::Rgb,
                        gstreamer_video::VideoFormat::Rgba,
                        gstreamer_video::VideoFormat::Bgr,
                        gstreamer_video::VideoFormat::Bgra,
                        gstreamer_video::VideoFormat::Bgrx,
                    ])
                    .build();
                vec![
                    gst::PadTemplate::new("src", gst::PadDirection::Src, gst::PadPresence::Always, &caps).unwrap(),
                    gst::PadTemplate::new("sink", gst::PadDirection::Sink, gst::PadPresence::Always, &caps).unwrap(),
                ]
            });
            PAD_TEMPLATES.as_ref()
        }
    }

    impl BaseTransformImpl for Madeira {
        const MODE: BaseTransformMode = BaseTransformMode::AlwaysInPlace;
        const PASSTHROUGH_ON_SAME_CAPS: bool = false;
        const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

        fn start(&self) -> Result<(), gst::ErrorMessage> {
            gst::info!(CAT, "Starting");

            let (ep, _ep_name) = (CPUExecutionProvider::default().build(), "cpu");

            ort::init().with_execution_providers([ep]).commit()
                .map_err(|_e| gst::ErrorMessage::new(
                    &gst::LibraryError::Failed,
                    Some("ORT init error"),
                    Some("Unable to start pipeline due to ORT init failure."),
                    file!(),
                    "start/1",
                    line!(),
                ))?;

            let settings = self.settings.lock().unwrap();
            let encoder_path = settings.encoder_path.clone();
            let decoder_path = settings.decoder_path.clone();
            drop(settings);

            let encoder_session = SessionBuilder::new()
                .map_err(|_e| gstreamer::ErrorMessage::new(
                    &gst::LibraryError::Failed,
                    Some("SessionBuilder error"),
                    Some("Unable to start pipeline due to ORT SessionBuilder failure."),
                    file!(),
                    "start/1",
                    line!(),
                ))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|_e| gstreamer::ErrorMessage::new(
                    &gst::LibraryError::Failed,
                    Some("Set optimization level error"),
                    Some("Unable to start pipeline due to ORT optimization level failure."),
                    file!(),
                    "start/1",
                    line!(),
                ))?
                .commit_from_file(&encoder_path)
                .map_err(|_e| gstreamer::ErrorMessage::new(
                    &gst::LibraryError::Failed,
                    Some("Commit from file error"),
                    Some("Unable to start pipeline due to ORT model file load failure."),
                    file!(),
                    "start/1",
                    line!(),
                ))?;

            let decoder_session = SessionBuilder::new()
                .map_err(|_e| gstreamer::ErrorMessage::new(
                    &gst::LibraryError::Failed,
                    Some("SessionBuilder error"),
                    Some("Unable to start pipeline due to ORT SessionBuilder failure."),
                    file!(),
                    "start/1",
                    line!(),
                ))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|_e| gstreamer::ErrorMessage::new(
                    &gst::LibraryError::Failed,
                    Some("Set optimization level error"),
                    Some("Unable to start pipeline due to ORT optimization level failure."),
                    file!(),
                    "start/1",
                    line!(),
                ))?
                .commit_from_file(&decoder_path)
                .map_err(|_e| gstreamer::ErrorMessage::new(
                    &gst::LibraryError::Failed,
                    Some("Commit from file error"),
                    Some("Unable to start pipeline due to ORT model file load failure."),
                    file!(),
                    "start/1",
                    line!(),
                ))?;

                let encoder = Arc::new(Mutex::new(encoder_session));
                let decoder = Arc::new(Mutex::new(decoder_session));

                *self.state.lock().unwrap() = Some(State { encoder, decoder });

            Ok(())
        }
    
        fn stop(&self) -> Result<(), gst::ErrorMessage> {
            gst::info!(CAT, "Stopping");
            Ok(())
        }

        fn set_caps(&self, incaps: &gst::Caps, outcaps: &gst::Caps) -> Result<(), gst::LoggableError> {
            let video_info = gstreamer_video::VideoInfo::from_caps(incaps)
                .expect("Failed to parse input caps");

            let obj = self.obj();
            gst::debug!(CAT, obj: &obj, "Negotiated caps: {:?}", incaps);

            *self.video_info.lock().unwrap() = Some(video_info);
            self.parent_set_caps(incaps, outcaps)
        }

        fn transform_ip(&self, buffer: &mut gst::BufferRef) -> Result<FlowSuccess, gst::FlowError> {
            let state_guard = self.state.lock().unwrap();
            let state = state_guard.as_ref().expect("State not initialized");

            let video_info = self.video_info.lock().unwrap();
            let video_info = video_info.as_ref().unwrap();

            let mut video_frame =
                match VideoFrameRef::from_buffer_ref_writable(buffer, video_info) {
                    Ok(frame) => frame,
                    Err(_) => {
                        let obj = self.obj();
                        gst::error!(CAT, obj: &obj, "Failed to map buffer writable");
                        return Err(gst::FlowError::Error);
                    }
                };

            let format = video_frame.format();
            let width = video_frame.width();
            let height = video_frame.height();
            let data = video_frame.plane_data_mut(0).unwrap();

            let settings = self.settings.lock().unwrap();
            let mask_threshold = settings.mask_threshold;
            let grid_stride = settings.grid_stride;
            drop(settings);

            // pre-processing
            let image = match format {
                gst_video::VideoFormat::Rgb => ImageBuffer::from_raw(width, height, data.to_vec()).map(DynamicImage::ImageRgb8),
                gst_video::VideoFormat::Rgba => ImageBuffer::from_raw(width, height, data.to_vec()).map(DynamicImage::ImageRgba8),
                gst_video::VideoFormat::Bgrx | gst_video::VideoFormat::Bgr => {
                    let mut rgb_data = Vec::with_capacity(data.len() / 4 * 3);
                    for chunk in data.chunks_exact(4) {
                        rgb_data.extend_from_slice(&[chunk[2], chunk[1], chunk[0]]);
                    }
                    ImageBuffer::from_raw(width, height, rgb_data).map(DynamicImage::ImageRgb8)
                }
                gst_video::VideoFormat::Bgra => {
                    let mut rgba_data = data.to_vec();
                    rgba_data.chunks_exact_mut(4).for_each(|c| c.swap(0, 2));
                    ImageBuffer::from_raw(width, height, rgba_data).map(DynamicImage::ImageRgba8)
                }
                _ => None,
            };
            let frame_img = image.ok_or_else(|| {
                gst::FlowError::NotSupported
            });

            let resized_img = frame_img?.resize_exact(1024, 1024, image::imageops::FilterType::Triangle);
            let resized_rgb_img = resized_img.to_rgb8();

            let array = Array::from_shape_vec((1024 as usize, 1024 as usize, 3), resized_rgb_img.to_vec()).unwrap().mapv(|x| x as f32 / 255.0);
            let mean = Array::from_shape_vec((1, 1, 3), vec![0.485, 0.456, 0.406]).unwrap();
            let std = Array::from_shape_vec((1, 1, 3), vec![0.229, 0.224, 0.225]).unwrap();
            let normalized_array = (array - &mean) / &std;

            let image_tensor = normalized_array.permuted_axes([2, 0, 1]).insert_axis(Axis(0));

            let image_tensor_dyn: ArrayD<f32> = image_tensor.into_dyn();
            let image_shape: Vec<usize> = image_tensor_dyn.shape().to_vec();
            let image_data: Vec<f32> = image_tensor_dyn.into_raw_vec();
        
            // inference
            let ort_input_value = OrtValue::from_array((image_shape.into_iter().map(|d| d as i64).collect::<Vec<_>>(), image_data))
                .map_err(|_e| {
                    gst::FlowError::NotSupported
                })?;
            let input = ort::inputs![ort_input_value];

            let mut encoder_guard = state.encoder.lock().unwrap();
            let enc_out = encoder_guard.run(input)
                .map_err(|_e| {
                    gst::FlowError::Error
                })?;

            let h0 = enc_out[0].try_extract_tensor::<f32>()
                .map_err(|_e| {
                    gst::FlowError::Error
                })?.to_owned();
            let h1 = enc_out[1].try_extract_tensor::<f32>()
                .map_err(|_e| {
                    gst::FlowError::Error
                })?.to_owned();
            let image_embeddings = enc_out[2].try_extract_tensor::<f32>()
                .map_err(|_e| {
                    gst::FlowError::Error
                })?;

            let (hc, wc) = (image_embeddings.0[2] as usize, image_embeddings.0[3] as usize);

            let im_emb_in = OrtValue::from_array((image_embeddings.0.clone(), image_embeddings.1.to_vec()))
                .map_err(|_e| { gst::FlowError::Error })?;
            let h0_in = OrtValue::from_array((h0.0.clone(), h0.1.to_vec()))
                .map_err(|_e| { gst::FlowError::Error })?;
            let h1_in = OrtValue::from_array((h1.0.clone(), h1.1.to_vec()))
                .map_err(|_e| { gst::FlowError::Error })?;

            let point_labels = arr2(&[[1.0f32]]).into_dyn();
            let point_labels_in = OrtValue::from_array((point_labels.shape().to_vec(), point_labels.into_raw_vec()))
                        .map_err(|_e| { gst::FlowError::Error })?;

            let has_mask_input = arr1(&[0.0f32]).into_dyn();
            let has_mask_input_in = OrtValue::from_array((has_mask_input.shape().to_vec(), has_mask_input.into_raw_vec()))
                        .map_err(|_e| { gst::FlowError::Error })?;

            let mask_shape = [1, 1, hc * 4, wc * 4];
            let mask_data = vec![0.0f32; mask_shape.iter().product()];
            let mask_input = Tensor::from_array((mask_shape, mask_data))
                .map_err(|_| gst::FlowError::Error)?;

            let grid = make_grid(hc, wc, grid_stride);
            let mut all_masks: Vec<ndarray::Array2<bool>> = Vec::new();
            let mut mask_height = 0;
            let mut mask_width = 0;
            let mut decoder_guard = state.decoder.lock().unwrap();

            for (x, y) in grid {
                let point_coords = arr3(&[[[x, y]]]).into_dyn();
                let point_coords_in = OrtValue::from_array((point_coords.shape().to_vec(), point_coords.into_raw_vec()))
                        .map_err(|_e| { gst::FlowError::Error })?;

                let dec_input = ort::inputs![
                    "image_embed" => &im_emb_in,
                    "high_res_feats_0" => &h0_in,
                    "high_res_feats_1" => &h1_in,
                    "point_coords" => point_coords_in,
                    "point_labels" => &point_labels_in,
                    "has_mask_input" => &has_mask_input_in,
                    "mask_input" => &mask_input,
                ];
                let dec_out = match decoder_guard.run(dec_input) {
                    Ok(output) => {
                        output
                    },
                    Err(e) => {
                        gst::error!(CAT, imp: self, "Decoder failed for {}, {}: {:?}", x, y, e);
                        return Err(gst::FlowError::Error);
                    }
                };

                let mask_logits = dec_out[0]
                    .try_extract_tensor::<f32>()
                    .map_err(|_| gst::FlowError::Error)?;

                let height = mask_logits.0[2] as usize;
                let width = mask_logits.0[3] as usize;

                if mask_height == 0 {
                    mask_height = height;
                    mask_width = width;
                }

                let mask_size = height * width;
                let mask_data = &mask_logits.1[..mask_size];
                let mask = mask_data
                    .iter()
                    .map(|&v| v > mask_threshold)
                    .collect::<Vec<bool>>();
                let mask_array = Array::from_shape_vec((height, width), mask)
                    .map_err(|_| gst::FlowError::Error)?;
                all_masks.push(mask_array);
            }

            let combined_mask = if !all_masks.is_empty() {
                let mut combined = Array::from_elem((mask_height, mask_width), false);
                for mask in &all_masks {
                    combined
                        .iter_mut()
                        .zip(mask.iter())
                        .for_each(|(a, &b)| *a = *a || b);
                }
                combined
            } else {
                Array::from_elem((1024, 1024), false)
            };

            let mut result_img = DynamicImage::ImageRgb8(resized_rgb_img).to_rgba8();
            let (width, height) = result_img.dimensions();

            let mask_img = ImageBuffer::<Rgba<u8>, _>::from_fn(width, height, |x, y| {
                let mask_x = (x as f32 * mask_width as f32 / width as f32) as usize;
                let mask_y = (y as f32 * mask_height as f32 / height as f32) as usize;
                if combined_mask.get((mask_y, mask_x)).copied().unwrap_or(false) {
                    Rgba([255, 255, 255, 255])
                } else {
                    Rgba([0, 0, 0, 0])
                }
            });

            let mut overlay = RgbaImage::new(width, height);
            for (x, y, pixel) in overlay.enumerate_pixels_mut() {
                if mask_img.get_pixel(x, y).0[3] > 0 {
                    *pixel = Rgba([255, 0, 0, 128]);
                } else {
                    *pixel = Rgba([0, 0, 0, 0]);
                }
            }

            image::imageops::overlay(&mut result_img, &overlay, 0, 0);

            let final_image = DynamicImage::ImageRgba8(result_img).resize_exact(
                video_info.width(),
                video_info.height(),
                image::imageops::FilterType::Triangle,
            );

            let image_bytes = match format {
                gst_video::VideoFormat::Rgb => final_image.to_rgb8().into_raw(),
                gst_video::VideoFormat::Rgba => final_image.to_rgba8().into_raw(),
                gst_video::VideoFormat::Bgr | gst_video::VideoFormat::Bgrx => {
                    let rgb = final_image.to_rgb8();
                    let mut bgrx = Vec::with_capacity((width * height * 4) as usize);
                    for pixel in rgb.pixels() {
                        bgrx.extend_from_slice(&[pixel[2], pixel[1], pixel[0], 255]);
                    }
                    bgrx
                }
                gst_video::VideoFormat::Bgra => {
                    let mut rgba = final_image.to_rgba8();
                    rgba.pixels_mut().for_each(|p| p.0.swap(0, 2));
                    rgba.into_raw()
                }
                _ => return Err(gst::FlowError::NotSupported),
            };
            data.copy_from_slice(&image_bytes);

            Ok(FlowSuccess::Ok)
        }
    }
}
