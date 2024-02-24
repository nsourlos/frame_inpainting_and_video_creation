# Video Segmentation and Inpainting


[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/uses-badges.svg)](https://forthebadge.com)

<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT) -->
[![Maintenance](https://img.shields.io/badge/Maintained%3F-no-red.svg)]( https://github.com/nsourlos/semi-automated_installation_exe_msi_files-Windows_10)


## Introduction

This code implements a video segmentation and inpainting pipeline using the Stable Diffusion (SD) model and LangSAM (Language-based Segmentation) model. The goal is to segment specific objects in each frame of a video based on textual prompts and then inpaint the segmented regions.

## Functionality

The [script](./masks_inpaint_langseg.py) performs two main tasks:

- **Segmentation**: The script uses the LangSAM model to segment an image based on the `text_prompt`.
- **Inpainting**: After segmentation, the script uses the StableDiffusionInpaintPipeline to inpaint the segmented parts of the image based on the `prompt_sd`.

## Environment Setup
To run the code successfully, follow these steps to set up the required environment:

1. **Create a new Anaconda environment with Python 3.10(.12):**
   ```bash
   conda create --name your_environment_name python=3.10.12
   conda activate your_environment_name
   ```

2. **Install CUDA Toolkit (version 11.7.0) - For Nvidia GeForce GTX 1660Ti:**

   ```bash 
    conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
   ```

3. **Export CUDA Path:**

    ```bash 
    export CUDA_HOME=/usr/local/cuda-11.7/
    ```  

    If it fails, try also to find the path with ```echo $CUDA_HOME``` (not recommended).

4. **Install PyTorch:**

   ```bash 
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
   ```

5. **Install lang-sam:**

   ```bash 
    git clone https://github.com/luca-medeiros/lang-segment-anything
    cd lang-segment-anything
    pip install -e .
   ```

6. **Install diffusers and accelerate for GPU stable diffusion inference:**

   ```bash 
    pip install diffusers==0.20.2 accelerate==0.22.0
   ```

7. **Export Paths:**

    Ensure that these are the correct folders. This needs to be done only once.
    Taken from [PyTorch Issue #102535](https://github.com/pytorch/pytorch/issues/102535):

   ```bash 
    export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64 
   ``` 
   The above have to be added in the 'bashrc' file.

## Frame Extraction to be Inpainted and Combine Inpainted and Original DeepFake Videos

This [script](./video_to_img.py) performs the following tasks:

1. Resizes input videos to a specified resolution (1920x1080 by default) for compatibility with the deepfake processing script.
2. Extracts individual frames from a deepfake video to be used as input to an inpainting pipeline.
3. Recreates the deepfake video with the initial audio in it.
4. Crops resized videos from step 1 (those were not used in steps 2-3) to a smaller target size (1080x1080 by default) for final output.
5. Concatenates a deepfake created video (`final_deepfake_video.mp4`) created in steps 2-3 and using the deepfake creation pipeline, with any other cropped videos before and after it found in the `final_video_folder` and resizes the combined video to a desired resolution.




## External Dependencies

- **numpy**: Numerical computing library.
- **cv2**: OpenCV library for computer vision tasks.
- **matplotlib**: Library for creating static, animated, and interactive visualizations.
- **os**: Module for interacting with the operating system.
- **torch**: PyTorch deep learning framework.
- **diffusers**: Library providing implementations of various diffusion models.
- **PIL (Image, ImageFilter)**: Python Imaging Library for image processing tasks.
- **lang_sam**: Language-based segmentation model.

## External References

1. [Segment Anything GitHub Repository](https://github.com/facebookresearch/segment-anything)
2. [LangSAM GitHub Repository](https://github.com/luca-medeiros/lang-segment-anything)
3. [GroundingDINO Installation](https://github.com/IDEA-Research/GroundingDINO)
4. [Alternative to LangSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)
5. [Stable Diffusion Documentation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img)
6. [LabLab Stable Diffusion img2img](https://lablab.ai/t/stable-diffusion-img2img)
7. [Segment-Anything-Video GitHub Repository](https://github.com/kadirnar/segment-anything-video)
8. [Segment-Everything-Everywhere-All-At-Once GitHub Repository](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)

## Main Inpainting Script Structure

### 1. Prompts and Paths
- `text_prompt`: Text describing the object to segment (e.g., "costume").
- `prompt_sd`: Text prompt for inpainting (e.g., "A fancy summer shirt with flowers in a black background").
- `sam_checkpoint`: Path to the SAM model checkpoint.
- `model_type`: SAM model type (e.g., "vit_b").
- `frame_path`: Path to the directory containing video frames.
- `save_path`: Path to the directory for saving inpainted frames.
- `test_save_path`: Path to save images for manual checking (debugging).
- `target_width`: Desired width for cropped images and masks.
- `target_height`: Desired height for cropped images and masks.

### 2. Initialize Dependencies

- Creates directories for saving inpainted frames if they don't exist.
- Initializes the Stable Diffusion inpainting pipeline, enabling CPU offloading and attention slicing for efficiency.
- Loads the SAM model.

### 3. Crop and Resize Function

- `crop_and_resize`: Crops an image and its mask around the center, resizing them to the specified dimensions.

### 4. Inpainting Function

- Crops the image and mask using crop_and_resize.
- Inpaints the cropped image using Stable Diffusion based on the provided prompt and mask.
- Returns the inpainted image.

### 5. Blend Image Function

- Blends the inpainted image with the original image using the mask and applies Gaussian blur for smoother transitions.
- Returns the blended image.

### 6. Main Loop

- Loop through each frame in the video.
- For the first frame:
  - Loads the frame, predicts masks using SAM, and converts them to uint8 format.
  - Creates a copy of the first frame's mask.
  - Crops and resizes the image and mask.
  - Inpaints the cropped image using Stable Diffusion.
  - Blends the inpainted image with the original cropped image and saves it.
- For subsequent frames:
  - Loads the inpainted image from the previous iteration.
  - Loads, crops and resizes the current frame
  - Predicts masks for the current frame using SAM.
  - Extracts the inpainted portion from the previous frame using the current frame's mask.
  - Replaces pixels in the cropped current frame with the corresponding pixels from the extracted inpainted portion.
  - Inpaints the remaining unmasked pixels in the cropped current frame using Stable Diffusion.
  - Blends the inpainted cropped image with the original cropped image and saves it.

### Notes

- The code uses Stable Diffusion for inpainting based on masks generated by LangSAM.
- The inpainting process involves blending the inpainted mask with the original image using the provided mask.
- The code contains debugging and manual checking options, which can be activated by uncommenting specific sections.

### Recommendations

- Ensure the necessary dependencies are installed.
- Verify that paths and checkpoints are correctly set.
- Adjust parameters and prompts based on specific use cases.
- Carefully review and understand the inpainting process, as it involves blending and masking operations.

### Important

- This documentation assumes familiarity with the referenced models and libraries.
- It is recommended to refer to the linked repositories and documentation for detailed explanations of the models and methods used.
- Some external links may be subject to changes, and it is advisable to refer to the latest documentation.