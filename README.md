# Waste Detection using YOLOE

This repository contains notebooks and utilities for detecting litter/waste items in images and live camera feeds using a YOLOE-based object detection workflow. The project demonstrates how to load a trained YOLOE-style detector, run inference on images and webcam video, and visualize detections with bounding boxes and confidence scores.

**Repository structure**
- `Detection on image.ipynb` — Notebook to run detection on single images and visualize/save results.
- `Detection using camera.ipynb` — Notebook to run real-time detection from a webcam (or video file).
- `YOLOE_visual_prompt_.ipynb` — Notebook demonstrating the visual prompt / guided detection features (if available in your model code).

## Quick overview

- Goal: Detect common waste items (plastic, paper, glass, organic waste, etc.) in images and video streams.
- Approach: Use a YOLOE-like object detection model to predict bounding boxes, class labels, and confidence scores, then draw boxes on the source images and optionally save results to disk.

## Requirements

- Python 3.8 or newer
- Recommended packages (install into a virtual environment):

  - `torch` and `torchvision` (for model runtime)
  - `opencv-python` (for image I/O and webcam capture)
  - `matplotlib` (for plotting results inline)
  - `numpy`
  - `jupyter` / `notebook` (to run the .ipynb files)

If this repository includes a `requirements.txt`, install with:

PowerShell
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If there is no `requirements.txt`, install the typical packages manually:

PowerShell
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision opencv-python matplotlib numpy jupyter
```

Note: Install a `torch` build compatible with your CUDA setup if you plan to use GPU acceleration. See the official PyTorch site for platform-specific install commands.

## How to run

1. Open the repository folder in VS Code or start Jupyter Notebook:

PowerShell
```
jupyter notebook
```

2. Open one of the notebooks:
   - `Detection on image.ipynb` — for running detection on saved images.
   - `Detection using camera.ipynb` — for webcam/live detection (allow camera access).
   - `YOLOE_visual_prompt_.ipynb` — for experiments with visual prompting.

3. Follow the cells' instructions. Typical steps inside notebooks:
   - Install or import dependencies.
   - Load the model weights (path or checkpoint may be prompted by the notebook).
   - Prepare an input image or configure the camera/video source.
   - Run inference and visualize detections.
   - Save detection images to a local `results/` folder (create the folder if missing).

## Example usage (single image)

Inside `Detection on image.ipynb` you will typically:

- Set `image_path = "path/to/your/image.jpg"`.
- Load the model checkpoint: `model = load_model("path/to/checkpoint.pth")` (notebook cell may show an API).
- Run `predictions = model.detect(image)` and then `visualize(image, predictions)`.

Example output (text summary):

```
Image: beach_waste_001.jpg
Detections: 3
 - plastic_bottle: 0.94  (x1=120,y1=230,x2=240,y2=410)
 - banana_peel: 0.88     (x1=300,y1=220,x2=360,y2=320)
 - paper_cup: 0.76       (x1=400,y1=150,x2=460,y2=230)
```

And an output image saved to `results/example_detection.jpg` with bounding boxes and labels drawn.

## Example results (illustrative)

Below is an example detection summary you can expect after running the notebook (this is an illustrative example — run the notebooks to generate actual outputs for your images):

- Total images processed: 1
- Average detections per image: 2.8
- Sample high-confidence detection:
  - class: `plastic_bottle`, confidence: 0.94

To save result images, create a `results/` directory and ensure the notebook's save paths point there. Many notebook cells include a `save=True` flag when calling visualization utilities.

## Results (sample outputs)

Below are sample detection outputs produced while developing and testing this project. These are included in the repository so others can quickly see what the detector produces.

- Example 1: `Result 1.png`

<img src="./Result%201.png" alt="Result 1" width="600" />

- Example 2: `Result 2.png`

<img src="./Result%202.png" alt="Result 2" width="600" />


- Raspberry Pi demo: `Rasberry Pi 5 detection result.jpg`

<img src="./Rasberry%20Pi%205%20detection%20result.jpg" alt="Raspberry Pi demo" width="600" />

If you prefer images inside a `results/` folder, move these files into `results/` and update the paths above (or I can do that for you).

## Tips & troubleshooting

- If the model fails to load, check the checkpoint path and PyTorch version compatibility.
- For webcam detection, verify that your camera index (0, 1, ...) is correct. In PowerShell, close other apps using the camera.
- If inference is slow, try installing a CUDA-enabled `torch` build and run on GPU.
- If detections are poor, consider fine-tuning the model with additional annotated waste images or improving pre/post-processing (NMS thresholds, image scaling).


