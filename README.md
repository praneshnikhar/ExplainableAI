<img width="1422" height="776" alt="Image" src="https://github.com/user-attachments/assets/09c77f79-69b4-4075-855d-458ce03a7aa6" />

features-
- Upload any image from your browser
- Auto-detects multiple objects using **Detectron2**
- Classifies each object with **CLIP**
- Highlights important pixels using **LIME**
- Clean Gradio web interface

1. **Detectron2** finds objects in the image.
2. Each object is cropped and sent to **CLIP** for classification.
3. **LIME** explains why CLIP made that decision by highlighting influential pixels.

Install dependencies inside a virtual environment (venv or conda recommended):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python gradio matplotlib scikit-image transformers lime
