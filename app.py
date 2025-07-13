# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np

# from transformers import CLIPProcessor, CLIPModel
# from lime import lime_image
# from skimage.segmentation import mark_boundaries

# # -------- Load Image --------
# img = cv2.imread("images/cat.jpg")
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# pil_img = Image.fromarray(img_rgb)

# # -------- Load CLIP Model --------
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# labels = ["cat", "dog", "car", "tree"]
# inputs = clip_processor(text=labels, images=pil_img, return_tensors="pt", padding=True)
# outputs = clip_model(**inputs)
# probs = outputs.logits_per_image.softmax(dim=1)

# print("🔍 Classification Results:")
# for label, prob in zip(labels, probs[0]):
#     print(f"{label}: {prob.item():.4f}")

# # -------- LIME Explainability --------
# explainer = lime_image.LimeImageExplainer()

# def predict_fn(images):
#     pil_imgs = [Image.fromarray(img) for img in images]
#     inputs = clip_processor(text=["cat"], images=pil_imgs, return_tensors="pt", padding=True)
#     outputs = clip_model(**inputs)
#     return outputs.logits_per_image.softmax(dim=1).detach().numpy()

# explanation = explainer.explain_instance(
#     img_rgb,
#     predict_fn,
#     top_labels=1,
#     hide_color=0,
#     num_samples=1000
# )

# temp, mask = explanation.get_image_and_mask(
#     label=0,
#     positive_only=True,
#     hide_rest=False,
#     num_features=5,
#     min_weight=0.01
# )

# # -------- Show Output --------
# plt.imshow(mark_boundaries(temp, mask))
# plt.axis('off')
# plt.title("LIME Explanation")
# plt.show()









import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from transformers import CLIPProcessor, CLIPModel
from lime import lime_image
from skimage.segmentation import mark_boundaries

import gradio as gr

# Load CLIP model + processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# LIME setup
explainer = lime_image.LimeImageExplainer()

# 🧠 Prediction wrapper for LIME
def predict_fn(images, class_text):
    pil_imgs = [Image.fromarray(img) for img in images]
    inputs = clip_processor(text=[class_text], images=pil_imgs, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()
    return probs

# # 🔍 Main Explain function
# def explain_fn(image, label_string):
#     # Step 1: Preprocess
#     img_np = np.array(image.convert("RGB"))
#     labels = [lbl.strip() for lbl in label_string.split(",") if lbl.strip()]
    
#     if not labels:
#         return "No labels provided."

#     # Step 2: Predict with CLIP
#     inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True)
#     outputs = clip_model(**inputs)
#     probs = outputs.logits_per_image.softmax(dim=1)[0]
#     top_label_idx = torch.argmax(probs).item()
#     top_label = labels[top_label_idx]
#     confidence = probs[top_label_idx].item()

#     # Step 3: Explain with LIME
#     def lime_wrapper(imgs):
#         pil_imgs = [Image.fromarray(img) for img in imgs]
#         inputs = clip_processor(text=[top_label], images=pil_imgs, return_tensors="pt", padding=True)
#         outputs = clip_model(**inputs)
#         return outputs.logits_per_image.softmax(dim=1).detach().numpy()

#     explanation = explainer.explain_instance(
#         img_np,
#         lime_wrapper,
#         top_labels=1,
#         hide_color=0,
#         num_samples=1000
#     )

#     temp, mask = explanation.get_image_and_mask(
#         label=0,
#         positive_only=True,
#         hide_rest=False,
#         num_features=5,
#         min_weight=0.01
#     )

#     result = mark_boundaries(temp, mask)

#     # Add text overlay (optional)
#     result_image = Image.fromarray((result * 255).astype(np.uint8))
#     return result_image




from matplotlib import cm

def explain_fn(image, label_string):
    img_np = np.array(image.convert("RGB"))
    labels = [lbl.strip() for lbl in label_string.split(",") if lbl.strip()]
    
    if not labels:
        return "❌ No labels provided."

    # Predict with CLIP
    inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]
    top_label_idx = torch.argmax(probs).item()
    top_label = labels[top_label_idx]
    confidence = probs[top_label_idx].item()
    
    print(f"🔍 Predicted: {top_label} ({confidence:.2f})")
    if confidence < 0.2:
        return f"⚠️ Prediction confidence too low: {top_label} ({confidence:.2f})"

    # LIME explanation
    def lime_wrapper(imgs):
        pil_imgs = [Image.fromarray(img) for img in imgs]
        inputs = clip_processor(text=[top_label], images=pil_imgs, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        return outputs.logits_per_image.softmax(dim=1).detach().numpy()

    explanation = explainer.explain_instance(
        img_np,
        lime_wrapper,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        label=0,
        positive_only=True,
        hide_rest=False,
        num_features=10,
        min_weight=0.01
    )

    # Apply color heatmap overlay
    # overlay = cm.jet(mask.astype(float) / mask.max())[:, :, :3]  # RGB heatmap
    # heatmap = (overlay * 255).astype(np.uint8)
    # blended = cv2.addWeighted(img_np, 0.7, heatmap, 0.3, 0)

    # result_img = Image.fromarray(blended)
    # # Caption for user understanding
    # result_img.info["label"] = f"Top Prediction: {top_label} ({confidence:.2f})"
    # return result_img
    
        # Apply heatmap overlay using a bright colormap
    overlay = cm.inferno(mask.astype(float) / mask.max())[:, :, :3]  # Bright, colorful mask
    heatmap = (overlay * 255).astype(np.uint8)

    # Blend with original image for better visibility
    blended = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

    # Add label as caption at bottom (optional)
    result_img = Image.fromarray(blended)
    caption = f"✅ Prediction: {top_label} ({confidence:.2f}) — Highlighted areas contributed most"
    print(caption)
    return result_img







# 🌐 Gradio UI
gr.Interface(
    fn=explain_fn,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Enter comma-separated labels (e.g., cat, dog, statue)")
    ],
    outputs=gr.Image(label="LIME Explanation", type="pil", image_mode="RGB"),

    title="🧠 Explainable Vision AI with CLIP + LIME",
    description="Upload an image and provide labels. The model will predict the top label using CLIP, then highlight the most important regions using LIME."
).launch()

