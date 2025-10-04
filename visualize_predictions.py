
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np
from pathlib import Path

# VOC2007 categories
VOC_CATEGORIES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def get_transform():
    """Get the image preprocessing transform"""
    return transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.CenterCrop(576),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_model(model_path, num_labels=20):
    """Load trained C-Tran model"""
    from models import CTranModel
    
    model = CTranModel(num_labels=num_labels, use_lmt=True, layers=3, heads=4)
    
    # Load checkpoint with weights_only=False for compatibility
    checkpoint = torch.load(model_path, 
                          map_location='cuda' if torch.cuda.is_available() else 'cpu',
                          weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel models
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    return model, device

def predict_image(model, image_path, device, threshold=0.5):
    """Run prediction on a single image"""
    transform = get_transform()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Create dummy mask (all unknown during inference)
    mask = torch.zeros(1, len(VOC_CATEGORIES)).to(device)
    
    # Predict
    with torch.no_grad():
        output, _, _ = model(img_tensor, mask)
        probs = torch.sigmoid(output).cpu().numpy()[0]
    
    # Get predictions above threshold
    predictions = []
    for i, prob in enumerate(probs):
        if prob > threshold:
            predictions.append((VOC_CATEGORIES[i], prob))
    
    # Sort by confidence
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    return predictions, probs, image

def visualize_single_prediction(image_path, model_path, threshold=0.5, save_path=None):
    """Visualize predictions for a single image"""
    model, device = load_model(model_path)
    predictions, probs, original_image = predict_image(model, image_path, device, threshold)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Show original image
    ax1.imshow(original_image)
    ax1.axis('off')
    ax1.set_title('Input Image', fontsize=14, fontweight='bold')
    
    # Show predictions as bar chart
    if predictions:
        labels, scores = zip(*predictions)
        y_pos = np.arange(len(labels))
        
        ax2.barh(y_pos, scores, color='skyblue')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_title(f'Predictions (threshold={threshold})', fontsize=14, fontweight='bold')
        ax2.set_xlim([0, 1])
        ax2.invert_yaxis()
        
        # Add percentage labels
        for i, score in enumerate(scores):
            ax2.text(score + 0.02, i, f'{score*100:.1f}%', va='center')
    else:
        ax2.text(0.5, 0.5, 'No predictions above threshold', 
                ha='center', va='center', fontsize=12)
        ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    return predictions

def visualize_multiple_predictions(image_dir, model_path, num_images=4, threshold=0.5):
    """Visualize predictions for multiple images"""
    model, device = load_model(model_path)
    
    # Get image paths
    image_paths = list(Path(image_dir).glob('*.jpg'))[:num_images]
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
    
    # Create grid
    rows = (len(image_paths) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(15, 5*rows))
    axes = axes.flatten() if len(image_paths) > 1 else [axes]
    
    for idx, img_path in enumerate(image_paths):
        predictions, probs, image = predict_image(model, img_path, device, threshold)
        
        # Show image with predictions
        axes[idx].imshow(image)
        axes[idx].axis('off')
        
        # Create title with predictions
        if predictions:
            pred_text = ', '.join([f"{label} ({score*100:.0f}%)" for label, score in predictions[:3]])
            axes[idx].set_title(pred_text, fontsize=10)
        else:
            axes[idx].set_title('No predictions', fontsize=10)
    
    # Hide unused subplots
    for idx in range(len(image_paths), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
