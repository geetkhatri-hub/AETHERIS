import cv2
import numpy as np

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to a grayscale image.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to YUV to apply CLAHE on luminance channel
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

def unsharp_mask(image, sigma=1.0, strength=1.5):
    """
    Applies an unsharp mask filter to sharpen fine structures and edges.
    """
    # Soft blur to capture low frequencies
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    # Subtract blur and scale to get high frequencies
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def enhance_ir_image(image):
    """
    Full preprocessing pipeline for IR images:
    1. Denoise slightly (Bilateral Filter to keep edges sharp)
    2. CLAHE (Adaptive Contrast)
    3. Unsharp Mask (Edge Enhancement)
    """
    # Make sure we work with a single channel if it's grayscale
    is_color = len(image.shape) == 3 and image.shape[2] == 3
    if is_color:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # 1. Bilateral filter for edge-preserving denoising
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    
    # 2. CLAHE to bring out details in shadows and highlights
    enhanced = apply_clahe(denoised, clip_limit=3.0, tile_grid_size=(8, 8))
    
    # 3. Sharpen edges
    sharpened = unsharp_mask(enhanced, sigma=1.0, strength=1.2)
    
    return sharpened
