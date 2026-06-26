import numpy as np
import cv2
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None

def calculate_psnr(target, prediction):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between target (RGB) and prediction (RGB).
    Images should be numpy arrays of shape (H, W, 3) in range [0, 255].
    """
    # Convert to float
    t = target.astype(np.float64)
    p = prediction.astype(np.float64)
    
    mse = np.mean((t - p) ** 2)
    if mse == 0:
        return 100.0
        
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)

def calculate_ssim(target, prediction):
    """
    Computes Structural Similarity Index (SSIM) between target and prediction.
    Images should be numpy arrays of shape (H, W, 3) in range [0, 255].
    """
    # If skimage is available, use it
    if ssim is not None:
        # Determine if grayscale or RGB
        is_multichannel = len(target.shape) == 3 and target.shape[2] == 3
        # Specify channel_axis if newer skimage version, else multichannel=True
        try:
            if is_multichannel:
                score, _ = ssim(target, prediction, full=True, channel_axis=2)
            else:
                score, _ = ssim(target, prediction, full=True)
        except TypeError:
            # Fallback for older versions
            if is_multichannel:
                score, _ = ssim(target, prediction, full=True, multichannel=True)
            else:
                score, _ = ssim(target, prediction, full=True)
        return float(score)
    
    # Simple manual fallback for SSIM if skimage fails
    # (Simplified structural similarity formula over 11x11 sliding window)
    t_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if len(target.shape) == 3 else target
    p_gray = cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY) if len(prediction.shape) == 3 else prediction
    
    t_gray = t_gray.astype(np.float64)
    p_gray = p_gray.astype(np.float64)
    
    mu_x = cv2.GaussianBlur(t_gray, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(p_gray, (11, 11), 1.5)
    
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = cv2.GaussianBlur(t_gray ** 2, (11, 11), 1.5) - mu_x_sq
    sigma_y_sq = cv2.GaussianBlur(p_gray ** 2, (11, 11), 1.5) - mu_y_sq
    sigma_xy = cv2.GaussianBlur(t_gray * p_gray, (11, 11), 1.5) - mu_xy
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return float(np.mean(ssim_map))
