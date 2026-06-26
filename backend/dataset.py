import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class FLIRDataset(Dataset):
    """
    Dataset loader for FLIR Thermal Dataset (paired Thermal IR and RGB images).
    If the directory does not exist or has no images, it falls back to generating
    procedural mock thermal-RGB pairs on-the-fly for seamless testing and demonstration.
    """
    def __init__(self, root_dir, split="train", transform_ir=None, transform_rgb=None, target_size=(256, 256)):
        self.root_dir = root_dir
        self.split = split
        self.transform_ir = transform_ir
        self.transform_rgb = transform_rgb
        self.target_size = target_size
        
        # Directories for paired thermal and RGB
        self.thermal_dir = os.path.join(root_dir, split, "thermal")
        self.rgb_dir = os.path.join(root_dir, split, "rgb")
        
        self.image_names = []
        self.use_mock = True
        
        if os.path.exists(self.thermal_dir) and os.path.exists(self.rgb_dir):
            thermal_files = sorted(os.listdir(self.thermal_dir))
            # Match paired images (assuming they share base filenames or indexes)
            for f in thermal_files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Check if matching RGB file exists
                    rgb_path = os.path.join(self.rgb_dir, f)
                    if os.path.exists(rgb_path):
                        self.image_names.append(f)
                        
            if len(self.image_names) > 0:
                self.use_mock = False
                print(f"Loaded {len(self.image_names)} paired images from {root_dir} ({split} split)")
                
        if self.use_mock:
            print(f"Dataset directory not found or empty. Using procedural mock generator for '{split}' split.")
            # Set a fixed size for the mock dataset
            self.image_names = [f"mock_image_{i:04d}.png" for i in range(100 if split == "train" else 20)]

    def __len__(self):
        return len(self.image_names)

    def _generate_procedural_pair(self, index):
        """
        Generates a paired synthetic IR (1-channel) and RGB (3-channel) image.
        Simulates roads, building rectangles, and vehicle shapes.
        """
        # Create blank images
        h, w = self.target_size
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        # Background: green grass in RGB, cool dark color in IR
        rgb[:] = [34, 139, 34] # Forest Green
        thermal = np.zeros((h, w), dtype=np.uint8)
        thermal[:] = 40 # Cool background temperature
        
        # Seed generator based on index to ensure deterministic rendering
        rng = np.random.default_rng(index)
        
        # 1. Draw a road (warm road in IR, gray asphalt in RGB)
        road_width = rng.integers(40, 80)
        road_type = rng.choice(['horizontal', 'vertical', 'diagonal'])
        
        if road_type == 'horizontal':
            y_start = rng.integers(h // 4, 3 * h // 4)
            # Draw road on RGB
            cv2.rectangle(rgb, (0, y_start), (w, y_start + road_width), (100, 100, 100), -1)
            # Draw dashed yellow center line on road
            for x in range(0, w, 20):
                cv2.line(rgb, (x, y_start + road_width//2), (x + 10, y_start + road_width//2), (0, 255, 255), 2)
            # Draw on IR (asphalt absorbs heat, warm: 130)
            cv2.rectangle(thermal, (0, y_start), (w, y_start + road_width), 130, -1)
            
            # Draw cars on the road
            for _ in range(rng.integers(1, 3)):
                car_x = rng.integers(10, w - 40)
                car_y = y_start + road_width//2 + rng.choice([-15, 5])
                # Car in RGB (random color)
                car_color = tuple(map(int, rng.integers(0, 255, size=3)))
                cv2.rectangle(rgb, (car_x, car_y), (car_x + 30, car_y + 12), car_color, -1)
                # Windshield
                cv2.rectangle(rgb, (car_x + 8, car_y + 2), (car_x + 22, car_y + 10), (200, 200, 200), -1)
                # Car in IR (engine is hot: 250, body is warm: 180)
                cv2.rectangle(thermal, (car_x, car_y), (car_x + 30, car_y + 12), 180, -1)
                # Engine bay (front)
                cv2.rectangle(thermal, (car_x + 20, car_y + 2), (car_x + 28, car_y + 10), 250, -1)
                
        elif road_type == 'vertical':
            x_start = rng.integers(w // 4, 3 * w // 4)
            # Road RGB
            cv2.rectangle(rgb, (x_start, 0), (x_start + road_width, h), (100, 100, 100), -1)
            for y in range(0, h, 20):
                cv2.line(rgb, (x_start + road_width//2, y), (x_start + road_width//2, y + 10), (0, 255, 255), 2)
            # Road IR
            cv2.rectangle(thermal, (x_start, 0), (x_start + road_width, h), 130, -1)
            
            # Draw cars
            for _ in range(rng.integers(1, 3)):
                car_x = x_start + road_width//2 + rng.choice([-15, 5])
                car_y = rng.integers(10, h - 40)
                car_color = tuple(map(int, rng.integers(0, 255, size=3)))
                cv2.rectangle(rgb, (car_x, car_y), (car_x + 12, car_y + 30), car_color, -1)
                # Car IR
                cv2.rectangle(thermal, (car_x, car_y), (car_x + 12, car_y + 30), 180, -1)
                cv2.rectangle(thermal, (car_x + 2, car_y + 20), (car_x + 10, car_y + 28), 250, -1) # engine
        else:
            # Diagonal Road
            cv2.line(rgb, (0, 0), (w, h), (100, 100, 100), road_width)
            cv2.line(thermal, (0, 0), (w, h), 130, road_width)
            
        # 2. Draw Buildings (large rectangular structures, cool roofs but warm edges/vents)
        for _ in range(rng.integers(1, 4)):
            bw = rng.integers(50, 90)
            bh = rng.integers(50, 90)
            bx = rng.integers(10, w - bw - 10)
            by = rng.integers(10, h - bh - 10)
            
            # Avoid placing building on the center line of vertical/horizontal roads
            if road_type == 'horizontal' and abs(by + bh//2 - y_start) < road_width:
                continue
            if road_type == 'vertical' and abs(bx + bw//2 - x_start) < road_width:
                continue
                
            # Building RGB (beige or brick color roof)
            roof_color = tuple(map(int, rng.choice([[210, 180, 140], [188, 143, 143], [112, 128, 144]])))
            cv2.rectangle(rgb, (bx, by), (bx + bw, by + bh), roof_color, -1)
            # Add a border line to mimic parapets
            cv2.rectangle(rgb, (bx, by), (bx + bw, by + bh), (50, 50, 50), 2)
            # Draw AC unit vents on roof
            cv2.rectangle(rgb, (bx + bw//3, by + bh//3), (bx + bw//3 + 12, by + bh//3 + 12), (200, 200, 200), -1)
            
            # Building IR (Roofs reflect heat, can be cool (~90), but AC vents are hot (~240))
            cv2.rectangle(thermal, (bx, by), (bx + bw, by + bh), 90, -1)
            cv2.rectangle(thermal, (bx + bw//3, by + bh//3), (bx + bw//3 + 12, by + bh//3 + 12), 240, -1)
            
        # 3. Add Gaussian noise to IR to simulate sensor noise
        noise = rng.normal(0, 5, thermal.shape).astype(np.int16)
        thermal = np.clip(thermal.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Make thermal low-contrast as specified (e.g. compress range to 50-180)
        thermal = (thermal * 0.5 + 40).astype(np.uint8)
        
        return Image.fromarray(thermal), Image.fromarray(rgb)

    def __getitem__(self, idx):
        if self.use_mock:
            # Generate synthetic image pairs
            thermal_img, rgb_img = self._generate_procedural_pair(idx)
        else:
            # Load real image pairs
            file_name = self.image_names[idx]
            thermal_path = os.path.join(self.thermal_dir, file_name)
            rgb_path = os.path.join(self.rgb_dir, file_name)
            
            thermal_img = Image.open(thermal_path).convert("L") # Ensure grayscale
            rgb_img = Image.open(rgb_path).convert("RGB") # Ensure RGB
            
            # Resize
            thermal_img = thermal_img.resize(self.target_size, Image.BILINEAR)
            rgb_img = rgb_img.resize(self.target_size, Image.BILINEAR)

        # Apply transforms if provided
        if self.transform_ir:
            ir_tensor = self.transform_ir(thermal_img)
        else:
            # Default transform to Tensor and Normalize to [-1, 1]
            ir_tensor = torch.tensor(np.array(thermal_img), dtype=torch.float32).unsqueeze(0) / 127.5 - 1.0
            
        if self.transform_rgb:
            rgb_tensor = self.transform_rgb(rgb_img)
        else:
            # Default transform to Tensor and Normalize to [-1, 1]
            rgb_tensor = torch.tensor(np.array(rgb_img), dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1.0
            
        return ir_tensor, rgb_tensor

if __name__ == "__main__":
    # Test dataset loading
    dataset = FLIRDataset(root_dir="dataset_path_does_not_exist", split="train")
    ir_t, rgb_t = dataset[0]
    print("Dataset item loaded successfully.")
    print("IR Tensor Shape:", ir_t.shape, "Min:", ir_t.min().item(), "Max:", ir_t.max().item())
    print("RGB Tensor Shape:", rgb_t.shape, "Min:", rgb_t.min().item(), "Max:", rgb_t.max().item())
    assert ir_t.shape == (1, 256, 256), "IR channel count must be 1"
    assert rgb_t.shape == (3, 256, 256), "RGB channel count must be 3"
    print("Procedural dataset verification successful!")
