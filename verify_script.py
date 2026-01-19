import cv2
import numpy as np
import sys
import os

# Ensure we can import processing
sys.path.append("d:/antigravity/GradeSense-AI")

try:
    from processing import ColorEngine
    print("Module loaded successfully.")
except ImportError as e:
    print(f"Failed to load module: {e}")
    sys.exit(1)

def test_logic():
    print("Creating dummy images...")
    # Create random images
    img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 1. Test Dominant Colors
    print("Testing K-Means...")
    colors = ColorEngine.get_dominant_colors(img1, k=5)
    print(f"Dominant Colors: {colors}")
    if len(colors) != 5:
        print("Error: Did not return 5 colors.")
        sys.exit(1)
        
    # 2. Test Match
    print("Testing Color Match (Smart Scale)...")
    matched = ColorEngine.apply_color_match_smart(img1, img2)
    if matched.shape != img1.shape:
        print(f"Error: Shape mismatch. Orig: {img1.shape}, Matched: {matched.shape}")
        sys.exit(1)
    
    # 3. Test Metrics
    print("Testing Metrics...")
    metrics = ColorEngine.calculate_grade_metrics(img1, matched)
    print(f"Metrics: {metrics}")
    
    # 4. Test Consultant Logic
    print("Testing Consultant Logic...")
    offsets = ColorEngine.calculate_pro_offsets(img1, img2)
    print(f"Offsets: {offsets}")
    if "Lift" not in offsets or "Gamma" not in offsets or "Gain" not in offsets:
        print("Error: Missing keys in offsets.")
        sys.exit(1)
        
    # 5. Test LUT Gen
    print("Testing LUT Generation...")
    lut = ColorEngine.create_3d_lut(colors, {"Midtones": "#A0A0A0"})
    if "LUT_3D_SIZE 33" not in lut:
        print("Error: Invalid LUT header.")
        sys.exit(1)
        
    # 6. Test Match Consultant
    print("Testing Match Consultant...")
    advice = ColorEngine.analyze_advanced_grading(img1, img2)
    print(f"Advice: {advice}")
    if "Black Point" not in advice:
        print("Error: Missing advice keys.")
        sys.exit(1)

    # 7. Test Match LUT
    print("Testing Match LUT...")
    match_lut = ColorEngine.generate_match_lut(img1, img2)
    if "GradeSense AI Match LUT" not in match_lut:
         print("Error: Invalid Match LUT header.")
         sys.exit(1)
    
    print("ALL TESTS PASSED.")

if __name__ == "__main__":
    test_logic()
