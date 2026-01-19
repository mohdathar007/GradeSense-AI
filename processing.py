import cv2
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from skimage import exposure, color, transform, metrics
from collections import Counter

class ColorEngine:
    @staticmethod
    @st.cache_data
    def get_dominant_colors(image_rgb, k=5):
        """
        Extracts dominant colors using K-Means clustering.
        Args:
            image_rgb (numpy.ndarray): Input image in RGB format.
            k (int): Number of clusters.
        Returns:
            list: List of hex color strings.
        """
        # Resize for speed (proxy)
        max_dim = 150
        h, w, _ = image_rgb.shape
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_small = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_small = image_rgb

        pixels = img_small.reshape(-1, 3)
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        
        # Sort colors by frequency
        counts = Counter(kmeans.labels_)
        sorted_indices = [i[0] for i in sorted(counts.items(), key=lambda x: x[1], reverse=True)]
        sorted_colors = colors[sorted_indices]
        
        hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in sorted_colors]
        return hex_colors

    @staticmethod
    def _create_lab_transfer_function(src_channel, matched_channel):
        """
        Creates a transfer function (interpolation) map from source values to matched values.
        """
        # Determine unique values and their mappings
        # To make it robust, we can use a binned approach or just sort and interp
        # A simple method: Sort both defined lists. T(src_sorted[i]) = matched_sorted[i]
        # But we need T(x).
        
        src_flat = src_channel.ravel()
        dst_flat = matched_channel.ravel()
        
        # Subsample if too large to sort quickly, though 720p is fine
        if len(src_flat) > 100000:
            indices = np.linspace(0, len(src_flat) - 1, 100000).astype(int)
            src_flat = src_flat[indices]
            dst_flat = dst_flat[indices]

        src_sorted = np.sort(src_flat)
        dst_sorted = np.sort(dst_flat)
        
        # Remove duplicates from src for unique mapping x -> y?
        # Actually np.interp needs x to be increasing.
        # But simply mapping sorted quantiles is the definition of histogram matching.
        # We want to map value v from src to a value v' such that CDF_src(v) = CDF_dst(v')
        
        # We can just return the sorted arrays to use as a lookup via interpolation
        return src_sorted, dst_sorted

    @staticmethod
    @st.cache_data
    def apply_color_match_smart(source_rgb, reference_rgb):
        """
        Applies histogram matching using a Dual-Stream (Smart Scale) approach in LAB space.
        Analyzes 720p proxies to learn the transform, applies to full-res.
        """
        # 1. Create Proxies (max 720p)
        proxy_size = 720
        h, w, _ = source_rgb.shape
        scale_s = proxy_size / max(h, w) if max(h, w) > proxy_size else 1.0
        
        h_r, w_r, _ = reference_rgb.shape
        scale_r = proxy_size / max(h_r, w_r) if max(h_r, w_r) > proxy_size else 1.0
        
        if scale_s < 1.0:
            s_proxy = cv2.resize(source_rgb, (int(w * scale_s), int(h * scale_s)), interpolation=cv2.INTER_AREA)
        else:
            s_proxy = source_rgb
            
        if scale_r < 1.0:
            r_proxy = cv2.resize(reference_rgb, (int(w_r * scale_r), int(h_r * scale_r)), interpolation=cv2.INTER_AREA)
        else:
            r_proxy = reference_rgb

        # 2. Convert Proxies to LAB
        s_lab = color.rgb2lab(s_proxy)
        r_lab = color.rgb2lab(r_proxy)

        # 3. Match Histograms on Proxies
        # match_histograms returns the transformed source
        matched_proxy_lab = exposure.match_histograms(s_lab, r_lab, channel_axis=2)

        # 4. Learn Transfer Function from Proxy -> Matched Proxy
        # For each channel L, A, B
        transforms = []
        for i in range(3):
            src_sorted, dst_sorted = ColorEngine._create_lab_transfer_function(s_lab[..., i], matched_proxy_lab[..., i])
            transforms.append((src_sorted, dst_sorted))

        # 5. Apply Transfer Function to Full-Res Source
        full_source_lab = color.rgb2lab(source_rgb)
        
        final_lab = np.zeros_like(full_source_lab)
        
        for i in range(3):
            src_vals, dst_vals = transforms[i]
            # Interpolate the mapping for the full image pixels
            # np.interp(x, xp, fp)
            final_lab[..., i] = np.interp(full_source_lab[..., i].ravel(), src_vals, dst_vals).reshape(full_source_lab[..., i].shape)

        # 6. Convert back to RGB
        final_rgb = color.lab2rgb(final_lab)
        
        # Ensure range 0-1 (scikit-image lab2rgb output) -> 0-255 uint8 if input was uint8
        if source_rgb.dtype == np.uint8:
            final_rgb = (final_rgb * 255).clip(0, 255).astype(np.uint8)
            
        return final_rgb

    @staticmethod
    def calculate_grade_metrics(source_rgb, target_rgb):
        """
        Calculates MSE, PSNR, and RGB Channel Shifts.
        Input images should be resized to the same dimensions before calling this, ideally.
        But we will handle resizing here to match source's dimensions for comparison.
        """
        # Resize target to match source for pixel-wise comparison
        if source_rgb.shape != target_rgb.shape:
             h, w, _ = source_rgb.shape
             target_rgb = cv2.resize(target_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        # Metrics
        mse = metrics.mean_squared_error(source_rgb, target_rgb)
        psnr = metrics.peak_signal_noise_ratio(source_rgb, target_rgb) if mse > 0 else 100
        
        # Channel Shift (Delta)
        # Average(Target) - Average(Source)
        # Assuming uint8
        avg_src = source_rgb.mean(axis=(0, 1))
        avg_tgt = target_rgb.mean(axis=(0, 1))
        diff = avg_tgt - avg_src # R, G, B
        
        return {
            "MSE": mse,
            "PSNR": psnr,
            "R_Shift": diff[0],
            "G_Shift": diff[1],
            "B_Shift": diff[2]
        }

    @staticmethod
    def calculate_pro_offsets(source_rgb, reference_rgb):
        """
        Calculates specific Lift/Gamma/Gain offsets based on luminance regions.
        Lift (Shadows): 0-30%
        Gamma (Midtones): 30-70%
        Gain (Highlights): 70-100%
        Returns dictionary with offsets for Premiere (-100 to 100) and Resolve (0.0 to 1.0 base).
        """
        # Ensure resized for comparison
        if source_rgb.shape != reference_rgb.shape:
             h, w, _ = source_rgb.shape
             reference_rgb = cv2.resize(reference_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
             
        # Convert to floats 0-1
        src_f = source_rgb.astype(np.float32) / 255.0
        ref_f = reference_rgb.astype(np.float32) / 255.0
        
        # Luminance for masking (using Rec.709 weights)
        lum_src = 0.2126 * src_f[..., 0] + 0.7152 * src_f[..., 1] + 0.0722 * src_f[..., 2]
        
        # Masks
        shadows_mask = lum_src < 0.3
        highlights_mask = lum_src > 0.7
        midtones_mask = (lum_src >= 0.3) & (lum_src <= 0.7)
        
        # Helper to get mean diff per channel in a masked region
        def get_diff(mask):
            if np.sum(mask) == 0:
                return np.array([0.0, 0.0, 0.0])
            
            s_mean = src_f[mask].mean(axis=0)
            r_mean = ref_f[mask].mean(axis=0)
            return r_mean - s_mean

        lift_diff = get_diff(shadows_mask)   # RGB delta
        gamma_diff = get_diff(midtones_mask)
        gain_diff = get_diff(highlights_mask)
        
        # Scale for Premiere (-100 to 100, approx)
        # Visual range for lift/gamma/gain wheels often maps small float shifts to larger slider moves
        # Let's approximate: 0.1 float diff ~ 20 units in Premiere? This is heuristic.
        # But user requested mapping RGB mean diffs.
        
        return {
            "Lift": lift_diff, # Float delta
            "Gamma": gamma_diff,
            "Gain": gain_diff
        }

    @staticmethod
    def create_3d_lut(source_palette, params):
        """
        Generates a 33pt .cube LUT string using Professional Grading Math.
        Supports: Lift (Offset), Gamma (Power), Gain (Slope), Contrast (S-Curve), Saturation.
        """
        size = 33
        lut_str = [
            "# GradeSense AI Professional LUT",
            f"LUT_3D_SIZE {size}",
        ]
        
        # Extract Params (Defaults if missing)
        lift = params.get("Lift", 0.0)   # -1.0 to 1.0 (Approx)
        gamma = params.get("Gamma", 0.0) # -1.0 to 1.0
        gain = params.get("Gain", 0.0)   # -1.0 to 1.0
        contrast = params.get("Contrast", 1.0) # 0.0 to 2.0+
        sat = params.get("Sat", 1.0)     # 0.0 to 2.0+
        mid_tint_hex = params.get("Midtones", "#808080")
        
        # Convert Tint
        def hex_to_rgb(hex_code):
            hex_code = hex_code.lstrip('#')
            return np.array([int(hex_code[i:i+2], 16) for i in (0, 2, 4)]) / 255.0
        
        tint_rgb = hex_to_rgb(mid_tint_hex)
        tint_strength = 0.6 # Boosted to Ensure Visibility
        
        # Pre-calc Gamma Power (Simple mapping: -0.5 -> 0.5 power? No.)
        # Resolve Gamma: 0.0 is Identity (1.0). +0.1 -> Brighten.
        # Math: val ^ (1 / (1 + gamma_val))
        gamma_power = 1.0 / (1.0 + gamma * 2.0) # Scaling gamma effect
        
        step = 1.0 / (size - 1)
        
        for b in range(size):
            for g in range(size):
                for r in range(size):
                    # 1. Base RGB
                    col = np.array([r * step, g * step, b * step], dtype=np.float32)
                    
                    # 2. LIFT (Shadow Offset) - Weighted by inverse Luma slightly to protect absolute white? 
                    # Resolve Lift is mostly a pure offset with soft clip. We'll do offset.
                    col = col + lift
                    
                    # 3. GAIN (Highlight Scale) - Pivot is 0
                    col = col * (1.0 + gain)
                    col = np.clip(col, 0.001, 2.0) # Avoid neg/div0
                    
                    # 4. GAMMA (Midtone Power)
                    col = np.power(col, gamma_power)
                    
                    # 5. CONTRAST (S-Curve)
                    # Pivot 0.5
                    col = (col - 0.5) * contrast + 0.5
                    
                    # 6. MIDTONE TINT (Split Tone)
                    # Mix tint based on parabolic midtone mask (peaking at 0.5 luma)
                    luma = np.dot(col, [0.299, 0.587, 0.114])
                    mid_mask = 1.0 - 4.0 * ((luma - 0.5) ** 2) # Parabola 0->1->0
                    mid_mask = np.clip(mid_mask, 0.0, 1.0)
                    
                    if mid_tint_hex != "#808080":
                         # Tint towards target color in midtones
                         # Blend "col" with "tint_rgb * luma" ?
                         # Let's just push vector
                         tint_vector = (tint_rgb - 0.5) * 2.0 # -1 to 1 shift
                         col += tint_vector * (mid_mask * tint_strength)

                    # 7. SATURATION
                    # Interpolate between Luma (Greyscale) and Color
                    luma_v = np.dot(col, [0.299, 0.587, 0.114])
                    grey = np.array([luma_v, luma_v, luma_v])
                    col = grey + (col - grey) * sat

                    # Final Clip
                    col = np.clip(col, 0.0, 1.0)
                    
                    lut_str.append(f"{col[0]:.6f} {col[1]:.6f} {col[2]:.6f}")
                    
        return "\n".join(lut_str)
        
    @staticmethod
    def apply_grade_to_palette(palette, params):
        """
        Applies grading math to a 1D palette list.
        palette: (N, 3) Float RGB array OR list of Hex strings
        params: dict of grading params
        Returns: (N, 3) formatted hex string list
        """
        def hex_to_rgb(hex_code):
            hex_code = hex_code.lstrip('#')
            return np.array([int(hex_code[i:i+2], 16) for i in (0, 2, 4)]) / 255.0
            
        # Parse Input format
        if isinstance(palette, list) and isinstance(palette[0], str) and palette[0].startswith("#"):
             # Convert Hex List to RGB Float Array
             palette = np.array([hex_to_rgb(c) for c in palette])
        else:
             # Assume numeric
             if isinstance(palette, list): palette = np.array(palette)
             if palette.max() > 1.0: palette = palette / 255.0
        
        lift = params.get("Lift", 0.0)
        gamma = params.get("Gamma", 0.0)
        gain = params.get("Gain", 0.0)
        contrast = params.get("Contrast", 1.0)
        sat = params.get("Sat", 1.0)
        mid_tint_hex = params.get("Midtones", "#808080")
        
        tint_rgb = hex_to_rgb(mid_tint_hex)
        tint_strength = 0.6 # Boosted to 0.6 for Visibility
        
        gamma_power = 1.0 / (1.0 + gamma * 2.0)
        
        result_hex = []
        for i in range(len(palette)):
            col = palette[i].astype(np.float32)
            
            # Same pipeline as LUT
            col = col + lift
            col = col * (1.0 + gain)
            col = np.clip(col, 0.001, 2.0)
            col = np.power(col, gamma_power)
            col = (col - 0.5) * contrast + 0.5
            
            luma = np.dot(col, [0.299, 0.587, 0.114])
            mid_mask = np.clip(1.0 - 4.0 * ((luma - 0.5) ** 2), 0.0, 1.0)
            
            if mid_tint_hex != "#808080":
                 tint_vector = (tint_rgb - 0.5) * 2.0
                 col += tint_vector * (mid_mask * tint_strength)
            
            grey = np.array([luma, luma, luma])
            col = grey + (col - grey) * sat
            col = np.clip(col, 0.0, 1.0)
            
            # To Hex
            r, g, b = (col * 255).astype(int)
            result_hex.append(f"#{r:02x}{g:02x}{b:02x}")
            
        return result_hex

    @staticmethod
    def generate_match_lut(source_rgb, reference_rgb):
        """
        Generates a 33pt .cube LUT that replicates the histogram match transform.
        Unlike create_3d_lut which used a simple tint, this learns the actual transform
        from Source -> Reference using the same logic as apply_color_match_smart.
        """
        # 1. Learn the transform (Reuse logic efficiently)
        # We need the transfer functions.
        # Let's extract the transfer connection logic or re-compute it on proxies.
        
        proxy_size = 720
        h, w, _ = source_rgb.shape
        scale_s = proxy_size / max(h, w) if max(h, w) > proxy_size else 1.0
        
        h_r, w_r, _ = reference_rgb.shape
        scale_r = proxy_size / max(h_r, w_r) if max(h_r, w_r) > proxy_size else 1.0
        
        if scale_s < 1.0:
            s_proxy = cv2.resize(source_rgb, (int(w * scale_s), int(h * scale_s)), interpolation=cv2.INTER_AREA)
        else:
            s_proxy = source_rgb
            
        if scale_r < 1.0:
            r_proxy = cv2.resize(reference_rgb, (int(w_r * scale_r), int(h_r * scale_r)), interpolation=cv2.INTER_AREA)
        else:
            r_proxy = reference_rgb

        s_lab = color.rgb2lab(s_proxy)
        r_lab = color.rgb2lab(r_proxy)
        matched_proxy_lab = exposure.match_histograms(s_lab, r_lab, channel_axis=2)

        transforms = []
        for i in range(3):
            src_sorted, dst_sorted = ColorEngine._create_lab_transfer_function(s_lab[..., i], matched_proxy_lab[..., i])
            transforms.append((src_sorted, dst_sorted))
            
        # 2. Apply to Identity Cube
        size = 33
        lut_str = [
            "# GradeSense AI Match LUT",
            f"LUT_3D_SIZE {size}",
        ]
        
        step = 1.0 / (size - 1)
        
        # We need to map RGB cube point -> LAB -> Transform LAB -> RGB'
        
        # To speed up, we can process the whole cube as a reshape
        # But for 33^3 = 35937 points, loop is 'okay' but vectorized is better.
        # Let's try vectorized approach for the cube.
        
        # Create meshgrid
        r = np.linspace(0, 1, size)
        g = np.linspace(0, 1, size)
        b = np.linspace(0, 1, size)
        
        # Note: .cube order is usually: outer Red, middle Green, inner Blue ? 
        # Actually spec: B changes fastest, then G, then R.
        # So loop R, then G, then B.
        
        rr, gg, bb = np.meshgrid(r, g, b, indexing='ij')
        cube_rgb = np.stack([rr, gg, bb], axis=-1).reshape(-1, 3) 
        
        # Convert cube to LAB
        cube_lab = color.rgb2lab(cube_rgb) # Scikit image expects 0-1 floats for RGB? Yes.
        
        # Apply transforms
        final_cube_lab = np.zeros_like(cube_lab)
        for i in range(3):
            src_vals, dst_vals = transforms[i]
            final_cube_lab[..., i] = np.interp(cube_lab[..., i], src_vals, dst_vals)
            
        # Convert back to RGB
        final_cube_rgb = color.lab2rgb(final_cube_lab)
        
        # Clip
        final_cube_rgb = np.clip(final_cube_rgb, 0.0, 1.0)
        
        # Format string
        for val in final_cube_rgb:
            lut_str.append(f"{val[0]:.6f} {val[1]:.6f} {val[2]:.6f}")
            
        return "\n".join(lut_str)

    @staticmethod
    def prompt_to_numerical_offsets(prompt):
        """
        Parses natural language prompt to generate grading offsets.
        Uses a deterministic keyword-based engine to simulate AI "understanding".
        Returns a dictionary with 'Lift', 'Gamma', 'Gain' (RGB deltas) and 'Explanation'.
        """
        prompt = prompt.lower()
        
        # Default State (Neutral)
        # Offsets are delta factors. 1.0 = Neutral.
        # But our rendering logic uses additive deltas or direct colors.
        # Let's align with the create_3d_lut logic which expects a Target Tint for Midtones/Highs.
        
        # We will generate a "Target Midtone Tint" (Hex) and "Target Highlight Tint" (Hex)
        # And some prose.
        
        param_state = {
            "Midtones": np.array([0.5, 0.5, 0.5]), # Normalized RGB
            "Contrast": 1.0,
            "Sat": 1.0,
            "Explanation": "No specific mood detected. Applying neutral balanced grade."
        }
        
        explanation_parts = []
        
        # -- DECISION TREES --
        
        # 1. Temperature (Warm vs Cold)
        if any(w in prompt for w in ["warm", "gold", "summer", "heat", "sepia", "desert", "sun", "orange"]):
            param_state["Midtones"] += np.array([0.25, 0.05, -0.2]) # Aggressive Warmth
            param_state["Gamma_Scalar"] = 0.06 
            param_state["Contrast"] = 1.1 
            explanation_parts.append("Deep warmth engaged: Boosted red/orange in mids")
        elif any(w in prompt for w in ["cold", "blue", "winter", "ice", "steel", "teal", "cool", "frost"]):
            param_state["Midtones"] += np.array([-0.25, 0.0, 0.35]) # Aggressive Cool
            param_state["Gamma_Scalar"] = -0.08 
            param_state["Contrast"] = 1.15 
            explanation_parts.append("Deep freeze engaged: Strong teal/blue push in mids")

        # 2. Vibe / Era
        if any(w in prompt for w in ["vintage", "retro", "70s", "80s", "film", "nostalgic", "classic"]):
            param_state["Contrast"] = 0.85 
            param_state["Midtones"] += np.array([0.1, 0.1, -0.1]) 
            param_state["Lift_Scalar"] = 0.08 
            explanation_parts.append("Faded film stock: Lifted blacks and warm wash")
        elif any(w in prompt for w in ["cyberpunk", "neon", "future", "sci-fi", "matrix", "blade runner"]):
            param_state["Contrast"] = 1.3
            param_state["Sat"] = 1.6
            if "matrix" in prompt or "green" in prompt:
                param_state["Midtones"] = np.array([0.1, 0.9, 0.1]) 
                explanation_parts.append("Matrix Green: Hard gamma shift")
            else:
                param_state["Midtones"] += np.array([0.2, -0.1, 0.4]) 
                explanation_parts.append("Vaporwave: Split-tone pink/cyan aggression")
                
        if any(w in prompt for w in ["noir", "bw", "black and white", "monochrome", "grey"]):
            param_state["Sat"] = 0.0
            param_state["Contrast"] = 1.6
            explanation_parts.append("Noir: Hard Black & White conversion")
            
        if any(w in prompt for w in ["corporate", "clean", "minimal", "white", "bright", "modern"]):
            param_state["Midtones"] = np.array([0.45, 0.45, 0.55]) 
            param_state["Contrast"] = 1.05
            param_state["Gamma_Scalar"] = 0.02
            explanation_parts.append("Corporate Clean: Neutralized and brightened")

        # Cleanup
        param_state["Midtones"] = np.clip(param_state["Midtones"], 0.0, 1.0)
        
        # Convert Midtone RGB to Hex for LUT consumption
        r, g, b = (param_state["Midtones"] * 255).astype(int)
        hex_tint = f"#{r:02x}{g:02x}{b:02x}"
        
        if explanation_parts:
            param_state["Explanation"] = ". ".join(explanation_parts) + "."
            
        # -- MAP TO RESOLVE OFFSETS --
        # Base delta = (Contrast - 1.0) * Factor
        c_delta = param_state["Contrast"] - 1.0
        
        # Lift/Gain primarily driven by Contrast
        lift_val = -c_delta * 0.15 
        gain_val = c_delta * 0.15
        
        # Add explicit scalar overrides if set
        if "Lift_Scalar" in param_state: lift_val += param_state["Lift_Scalar"]
        
        # Gamma is driven by explicitly scalar OR deduced from Luma shift of Midtones
        gamma_val = param_state.get("Gamma_Scalar", 0.0)
        
        return {
            "Midtones": hex_tint, 
            "Contrast": param_state["Contrast"],
            "Sat": param_state["Sat"],
            "Lift": lift_val,
            "Gamma": gamma_val,
            "Gain": gain_val,
            "Explanation": param_state["Explanation"]
        }

    @staticmethod
    def analyze_advanced_grading(source_rgb, reference_rgb):
        """
        Analyzes YCrCb diffs to provide semantic advice.
        Returns detailed dictionary of issues.
        """
        # Resize Ref
        if source_rgb.shape != reference_rgb.shape:
             h, w, _ = source_rgb.shape
             reference_rgb = cv2.resize(reference_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
             
        # Convert to YCrCb (OpenCV uses BGR internally usually, but our logic passes RGB. 
        # cv2.COLOR_RGB2YCrCb exists)
        s_ycc = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        r_ycc = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        
        # Channels: Y (Luma), Cr (Red-diff), Cb (Blue-diff)
        # Ranges: Y [0, 255], Cr/Cb [0, 255]
        
        advice = {}
        
        # 1. Black Point / Shadows
        # Compare lowest 5 percentile of Luma
        s_black = np.percentile(s_ycc[..., 0], 5)
        r_black = np.percentile(r_ycc[..., 0], 5)
        
        if s_black > r_black + 10:
            advice["Black Point"] = "Your shadows are lifted. Crush the blacks to match density."
        elif s_black < r_black - 10:
            advice["Black Point"] = "Your shadows are too crushed. Lift the blacks slightly."
        else:
            advice["Black Point"] = "Shadow density matches well."
            
        # 2. Highlight Roll-off
        # Compare highest 5 percentile
        s_white = np.percentile(s_ycc[..., 0], 95)
        r_white = np.percentile(r_ycc[..., 0], 95)
        
        if s_white > r_white + 15:
            advice["Highlights"] = "Source matches are brighter. Apply a highlight roll-off."
        elif s_white < r_white - 15:
            advice["Highlights"] = "Source lacks dynamic range in highlights. Boost gain."
        else:
            advice["Highlights"] = "Dynamic range at the top end matches."
            
        # 3. Color Cast (Average Cr/Cb)
        s_cr = s_ycc[..., 1].mean()
        r_cr = r_ycc[..., 1].mean()
        s_cb = s_ycc[..., 2].mean()
        r_cb = r_ycc[..., 2].mean()
        
        casts = []
        if s_cr > r_cr + 5: casts.append("Source is warmer/redder.")
        if s_cr < r_cr - 5: casts.append("Source is cooler/greener.")
        if s_cb > r_cb + 5: casts.append("Source matches are too Blue.")
        if s_cb < r_cb - 5: casts.append("Source matches are too Yellow.")
        
        advice["Color Cast"] = " ".join(casts) if casts else "Overall tint is similar."
        
        return advice

    @staticmethod
    def generate_mood_explanation(mood, palette):
        """
        Returns 3 distinct "Why" sentences based on the mood.
        """
        # We can make this dynamic based on the palette if we want, but for now, distinct mood theory.
        # Palette is available if we want to say "Your dominant Teals..."
        
        explanations = {
            "Cyberpunk": [
                "Increases Cyan/Magenta separation to mimic neon-lit environments.",
                "Crushes blacks while boosting highlight saturation for high-contrast impact.",
                "Shifts skin tones towards cool magentas to detach them from reality."
            ],
            "Noir": [
                "Eliminates saturation to focus purely on lighting ratios and texture.",
                "Deepens shadows significantly to create mystery and negative space.",
                "Increases local contrast (clarity) to emphasize grit and grain structures."
            ],
            "Desert": [
                "Warms up the white balance to simulate the golden hour or harsh sun.",
                "Lifts yellow/orange midtones to create a 'heat haze' atmosphere.",
                "Rolls off blue channels in the shadows to maintain a dusty warmth throughout."
            ],
            "Corporate": [
                "Neutralizes white balance for a clean, trustworthy aesthetic.",
                "Slightly cools shadows (Teal) to contrast with warm skin tones.",
                "Maintains high midtone detail and safeguards highlights from clipping."
            ],
            "Cinematic (Default)": [
                "Applies a subtle S-Curve to expand contrast while preserving midtone roll-off.",
                "Balances vector scope skin tone lines for naturalistic appeal.",
                "Introduces a very slight teal/orange split to separate subject from background."
            ]
        }
        return explanations.get(mood, explanations["Cinematic (Default)"])

    @staticmethod
    def get_safe_resolve_offsets(mood):
        """
        Returns Resolve-compatible (0.0 - 1.0 scale approx) offsets for display.
        This provides 'suggested' values for the user to dial in manually if they want.
        """
        # These are heuristic "cookbook" values.
        offsets = {
            "Cyberpunk": {"Lift": -0.02, "Gamma": 0.05, "Gain": 1.10, "Sat": 1.2},
            "Noir": {"Lift": -0.05, "Gamma": -0.10, "Gain": 0.90, "Sat": 0.0},
            "Desert": {"Lift": 0.01, "Gamma": 0.08, "Gain": 1.05, "Sat": 1.1},
            "Corporate": {"Lift": 0.00, "Gamma": 0.00, "Gain": 1.00, "Sat": 1.0},
            "Cinematic (Default)": {"Lift": -0.01, "Gamma": -0.02, "Gain": 1.02, "Sat": 1.0}
        }
        return offsets.get(mood, offsets["Cinematic (Default)"])

import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ScopeEngine:
    @staticmethod
    @st.cache_data
    def generate_waveform(image, mode='luma'):
        """
        Generates a Luma Waveform Density Plot using Plotly Heatmap.
        image: RGB float (0-1) or int (0-255). We will convert to 0-255 uint8.
        """
        # Ensure 0-255
        if image.dtype != np.uint8:
             src_img = (image * 255).clip(0, 255).astype(np.uint8)
        else:
             src_img = image.copy()
             
        # 1. Downsample for Performance
        h, w, c = src_img.shape
        if w > 1024:
            scale = 1024 / w
            src_img = cv2.resize(src_img, (1024, int(h * scale)), interpolation=cv2.INTER_AREA)
        
        # 2. Convert to Luma (IRE Scale 0-100)
        # Rec. 709
        # cv2 uses BGR usually, but load_image returns RGB.
        luma = np.dot(src_img, [0.2126, 0.7152, 0.0722])
        luma = (luma / 255.0 * 100).astype(np.uint8) # 0-100 IRE
        
        # 3. Create 2D Density Map
        h_resized, w_resized = luma.shape
        
        # Vectorized 2D Histogram
        # Flatten x (cols) and y (luma) maps
        x_indices = np.tile(np.arange(w_resized), (h_resized, 1)).flatten()
        y_vals = luma.flatten()
        
        # Bin into (W, 101)
        H, _, _ = np.histogram2d(x_indices, y_vals, bins=[w_resized, 101], range=[[0, w_resized], [0, 100]])
        
        # Transpose to (101, W) for plotting (Y is rows)
        density_map = H.T 
        
        # Log scale
        density_map = np.log1p(density_map)
        
        # Plot
        fig = go.Figure(data=go.Heatmap(
            z=density_map,
            x=np.arange(w_resized),
            y=np.arange(101),
            colorscale='Inferno',
            showscale=False,
            zmin=0,
            zmax=np.max(density_map) * 0.8
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=20, t=10, b=30),
            xaxis=dict(title="Image Width", showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(title="IRE", showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[0, 100]),
            height=300
        )
        return fig

    @staticmethod
    @st.cache_data
    def generate_rgb_parade(image):
        """
        Generates RGB Parade.
        """
        if image.dtype != np.uint8:
             src_img = (image * 255).clip(0, 255).astype(np.uint8)
        else:
             src_img = image.copy()
             
        # 1. Downsample (Parade is usually 3x smaller width effectively per channel)
        h, w, c = src_img.shape
        target_w = 340 
        scale = target_w / w
        if w > target_w:
             src_img = cv2.resize(src_img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        
        h_s, w_s, _ = src_img.shape
        
        channels = [src_img[..., 0], src_img[..., 1], src_img[..., 2]] 
        names = ['Red', 'Green', 'Blue']
        cmaps = ['Reds', 'Greens', 'Blues'] 
        
        fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.02)
        
        x_indices = np.tile(np.arange(w_s), (h_s, 1)).flatten()
        
        for i, (chan, name, cmap) in enumerate(zip(channels, names, cmaps)):
            y_vals = chan.flatten()
            H, _, _ = np.histogram2d(x_indices, y_vals, bins=[w_s, 256], range=[[0, w_s], [0, 255]])
            density = np.log1p(H.T)
            
            fig.add_trace(
                go.Heatmap(z=density, colorscale=cmap, showscale=False),
                row=1, col=i+1
            )
            
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=30, r=10, t=20, b=20),
            yaxis=dict(title="8-bit", range=[0, 255], gridcolor='rgba(255,255,255,0.1)'),
            xaxis=dict(showticklabels=False),
            height=250,
            showlegend=False
        )
        return fig

    @staticmethod
    @st.cache_data
    def generate_histogram(image):
        """
        Generates Multi-Channel Histogram.
        """
        if image.dtype != np.uint8:
             src_img = (image * 255).clip(0, 255).astype(np.uint8)
        else:
             src_img = image.copy()
             
        if src_img.shape[1] > 1024:
             src_img = src_img[::2, ::2, :]
             
        fig = go.Figure()
        
        range_max = 255
        bins = 256
        
        # R
        rh, _ = np.histogram(src_img[..., 0], bins=bins, range=(0, range_max))
        fig.add_trace(go.Scatter(y=rh, fill='tozeroy', name='Red', line=dict(color='rgba(255, 50, 50, 0.8)', width=1)))
        
        # G
        gh, _ = np.histogram(src_img[..., 1], bins=bins, range=(0, range_max))
        fig.add_trace(go.Scatter(y=gh, fill='tozeroy', name='Green', line=dict(color='rgba(50, 255, 50, 0.8)', width=1)))
        
        # B
        bh, _ = np.histogram(src_img[..., 2], bins=bins, range=(0, range_max))
        fig.add_trace(go.Scatter(y=bh, fill='tozeroy', name='Blue', line=dict(color='rgba(50, 50, 255, 0.8)', width=1)))
        
        # Luma
        luma = np.dot(src_img, [0.299, 0.587, 0.114])
        lh, _ = np.histogram(luma, bins=bins, range=(0, range_max))
        fig.add_trace(go.Scatter(y=lh, name='Luma', line=dict(color='rgba(255, 255, 255, 0.6)', width=2, dash='dot')))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=20),
            yaxis=dict(showgrid=False, showticklabels=False),
            xaxis=dict(title="Level (0-255)", showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            height=200,
            showlegend=True,
            legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1)
        )
        return fig
