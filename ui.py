import streamlit as st
import pandas as pd

def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        
        /* Ultraviolet Tech Theme */
        .stApp {
            background-color: #0a0a12;
            color: #FAFAFA;
            font-family: 'Inter', sans-serif;
            /* Layered: Grid on top, Radial Glow, Deep Base */
            background-image: 
                linear-gradient(rgba(124, 58, 237, 0.05) 1px, transparent 1px),
                linear-gradient(90deg, rgba(124, 58, 237, 0.05) 1px, transparent 1px),
                radial-gradient(circle at 50% 0%, #1e1b2e 0%, #0a0a12 70%);
            background-size: 40px 40px, 40px 40px, 100% 100%;
            background-attachment: fixed;
        }

        /* Hide Streamlit Anchors */
        .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a {
            display: none !important;
            pointer-events: none;
        }
        
        /* Control Deck Sidebar */
        section[data-testid="stSidebar"] {
            background-color: rgba(10, 10, 18, 0.85);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(124, 58, 237, 0.1);
        }

        /* Titles - Glowing */
        h1, h2, h3 {
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #ffffff;
            text-shadow: 0 0 20px rgba(124, 58, 237, 0.5);
        }
        h1 {
            background: linear-gradient(90deg, #ffffff, #e2e8f0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.2rem;
            filter: drop-shadow(0 0 10px rgba(124, 58, 237, 0.3));
        }
        .subtitle {
            text-align: center;
            color: #a855f7;
            font-size: 1.0em;
            margin-bottom: 3rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            text-shadow: 0 0 10px rgba(124, 58, 237, 0.3);
        }

        /* Glass Modules (Ultraviolet Spec) */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 20px;
            padding: 24px;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(124, 58, 237, 0.3);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5), inset 0 0 20px rgba(124, 58, 237, 0.1);
            margin-bottom: 24px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .glass-card:hover {
            border-color: rgba(168, 85, 247, 0.6);
            box-shadow: 0 0 30px rgba(124, 58, 237, 0.2), inset 0 0 20px rgba(124, 58, 237, 0.2);
            transform: translateY(-2px);
        }

        /* Energy Buttons */
        .stButton button {
            background: linear-gradient(135deg, rgba(124, 58, 237, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
            border: 1px solid rgba(124, 58, 237, 0.5);
            color: #e2e8f0;
            border-radius: 12px;
            font-weight: 600;
            letter-spacing: 0.05em;
            transition: all 0.3s ease;
            text-transform: uppercase;
            font-size: 0.9em;
        }
        .stButton button:hover {
            background: linear-gradient(135deg, rgba(124, 58, 237, 0.8) 0%, rgba(168, 85, 247, 0.8) 100%);
            color: #fff;
            box-shadow: 0 0 25px rgba(124, 58, 237, 0.6);
            border-color: #fff;
            transform: scale(1.02);
        }
        /* Primary Action Button Overrides */
        button[kind="primary"] {
            background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: white !important;
            box-shadow: 0 0 20px rgba(124, 58, 237, 0.4);
        }
        button[kind="primary"]:hover {
            box-shadow: 0 0 40px rgba(124, 58, 237, 0.6);
        }

        /* File Uploader - Energy Intake */
        div[data-testid="stFileUploader"] {
            border: 1px dashed rgba(124, 58, 237, 0.6);
            background: rgba(124, 58, 237, 0.05);
            border-radius: 16px;
            padding: 20px;
            transition: all 0.3s ease;
        }
        div[data-testid="stFileUploader"]:hover {
            background: rgba(124, 58, 237, 0.1);
            border-color: #a855f7;
            box-shadow: 0 0 20px rgba(124, 58, 237, 0.2);
        }

        /* Glass Input Area */
        .stTextArea textarea {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(124, 58, 237, 0.4) !important;
            color: #fff !important;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        .stTextArea textarea:focus {
            border-color: #a855f7 !important;
            box-shadow: 0 0 15px rgba(124, 58, 237, 0.4);
            background-color: rgba(255, 255, 255, 0.08) !important;
        }

        /* Image Frames */
        div[data-testid="stImage"] img {
            border: 1px solid rgba(124, 58, 237, 0.3);
            border-radius: 16px;
            box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.6);
        }

        /* Custom Metrics */
        .stat-label {
            color: #a855f7;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            margin-bottom: 8px;
            font-weight: 700;
        }
        .stat-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #ffffff;
            text-shadow: 0 0 15px rgba(124, 58, 237, 0.4);
        }
        .stat-desc {
            font-size: 0.85rem;
            color: rgba(226, 232, 240, 0.6);
            margin-top: 6px;
            font-weight: 400;
        }
        .stat-value.accent { color: #ffffff; }
        .stat-value.teal { color: #2dd4bf; text-shadow: 0 0 15px rgba(45, 212, 191, 0.4); }
        .stat-value.amber { color: #f59e0b; text-shadow: 0 0 15px rgba(245, 158, 11, 0.4); }

        </style>
    """, unsafe_allow_html=True)

def draw_glass_card(title, value, description="", color_class="accent"):
    """
    Renders a custom HTML metric card.
    """
    st.markdown(f"""
    <div class="glass-card">
        <div class="stat-label">{title}</div>
        <div class="stat-value {color_class}">{value}</div>
        <div class="stat-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown("<h1>GradeSense AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-Powered Color Science for Visual Storytellers</p>", unsafe_allow_html=True)

def render_swatches(hex_colors):
    if not hex_colors:
        return
    
    st.markdown("### Dominant Palette")
    cols = st.columns(len(hex_colors))
    for i, color in enumerate(hex_colors):
        with cols[i]:
            # Use st.color_picker for visual, but disabled/read-only sort of
            # Streamlit color picker is an input, but we can display a box
            st.markdown(
                f"""
                <div style="background-color: {color}; height: 80px; border-radius: 8px; border: 1px solid #333;"></div>
                <p style="text-align: center; font-family: monospace; margin-top: 5px;">{color}</p>
                """,
                unsafe_allow_html=True
            )

def render_grading_advice(mood, dominant_colors):
    """
    Returns grading advice based on mood and potentially dominant colors.
    """
    mood_profiles = {
        "Cinematic (Default)": {
            "Highlights": "#FFF8E7",
            "Midtones": "#A0A0A0",
            "Shadows": "#1A1A2E",
            "Tip": "Balanced natural contrast with rolled-off highlights."
        },
        "Cyberpunk": {
            "Highlights": "#00F0FF", # Cyan push
            "Midtones": "#6D28D9", # Purple push
            "Shadows": "#050510",
            "Tip": "Push Cyan/Magenta separation. Maintain high contrast."
        },
        "Noir": {
            "Highlights": "#E5E5E5",
            "Midtones": "#525252",
            "Shadows": "#000000",
            "Tip": "Desaturate completely or keep single hue. Crush blacks."
        },
        "Desert": {
            "Highlights": "#FFD700", # Gold
            "Midtones": "#D97706", # Amber
            "Shadows": "#451a03", # Warm brown
            "Tip": "Warm tint (Amber). Raise midtones for heat haze effect."
        },
        "Corporate": {
            "Highlights": "#FFFFFF",
            "Midtones": "#64748B",
            "Shadows": "#0F172A", # Teal lean
            "Tip": "Clean whites, teal-leaning shadows. High clarity/sharpness."
        }
    }
    
    advice = mood_profiles.get(mood, mood_profiles["Cinematic (Default)"])
    
    st.markdown(f"### {mood} Grading Advice")
    st.info(advice["Tip"])
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Target Highlights")
        st.color_picker("Highs", advice["Highlights"], disabled=True, key="h_pick")
    with c2:
        st.caption("Target Midtones")
        st.color_picker("Mids", advice["Midtones"], disabled=True, key="m_pick")
    with c3:
        st.caption("Target Shadows")
        st.color_picker("Shadows", advice["Shadows"], disabled=True, key="s_pick")

def render_grade_report(metrics):
    st.markdown("### Technical Grade Sheet")
    
    # 1. Accuracy
    col1, col2 = st.columns(2)
    col1.metric("SSIM/MSE Accuracy", f"{metrics['PSNR']:.2f} dB", delta="Signal-to-Noise")
    col2.metric("Mean Squared Error", f"{metrics['MSE']:.4f}", delta_color="inverse")
    
    # 2. Channel Shifts Table
    shifts = {
        "Channel": ["Red", "Green", "Blue"],
        "Average Shift (Delta)": [
            f"{metrics['R_Shift']:+.2f}",
            f"{metrics['G_Shift']:+.2f}",
            f"{metrics['B_Shift']:+.2f}"
        ],
        "Advice": [
            "Lift Red" if metrics['R_Shift'] > 5 else "Cut Red" if metrics['R_Shift'] < -5 else "Neutral",
            "Lift Green" if metrics['G_Shift'] > 5 else "Cut Green" if metrics['G_Shift'] < -5 else "Neutral",
            "Lift Blue" if metrics['B_Shift'] > 5 else "Cut Blue" if metrics['B_Shift'] < -5 else "Neutral"
        ]
    }
    
    st.table(pd.DataFrame(shifts))

def render_grading_consultant(values):
    st.markdown("### Grading Consultant")
    
    # 3-Zone Report
    zones = ["Lift", "Gamma", "Gain"]
    friendly_zones = {
        "Lift": "Shadows (Black Levels)",
        "Gamma": "Midtones (Skin & Exposure)",
        "Gain": "Highlights (White Balance)"
    }
    
    col_layout = st.columns(3)
    
    for i, zone in enumerate(zones):
        deltas = values[zone] * 100 # Scale for readability
        r_d, g_d, b_d = deltas
        
        advice_html = ""
        advice_list = []
        if abs(r_d) > 10:
            direction = "Cut" if r_d < 0 else "Boost"
            advice_list.append(f"{direction} Red by {abs(r_d):.1f}")
        if abs(g_d) > 10:
             direction = "Cut" if g_d < 0 else "Boost"
             advice_list.append(f"{direction} Green by {abs(g_d):.1f}")
        if abs(b_d) > 10:
             direction = "Cut" if b_d < 0 else "Boost"
             advice_list.append(f"{direction} Blue by {abs(b_d):.1f}")
        
        if advice_list:
            for item in advice_list:
                advice_html += f"<li>{item}</li>"
            advice_html = f"<ul>{advice_html}</ul>"
        else:
             advice_html = "<p style='color: #4ade80;'>Balanced</p>"
            
        with col_layout[i]:
            st.markdown(f"""
            <div class="glass-card">
                <strong>{friendly_zones[zone]}</strong><br>
                <span style="color: #9CA3AF; font-size: 0.8em;">Primary Color Wheels</span>
                {advice_html}
            </div>
            """, unsafe_allow_html=True)

def render_match_consultant(advice_dict):
    st.markdown("### Technical Grading Breakdown")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""
        <div class="glass-card">
            <strong>Density (Black Point)</strong><br>
            <p style="margin-top: 10px; color: #60a5fa;">{advice_dict["Black Point"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="glass-card">
            <strong>Roll-off (Highlights)</strong><br>
            <p style="margin-top: 10px; color: #60a5fa;">{advice_dict["Highlights"]}</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="glass-card">
            <strong>Color Balance</strong><br>
            <p style="margin-top: 10px; color: #60a5fa;">{advice_dict["Color Cast"]}</p>
        </div>
        """, unsafe_allow_html=True)

def lut_download_section(lut_str, filename="GradeSense_Look.cube"):
    st.download_button(
        label="Download 3D LUT (.cube)",
        data=lut_str,
        file_name=filename,
        mime="text/plain",
        type="primary"
    )

def render_glass_container(content_func, *args, **kwargs):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    content_func(*args, **kwargs)
    st.markdown('</div>', unsafe_allow_html=True)

def render_scope_dashboard(image_rgb):
    """
    Renders the collapsible Technical Scope Suite.
    """
    with st.expander("Digital Scopes (Waveform / Parade / Histogram)", expanded=False):
        st.markdown('<div style="margin-bottom: 15px; color: #a855f7; font-family: monospace; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px;">/// Signal Analysis Module</div>', unsafe_allow_html=True)
        
        # Tabs for layout
        tab_wave, tab_parade, tab_hist = st.tabs(["Luma Waveform", "RGB Parade", "Histogram"])
        
        from processing import ScopeEngine
        
        with tab_wave:
            with st.spinner("Tracing Luma Signal..."):
                fig_wave = ScopeEngine.generate_waveform(image_rgb)
                st.plotly_chart(fig_wave, use_container_width=True, config={'displayModeBar': False})
                
        with tab_parade:
            with st.spinner("Separating Channels..."):
                fig_parade = ScopeEngine.generate_rgb_parade(image_rgb)
                st.plotly_chart(fig_parade, use_container_width=True, config={'displayModeBar': False})
                
        with tab_hist:
            fig_hist = ScopeEngine.generate_histogram(image_rgb)
            st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
