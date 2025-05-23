import streamlit as st

# Configure page - MUST be absolute first command
st.set_page_config(
    page_title="Triangle Visualizer",
    page_icon="🔺",
    layout="wide",
    initial_sidebar_state="expanded"
)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import sympy
import io
import base64
import pandas as pd

"""
Recursive Difference Triangle Visualization Tool
A mathematical analysis tool for studying numerical sequences through recursive difference patterns.

This application computes triangular arrangements of absolute differences between consecutive 
terms in numerical sequences, revealing structural properties and convergence behaviors.
"""

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    .info-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box h4 {
        color: #495057;
        margin-bottom: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .info-box p {
        color: #6c757d;
        line-height: 1.6;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
    .info-box ul {
        color: #6c757d;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    .info-box li {
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        border-radius: 0.5rem;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 500;
    }
    .zoom-controls {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_sequence(seq_type, n):
    """Generate a sequence of the specified type"""
    if seq_type == "Prime Numbers":
        return list(sympy.primerange(2, sympy.prime(n) + 1))[:n]
    elif seq_type == "Fibonacci":
        fib = [1, 1]
        while len(fib) < n:
            fib.append(fib[-1] + fib[-2])
        return fib[:n]
    elif seq_type == "Natural Numbers":
        return list(range(1, n + 1))
    elif seq_type == "Square Numbers":
        return [i**2 for i in range(1, n + 1)]
    elif seq_type == "Triangular Numbers":
        return [i*(i+1)//2 for i in range(1, n + 1)]
    else:
        return list(range(1, n + 1))

@st.cache_data
def compute_triangle(sequence):
    """Compute the recursive difference triangle using NumPy"""
    if not sequence:
        return []
    
    triangle = [np.array(sequence, dtype=np.int64)]
    
    for row_idx in range(1, len(sequence)):
        current_row = triangle[-1]
        if len(current_row) <= 1:
            break
        
        # Use NumPy for faster computation
        new_row = np.abs(np.diff(current_row))
        triangle.append(new_row)
    
    return triangle

def parse_csv_robust(uploaded_file):
    """
    Robust CSV parser that handles multiple formats:
    - Headers or no headers
    - Comma-separated or line-separated
    - Mixed whitespace
    - Various delimiters
    """
    try:
        # Read the raw content
        content = uploaded_file.read().decode('utf-8')
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        numbers = []
        
        # Try different parsing strategies
        for line in lines:
            # Skip obvious header lines
            if any(char.isalpha() for char in line) and not line.replace(',', '').replace('.', '').replace('-', '').isdigit():
                continue
            
            # Try comma-separated values first
            if ',' in line:
                parts = [part.strip() for part in line.split(',')]
                for part in parts:
                    if part and part.replace('.', '').replace('-', '').isdigit():
                        try:
                            numbers.append(int(float(part)))
                        except ValueError:
                            continue
            # Try space-separated values
            elif ' ' in line:
                parts = line.split()
                for part in parts:
                    if part.replace('.', '').replace('-', '').isdigit():
                        try:
                            numbers.append(int(float(part)))
                        except ValueError:
                            continue
            # Try single number per line
            else:
                if line.replace('.', '').replace('-', '').isdigit():
                    try:
                        numbers.append(int(float(line)))
                    except ValueError:
                        continue
        
        return numbers if numbers else None
        
    except Exception:
        return None

def create_detailed_plot(triangle, sequence_name, max_terms, figsize_multiplier=1.0):
    """Create detailed cell-by-cell visualization with zoom support"""
    if not triangle:
        return None
    
    max_width = len(triangle[0])
    max_height = len(triangle)
    
    # Adaptive sizing with zoom multiplier
    base_cell_size = 0.8 if max_width <= 50 else 0.5 if max_width <= 100 else 0.3
    cell_size = base_cell_size * figsize_multiplier
    
    base_font_size = 10 if max_width <= 50 else 8 if max_width <= 100 else 6
    font_size = max(4, int(base_font_size * figsize_multiplier))
    
    show_text = font_size >= 5 and cell_size >= 0.4
    
    # Create figure with better proportions
    fig_width = max(12, min(20, max_width * cell_size / 2))
    fig_height = max(8, min(16, max_height * cell_size / 2))
    fig, ax = plt.subplots(figsize=(fig_width * figsize_multiplier, fig_height * figsize_multiplier))
    
    # Set high DPI for crisp rendering
    fig.dpi = 100
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Enhanced color scheme
    colors = {
        0: {'bg': '#2c3e50', 'text': '#ffffff'},  # Dark blue-gray for zeros
        2: {'bg': '#3498db', 'text': '#ffffff'},  # Bright blue for twos
        'default': {'bg': '#e74c3c', 'text': '#ffffff'}  # Vibrant red for others
    }
    
    rectangles_data = []
    texts_data = []
    
    for row_idx, row in enumerate(triangle):
        if len(row) == 0:
            continue
        
        row_width = len(row)
        start_x = (max_width - row_width) * cell_size / 2
        y_pos = (max_height - row_idx - 1) * cell_size
        
        for col_idx, value in enumerate(row):
            x_pos = start_x + col_idx * cell_size
            
            # Get enhanced colors
            color_info = colors.get(value, colors['default'])
            
            rectangles_data.append((x_pos, y_pos, cell_size, color_info['bg']))
            
            if show_text:
                texts_data.append((x_pos + cell_size/2, y_pos + cell_size/2, 
                                str(value), color_info['text'], font_size))
    
    # Create rectangles with better styling
    rect_patches = []
    rect_colors = []
    
    for x, y, size, color in rectangles_data:
        rect = Rectangle((x, y), size, size, linewidth=0.5)
        rect_patches.append(rect)
        rect_colors.append(color)
    
    if rect_patches:
        collection = PatchCollection(rect_patches, facecolors=rect_colors, 
                                   edgecolors='#34495e', linewidths=0.3, alpha=0.9)
        ax.add_collection(collection)
    
    # Add text with better styling
    if show_text:
        for x, y, text, color, size in texts_data:
            ax.text(x, y, text, ha='center', va='center', 
                   fontsize=size, color=color, weight='bold',
                   fontfamily='monospace')
    
    # Set limits with proper padding
    padding = cell_size * 1.5
    ax.set_xlim(-padding, max_width * cell_size + padding)
    ax.set_ylim(-padding, max_height * cell_size + padding)
    
    # Enhanced title
    title_size = max(12, min(18, int(16 * figsize_multiplier)))
    ax.set_title(f'Detailed View - {sequence_name} ({max_terms} terms)\n'
                f'Blue: 2s  •  Dark Gray: 0s  •  Red: Other Values', 
                fontsize=title_size, pad=20, color='#2c3e50', weight='bold')
    
    # Add subtle grid for better readability if zoomed in
    if figsize_multiplier > 1.2 and max_width <= 100:
        ax.grid(True, alpha=0.1, color='#7f8c8d', linewidth=0.5)
    
    plt.tight_layout()
    return fig

def create_structure_plot(triangle, sequence_name, max_terms, figsize_multiplier=1.0):
    """Create structure view with color-coded squares and zoom support"""
    if not triangle:
        return None
    
    max_width = len(triangle[0])
    max_height = len(triangle)
    
    # Adaptive sizing with zoom
    base_cell_size = 0.8 if max_width <= 500 else 0.4 if max_width <= 1000 else 0.2
    cell_size = base_cell_size * figsize_multiplier
    
    # Create figure with better proportions
    fig_width = max(14, min(24, max_width * cell_size / 8))
    fig_height = max(10, min(20, max_height * cell_size / 8))
    fig, ax = plt.subplots(figsize=(fig_width * figsize_multiplier, fig_height * figsize_multiplier))
    
    # Enhanced styling
    fig.dpi = 100
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    
    # Enhanced color scheme
    colors = {
        0: '#2c3e50',    # Dark blue-gray for zeros
        2: '#3498db',    # Bright blue for twos  
        'default': '#e74c3c'  # Vibrant red for others
    }
    
    # Collect rectangles by color for efficient rendering
    red_rects = []
    blue_rects = []
    black_rects = []
    
    for row_idx, row in enumerate(triangle):
        if len(row) == 0:
            continue
        
        row_width = len(row)
        start_x = (max_width - row_width) * cell_size / 2
        y_pos = (max_height - row_idx - 1) * cell_size
        
        for col_idx, value in enumerate(row):
            x_pos = start_x + col_idx * cell_size
            rect_data = (x_pos, y_pos, cell_size, cell_size)
            
            if value == 0:
                black_rects.append(rect_data)
            elif value == 2:
                blue_rects.append(rect_data)
            else:
                red_rects.append(rect_data)
    
    # Create all patches with enhanced styling
    all_patches = []
    all_colors = []
    
    for x, y, w, h in red_rects:
        rect = Rectangle((x, y), w, h, linewidth=0.2)
        all_patches.append(rect)
        all_colors.append(colors['default'])
    
    for x, y, w, h in blue_rects:
        rect = Rectangle((x, y), w, h, linewidth=0.2)
        all_patches.append(rect)
        all_colors.append(colors[2])
    
    for x, y, w, h in black_rects:
        rect = Rectangle((x, y), w, h, linewidth=0.2)
        all_patches.append(rect)
        all_colors.append(colors[0])
    
    if all_patches:
        collection = PatchCollection(all_patches, facecolors=all_colors, 
                                   edgecolors='#34495e', linewidths=0.1, alpha=0.9)
        ax.add_collection(collection)
    
    # Set limits with proper padding
    padding = cell_size * 2
    ax.set_xlim(-padding, max_width * cell_size + padding)
    ax.set_ylim(-padding, max_height * cell_size + padding)
    
    # Enhanced title and legend
    title_size = max(12, min(18, int(16 * figsize_multiplier)))
    ax.set_title(f'Structure View - {sequence_name} ({max_terms} terms)\n'
                f'Blue: 2s  •  Dark Gray: 0s  •  Red: Other Values', 
                fontsize=title_size, pad=25, color='#2c3e50', weight='bold')
    
    # Enhanced legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['default'], 
               markersize=12, label='Other Values', markeredgecolor='#34495e'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[2], 
               markersize=12, label='2s', markeredgecolor='#34495e'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[0], 
               markersize=12, label='0s', markeredgecolor='#34495e')
    ]
    legend = ax.legend(handles=legend_elements, loc='upper right', 
                      fontsize=max(10, int(12 * figsize_multiplier)),
                      frameon=True, fancybox=True, shadow=True,
                      facecolor='white', edgecolor='#dee2e6')
    
    plt.tight_layout()
    return fig

def get_download_link(fig, filename):
    """Generate download link for the plot"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.read()).decode()
    
    href = f'<a href="data:image/png;base64,{img_data}" download="{filename}">Download PNG</a>'
    return href

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🔺 Recursive Difference Triangle Analyzer</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # Sequence selection
    sequence_type = st.sidebar.selectbox(
        "Sequence Type:",
        ["Prime Numbers", "Fibonacci", "Natural Numbers", "Square Numbers", "Triangular Numbers"]
    )
    
    # Number of terms
    max_terms = st.sidebar.number_input(
        "Number of Terms:",
        min_value=1,
        max_value=1000,
        value=50,
        step=1
    )
    
    # Recommendations
    if max_terms <= 100:
        st.sidebar.success("✅ Perfect for detailed view")
    elif max_terms <= 300:
        st.sidebar.info("ℹ️ Good for both views")
    else:
        st.sidebar.warning("⚠️ Structure view recommended")
    
    # View mode and zoom controls
    view_mode = st.sidebar.radio(
        "View Mode:",
        ["Detailed View", "Structure View"]
    )
    
    # Zoom controls
    st.sidebar.header("🔍 Display Options")
    zoom_level = st.sidebar.slider(
        "Zoom Level:",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Adjust the size of the visualization"
    )
    
    if zoom_level != 1.0:
        zoom_text = f"{'Zoomed in' if zoom_level > 1.0 else 'Zoomed out'} to {zoom_level:.1f}x"
        st.sidebar.info(zoom_text)
    
    # Info section
    with st.sidebar.expander("ℹ️ How it works"):
        st.write("""
        **Recursive Difference Triangle:**
        1. Start with your sequence
        2. Take absolute differences between consecutive numbers
        3. Repeat until you reach a single number
        4. Forms an upside-down triangle
        
        **Colors:**
        - 🔴 Red: Other values
        - 🔵 Blue: 2s
        - ⚫ Black: 0s
        """)
    
    # CSV upload with detailed format info
    st.sidebar.header("📁 Custom Sequence")
    st.sidebar.markdown("""
    **Supported CSV formats:**
    - One number per line
    - Comma-separated values
    - With or without headers
    - Mixed whitespace handling
    
    **Example formats:**
    ```
    1,2,3,4,5
    ```
    ```
    Number
    1
    2
    3
    ```
    """)
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv', 'txt'])
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.header("📊 Analysis Overview")
        st.markdown("""
        <div class="info-box">
        <h4>Mathematical Purpose</h4>
        <p>This tool analyzes numerical sequences by computing recursive absolute differences, creating triangular patterns that reveal underlying mathematical structures and behaviors in data.</p>
        
        <h4>Visual Analysis Guide</h4>
        <ul>
        <li><strong>Blue cells (value = 2)</strong> - Indicate specific difference magnitudes</li>
        <li><strong>Black cells (value = 0)</strong> - Show convergence points in the sequence</li>
        <li><strong>Red cells</strong> - Represent all other difference values</li>
        <li><strong>Triangle convergence</strong> - Demonstrates sequence stability properties</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col1:
        # Generate or load sequence
        if uploaded_file is not None:
            sequence = parse_csv_robust(uploaded_file)
            if sequence:
                sequence = sequence[:1000]  # Limit to 1000 for performance
                sequence_name = f"Custom Data ({len(sequence)} values)"
                max_terms = len(sequence)
                st.success(f"✅ Successfully parsed {len(sequence)} values from file")
            else:
                st.error("❌ Could not parse the uploaded file. Please check the format.")
                st.info("""
                **Supported formats:**
                - `1,2,3,4,5` (comma-separated)
                - `1 2 3 4 5` (space-separated)
                - One number per line
                - Files with headers (automatically skipped)
                """)
                sequence = generate_sequence(sequence_type, max_terms)
                sequence_name = sequence_type
        else:
            sequence = generate_sequence(sequence_type, max_terms)
            sequence_name = sequence_type
        
        # Show sequence preview
        with st.expander("🔍 Sequence Preview"):
            preview = sequence[:20]
            if len(sequence) > 20:
                preview_text = ", ".join(map(str, preview)) + f"... ({len(sequence)} total)"
            else:
                preview_text = ", ".join(map(str, preview))
            st.code(preview_text)
        
        # Generate triangle
        with st.spinner("🔄 Computing triangle..."):
            triangle = compute_triangle(sequence)
        
        # Create visualization with zoom
        with st.spinner("🎨 Creating visualization..."):
            if view_mode == "Detailed View":
                fig = create_detailed_plot(triangle, sequence_name, max_terms, zoom_level)
            else:
                fig = create_structure_plot(triangle, sequence_name, max_terms, zoom_level)
        
        # Display plot with better container
        if fig:
            # Add zoom information
            if zoom_level != 1.0:
                zoom_info = f"🔍 **Display:** {zoom_level:.1f}x zoom"
                if zoom_level > 1.5:
                    zoom_info += " (High detail - may take longer to render)"
                st.info(zoom_info)
            
            st.pyplot(fig, use_container_width=True)
            
            # Download button
            filename = f"triangle_{sequence_name.lower().replace(' ', '_')}_{max_terms}terms_zoom{zoom_level:.1f}x.png"
            st.markdown(get_download_link(fig, filename), unsafe_allow_html=True)
            
            plt.close(fig)  # Clean up memory
        
        # Statistics
        st.header("📈 Triangle Statistics")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Triangle Height", len(triangle))
        
        with col_b:
            total_cells = sum(len(row) for row in triangle)
            st.metric("Total Cells", f"{total_cells:,}")
        
        with col_c:
            # Count zeros in triangle
            zero_count = sum(np.sum(row == 0) for row in triangle)
            st.metric("Zero Values", zero_count)

if __name__ == "__main__":
    main()
