import streamlit as st

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
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .info-box h4 {
        color: #495057;
        margin-bottom: 0.5rem;
    }
    .info-box p, .info-box ul {
        color: #6c757d;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_sequence(seq_type, n):
    """Generate a sequence of the specified type - optimized with numpy where possible"""
    if seq_type == "Prime Numbers":
        return list(sympy.primerange(2, sympy.prime(n) + 1))[:n]
    elif seq_type == "Fibonacci":
        if n <= 0:
            return []
        elif n == 1:
            return [1]
        elif n == 2:
            return [1, 1]
        # Optimized fibonacci using matrix multiplication for large n
        fib = np.ones(n, dtype=np.int64)
        for i in range(2, n):
            fib[i] = fib[i-1] + fib[i-2]
        return fib.tolist()
    elif seq_type == "Natural Numbers":
        return np.arange(1, n + 1, dtype=np.int64).tolist()
    elif seq_type == "Square Numbers":
        return (np.arange(1, n + 1, dtype=np.int64) ** 2).tolist()
    elif seq_type == "Triangular Numbers":
        arr = np.arange(1, n + 1, dtype=np.int64)
        return ((arr * (arr + 1)) // 2).tolist()
    else:
        return list(range(1, n + 1))

@st.cache_data
def compute_triangle(sequence):
    """Compute the recursive difference triangle using NumPy - optimized version"""
    if not sequence:
        return []
    
    # Pre-allocate memory for better performance
    n = len(sequence)
    triangle = []
    
    # Use numpy arrays for faster computation
    current = np.array(sequence, dtype=np.int64)
    triangle.append(current.copy())
    
    for i in range(1, n):
        if len(current) <= 1:
            break
        # Vectorized absolute difference
        current = np.abs(np.diff(current))
        triangle.append(current.copy())
    
    return triangle

def parse_csv_robust(uploaded_file):
    """Optimized CSV parser with numpy for large files"""
    try:
        content = uploaded_file.read().decode('utf-8')
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        numbers = []
        
        for line in lines:
            # Skip header lines
            if any(char.isalpha() for char in line) and not line.replace(',', '').replace('.', '').replace('-', '').isdigit():
                continue
            
            # Parse numbers more efficiently
            line_numbers = []
            
            # Try different delimiters
            for delimiter in [',', ' ', '\t', ';']:
                if delimiter in line:
                    parts = line.split(delimiter)
                    for part in parts:
                        part = part.strip()
                        if part and part.replace('.', '').replace('-', '').lstrip('-').isdigit():
                            try:
                                line_numbers.append(int(float(part)))
                            except ValueError:
                                continue
                    break
            else:
                # Single number per line
                if line.replace('.', '').replace('-', '').lstrip('-').isdigit():
                    try:
                        line_numbers.append(int(float(line)))
                    except ValueError:
                        pass
            
            numbers.extend(line_numbers)
        
        return numbers if numbers else None
        
    except Exception:
        return None

def create_optimized_plot(triangle, sequence_name, max_terms, dpi=100, high_quality=False):
    """Optimized plotting with vectorized operations"""
    if not triangle:
        return None
    
    max_width = len(triangle[0])
    max_height = len(triangle)
    total_cells = sum(len(row) for row in triangle)
    
    # Adaptive sizing
    if max_width <= 50:
        cell_size = 0.8
        font_size = 10 if not high_quality else 14
        show_text = True
    elif max_width <= 100:
        cell_size = 0.6
        font_size = 8 if not high_quality else 12
        show_text = True
    elif max_width <= 200:
        cell_size = 0.4
        font_size = 6 if not high_quality else 10
        show_text = True
    elif max_width <= 400:
        cell_size = 0.2
        font_size = 4 if not high_quality else 8
        show_text = max_width <= 300
    else:
        cell_size = 0.1
        font_size = 3 if not high_quality else 6
        show_text = False
    
    # Increase figure size for high quality export
    if high_quality:
        fig_width = min(40, max(16, max_width * cell_size))
        fig_height = min(32, max(12, max_height * cell_size))
    else:
        fig_width = min(20, max(8, max_width * cell_size / 2))
        fig_height = min(16, max(6, max_height * cell_size / 2))
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Updated colors - WHITE for 0s
    colors = {0: '#FFFFFF', 2: '#3498db', 'default': '#e74c3c'}
    
    # Pre-allocate arrays for better performance
    rect_patches = []
    rect_colors = []
    texts = []
    
    # Vectorized rectangle creation
    for row_idx, row in enumerate(triangle):
        if len(row) == 0:
            continue
        
        row_array = np.array(row)
        row_width = len(row)
        start_x = (max_width - row_width) * cell_size / 2
        y_pos = (max_height - row_idx - 1) * cell_size
        
        # Create all rectangles for this row at once
        x_positions = start_x + np.arange(row_width) * cell_size
        
        for col_idx, (x_pos, value) in enumerate(zip(x_positions, row_array)):
            color = colors.get(int(value), colors['default'])
            
            rect = Rectangle((x_pos, y_pos), cell_size, cell_size, 
                           linewidth=max(0.1, cell_size/10))
            rect_patches.append(rect)
            rect_colors.append(color)
            
            if show_text:
                # For white squares (0s), use black text
                text_color = 'black' if value == 0 else 'white'
                texts.append((x_pos + cell_size/2, y_pos + cell_size/2, 
                            str(value), text_color))
    
    # Add all rectangles at once
    if rect_patches:
        collection = PatchCollection(rect_patches, facecolors=rect_colors, 
                                   edgecolors='gray', linewidths=max(0.05, cell_size/20))
        ax.add_collection(collection)
    
    # Add all text at once
    if show_text and texts:
        for x, y, text, color in texts:
            ax.text(x, y, text, ha='center', va='center', 
                   fontsize=font_size, color=color, weight='bold')
    
    # Set limits
    padding = max(cell_size, 0.5)
    ax.set_xlim(-padding, max_width * cell_size + padding)
    ax.set_ylim(-padding, max_height * cell_size + padding)
    
    # Title
    title_text = f'{sequence_name} ({max_terms} terms)'
    if not show_text:
        title_text += ' - Numbers hidden (too small to display)'
    else:
        title_text += ' - Blue: 2s, White: 0s, Red: Others'
    
    title_size = min(14, max(8, 100/max_width))
    if high_quality:
        title_size *= 1.5
    
    ax.set_title(title_text, fontsize=title_size, pad=15)
    
    plt.tight_layout()
    return fig

def get_download_links(fig, filename_base, triangle, sequence_name, max_terms):
    """Generate multiple download options including high-resolution"""
    links = []
    
    # Standard quality (300 DPI)
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.read()).decode()
    links.append(f'<a href="data:image/png;base64,{img_data}" download="{filename_base}_300dpi.png" class="download-link">📥 Download PNG (300 DPI)</a>')
    
    # High quality (600 DPI) - only for smaller triangles
    total_cells = sum(len(row) for row in triangle)
    if total_cells < 10000:  # Limit for performance
        img_buffer_hq = io.BytesIO()
        fig.savefig(img_buffer_hq, format='png', dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        img_buffer_hq.seek(0)
        img_data_hq = base64.b64encode(img_buffer_hq.read()).decode()
        links.append(f'<a href="data:image/png;base64,{img_data_hq}" download="{filename_base}_600dpi.png" class="download-link">📥 Download PNG (600 DPI)</a>')
    
    # Maximum quality export button
    max_quality_button = f"""
    <div style="margin-top: 10px;">
        <p style="font-size: 0.9em; color: #666;">For maximum resolution, click the button below:</p>
    </div>
    """
    links.append(max_quality_button)
    
    return '<br>'.join(links)

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🔺 Recursive Difference Triangle Analyzer</h1>', 
                unsafe_allow_html=True)
    
    # Add custom CSS for download links
    st.markdown("""
    <style>
    .download-link {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        background-color: #3498db;
        color: white;
        text-decoration: none;
        border-radius: 0.25rem;
        transition: background-color 0.3s;
    }
    .download-link:hover {
        background-color: #2980b9;
        color: white;
        text-decoration: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
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
        step=1,
        help="Large sequences may be slow or crash your browser"
    )
    
    # Dynamic warnings
    if max_terms <= 100:
        st.sidebar.success("✅ Good size - fast rendering")
    elif max_terms <= 200:
        st.sidebar.info("ℹ️ Medium size - should work fine")
    elif max_terms <= 400:
        st.sidebar.warning("⚠️ Large size - may be slow to render")
    else:
        st.sidebar.error("🚨 Very large - high risk of browser crash")
    
    # CSV upload
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
        - ⚪ White: 0s
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.header("📊 About")
        st.markdown("""
        <div class="info-box">
        <h4>What it does</h4>
        <p>Takes differences between consecutive numbers repeatedly to form a triangle pattern.</p>
        
        <h4>Colors</h4>
        <ul>
        <li><strong>Blue</strong> - Value is 2</li>
        <li><strong>White</strong> - Value is 0</li>
        <li><strong>Red</strong> - All other values</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col1:
        # Generate or load sequence
        if uploaded_file is not None:
            sequence = parse_csv_robust(uploaded_file)
            if sequence:
                sequence = sequence[:1000]
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
        
        # Create visualization
        with st.spinner("🎨 Creating visualization..."):
            fig = create_optimized_plot(triangle, sequence_name, max_terms)
        
        # Display plot
        if fig:
            st.pyplot(fig, use_container_width=True)
            
            # Download options
            filename_base = f"triangle_{sequence_name.lower().replace(' ', '_')}_{max_terms}terms"
            st.markdown("### 📥 Download Options")
            st.markdown(get_download_links(fig, filename_base, triangle, sequence_name, max_terms), 
                       unsafe_allow_html=True)
            
            # Maximum quality export button
            if st.button("🚀 Generate Maximum Quality Image (1200 DPI)", type="primary"):
                with st.spinner("Creating ultra high-resolution image... This may take a moment."):
                    # Create high quality version
                    fig_hq = create_optimized_plot(triangle, sequence_name, max_terms, 
                                                  dpi=1200, high_quality=True)
                    if fig_hq:
                        img_buffer_max = io.BytesIO()
                        fig_hq.savefig(img_buffer_max, format='png', dpi=1200, 
                                     bbox_inches='tight', facecolor='white', 
                                     edgecolor='none')
                        img_buffer_max.seek(0)
                        
                        st.download_button(
                            label="⬇️ Download Maximum Quality PNG (1200 DPI)",
                            data=img_buffer_max,
                            file_name=f"{filename_base}_1200dpi.png",
                            mime="image/png"
                        )
                        plt.close(fig_hq)
                
                st.success("✅ High-resolution image generated!")
            
            plt.close(fig)  # Clean up memory
        
        # Statistics
        st.header("📈 Triangle Statistics")
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric("Triangle Height", len(triangle))
        
        with col_b:
            total_cells = sum(len(row) for row in triangle)
            st.metric("Total Cells", f"{total_cells:,}")
        
        with col_c:
            # Count zeros in triangle
            zero_count = sum(np.sum(row == 0) for row in triangle)
            st.metric("Zero Values", zero_count)
        
        with col_d:
            # Count twos in triangle
            two_count = sum(np.sum(row == 2) for row in triangle)
            st.metric("Two Values", two_count)

# Run the main app
if __name__ == "__main__":
    main()
