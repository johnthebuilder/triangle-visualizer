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

def create_detailed_plot(triangle, sequence_name, max_terms):
    """Create detailed cell-by-cell visualization with warnings for large sequences"""
    if not triangle:
        return None
    
    max_width = len(triangle[0])
    max_height = len(triangle)
    
    # Calculate total cells for warning
    total_cells = sum(len(row) for row in triangle)
    
    # Show warning for very large triangles but still render
    if total_cells > 100000:
        st.warning(f"⚠️ Large triangle ({total_cells:,} cells) - rendering may take time and use significant memory.")
    elif total_cells > 50000:
        st.info(f"ℹ️ Medium-large triangle ({total_cells:,} cells) - please wait for rendering...")
    
    # Adaptive sizing based on width
    if max_width <= 50:
        cell_size = 0.8
        font_size = 10
        show_text = True
    elif max_width <= 100:
        cell_size = 0.6
        font_size = 8
        show_text = True
    elif max_width <= 200:
        cell_size = 0.4
        font_size = 6
        show_text = True
    elif max_width <= 400:
        cell_size = 0.2
        font_size = 4
        show_text = max_width <= 300  # Hide text for very large triangles
    else:
        cell_size = 0.1
        font_size = 3
        show_text = False  # Too small to read
    
    # Create figure with reasonable size limits
    fig_width = min(20, max(8, max_width * cell_size / 2))
    fig_height = min(16, max(6, max_height * cell_size / 2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Simple colors
    colors = {0: '#2c3e50', 2: '#3498db', 'default': '#e74c3c'}
    
    # Build rectangles and text
    rect_patches = []
    rect_colors = []
    
    for row_idx, row in enumerate(triangle):
        if len(row) == 0:
            continue
        
        row_width = len(row)
        start_x = (max_width - row_width) * cell_size / 2
        y_pos = (max_height - row_idx - 1) * cell_size
        
        for col_idx, value in enumerate(row):
            x_pos = start_x + col_idx * cell_size
            color = colors.get(value, colors['default'])
            
            rect = Rectangle((x_pos, y_pos), cell_size, cell_size, linewidth=max(0.1, cell_size/10))
            rect_patches.append(rect)
            rect_colors.append(color)
            
            # Add text if readable
            if show_text:
                ax.text(x_pos + cell_size/2, y_pos + cell_size/2, str(value), 
                       ha='center', va='center', fontsize=font_size, 
                       color='white', weight='bold')
    
    # Add all rectangles at once
    if rect_patches:
        collection = PatchCollection(rect_patches, facecolors=rect_colors, 
                                   edgecolors='gray', linewidths=max(0.05, cell_size/20))
        ax.add_collection(collection)
    
    # Set limits
    padding = max(cell_size, 0.5)
    ax.set_xlim(-padding, max_width * cell_size + padding)
    ax.set_ylim(-padding, max_height * cell_size + padding)
    
    # Simple title
    title_text = f'{sequence_name} ({max_terms} terms)'
    if not show_text:
        title_text += ' - Numbers hidden (too small to display)'
    else:
        title_text += ' - Blue: 2s, Gray: 0s, Red: Others'
    
    ax.set_title(title_text, fontsize=min(14, max(8, 100/max_width)), pad=15)
    
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
    
    # Number of terms with warnings but higher limit
    max_terms = st.sidebar.number_input(
        "Number of Terms:",
        min_value=1,
        max_value=1000,  # Increased back to 1000
        value=50,
        step=1,
        help="Large sequences may be slow or crash your browser"
    )
    
    # Dynamic warnings based on size
    if max_terms <= 100:
        st.sidebar.success("✅ Good size - fast rendering")
    elif max_terms <= 200:
        st.sidebar.info("ℹ️ Medium size - should work fine")
    elif max_terms <= 400:
        st.sidebar.warning("⚠️ Large size - may be slow to render")
    else:
        st.sidebar.error("🚨 Very large - high risk of browser crash")
    
    # CSV upload with detailed format info (moved above "How it works")
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
        2. Take absolute differences between number sequences
        3. Repeat until you reach a single number
        4. Forms an upside-down triangle
        
        **Colors:**
        - 🔴 Red: Other values
        - 🔵 Blue: 2s
        - ⚫ Black: 0s
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.header("📊 About")
        st.markdown("""
        <div class="info-box">
        <h4>What it does</h4>
        <p>Takes differences between number sequences repeatedly to form a triangle pattern.</p>
        
        <h4>Colors</h4>
        <ul>
        <li><strong>Blue</strong> - Value is 2</li>
        <li><strong>Gray</strong> - Value is 0</li>
        <li><strong>Red</strong> - All other values</li>
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
        
        # Create visualization (single view only)
        with st.spinner("🎨 Creating visualization..."):
            fig = create_detailed_plot(triangle, sequence_name, max_terms)
        
        # Display plot
        if fig:
            st.pyplot(fig, use_container_width=True)
            
            # Download button
            filename = f"triangle_{sequence_name.lower().replace(' ', '_')}_{max_terms}terms.png"
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

# Run the main app
main()
