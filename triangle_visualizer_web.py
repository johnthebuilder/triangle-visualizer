import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import sympy
import io
import base64
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Triangle Visualizer",
    page_icon="🔺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
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

def create_detailed_plot(triangle, sequence_name, max_terms):
    """Create detailed cell-by-cell visualization"""
    if not triangle:
        return None
    
    max_width = len(triangle[0])
    max_height = len(triangle)
    
    # Adaptive sizing
    if max_width > 200:
        cell_size = 0.3
        font_size = 6
        show_text = False
    elif max_width > 100:
        cell_size = 0.5
        font_size = 8
        show_text = True
    else:
        cell_size = max(0.8, min(2.0, 15.0 / max_width))
        font_size = max(6, min(12, cell_size * 8))
        show_text = True
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Color mapping
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
            
            # Determine color
            if value == 0:
                color = '#000000'
                text_color = '#FFFFFF'
            elif value == 2:
                color = '#0000FF'
                text_color = '#FFFFFF'
            else:
                color = '#FF0000'
                text_color = '#FFFFFF'
            
            rectangles_data.append((x_pos, y_pos, cell_size, color))
            
            if show_text:
                texts_data.append((x_pos + cell_size/2, y_pos + cell_size/2, 
                                str(value), text_color, font_size))
    
    # Create rectangles
    rect_patches = []
    rect_colors = []
    
    for x, y, size, color in rectangles_data:
        rect = Rectangle((x, y), size, size, linewidth=0.1)
        rect_patches.append(rect)
        rect_colors.append(color)
    
    if rect_patches:
        collection = PatchCollection(rect_patches, facecolors=rect_colors, 
                                   edgecolors='gray', linewidths=0.1)
        ax.add_collection(collection)
    
    # Add text
    if show_text:
        for x, y, text, color, size in texts_data:
            ax.text(x, y, text, ha='center', va='center', 
                   fontsize=size, color=color, weight='bold')
    
    # Set limits
    padding = cell_size
    ax.set_xlim(-padding, max_width * cell_size + padding)
    ax.set_ylim(-padding, max_height * cell_size + padding)
    
    # Title
    ax.set_title(f'Detailed View - {sequence_name} ({max_terms} terms)\n'
                f'Red=Other Values, Blue=2s, Black=0s', 
                fontsize=14, pad=20)
    
    return fig

def create_structure_plot(triangle, sequence_name, max_terms):
    """Create structure view with color-coded squares"""
    if not triangle:
        return None
    
    max_width = len(triangle[0])
    max_height = len(triangle)
    
    # Adaptive sizing for structure view
    if max_width > 1000:
        cell_size = 0.2
    elif max_width > 500:
        cell_size = 0.4
    else:
        cell_size = 0.8
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Collect rectangles by color
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
    
    # Create all patches
    all_patches = []
    all_colors = []
    
    for x, y, w, h in red_rects:
        rect = Rectangle((x, y), w, h, linewidth=0)
        all_patches.append(rect)
        all_colors.append('#FF0000')
    
    for x, y, w, h in blue_rects:
        rect = Rectangle((x, y), w, h, linewidth=0)
        all_patches.append(rect)
        all_colors.append('#0000FF')
    
    for x, y, w, h in black_rects:
        rect = Rectangle((x, y), w, h, linewidth=0)
        all_patches.append(rect)
        all_colors.append('#000000')
    
    if all_patches:
        collection = PatchCollection(all_patches, facecolors=all_colors, edgecolors='none')
        ax.add_collection(collection)
    
    # Set limits
    padding = cell_size * 2
    ax.set_xlim(-padding, max_width * cell_size + padding)
    ax.set_ylim(-padding, max_height * cell_size + padding)
    
    # Title and legend
    ax.set_title(f'Structure View - {sequence_name} ({max_terms} terms)\n'
                f'Red=Other Values, Blue=2s, Black=0s', 
                fontsize=14, pad=20)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Other Values'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='2s'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10, label='0s')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
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
    st.markdown('<h1 class="main-header">🔺 Interactive Recursive Difference Triangle Visualizer</h1>', 
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
    
    # View mode
    view_mode = st.sidebar.radio(
        "View Mode:",
        ["Detailed View", "Structure View"]
    )
    
    # CSV upload
    st.sidebar.header("📁 Custom Sequence")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    
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
        - 🔵 Blue: 2s (often form patterns)
        - ⚫ Black: 0s (where differences cancel)
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.header("📊 About This Tool")
        st.markdown("""
        <div class="info-box">
        <h4>What does this show?</h4>
        <p>This visualization reveals hidden patterns in number sequences by repeatedly calculating differences between consecutive numbers.</p>
        
        <h4>What to look for:</h4>
        <ul>
        <li><strong>Blue squares (2s)</strong> - Often form geometric patterns</li>
        <li><strong>Black squares (0s)</strong> - Where the sequence "settles"</li>
        <li><strong>Triangle shape</strong> - How quickly it converges</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col1:
        # Generate or load sequence
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, header=None)
                sequence = df[0].dropna().astype(int).tolist()[:1000]  # Limit to 1000
                sequence_name = f"Custom CSV ({len(sequence)} terms)"
                max_terms = len(sequence)
                st.success(f"✅ Loaded {len(sequence)} values from CSV")
            except Exception as e:
                st.error(f"❌ Error loading CSV: {str(e)}")
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
            if view_mode == "Detailed View":
                fig = create_detailed_plot(triangle, sequence_name, max_terms)
            else:
                fig = create_structure_plot(triangle, sequence_name, max_terms)
        
        # Display plot
        if fig:
            st.pyplot(fig)
            
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

if __name__ == "__main__":
    main()
