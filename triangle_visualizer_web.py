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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for better memory usage
import sympy
import io
import base64
import gc  # Garbage collection

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
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #ffeeba;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_sequence(seq_type, n):
    """Generate a sequence of the specified type - memory efficient version"""
    try:
        if seq_type == "Prime Numbers":
            return list(sympy.primerange(2, sympy.prime(min(n, 10000)) + 1))[:n]
        elif seq_type == "Fibonacci":
            if n <= 0:
                return []
            elif n == 1:
                return [1]
            elif n == 2:
                return [1, 1]
            # Generate fibonacci iteratively to save memory
            fib = [1, 1]
            for i in range(2, n):
                fib.append(fib[-1] + fib[-2])
            return fib
        elif seq_type == "Natural Numbers":
            return list(range(1, n + 1))
        elif seq_type == "Square Numbers":
            return [i**2 for i in range(1, n + 1)]
        elif seq_type == "Triangular Numbers":
            return [i*(i+1)//2 for i in range(1, n + 1)]
        else:
            return list(range(1, n + 1))
    except MemoryError:
        st.error("Out of memory generating sequence. Try a smaller number of terms.")
        return []

@st.cache_data
def compute_triangle_memory_efficient(sequence, max_rows=None):
    """Memory-efficient triangle computation with row limit"""
    if not sequence:
        return []
    
    triangle = []
    current = np.array(sequence, dtype=np.int64)
    triangle.append(current.copy())
    
    # Limit the number of rows to prevent memory issues
    n_rows = len(sequence) if max_rows is None else min(len(sequence), max_rows)
    
    for i in range(1, n_rows):
        if len(current) <= 1:
            break
        current = np.abs(np.diff(current))
        triangle.append(current.copy())
        
        # Early termination if triangle gets too large
        total_elements = sum(len(row) for row in triangle)
        if total_elements > 100000:  # Limit total elements
            st.warning(f"Triangle truncated at row {i+1} to prevent memory issues")
            break
    
    return triangle

def parse_csv_robust(uploaded_file):
    """Memory-efficient CSV parser"""
    try:
        content = uploaded_file.read().decode('utf-8')
        lines = content.strip().split('\n')
        
        numbers = []
        max_numbers = 1000  # Limit to prevent memory issues
        
        for line in lines[:max_numbers]:  # Process limited lines
            line = line.strip()
            if not line:
                continue
                
            # Skip obvious headers
            if any(c.isalpha() for c in line) and not line.replace(',', '').replace('.', '').replace('-', '').isdigit():
                continue
            
            # Try to parse numbers
            for delimiter in [',', ' ', '\t', ';']:
                if delimiter in line:
                    parts = line.split(delimiter)
                    for part in parts:
                        part = part.strip()
                        if part and part.replace('.', '').replace('-', '').lstrip('-').isdigit():
                            try:
                                numbers.append(int(float(part)))
                                if len(numbers) >= max_numbers:
                                    return numbers
                            except:
                                continue
                    break
            else:
                # Single number per line
                if line.replace('.', '').replace('-', '').lstrip('-').isdigit():
                    try:
                        numbers.append(int(float(line)))
                    except:
                        pass
            
            if len(numbers) >= max_numbers:
                break
        
        return numbers if numbers else None
    except:
        return None

def create_memory_efficient_plot(triangle, sequence_name, max_terms, quality='standard'):
    """Memory-efficient plotting function"""
    if not triangle:
        return None
    
    max_width = len(triangle[0])
    max_height = len(triangle)
    total_cells = sum(len(row) for row in triangle)
    
    # Memory-based size limits
    if total_cells > 50000:
        st.warning("⚠️ Large triangle detected. Rendering simplified view to prevent crashes.")
        # Reduce triangle size for visualization
        sample_rate = max(2, total_cells // 25000)
        triangle = [row[::sample_rate] if len(row) > sample_rate else row for row in triangle[::sample_rate]]
        max_width = len(triangle[0])
        max_height = len(triangle)
        total_cells = sum(len(row) for row in triangle)
    
    # Adaptive sizing based on quality
    if quality == 'high':
        scale_factor = 1.5
        dpi = 150
    elif quality == 'maximum':
        scale_factor = 2
        dpi = 300
    else:  # standard
        scale_factor = 1
        dpi = 100
    
    # Cell sizing
    if max_width <= 50:
        cell_size = 0.8 * scale_factor
        font_size = 10 * scale_factor
        show_text = True
    elif max_width <= 100:
        cell_size = 0.5 * scale_factor
        font_size = 8 * scale_factor
        show_text = True
    elif max_width <= 200:
        cell_size = 0.3 * scale_factor
        font_size = 6 * scale_factor
        show_text = quality != 'standard'
    else:
        cell_size = 0.15 * scale_factor
        font_size = 4 * scale_factor
        show_text = False
    
    # Figure size with memory limits
    max_fig_size = 20 if quality == 'standard' else 30
    fig_width = min(max_fig_size, max(8, max_width * cell_size / 2))
    fig_height = min(max_fig_size * 0.8, max(6, max_height * cell_size / 2))
    
    # Create figure
    plt.ioff()  # Turn off interactive mode
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colors - WHITE for 0s
    colors = {0: '#FFFFFF', 2: '#3498db', 'default': '#e74c3c'}
    
    # Create patches efficiently
    patches = []
    facecolors = []
    
    # Limit text rendering for performance
    max_text_cells = 5000 if quality == 'standard' else 10000
    text_count = 0
    
    for row_idx, row in enumerate(triangle):
        if len(row) == 0:
            continue
            
        row_width = len(row)
        start_x = (max_width - row_width) * cell_size / 2
        y_pos = (max_height - row_idx - 1) * cell_size
        
        for col_idx, value in enumerate(row):
            x_pos = start_x + col_idx * cell_size
            
            # Create rectangle
            rect = Rectangle((x_pos, y_pos), cell_size, cell_size)
            patches.append(rect)
            facecolors.append(colors.get(int(value), colors['default']))
            
            # Add text if enabled and under limit
            if show_text and text_count < max_text_cells:
                text_color = 'black' if value == 0 else 'white'
                ax.text(x_pos + cell_size/2, y_pos + cell_size/2, str(value),
                       ha='center', va='center', fontsize=font_size,
                       color=text_color, weight='bold')
                text_count += 1
    
    # Add all patches at once
    collection = PatchCollection(patches, facecolors=facecolors,
                               edgecolors='gray', linewidths=0.5)
    ax.add_collection(collection)
    
    # Set limits
    padding = cell_size
    ax.set_xlim(-padding, max_width * cell_size + padding)
    ax.set_ylim(-padding, max_height * cell_size + padding)
    
    # Title
    title = f'{sequence_name} ({max_terms} terms) - Blue: 2s, White: 0s, Red: Others'
    if not show_text:
        title += ' (Text hidden)'
    ax.set_title(title, fontsize=12 * scale_factor, pad=15)
    
    plt.tight_layout()
    
    # Force garbage collection
    gc.collect()
    
    return fig

def get_image_download(fig, filename, dpi=150):
    """Generate download link with memory management"""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        
        # Clear the figure to free memory
        plt.close(fig)
        gc.collect()
        
        return buf
    except MemoryError:
        st.error("Out of memory while generating image. Try a smaller triangle or lower quality.")
        plt.close(fig)
        gc.collect()
        return None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🔺 Recursive Difference Triangle Analyzer</h1>', 
                unsafe_allow_html=True)
    
    # Memory warning
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Memory Usage Warning:</strong> Large triangles can cause browser crashes. 
        Start with smaller sequences (≤100 terms) and increase gradually.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    # Sequence selection
    sequence_type = st.sidebar.selectbox(
        "Sequence Type:",
        ["Prime Numbers", "Fibonacci", "Natural Numbers", "Square Numbers", "Triangular Numbers"]
    )
    
    # Number of terms with stricter limits
    max_terms = st.sidebar.slider(
        "Number of Terms:",
        min_value=1,
        max_value=500,  # Reduced from 1000
        value=50,
        step=1,
        help="Start small to avoid memory issues"
    )
    
    # Memory usage indicator
    if max_terms <= 50:
        st.sidebar.success("✅ Low memory usage")
    elif max_terms <= 100:
        st.sidebar.info("ℹ️ Moderate memory usage")
    elif max_terms <= 200:
        st.sidebar.warning("⚠️ High memory usage - may be slow")
    else:
        st.sidebar.error("🚨 Very high memory usage - risk of crash")
    
    # Quality settings
    st.sidebar.header("🎨 Quality Settings")
    render_quality = st.sidebar.radio(
        "Render Quality:",
        ["Standard (Fast)", "High (Slower)", "Maximum (Slowest)"],
        index=0
    )
    
    quality_map = {
        "Standard (Fast)": "standard",
        "High (Slower)": "high",
        "Maximum (Slowest)": "maximum"
    }
    quality = quality_map[render_quality]
    
    # CSV upload
    st.sidebar.header("📁 Custom Sequence")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file (max 1000 values)", 
        type=['csv', 'txt']
    )
    
    # Info
    with st.sidebar.expander("ℹ️ How it works"):
        st.write("""
        **Triangle Generation:**
        1. Start with sequence
        2. Calculate absolute differences
        3. Repeat until one number remains
        
        **Colors:**
        - 🔵 Blue: 2s
        - ⚪ White: 0s  
        - 🔴 Red: Other values
        
        **Memory Tips:**
        - Start with small sequences
        - Use Standard quality for large triangles
        - Close other browser tabs
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.header("📊 About")
        st.markdown("""
        <div class="info-box">
        <h4>Memory-Optimized Version</h4>
        <p>This version includes:</p>
        <ul>
        <li>Automatic triangle truncation</li>
        <li>Memory-efficient rendering</li>
        <li>Garbage collection</li>
        <li>Simplified visualization for large triangles</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col1:
        # Generate sequence
        if uploaded_file is not None:
            sequence = parse_csv_robust(uploaded_file)
            if sequence:
                sequence_name = f"Custom ({len(sequence)} values)"
                max_terms = len(sequence)
                st.success(f"✅ Loaded {len(sequence)} values")
            else:
                st.error("❌ Could not parse file")
                sequence = generate_sequence(sequence_type, max_terms)
                sequence_name = sequence_type
        else:
            sequence = generate_sequence(sequence_type, max_terms)
            sequence_name = sequence_type
        
        # Preview
        with st.expander("🔍 Sequence Preview"):
            preview = sequence[:20]
            preview_text = ", ".join(map(str, preview))
            if len(sequence) > 20:
                preview_text += f"... ({len(sequence)} total)"
            st.code(preview_text)
        
        # Generate triangle with memory management
        with st.spinner("🔄 Computing triangle..."):
            # Limit triangle size for very large sequences
            max_triangle_rows = 200 if max_terms > 200 else None
            triangle = compute_triangle_memory_efficient(sequence, max_triangle_rows)
            
            # Clean up
            gc.collect()
        
        if triangle:
            # Statistics first (before heavy rendering)
            st.header("📈 Statistics")
            cols = st.columns(4)
            
            with cols[0]:
                st.metric("Height", len(triangle))
            with cols[1]:
                total_cells = sum(len(row) for row in triangle)
                st.metric("Total Cells", f"{total_cells:,}")
            with cols[2]:
                zero_count = sum(np.sum(row == 0) for row in triangle)
                st.metric("Zeros", zero_count)
            with cols[3]:
                two_count = sum(np.sum(row == 2) for row in triangle)
                st.metric("Twos", two_count)
            
            # Render visualization
            st.header("📊 Visualization")
            
            # Check memory before rendering
            if total_cells > 25000 and quality != 'standard':
                st.warning("Large triangle detected. Switching to Standard quality to prevent crashes.")
                quality = 'standard'
            
            with st.spinner("🎨 Rendering... (this may take a moment)"):
                try:
                    fig = create_memory_efficient_plot(triangle, sequence_name, max_terms, quality)
                    
                    if fig:
                        # Display with container width
                        st.pyplot(fig, use_container_width=True)
                        
                        # Download options
                        st.header("📥 Download")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Standard quality
                            buf = get_image_download(fig, "standard", dpi=150)
                            if buf:
                                st.download_button(
                                    "📥 Download (150 DPI)",
                                    data=buf,
                                    file_name=f"triangle_{sequence_name}_{max_terms}_150dpi.png",
                                    mime="image/png"
                                )
                        
                        with col2:
                            # High quality (only for smaller triangles)
                            if total_cells < 10000:
                                buf_hq = get_image_download(fig, "high", dpi=300)
                                if buf_hq:
                                    st.download_button(
                                        "📥 Download (300 DPI)",
                                        data=buf_hq,
                                        file_name=f"triangle_{sequence_name}_{max_terms}_300dpi.png",
                                        mime="image/png"
                                    )
                            else:
                                st.info("High quality disabled for large triangles")
                        
                        with col3:
                            # Maximum quality (only for small triangles)
                            if total_cells < 5000:
                                if st.button("🚀 Generate Max Quality"):
                                    with st.spinner("Creating maximum quality image..."):
                                        fig_max = create_memory_efficient_plot(
                                            triangle, sequence_name, max_terms, 'maximum'
                                        )
                                        if fig_max:
                                            buf_max = get_image_download(fig_max, "max", dpi=600)
                                            if buf_max:
                                                st.download_button(
                                                    "⬇️ Download (600 DPI)",
                                                    data=buf_max,
                                                    file_name=f"triangle_{sequence_name}_{max_terms}_600dpi.png",
                                                    mime="image/png"
                                                )
                            else:
                                st.info("Max quality disabled for large triangles")
                        
                        # Cleanup
                        plt.close('all')
                        gc.collect()
                        
                except MemoryError:
                    st.error("Out of memory! Please try:")
                    st.write("- Reduce the number of terms")
                    st.write("- Use Standard quality")
                    st.write("- Close other browser tabs")
                    st.write("- Refresh the page")
                    plt.close('all')
                    gc.collect()
        else:
            st.error("Could not generate triangle. Try fewer terms.")

if __name__ == "__main__":
    # Set memory-efficient matplotlib settings
    plt.rcParams['figure.max_open_warning'] = 5
    plt.rcParams['figure.autolayout'] = True
    
    # Run app
    main()
