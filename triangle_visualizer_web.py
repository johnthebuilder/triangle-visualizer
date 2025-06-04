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
matplotlib.use('Agg')
import sympy
import io
import base64
import gc

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
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_sequence(seq_type, n):
    """Generate a sequence of the specified type"""
    if seq_type == "Prime Numbers":
        return list(sympy.primerange(2, sympy.prime(min(n, 10000)) + 1))[:n]
    elif seq_type == "Fibonacci":
        if n <= 0:
            return []
        elif n == 1:
            return [1]
        elif n == 2:
            return [1, 1]
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

@st.cache_data
def compute_triangle(sequence):
    """Compute the recursive difference triangle"""
    if not sequence:
        return []
    
    triangle = []
    current = np.array(sequence, dtype=np.int64)
    triangle.append(current.copy())
    
    for i in range(1, len(sequence)):
        if len(current) <= 1:
            break
        current = np.abs(np.diff(current))
        triangle.append(current.copy())
    
    return triangle

def parse_csv_robust(uploaded_file):
    """Parse CSV file"""
    try:
        content = uploaded_file.read().decode('utf-8')
        lines = content.strip().split('\n')
        
        numbers = []
        max_numbers = 2000
        
        for line in lines[:max_numbers]:
            line = line.strip()
            if not line:
                continue
                
            if any(c.isalpha() for c in line) and not line.replace(',', '').replace('.', '').replace('-', '').isdigit():
                continue
            
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

def create_vector_plot(triangle, sequence_name, max_terms, format='svg', show_text_threshold=200):
    """Create vector-based plot (SVG or PDF)"""
    if not triangle:
        return None
    
    max_width = len(triangle[0])
    max_height = len(triangle)
    total_cells = sum(len(row) for row in triangle)
    
    # Adaptive sizing
    if max_width <= 50:
        cell_size = 0.8
        font_size = 10
    elif max_width <= 100:
        cell_size = 0.5
        font_size = 8
    elif max_width <= 200:
        cell_size = 0.3
        font_size = 6
    elif max_width <= 500:
        cell_size = 0.15
        font_size = 4
    else:
        cell_size = 0.08
        font_size = 3
    
    # Show text only for smaller triangles
    show_text = max_width <= show_text_threshold
    
    # Figure size
    fig_width = max(8, min(40, max_width * cell_size / 2))
    fig_height = max(6, min(30, max_height * cell_size / 2))
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colors - WHITE for 0s
    colors = {0: '#FFFFFF', 2: '#3498db', 'default': '#e74c3c'}
    
    # Create patches
    patches = []
    facecolors = []
    texts = []
    
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
            
            # Store text info
            if show_text:
                text_color = 'black' if value == 0 else 'white'
                texts.append((x_pos + cell_size/2, y_pos + cell_size/2, str(value), text_color))
    
    # Add patches
    collection = PatchCollection(patches, facecolors=facecolors,
                               edgecolors='gray', linewidths=0.5)
    ax.add_collection(collection)
    
    # Add texts if enabled
    if show_text:
        for x, y, text, color in texts:
            ax.text(x, y, text, ha='center', va='center',
                   fontsize=font_size, color=color, weight='bold')
    
    # Set limits
    padding = cell_size
    ax.set_xlim(-padding, max_width * cell_size + padding)
    ax.set_ylim(-padding, max_height * cell_size + padding)
    
    # Title
    title = f'{sequence_name} ({max_terms} terms) - Blue: 2s, White: 0s, Red: Others'
    if not show_text:
        title += f' (Text hidden for triangles >{show_text_threshold} width)'
    ax.set_title(title, fontsize=14, pad=15)
    
    plt.tight_layout()
    
    return fig

def get_svg_download(fig):
    """Generate SVG download"""
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight',
               facecolor='white', edgecolor='none')
    buf.seek(0)
    svg_data = buf.getvalue().decode('utf-8')
    plt.close(fig)
    gc.collect()
    return svg_data

def get_pdf_download(fig):
    """Generate PDF download"""
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', bbox_inches='tight',
               facecolor='white', edgecolor='none')
    buf.seek(0)
    pdf_data = buf.getvalue()
    plt.close(fig)
    gc.collect()
    return pdf_data

def get_png_download(fig, dpi=150):
    """Generate PNG download"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    gc.collect()
    return buf

def create_direct_svg(triangle, sequence_name, max_terms, max_width_for_text=100):
    """Create optimized SVG directly without matplotlib for very large triangles"""
    if not triangle:
        return None
    
    max_width = len(triangle[0])
    max_height = len(triangle)
    
    # Adaptive cell size
    if max_width <= 100:
        cell_size = 20
    elif max_width <= 500:
        cell_size = 10
    elif max_width <= 1000:
        cell_size = 5
    else:
        cell_size = 3
    
    width = max_width * cell_size + 2 * cell_size
    height = max_height * cell_size + 2 * cell_size
    
    # Start SVG
    svg_parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<g transform="translate({cell_size}, {cell_size})">'
    ]
    
    # Colors
    colors = {0: '#FFFFFF', 2: '#3498db'}
    default_color = '#e74c3c'
    
    # Add title
    svg_parts.append(
        f'<text x="{width/2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">'
        f'{sequence_name} ({max_terms} terms) - Blue: 2s, White: 0s, Red: Others</text>'
    )
    
    # Draw cells
    show_text = max_width <= max_width_for_text
    
    for row_idx, row in enumerate(triangle):
        if len(row) == 0:
            continue
            
        row_width = len(row)
        start_x = (max_width - row_width) * cell_size / 2
        y_pos = row_idx * cell_size + 30  # Offset for title
        
        for col_idx, value in enumerate(row):
            x_pos = start_x + col_idx * cell_size
            color = colors.get(int(value), default_color)
            
            # Rectangle
            svg_parts.append(
                f'<rect x="{x_pos}" y="{y_pos}" width="{cell_size}" height="{cell_size}" '
                f'fill="{color}" stroke="gray" stroke-width="0.5"/>'
            )
            
            # Text
            if show_text and cell_size >= 10:
                text_color = 'black' if value == 0 else 'white'
                font_size = min(cell_size * 0.6, 12)
                svg_parts.append(
                    f'<text x="{x_pos + cell_size/2}" y="{y_pos + cell_size/2}" '
                    f'text-anchor="middle" dominant-baseline="middle" '
                    f'font-size="{font_size}" fill="{text_color}" font-weight="bold">{value}</text>'
                )
    
    svg_parts.extend(['</g>', '</svg>'])
    
    return '\n'.join(svg_parts)

# Main app
def main():
    st.markdown('<h1 class="main-header">🔺 Recursive Difference Triangle Analyzer</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
        <strong>✨ Vector Export Edition:</strong> This version exports to SVG and PDF formats for infinite resolution! 
        Perfect for large triangles without memory issues.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    sequence_type = st.sidebar.selectbox(
        "Sequence Type:",
        ["Prime Numbers", "Fibonacci", "Natural Numbers", "Square Numbers", "Triangular Numbers"]
    )
    
    max_terms = st.sidebar.slider(
        "Number of Terms:",
        min_value=1,
        max_value=1000,  # Increased back since vectors handle it better
        value=50,
        step=1
    )
    
    # Export settings
    st.sidebar.header("📤 Export Settings")
    
    show_text_threshold = st.sidebar.slider(
        "Hide text when width exceeds:",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Text is hidden in large triangles for better performance"
    )
    
    export_format = st.sidebar.radio(
        "Preferred Export Format:",
        ["SVG (Vector - Recommended)", "PDF (Vector)", "PNG (Raster)"],
        index=0
    )
    
    # CSV upload
    st.sidebar.header("📁 Custom Sequence")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv', 'txt'])
    
    # Info
    with st.sidebar.expander("ℹ️ About Vector Formats"):
        st.write("""
        **SVG (Recommended):**
        - Infinite resolution
        - Editable in Illustrator/Inkscape
        - Small file size
        - Works in browsers
        
        **PDF:**
        - Infinite resolution
        - Universal format
        - Good for printing
        
        **PNG:**
        - Fixed resolution
        - Larger files
        - Use for compatibility
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.header("📊 Why Vectors?")
        st.markdown("""
        <div class="info-box">
        <h4>Benefits of Vector Export</h4>
        <ul>
        <li>🎯 <strong>Infinite zoom</strong> - no pixelation</li>
        <li>💾 <strong>Smaller files</strong> - geometric data only</li>
        <li>🖥️ <strong>No memory crashes</strong> - efficient format</li>
        <li>✏️ <strong>Editable</strong> - modify in vector programs</li>
        <li>🖨️ <strong>Print-ready</strong> - any size/resolution</li>
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
        
        # Generate triangle
        with st.spinner("🔄 Computing triangle..."):
            triangle = compute_triangle(sequence)
        
        if triangle:
            # Statistics
            st.header("📈 Statistics")
            cols = st.columns(4)
            
            total_cells = sum(len(row) for row in triangle)
            
            with cols[0]:
                st.metric("Height", len(triangle))
            with cols[1]:
                st.metric("Total Cells", f"{total_cells:,}")
            with cols[2]:
                zero_count = sum(np.sum(row == 0) for row in triangle)
                st.metric("Zeros", zero_count)
            with cols[3]:
                two_count = sum(np.sum(row == 2) for row in triangle)
                st.metric("Twos", two_count)
            
            # Visualization
            st.header("📊 Visualization")
            
            # For display, create a low-res version
            with st.spinner("🎨 Rendering preview..."):
                display_fig = create_vector_plot(triangle, sequence_name, max_terms, 
                                               show_text_threshold=min(show_text_threshold, 100))
                if display_fig:
                    st.pyplot(display_fig, use_container_width=True)
                    plt.close(display_fig)
            
            # Export section
            st.header("📥 Export Options")
            
            export_cols = st.columns(3)
            
            # Prepare filename base
            filename_base = f"triangle_{sequence_name.lower().replace(' ', '_')}_{max_terms}terms"
            
            with export_cols[0]:
                # SVG Export
                if st.button("🎨 Generate SVG", type="primary", use_container_width=True):
                    with st.spinner("Creating SVG..."):
                        # For very large triangles, use direct SVG generation
                        if total_cells > 50000:
                            svg_data = create_direct_svg(triangle, sequence_name, max_terms, show_text_threshold)
                        else:
                            fig = create_vector_plot(triangle, sequence_name, max_terms, 
                                                   show_text_threshold=show_text_threshold)
                            svg_data = get_svg_download(fig)
                        
                        if svg_data:
                            st.download_button(
                                "⬇️ Download SVG",
                                data=svg_data,
                                file_name=f"{filename_base}.svg",
                                mime="image/svg+xml",
                                use_container_width=True
                            )
                            st.success("✅ SVG ready!")
            
            with export_cols[1]:
                # PDF Export
                if st.button("📄 Generate PDF", type="secondary", use_container_width=True):
                    with st.spinner("Creating PDF..."):
                        fig = create_vector_plot(triangle, sequence_name, max_terms,
                                               show_text_threshold=show_text_threshold)
                        pdf_data = get_pdf_download(fig)
                        
                        if pdf_data:
                            st.download_button(
                                "⬇️ Download PDF",
                                data=pdf_data,
                                file_name=f"{filename_base}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                            st.success("✅ PDF ready!")
            
            with export_cols[2]:
                # PNG Export (lower res)
                if st.button("🖼️ Generate PNG", use_container_width=True):
                    with st.spinner("Creating PNG..."):
                        fig = create_vector_plot(triangle, sequence_name, max_terms,
                                               show_text_threshold=show_text_threshold)
                        png_data = get_png_download(fig, dpi=150)
                        
                        if png_data:
                            st.download_button(
                                "⬇️ Download PNG (150 DPI)",
                                data=png_data,
                                file_name=f"{filename_base}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                            st.success("✅ PNG ready!")
            
            # Advanced options
            with st.expander("🔧 Advanced Export Options"):
                st.write("**For extremely large triangles (1000+ width):**")
                st.code("""
# The SVG export will work best. After downloading:
# 1. Open in Inkscape (free) or Adobe Illustrator
# 2. Select all (Ctrl+A)
# 3. Export as PNG at any resolution you need
# 4. Or print directly at any size
                """)
                
                if total_cells > 100000:
                    st.info(f"Your triangle has {total_cells:,} cells. The direct SVG generator is optimized for this size!")

if __name__ == "__main__":
    main()
