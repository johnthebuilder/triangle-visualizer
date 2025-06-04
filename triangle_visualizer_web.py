import streamlit as st

st.set_page_config(
    page_title="Triangle Visualizer",
    page_icon="🔺",
    layout="wide",
    initial_sidebar_state="expanded"
)

import numpy as np
import sympy
import io
import base64
import gc
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib
matplotlib.use('Agg')

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
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_sequence(seq_type, n):
    """Generate sequence with progress indication for large n"""
    if seq_type == "Prime Numbers":
        # More efficient prime generation for large n
        primes = []
        if n > 0:
            primes.append(2)
        if n > 1:
            primes.append(3)
        
        candidate = 5
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 2
        
        return primes[:n]
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
def compute_triangle_chunked(sequence, max_rows=None):
    """Compute triangle with chunking for very large sequences"""
    if not sequence:
        return []
    
    n = len(sequence)
    if max_rows:
        n = min(n, max_rows)
    
    # For very large triangles, we'll compute in chunks and yield results
    triangle = []
    current = np.array(sequence, dtype=np.int64)
    triangle.append(current.tolist())  # Convert to list to save memory
    
    for i in range(1, n):
        if len(current) <= 1:
            break
        current = np.abs(np.diff(current))
        triangle.append(current.tolist())
        
        # Periodic garbage collection for very large triangles
        if i % 100 == 0:
            gc.collect()
    
    return triangle

def create_svg_chunked(triangle, sequence_name, max_terms, chunk_size=1000):
    """Create SVG in chunks to handle very large triangles"""
    if not triangle:
        return None
    
    max_width = len(triangle[0])
    max_height = len(triangle)
    total_cells = sum(len(row) for row in triangle)
    
    # Adaptive cell size based on triangle width
    if max_width <= 100:
        cell_size = 10
    elif max_width <= 500:
        cell_size = 5
    elif max_width <= 1000:
        cell_size = 3
    elif max_width <= 5000:
        cell_size = 1.5
    else:
        cell_size = 1
    
    # SVG dimensions
    width = max_width * cell_size + 2 * cell_size
    height = max_height * cell_size + 4 * cell_size  # Extra space for title
    
    # Colors
    colors = {0: '#FFFFFF', 2: '#3498db'}
    default_color = '#e74c3c'
    
    # Start building SVG
    svg_parts = []
    
    # SVG header
    svg_parts.append(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">')
    svg_parts.append('<rect width="100%" height="100%" fill="white"/>')
    
    # Title
    svg_parts.append(
        f'<text x="{width/2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">'
        f'{sequence_name} ({max_terms} terms) - Blue: 2s, White: 0s, Red: Others</text>'
    )
    
    # Main group with offset for title
    svg_parts.append(f'<g transform="translate({cell_size}, {2*cell_size})">')
    
    # Process triangle in chunks to avoid memory issues
    rows_processed = 0
    chunk_parts = []
    
    for row_idx, row in enumerate(triangle):
        if len(row) == 0:
            continue
        
        row_width = len(row)
        start_x = (max_width - row_width) * cell_size / 2
        y_pos = row_idx * cell_size
        
        # Build row SVG
        row_parts = []
        for col_idx, value in enumerate(row):
            x_pos = start_x + col_idx * cell_size
            color = colors.get(int(value), default_color)
            
            # Simple rectangle without stroke for very large triangles
            if total_cells > 100000:
                row_parts.append(
                    f'<rect x="{x_pos}" y="{y_pos}" width="{cell_size}" height="{cell_size}" fill="{color}"/>'
                )
            else:
                row_parts.append(
                    f'<rect x="{x_pos}" y="{y_pos}" width="{cell_size}" height="{cell_size}" '
                    f'fill="{color}" stroke="gray" stroke-width="0.2"/>'
                )
        
        chunk_parts.extend(row_parts)
        rows_processed += 1
        
        # Write chunk to avoid memory buildup
        if rows_processed % chunk_size == 0:
            svg_parts.extend(chunk_parts)
            chunk_parts = []
            gc.collect()
    
    # Add remaining parts
    if chunk_parts:
        svg_parts.extend(chunk_parts)
    
    # Close SVG
    svg_parts.append('</g>')
    svg_parts.append('</svg>')
    
    # Join all parts
    svg_content = '\n'.join(svg_parts)
    
    return svg_content

def create_preview_plot(triangle, sequence_name, max_terms, max_preview_width=200):
    """Create a preview plot for display - downsample if needed"""
    if not triangle:
        return None
    
    max_width = len(triangle[0])
    max_height = len(triangle)
    
    # Downsample for preview if too large
    if max_width > max_preview_width:
        sample_rate = max_width // max_preview_width
        triangle_preview = []
        for i in range(0, len(triangle), sample_rate):
            if i < len(triangle):
                row = triangle[i]
                sampled_row = row[::sample_rate] if len(row) > sample_rate else row
                triangle_preview.append(sampled_row)
        triangle = triangle_preview
        max_width = len(triangle[0]) if triangle else 0
        max_height = len(triangle)
        preview_text = f" (Preview - downsampled {sample_rate}x)"
    else:
        preview_text = ""
    
    # Create matplotlib figure for preview only
    cell_size = 0.5 if max_width > 100 else 0.8
    fig_width = min(12, max(6, max_width * cell_size / 2))
    fig_height = min(10, max(4, max_height * cell_size / 2))
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_aspect('equal')
    ax.axis('off')
    
    colors = {0: '#FFFFFF', 2: '#3498db', 'default': '#e74c3c'}
    
    patches = []
    facecolors = []
    
    for row_idx, row in enumerate(triangle):
        if len(row) == 0:
            continue
        
        row_width = len(row)
        start_x = (max_width - row_width) * cell_size / 2
        y_pos = (max_height - row_idx - 1) * cell_size
        
        for col_idx, value in enumerate(row):
            x_pos = start_x + col_idx * cell_size
            rect = Rectangle((x_pos, y_pos), cell_size, cell_size)
            patches.append(rect)
            facecolors.append(colors.get(int(value), colors['default']))
    
    collection = PatchCollection(patches, facecolors=facecolors,
                               edgecolors='gray', linewidths=0.3)
    ax.add_collection(collection)
    
    padding = cell_size
    ax.set_xlim(-padding, max_width * cell_size + padding)
    ax.set_ylim(-padding, max_height * cell_size + padding)
    
    title = f'{sequence_name} ({max_terms} terms){preview_text}'
    ax.set_title(title, fontsize=12, pad=10)
    
    plt.tight_layout()
    return fig

def parse_csv_robust(uploaded_file):
    """Parse CSV file with support for large files"""
    try:
        content = uploaded_file.read().decode('utf-8')
        lines = content.strip().split('\n')
        
        numbers = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip headers
            if any(c.isalpha() for c in line) and not line.replace(',', '').replace('.', '').replace('-', '').isdigit():
                continue
            
            # Parse numbers
            for delimiter in [',', ' ', '\t', ';']:
                if delimiter in line:
                    parts = line.split(delimiter)
                    for part in parts:
                        part = part.strip()
                        if part and part.replace('.', '').replace('-', '').lstrip('-').isdigit():
                            try:
                                numbers.append(int(float(part)))
                            except:
                                continue
                    break
            else:
                if line.replace('.', '').replace('-', '').lstrip('-').isdigit():
                    try:
                        numbers.append(int(float(line)))
                    except:
                        pass
        
        return numbers if numbers else None
    except:
        return None

# Main app
def main():
    st.markdown('<h1 class="main-header">🔺 Recursive Difference Triangle Analyzer</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
        <strong>✨ Ultra-Scale Edition:</strong> Optimized to handle triangles with 10,000+ terms! 
        Uses efficient SVG generation and chunked processing.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    sequence_type = st.sidebar.selectbox(
        "Sequence Type:",
        ["Prime Numbers", "Fibonacci", "Natural Numbers", "Square Numbers", "Triangular Numbers"]
    )
    
    # Larger range for ultra-scale
    max_terms = st.sidebar.number_input(
        "Number of Terms:",
        min_value=1,
        max_value=10000,
        value=50,
        step=1
    )
    
    # Scale indicator
    if max_terms <= 100:
        st.sidebar.success("✅ Small scale - Full preview")
    elif max_terms <= 1000:
        st.sidebar.info("ℹ️ Medium scale - Downsampled preview")
    elif max_terms <= 5000:
        st.sidebar.warning("⚠️ Large scale - Heavily downsampled preview")
    else:
        st.sidebar.error("🚨 Ultra scale - Minimal preview, full SVG export")
    
    # Advanced settings
    st.sidebar.header("⚙️ Advanced Settings")
    
    if max_terms > 1000:
        compute_full_triangle = st.sidebar.checkbox(
            "Compute full triangle", 
            value=True,
            help="Uncheck to limit triangle depth for very large sequences"
        )
        
        if not compute_full_triangle:
            max_triangle_rows = st.sidebar.slider(
                "Maximum triangle rows:",
                min_value=10,
                max_value=1000,
                value=min(500, max_terms),
                step=10
            )
        else:
            max_triangle_rows = None
    else:
        max_triangle_rows = None
    
    # CSV upload
    st.sidebar.header("📁 Custom Sequence")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv', 'txt'])
    
    # Info
    with st.sidebar.expander("ℹ️ Performance Tips"):
        st.write("""
        **For sequences > 1000 terms:**
        - Preview will be downsampled
        - Full resolution in SVG export
        - Expect longer processing times
        
        **For sequences > 5000 terms:**
        - Use "Maximum triangle rows" option
        - SVG export may take 30+ seconds
        - Files will be large (10+ MB)
        
        **Memory saving:**
        - Close other browser tabs
        - Use Chrome/Firefox for best performance
        """)
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.header("📊 Capabilities")
        st.markdown("""
        <div class="info-box">
        <h4>This version handles:</h4>
        <ul>
        <li>✅ Up to 10,000 terms</li>
        <li>✅ Millions of cells</li>
        <li>✅ Chunked SVG generation</li>
        <li>✅ Memory-efficient processing</li>
        <li>✅ Downsampled previews</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col1:
        # Generate or load sequence
        if uploaded_file is not None:
            with st.spinner("Parsing CSV file..."):
                sequence = parse_csv_robust(uploaded_file)
            if sequence:
                sequence_name = f"Custom ({len(sequence)} values)"
                max_terms = len(sequence)
                st.success(f"✅ Loaded {len(sequence)} values")
            else:
                st.error("❌ Could not parse file")
                sequence = []
        else:
            with st.spinner(f"Generating {max_terms} {sequence_type}..."):
                sequence = generate_sequence(sequence_type, max_terms)
            sequence_name = sequence_type
        
        if sequence:
            # Preview sequence
            with st.expander("🔍 Sequence Preview"):
                preview = sequence[:50]
                preview_text = ", ".join(map(str, preview))
                if len(sequence) > 50:
                    preview_text += f"... ({len(sequence)} total)"
                st.code(preview_text)
            
            # Compute triangle
            with st.spinner(f"Computing triangle for {len(sequence)} terms..."):
                triangle = compute_triangle_chunked(sequence, max_triangle_rows)
                gc.collect()
            
            if triangle:
                # Statistics
                st.header("📈 Statistics")
                cols = st.columns(4)
                
                total_cells = sum(len(row) for row in triangle)
                
                with cols[0]:
                    st.metric("Height", f"{len(triangle):,}")
                with cols[1]:
                    st.metric("Total Cells", f"{total_cells:,}")
                with cols[2]:
                    st.metric("Width", f"{len(triangle[0]):,}")
                with cols[3]:
                    # Estimate file size
                    est_size_mb = (total_cells * 100) / (1024 * 1024)  # ~100 bytes per cell
                    st.metric("Est. SVG Size", f"{est_size_mb:.1f} MB")
                
                # Preview visualization
                st.header("👁️ Preview")
                
                if total_cells > 1000000:
                    st.info("Triangle too large for preview. Use SVG export for full visualization.")
                else:
                    with st.spinner("Generating preview..."):
                        preview_fig = create_preview_plot(triangle, sequence_name, max_terms)
                        if preview_fig:
                            st.pyplot(preview_fig, use_container_width=True)
                            plt.close(preview_fig)
                            gc.collect()
                
                # Export section
                st.header("📥 Export Full Resolution")
                
                export_cols = st.columns(2)
                
                with export_cols[0]:
                    st.markdown("""
                    <div class="info-box">
                    <h4>SVG Export Info:</h4>
                    <p>• Full resolution export<br>
                    • No downsampling<br>
                    • May take time for large triangles<br>
                    • Open in browser/Inkscape/Illustrator</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with export_cols[1]:
                    if st.button("🎨 Generate Full SVG", type="primary", use_container_width=True):
                        
                        # Progress indication for large exports
                        if total_cells > 100000:
                            progress_text = st.empty()
                            progress_bar = st.progress(0)
                            
                            progress_text.text(f"Generating SVG for {total_cells:,} cells...")
                            progress_bar.progress(0.2)
                        
                        # Generate SVG
                        try:
                            svg_content = create_svg_chunked(triangle, sequence_name, max_terms)
                            
                            if total_cells > 100000:
                                progress_bar.progress(0.8)
                            
                            if svg_content:
                                # Prepare download
                                filename = f"triangle_{sequence_name.lower().replace(' ', '_')}_{max_terms}terms.svg"
                                
                                st.download_button(
                                    "⬇️ Download SVG",
                                    data=svg_content,
                                    file_name=filename,
                                    mime="image/svg+xml",
                                    use_container_width=True
                                )
                                
                                if total_cells > 100000:
                                    progress_bar.progress(1.0)
                                    progress_text.text("✅ SVG generated successfully!")
                                
                                st.success(f"✅ SVG ready! File size: ~{len(svg_content)/(1024*1024):.1f} MB")
                                
                                # Cleanup
                                gc.collect()
                                
                        except Exception as e:
                            st.error(f"Error generating SVG: {str(e)}")
                            st.info("Try reducing the number of terms or limiting triangle depth.")
                
                # Additional export options
                with st.expander("🔧 Alternative Export Options"):
                    st.write("""
                    **For extremely large triangles that still cause issues:**
                    
                    1. **Limit triangle depth** - Use the checkbox in Advanced Settings
                    2. **Export data as CSV** - Process with external tools
                    3. **Use a dedicated visualization tool** - Export the raw triangle data
                    """)
                    
                    if st.button("📊 Export Triangle Data as CSV"):
                        # Export raw triangle data
                        csv_lines = []
                        for i, row in enumerate(triangle):
                            csv_lines.append(",".join(map(str, row)))
                        csv_content = "\n".join(csv_lines)
                        
                        st.download_button(
                            "⬇️ Download Triangle Data (CSV)",
                            data=csv_content,
                            file_name=f"triangle_data_{sequence_name}_{max_terms}.csv",
                            mime="text/csv"
                        )

if __name__ == "__main__":
    # Streamlit config for large data
    st.set_option('deprecation.showPyplotGlobalUse', False)
    main()
