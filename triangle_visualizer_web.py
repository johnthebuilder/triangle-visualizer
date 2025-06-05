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
from PIL import Image
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
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_sequence(seq_type, n):
    """Generate sequence efficiently"""
    if seq_type == "Prime Numbers":
        # Efficient prime generation
        if n <= 0:
            return []
        primes = [2]
        if n == 1:
            return primes
        
        # Sieve for first batch of primes
        limit = max(30, n * 15)  # Estimate
        sieve = np.ones(limit, dtype=bool)
        sieve[0:2] = False
        
        for i in range(2, int(np.sqrt(limit)) + 1):
            if sieve[i]:
                sieve[i*i::i] = False
        
        primes = np.where(sieve)[0].tolist()
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
    """Compute triangle with memory efficiency"""
    if not sequence:
        return []
    
    n = len(sequence)
    if max_rows:
        n = min(n, max_rows)
    
    triangle = []
    current = np.array(sequence, dtype=np.int64)
    triangle.append(current.tolist())
    
    for i in range(1, n):
        if len(current) <= 1:
            break
        current = np.abs(np.diff(current))
        triangle.append(current.tolist())
        
        if i % 100 == 0:
            gc.collect()
    
    return triangle

def create_pixel_array(triangle, max_pixels=4000, high_quality=False):
    """Create a pixel array representation for very large triangles"""
    if not triangle:
        return None, 1
    
    max_width = len(triangle[0])
    max_height = len(triangle)
    
    # For high quality, use MUCH higher resolution limits
    if high_quality:
        # Allow up to 32K resolution for high quality exports
        max_pixels = min(32768, max(max_width * 2, max_height * 2, 16000))
    
    # Calculate scaling to fit within max_pixels constraint
    scale = max(1, max(max_width, max_height) / max_pixels)
    
    # Create pixel dimensions
    pixel_width = int(max_width / scale)
    pixel_height = int(max_height / scale)
    
    # Ensure minimum 1 pixel per cell for smaller triangles
    # For high quality, try to use at least 2 pixels per cell if possible
    if scale < 1:
        pixel_width = max_width * (2 if high_quality else 1)
        pixel_height = max_height * (2 if high_quality else 1)
        scale = 1 / (2 if high_quality else 1)
    
    # Initialize pixel array (RGB) - start with white background
    pixels = np.ones((pixel_height, pixel_width, 3), dtype=np.uint8) * 255
    
    # Colors (RGB)
    colors = {
        0: np.array([255, 255, 255], dtype=np.uint8),  # White
        2: np.array([52, 152, 219], dtype=np.uint8),   # Blue
        'default': np.array([231, 76, 60], dtype=np.uint8)  # Red
    }
    
    # Process each pixel
    for py in range(pixel_height):
        # Calculate which row in the triangle this pixel represents
        triangle_y = int(py * scale)
        
        if triangle_y >= len(triangle):
            continue
            
        row = triangle[triangle_y]
        row_width = len(row)
        
        # Calculate the starting position for this row (centering)
        row_start_x = (max_width - row_width) / 2.0
        
        for px in range(pixel_width):
            # Calculate which column in the triangle this pixel represents
            triangle_x = px * scale
            
            # Check if this pixel is within the row bounds
            col_idx = int(triangle_x - row_start_x)
            
            if 0 <= col_idx < row_width:
                value = row[col_idx]
                color = colors.get(value, colors['default'])
                pixels[py, px] = color
    
    return pixels, scale

def create_efficient_svg(triangle, sequence_name, max_terms, max_cells=50000):
    """Create an efficient SVG using run-length encoding for large triangles"""
    if not triangle:
        return None
    
    max_width = len(triangle[0])
    max_height = len(triangle)
    total_cells = sum(len(row) for row in triangle)
    
    # If too many cells, sample the triangle
    if total_cells > max_cells:
        sample_rate = int(np.sqrt(total_cells / max_cells))
        sampled_triangle = []
        for i in range(0, len(triangle), sample_rate):
            if i < len(triangle):
                row = triangle[i]
                sampled_row = row[::sample_rate] if len(row) > sample_rate else row
                sampled_triangle.append(sampled_row)
        triangle = sampled_triangle
        max_width = max(len(row) for row in triangle) if triangle else 0
        max_height = len(triangle)
        is_sampled = True
    else:
        is_sampled = False
    
    # Cell size
    cell_size = max(1, min(10, 1000 / max_width))
    
    # SVG dimensions
    width = max_width * cell_size + 2 * cell_size
    height = max_height * cell_size + 4 * cell_size
    
    # Start SVG
    svg_parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">',
        f'{sequence_name} ({max_terms} terms) - Blue: 2s, White: 0s, Red: Others',
    ]
    
    if is_sampled:
        svg_parts.append(f' [Sampled {sample_rate}x]')
    
    svg_parts.extend([
        '</text>',
        f'<g transform="translate({cell_size}, {2*cell_size})">'
    ])
    
    # Process triangle with run-length encoding
    for row_idx, row in enumerate(triangle):
        if not row:
            continue
            
        row_width = len(row)
        start_x = (max_width - row_width) * cell_size / 2
        y_pos = row_idx * cell_size
        
        # Run-length encode the row
        if row:
            runs = []
            current_val = row[0]
            run_start = 0
            
            for i in range(1, len(row)):
                if row[i] != current_val:
                    runs.append((current_val, run_start, i - run_start))
                    current_val = row[i]
                    run_start = i
            runs.append((current_val, run_start, len(row) - run_start))
            
            # Generate rectangles for runs
            for value, start, length in runs:
                x_pos = start_x + start * cell_size
                color = '#FFFFFF' if value == 0 else '#3498db' if value == 2 else '#e74c3c'
                
                svg_parts.append(
                    f'<rect x="{x_pos}" y="{y_pos}" width="{length * cell_size}" '
                    f'height="{cell_size}" fill="{color}"/>'
                )
    
    svg_parts.extend(['</g>', '</svg>'])
    return '\n'.join(svg_parts)

def create_preview_plot(triangle, sequence_name, max_terms, max_size=200):
    """Create a matplotlib preview for reasonable-sized triangles"""
    if not triangle:
        return None
    
    max_width = len(triangle[0])
    max_height = len(triangle)
    
    # Downsample if needed
    if max_width > max_size or max_height > max_size:
        scale = max(max_width / max_size, max_height / max_size)
        sample_rate = int(np.ceil(scale))
        
        sampled_triangle = []
        for i in range(0, len(triangle), sample_rate):
            if i < len(triangle):
                row = triangle[i]
                sampled_row = row[::sample_rate] if len(row) > sample_rate else row
                sampled_triangle.append(sampled_row)
        
        triangle = sampled_triangle
        max_width = max(len(row) for row in triangle) if triangle else 0
        max_height = len(triangle)
        preview_text = f" (Preview - {sample_rate}x downsampled)"
    else:
        preview_text = ""
    
    # Create plot
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
    """Parse CSV file"""
    try:
        content = uploaded_file.read().decode('utf-8')
        lines = content.strip().split('\n')
        
        numbers = []
        
        for line in lines:
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
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    sequence_type = st.sidebar.selectbox(
        "Sequence Type:",
        ["Prime Numbers", "Fibonacci", "Natural Numbers", "Square Numbers", "Triangular Numbers"]
    )
    
    max_terms = st.sidebar.number_input(
        "Number of Terms:",
        min_value=1,
        max_value=10000,
        value=50,
        step=1
    )
    
    # Scale indicator
    if max_terms <= 500:
        st.sidebar.success("✅ Optimal range")
    elif max_terms <= 2000:
        st.sidebar.info("ℹ️ Large triangle - consider PNG export")
    elif max_terms <= 5000:
        st.sidebar.warning("⚠️ Very large - PNG export recommended")
    else:
        st.sidebar.error("🚨 Extreme size - PNG only")
    
    # Advanced settings
    st.sidebar.header("⚙️ Advanced Settings")
    
    export_format = st.sidebar.radio(
        "Export Format:",
        ["Auto (Recommended)", "SVG (Vector)", "PNG (Raster)"],
        help="Auto selects best format based on size"
    )
    
    # PNG quality settings
    if export_format == "PNG (Raster)" or max_terms > 1000:
        png_quality = st.sidebar.select_slider(
            "PNG Resolution:",
            options=["Standard", "High", "Ultra", "Maximum", "Extreme", "Ultimate"],
            value="High",
            help="Higher quality = larger file size & longer processing"
        )
    else:
        png_quality = "High"
    
    if max_terms > 1000:
        limit_triangle = st.sidebar.checkbox(
            "Limit triangle depth",
            value=max_terms > 3000,
            help="Prevents extremely deep triangles"
        )
        
        if limit_triangle:
            max_rows = st.sidebar.slider(
                "Maximum rows:",
                min_value=100,
                max_value=min(2000, max_terms),
                value=min(1000, max_terms),
                step=100
            )
        else:
            max_rows = None
    else:
        max_rows = None
    
    # CSV upload
    st.sidebar.header("📁 Custom Sequence")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv', 'txt'])
    
    # Info
    with st.sidebar.expander("ℹ️ Performance Guide"):
        st.write("""
        **PNG Resolution Options:**
        - Standard: 4K (4000×4000 max)
        - High: 8K (8000×8000 max)
        - Ultra: 12K (12000×12000 max)
        - Maximum: 16K (16000×16000 max)
        
        **Why PNG for large triangles?**
        - Millions of SVG elements slow browsers
        - PNG uses efficient pixel representation
        - Still ultra-high quality at 16K resolution
        """)
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
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
        # Generate sequence
        if uploaded_file is not None:
            with st.spinner("Parsing CSV..."):
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
            # Preview
            with st.expander("🔍 Sequence Preview"):
                preview = sequence[:50]
                preview_text = ", ".join(map(str, preview))
                if len(sequence) > 50:
                    preview_text += f"... ({len(sequence)} total)"
                st.code(preview_text)
            
            # Compute triangle
            with st.spinner("Computing triangle..."):
                triangle = compute_triangle_chunked(sequence, max_rows)
                gc.collect()
            
            if triangle:
                # Statistics
                st.header("📈 Statistics")
                cols = st.columns(6)
                
                total_cells = sum(len(row) for row in triangle)
                
                with cols[0]:
                    st.metric("Height", f"{len(triangle):,}")
                with cols[1]:
                    st.metric("Total Cells", f"{total_cells:,}")
                with cols[2]:
                    st.metric("Width", f"{len(triangle[0]):,}")
                
                # Count values efficiently
                zero_count = 0
                two_count = 0
                other_count = 0
                
                for row in triangle:
                    for value in row:
                        if value == 0:
                            zero_count += 1
                        elif value == 2:
                            two_count += 1
                        else:
                            other_count += 1
                
                with cols[3]:
                    st.metric("Zeros", f"{zero_count:,}")
                with cols[4]:
                    st.metric("Twos", f"{two_count:,}")
                with cols[5]:
                    st.metric("Others", f"{other_count:,}")
                
                # Auto format selection
                if export_format == "Auto (Recommended)":
                    if total_cells < 10000:
                        auto_format = "svg"
                    elif total_cells < 100000:
                        auto_format = "efficient_svg"
                    else:
                        auto_format = "png"
                elif export_format == "SVG (Vector)":
                    auto_format = "efficient_svg"
                else:
                    auto_format = "png"
                
                # Show format recommendation
                if total_cells > 100000:
                    st.markdown("""
                    <div class="warning-box">
                    <strong>⚠️ Large Triangle Detected</strong><br>
                    PNG export is recommended for triangles with over 100,000 cells for best performance.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Preview
                st.header("👁️ Preview")
                
                if total_cells < 500000:
                    with st.spinner("Generating preview..."):
                        fig = create_preview_plot(triangle, sequence_name, max_terms)
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
                else:
                    # For very large triangles, show pixel preview
                    with st.spinner("Generating pixel preview..."):
                        pixels, scale = create_pixel_array(triangle, max_pixels=800)
                        if pixels is not None:
                            img = Image.fromarray(pixels)
                            st.image(img, caption=f"Pixel preview (scale: 1:{scale:.1f})")
                
                # Export section
                st.header("📥 Export Options")
                
                export_cols = st.columns(2)
                
                with export_cols[0]:
                    if auto_format in ["svg", "efficient_svg"]:
                        if st.button("🎨 Generate SVG", type="primary", use_container_width=True):
                            with st.spinner("Creating optimized SVG..."):
                                svg_content = create_efficient_svg(triangle, sequence_name, max_terms)
                                
                                if svg_content:
                                    filename = f"triangle_{sequence_name.lower().replace(' ', '_')}_{max_terms}.svg"
                                    
                                    st.download_button(
                                        "⬇️ Download SVG",
                                        data=svg_content,
                                        file_name=filename,
                                        mime="image/svg+xml",
                                        use_container_width=True
                                    )
                                    st.success("✅ SVG ready!")
                
                with export_cols[1]:
                    if st.button("🖼️ Generate PNG", type="primary" if auto_format == "png" else "secondary", 
                                use_container_width=True):
                        with st.spinner("Creating high-resolution PNG..."):
                            # Determine resolution based on quality setting
                            quality_settings = {
                                "Standard": {"max_pixels": 4000, "desc": "4K"},
                                "High": {"max_pixels": 8000, "desc": "8K"}, 
                                "Ultra": {"max_pixels": 12000, "desc": "12K"},
                                "Maximum": {"max_pixels": 16000, "desc": "16K"},
                                "Extreme": {"max_pixels": 24000, "desc": "24K"},
                                "Ultimate": {"max_pixels": 32000, "desc": "32K"}
                            }
                            
                            settings = quality_settings.get(png_quality, quality_settings["High"])
                            
                            # For smaller triangles, ensure we use full resolution
                            # For Ultimate quality, always use high quality mode
                            use_high_quality = png_quality in ["Maximum", "Extreme", "Ultimate"] or max_terms < 2000
                            
                            # Show warning for extreme resolutions
                            if png_quality in ["Extreme", "Ultimate"]:
                                st.warning("⚠️ Extreme resolution selected. This may take several minutes and produce very large files (100+ MB).")
                            
                            pixels, scale = create_pixel_array(
                                triangle, 
                                max_pixels=settings["max_pixels"],
                                high_quality=use_high_quality
                            )
                            
                            if pixels is not None:
                                # Create high-quality PNG
                                img = Image.fromarray(pixels, mode='RGB')
                                
                                # Add title to image for extreme resolutions
                                if png_quality in ["Extreme", "Ultimate"]:
                                    from PIL import ImageDraw
                                    try:
                                        draw = ImageDraw.Draw(img)
                                        # Add title with larger font for high res
                                        title_text = f"{sequence_name} ({max_terms} terms) - {settings['desc']} Resolution"
                                        # Position title based on image size
                                        title_y = min(50, img.height // 100)
                                        draw.text((img.width // 20, title_y), title_text, fill=(100, 100, 100))
                                    except:
                                        pass
                                
                                # Add metadata
                                from PIL import PngImagePlugin
                                metadata = PngImagePlugin.PngInfo()
                                metadata.add_text("Title", f"Triangle {sequence_name} ({max_terms} terms)")
                                metadata.add_text("Software", "Triangle Visualizer - Ultra High Resolution")
                                metadata.add_text("Description", f"Scale: 1:{scale:.2f}, Cells: {total_cells:,}, Quality: {png_quality} ({settings['desc']})")
                                metadata.add_text("Colors", "White=0, Blue=2, Red=Others")
                                metadata.add_text("Resolution", f"{img.width}x{img.height} pixels")
                                
                                # Save to buffer with optimization
                                buffer = io.BytesIO()
                                
                                # Use lower compression for extreme resolutions to save time
                                compress_level = 9 if png_quality in ["Standard", "High"] else 6
                                
                                # Progress message for large images
                                if img.width * img.height > 100_000_000:
                                    st.info("Compressing image... This may take a minute for extreme resolutions.")
                                
                                img.save(buffer, format="PNG", pnginfo=metadata, optimize=True, compress_level=compress_level)
                                buffer.seek(0)
                                
                                filename = f"triangle_{sequence_name.lower().replace(' ', '_')}_{max_terms}_{settings['desc']}.png"
                                
                                st.download_button(
                                    "⬇️ Download PNG",
                                    data=buffer,
                                    file_name=filename,
                                    mime="image/png",
                                    use_container_width=True
                                )
                                
                                # Show resolution info
                                file_size_mb = len(buffer.getvalue()) / (1024 * 1024)
                                total_pixels = img.width * img.height
                                megapixels = total_pixels / 1_000_000
                                
                                st.success(f"✅ PNG ready! Resolution: {img.width:,}×{img.height:,} ({megapixels:.1f} MP, {settings['desc']}) - {file_size_mb:.1f} MB")
                                
                                # Detailed stats for extreme resolutions
                                if png_quality in ["Extreme", "Ultimate"]:
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Megapixels", f"{megapixels:.1f} MP")
                                    with col2:
                                        st.metric("File Size", f"{file_size_mb:.1f} MB")
                                    with col3:
                                        if scale < 1:
                                            st.metric("Pixels per Cell", f"{1/scale:.1f}")
                                        else:
                                            st.metric("Cells per Pixel", f"{scale:.1f}")
                                
                                # Memory warning for very large images
                                if total_pixels > 200_000_000:
                                    st.info("💡 **Viewing Tips for Ultra-High Resolution Images:**\n"
                                           "- Use 64-bit image viewers (IrfanView 64-bit, GIMP, Photoshop)\n"
                                           "- Ensure you have sufficient RAM (8+ GB recommended)\n"
                                           "- Some web browsers may struggle with images this large\n"
                                           "- Consider using image pyramiding software for smooth zooming")
                
                # Data export option
                with st.expander("📊 Export Raw Data"):
                    if st.button("Export Triangle as CSV"):
                        csv_lines = []
                        for row in triangle:
                            csv_lines.append(",".join(map(str, row)))
                        csv_content = "\n".join(csv_lines)
                        
                        st.download_button(
                            "⬇️ Download Triangle Data",
                            data=csv_content,
                            file_name=f"triangle_data_{sequence_name}_{max_terms}.csv",
                            mime="text/csv"
                        )

if __name__ == "__main__":
    main()
