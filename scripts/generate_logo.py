import math
import os

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "quasar_logo.svg")

W, H = 800, 800
CX, CY = 360, 400
R = 210

def arc_path(cx, cy, r, start_deg, end_deg):
    s = math.radians(start_deg)
    e = math.radians(end_deg)
    sx = cx + r * math.cos(s)
    sy = cy + r * math.sin(s)
    ex = cx + r * math.cos(e)
    ey = cy + r * math.sin(e)
    diff = (end_deg - start_deg) % 360
    large = 1 if diff > 180 else 0
    return f"M {sx:.2f},{sy:.2f} A {r},{r} 0 {large},1 {ex:.2f},{ey:.2f}"

def tube_strand(cx, cy, r, s_deg, e_deg, base_w, offset_hl):
    # Generates layered paths to simulate a 3D highly polished glass/metallic tube
    paths = []
    # 1. Base shadow / ambient occlusion (darkest outer rim)
    d = arc_path(cx, cy, r, s_deg, e_deg)
    paths.append(f'<path d="{d}" fill="none" stroke="#05021F" stroke-width="{base_w + 3}" stroke-linecap="round" />')
    
    # 2. Main color body (deep blue)
    paths.append(f'<path d="{d}" fill="none" stroke="url(#tube-main)" stroke-width="{base_w}" stroke-linecap="round" />')
    
    # 3. Ambient inner glow (vibrant blue)
    paths.append(f'<path d="{d}" fill="none" stroke="url(#tube-glow)" stroke-width="{base_w * 0.6}" stroke-linecap="round" />')
    
    # 4. Specular highlight (cyan/white, shifted inward for 3D bevel effect)
    d_hl = arc_path(cx, cy, r - offset_hl, s_deg + 1, e_deg - 1)
    paths.append(f'<path d="{d_hl}" fill="none" stroke="url(#tube-hl)" stroke-width="{base_w * 0.25}" stroke-linecap="round" />')
    
    # 5. Core bright hot spot (very thin, pure white)
    d_hot = arc_path(cx, cy, r - offset_hl - 0.5, s_deg + 2, e_deg - 2)
    paths.append(f'<path d="{d_hot}" fill="none" stroke="#FFFFFF" stroke-width="{max(1, base_w * 0.08)}" stroke-linecap="round" opacity="0.9" />')
    
    return "\\n".join(paths)

def generate():
    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">')
    svg.append('<defs>')
    
    # Gradients for the 3D tubes
    svg.append('''
    <!-- Main deep blue for the tube -->
    <linearGradient id="tube-main" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#120A78" />
        <stop offset="50%" stop-color="#1C10A3" />
        <stop offset="100%" stop-color="#0E0652" />
    </linearGradient>
    
    <!-- Vibrant blue glow inside the tube -->
    <linearGradient id="tube-glow" x1="0%" y1="100%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#2D1ED1" />
        <stop offset="50%" stop-color="#3F2BED" />
        <stop offset="100%" stop-color="#190F8A" />
    </linearGradient>
    
    <!-- Specular highlight gradient (cyan/white) -->
    <linearGradient id="tube-hl" x1="0%" y1="100%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#80A4FF" />
        <stop offset="50%" stop-color="#C2D6FF" />
        <stop offset="100%" stop-color="#5C85E6" />
    </linearGradient>
    
    <!-- Tail main gradient -->
    <linearGradient id="tail-main" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#1C10A3" />
        <stop offset="100%" stop-color="#0E0652" />
    </linearGradient>
    
    <!-- General Glow Filters -->
    <filter id="glow-heavy" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="8" result="blur" />
        <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
        </feMerge>
    </filter>
    <filter id="glow-light" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="3" result="blur" />
        <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
        </feMerge>
    </filter>
    ''')
    
    # 4. Starburst definitions (Lens Flare effect)
    flare_ang = 315
    # The starburst is positioned on the main strand
    flare_x = CX + R * math.cos(math.radians(flare_ang))
    flare_y = CY + R * math.sin(math.radians(flare_ang))
    
    # Define the spikes of the lens flare
    spikes_config = [
        (0, 160, 4), (90, 160, 4), (180, 160, 4), (270, 160, 4), # Primary long spikes
        (45, 100, 2), (135, 100, 2), (225, 100, 2), (315, 100, 2), # Secondary diagonal spikes
        (22.5, 60, 1), (67.5, 60, 1), (112.5, 60, 1), (157.5, 60, 1), # Tertiary minor spikes
        (202.5, 60, 1), (247.5, 60, 1), (292.5, 60, 1), (337.5, 60, 1)
    ]
    
    spike_polys = []
    for ang, length, width in spikes_config:
        ex = flare_x + length * math.cos(math.radians(ang))
        ey = flare_y + length * math.sin(math.radians(ang))
        # Perpendicular points to create a sharp diamond polygon
        px = width * math.cos(math.radians(ang + 90))
        py = width * math.sin(math.radians(ang + 90))
        pts = [(flare_x + px, flare_y + py), (ex, ey), (flare_x - px, flare_y - py), (flare_x, flare_y)]
        p_str = " ".join([f"{x:.2f},{y:.2f}" for x,y in pts])
        
        gid = f"spike_grad_{str(ang).replace('.', '_')}"
        # Fading gradient so the spikes blend into the background
        svg.append(f'''<linearGradient id="{gid}" x1="{flare_x}" y1="{flare_y}" x2="{ex}" y2="{ey}" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stop-color="#FFFFFF" stop-opacity="1" />
            <stop offset="20%" stop-color="#E6EEFF" stop-opacity="0.8" />
            <stop offset="100%" stop-color="#A3BFFF" stop-opacity="0" />
        </linearGradient>''')
        spike_polys.append(f'<polygon points="{p_str}" fill="url(#{gid})" />')
        
    svg.append('</defs>')

    # 1. Circle Strands (Multi-strand tubular 3D effect)
    # The gap is at the bottom right. We draw clockwise.
    # Outer thin strand
    svg.append(tube_strand(CX, CY, R + 18, 90, 400, base_w=9, offset_hl=1.5))
    # Main thick center strand
    svg.append(tube_strand(CX, CY, R, 75, 415, base_w=16, offset_hl=2.5))
    # Inner medium strand
    svg.append(tube_strand(CX, CY, R - 16, 65, 430, base_w=11, offset_hl=1.5))

    # 2. Tail
    # The tail is a sharp, elegant needle crossing from inside to outside
    sx, sy = CX - 30, CY + 60
    ex, ey = CX + 180, CY + 230
    mx, my = (sx + ex) / 2, (sy + ey) / 2
    dx, dy = ex - sx, ey - sy
    length = math.hypot(dx, dy)
    px, py = -dy / length, dx / length
    w = 10
    
    # Main tail body (dark blue base)
    svg.append(f'<path d="M {sx},{sy} Q {mx+px*w},{my+py*w} {ex},{ey} Q {mx-px*w},{my-py*w} {sx},{sy}" fill="url(#tail-main)" />')
    
    # Tail inner vibrant glow
    w_glow = 6
    svg.append(f'<path d="M {sx},{sy} Q {mx+px*w_glow},{my+py*w_glow} {ex},{ey} Q {mx-px*w_glow},{my-py*w_glow} {sx},{sy}" fill="url(#tube-glow)" />')
    
    # Tail Specular Highlight (shifted to the upper edge for 3D bevel)
    hx = px * 3
    hy = py * 3
    w_hl = 2
    hl_pts = [(sx+hx, sy+hy), (mx+px*w_hl+hx, my+py*w_hl+hy), (ex, ey), (mx-px*w_hl+hx, my-py*w_hl+hy)]
    hl_str = " ".join([f"{x:.2f},{y:.2f}" for x,y in hl_pts])
    svg.append(f'<polygon points="{hl_str}" fill="url(#tube-hl)" />')

    # 3. Wisps
    # Elegant, crisp sweeping curves on the right side of the starburst
    wisps = [
        (25, 270, 390, 0.6, 2.5),
        (45, 260, 400, 0.4, 1.5),
        (65, 250, 380, 0.2, 1),
    ]
    for r_off, s, e, op, w_wisp in wisps:
        d = arc_path(CX, CY, R + r_off, s, e)
        # The wisps also have a slight highlight for realism
        svg.append(f'<path d="{d}" fill="none" stroke="#2D1ED1" stroke-width="{w_wisp+1.5}" stroke-linecap="round" opacity="{op}" />')
        svg.append(f'<path d="{d}" fill="none" stroke="#C2D6FF" stroke-width="{w_wisp}" stroke-linecap="round" opacity="{op}" />')

    # 4. Starburst Compositing
    # Underlying deep blue ambient glow behind the star
    svg.append(f'<circle cx="{flare_x}" cy="{flare_y}" r="60" fill="#2D1ED1" filter="url(#glow-heavy)" opacity="0.6" />')
    
    # Sharp lens flare spikes
    for poly in spike_polys:
        svg.append(poly)
        
    # Core bright hot spot
    svg.append(f'<circle cx="{flare_x}" cy="{flare_y}" r="12" fill="#FFFFFF" filter="url(#glow-light)" />')
    svg.append(f'<circle cx="{flare_x}" cy="{flare_y}" r="5" fill="#FFFFFF" />')

    svg.append('</svg>')
    
    with open(OUTPUT, "w") as f:
        f.write("\\n".join(svg))
    print(f"Generated {OUTPUT}")

if __name__ == "__main__":
    generate()
