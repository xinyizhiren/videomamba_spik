from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(r"D:\code\PYTHON\video_sm\outputs\monthly_report")
PNG_OUT = OUT_DIR / "ANN_VideoMamba_model_architecture.png"
SVG_OUT = OUT_DIR / "ANN_VideoMamba_model_architecture.svg"

W, H = 2200, 1200


class C:
    bg = (255, 255, 255)
    ink = (20, 27, 39)
    muted = (91, 106, 128)
    line = (214, 223, 235)
    soft = (247, 249, 252)
    blue = (37, 99, 235)
    cyan = (8, 145, 178)
    violet = (124, 58, 237)
    green = (22, 163, 74)
    amber = (217, 119, 6)
    red = (220, 38, 38)
    dark = (30, 41, 59)


FONT_REG = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


def text_center(draw, xy, text, size, color, bold=False):
    f = font(size, bold)
    x1, y1, x2, y2 = xy
    box = draw.multiline_textbbox((0, 0), text, font=f, spacing=8, align="center")
    tw = box[2] - box[0]
    th = box[3] - box[1]
    draw.multiline_text(
        ((x1 + x2 - tw) / 2, (y1 + y2 - th) / 2 - 2),
        text,
        font=f,
        fill=color,
        spacing=8,
        align="center",
    )


def text_left(draw, xy, text, size, color, bold=False, spacing=8):
    draw.multiline_text(xy, text, font=font(size, bold), fill=color, spacing=spacing)


def rounded(draw, xy, fill, outline, radius=26, width=4):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, start, end, color=C.line):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=6)
    draw.polygon([(x2, y2), (x2 - 18, y2 - 11), (x2 - 18, y2 + 11)], fill=color)


def box(draw, x, y, w, h, title, body, color):
    rounded(draw, (x, y, x + w, y + h), C.bg, color)
    text_center(draw, (x + 18, y + 22, x + w - 18, y + 62), title, 29, color, True)
    text_center(draw, (x + 20, y + 80, x + w - 20, y + h - 18), body, 22, C.muted)


def note(draw, x, y, w, h, title, body, color):
    rounded(draw, (x, y, x + w, y + h), C.bg, C.line, radius=22, width=3)
    draw.rectangle((x, y, x + 9, y + h), fill=color)
    text_left(draw, (x + 32, y + 24), title, 27, C.ink, True)
    text_left(draw, (x + 32, y + 72), body, 22, C.muted, spacing=7)


def build_png():
    im = Image.new("RGB", (W, H), C.bg)
    d = ImageDraw.Draw(im)

    d.rounded_rectangle((90, 70, 330, 120), radius=16, fill=C.dark)
    text_center(d, (90, 70, 330, 120), "ANN baseline", 25, (255, 255, 255), True)

    text_left(d, (90, 170), "原始 ANN VideoMamba 模型结构", 64, C.ink, True)
    text_left(
        d,
        (90, 265),
        "clean VideoMamba small：全浮点激活，无 LIF / spike 层，无 SNN 时间步重复；作为后续 SNN 训练和蒸馏的 teacher。",
        28,
        C.muted,
    )

    y = 460
    h = 160
    specs = [
        (90, 250, "Input video", "B x 3 x 16 x 224 x 224", C.blue),
        (405, 270, "PatchEmbed", "Conv3D patchify\n384 x 16 x 14 x 14", C.cyan),
        (740, 290, "Tokens + PE", "3136 patches + CLS\nB x 3137 x 384", C.violet),
        (1100, 330, "24 x VideoMamba Blocks", "Residual + LayerNorm\n+ Mamba mixer", C.green),
        (1495, 250, "Final Norm", "residual add\nLayerNorm", C.amber),
        (1810, 270, "Classifier Head", "MeanPool\nLinear 384 -> 12", C.red),
    ]
    for i, (x, w, title, body, color) in enumerate(specs):
        box(d, x, y, w, h, title, body, color)
        if i < len(specs) - 1:
            nx = specs[i + 1][0]
            arrow(d, (x + w + 22, y + h // 2), (nx - 24, y + h // 2))

    note(
        d,
        90,
        760,
        590,
        170,
        "双视角输入与融合",
        "训练时 view1 / view2 共享同一个 ANN 权重。\n两个视角分别 forward，logits 平均后用于分类。",
        C.blue,
    )
    note(
        d,
        805,
        760,
        590,
        170,
        "Block 内部保持浮点",
        "LayerNorm、Mamba mixer、residual 分支均保持连续激活。\n没有膜电位状态，也没有二值 spike 输出。",
        C.green,
    )
    note(
        d,
        1520,
        760,
        590,
        170,
        "SNN 实验的参照系",
        "后续 SNN 从 ANN best.pth 加载权重。\nclean ANN 同时作为 distillation teacher。",
        C.cyan,
    )

    d.line((90, 1070, 2110, 1070), fill=C.line, width=3)
    text_left(d, (90, 1095), "VideoMamba ANN baseline architecture | f16 x 224 | 24 blocks | 12 classes", 22, (145, 155, 171))

    im.save(PNG_OUT, quality=95)


def svg_rect(x, y, w, h, fill, stroke, rx=18, sw=3):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def svg_text(x, y, text, size, color, weight="400", anchor="start"):
    safe = escape(text)
    return f'<text x="{x}" y="{y}" font-family="Microsoft YaHei, Arial" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{safe}</text>'


def build_svg():
    def rgb(c):
        return f"rgb({c[0]},{c[1]},{c[2]})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        svg_rect(90, 70, 240, 50, rgb(C.dark), rgb(C.dark), 16, 0),
        svg_text(210, 104, "ANN baseline", 25, "white", "700", "middle"),
        svg_text(90, 220, "原始 ANN VideoMamba 模型结构", 64, rgb(C.ink), "700"),
        svg_text(90, 305, "clean VideoMamba small：全浮点激活，无 LIF / spike 层，无 SNN 时间步重复；作为后续 SNN 训练和蒸馏的 teacher。", 28, rgb(C.muted)),
    ]
    specs = [
        (90, 250, "Input video", "B x 3 x 16 x 224 x 224", C.blue),
        (405, 270, "PatchEmbed", "Conv3D patchify / 384 x 16 x 14 x 14", C.cyan),
        (740, 290, "Tokens + PE", "3136 patches + CLS / B x 3137 x 384", C.violet),
        (1100, 330, "24 x VideoMamba Blocks", "Residual + LayerNorm + Mamba mixer", C.green),
        (1495, 250, "Final Norm", "residual add / LayerNorm", C.amber),
        (1810, 270, "Classifier Head", "MeanPool / Linear 384 -> 12", C.red),
    ]
    y, h = 460, 160
    for i, (x, w, title, body, color) in enumerate(specs):
        parts.append(svg_rect(x, y, w, h, "white", rgb(color), 26, 4))
        parts.append(svg_text(x + w / 2, y + 55, title, 29, rgb(color), "700", "middle"))
        parts.append(svg_text(x + w / 2, y + 108, body, 22, rgb(C.muted), "400", "middle"))
        if i < len(specs) - 1:
            nx = specs[i + 1][0]
            parts.append(f'<line x1="{x+w+22}" y1="{y+h/2}" x2="{nx-32}" y2="{y+h/2}" stroke="{rgb(C.line)}" stroke-width="6"/>')
            parts.append(f'<polygon points="{nx-24},{y+h/2} {nx-44},{y+h/2-12} {nx-44},{y+h/2+12}" fill="{rgb(C.line)}"/>')

    notes = [
        (90, "双视角输入与融合", "训练时 view1 / view2 共享同一个 ANN 权重；两个视角 logits 平均后用于分类。", C.blue),
        (805, "Block 内部保持浮点", "LayerNorm、Mamba mixer、residual 分支均保持连续激活；没有膜电位状态或 spike 输出。", C.green),
        (1520, "SNN 实验的参照系", "后续 SNN 从 ANN best.pth 加载权重；clean ANN 同时作为 distillation teacher。", C.cyan),
    ]
    for x, title, body, color in notes:
        parts.append(svg_rect(x, 760, 590, 170, "white", rgb(C.line), 22, 3))
        parts.append(f'<rect x="{x}" y="760" width="9" height="170" fill="{rgb(color)}"/>')
        parts.append(svg_text(x + 32, 810, title, 27, rgb(C.ink), "700"))
        parts.append(svg_text(x + 32, 865, body, 22, rgb(C.muted)))

    parts.append(f'<line x1="90" y1="1070" x2="2110" y2="1070" stroke="{rgb(C.line)}" stroke-width="3"/>')
    parts.append(svg_text(90, 1120, "VideoMamba ANN baseline architecture | f16 x 224 | 24 blocks | 12 classes", 22, "rgb(145,155,171)"))
    parts.append("</svg>")
    SVG_OUT.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_png()
    build_svg()
    print(PNG_OUT)
    print(SVG_OUT)
