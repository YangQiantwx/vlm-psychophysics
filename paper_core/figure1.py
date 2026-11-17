"""Build the paper's five-panel figure and its editable draw.io source."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image
import pymupdf


HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1200, 716
TOP_OFFSET = 61
SCALE = 495 / WIDTH
INK, MUTED, RULE = "#192939", "#47586a", "#d2dae2"
WHITE = "#ffffff"
FREQUENCY_CROP = (160, 32, 608, 480)
PAIR_CROP = (124, 112, 644, 424)
PALETTE = (
    ("#215e90", "#eaf3f9", "#c2d9ea"),
    ("#9c451e", "#fcf0e8", "#ebcbb9"),
    ("#216453", "#eaf5ef", "#c1dbd0"),
    ("#62487e", "#f1ecf6", "#d6c8e3"),
    ("#973d58", "#f9edf0", "#eac8d5"),
)


class Figure:
    def __init__(self, arrays, receipt):
        self.arrays = arrays
        self.receipt = receipt
        self.expected = {
            key: digest
            for group in ("examples", "conditions")
            for example in receipt[group].values()
            for key, digest in example["frame_pixel_sha256"].items()
        }
        self.elements = []
        self.images = {}
        self.sources = set()

    def text(self, x, y, content, size=15, color=INK, align="left", face="sans"):
        self.elements.append(("text", (x, y - TOP_OFFSET, content, size, color, align, face)))

    def box(self, x, y, w, h, fill=None, stroke=None, width=1.2, rounded=False):
        self.elements.append(("box", (x, y - TOP_OFFSET, w, h, fill, stroke, width, rounded)))

    def line(self, x0, y0, x1, y1, color=RULE, width=1.2):
        self.elements.append(("line", (x0, y0 - TOP_OFFSET, x1, y1 - TOP_OFFSET, color, width)))

    def next_frame(self, x, y):
        self.line(x, y, x + 17, y, MUTED, 1.6)
        self.line(x + 11, y - 5, x + 17, y, MUTED, 1.6)
        self.line(x + 11, y + 5, x + 17, y, MUTED, 1.6)

    def ring(self, cx, cy, radius, color):
        self.elements.append(("ring", (cx - radius, cy - radius - TOP_OFFSET, 2 * radius, 2 * radius, color)))

    def image(self, key, x, y, w, crop=None):
        token = (key, crop)
        image = self.arrays[key]
        if token not in self.images:
            if image.shape != (512, 768, 3) or image.dtype != np.uint8:
                raise ValueError(f"Unexpected source geometry for {key}")
            if hashlib.sha256(image.tobytes()).hexdigest() != self.expected[key]:
                raise ValueError(f"Source RGB digest mismatch for {key}")
            source = Image.fromarray(image)
            self.images[token] = source.crop(crop) if crop else source
            self.sources.add(key)
        pixel_size = self.images[token].size
        self.elements.append(("image", (token, x, y - TOP_OFFSET, w, w * pixel_size[1] / pixel_size[0])))

    def card(self, x, y, w, h, theme, title, question):
        accent, tint, stroke = PALETTE[theme]
        self.box(x, y, w, h, WHITE, stroke, 1.45, rounded=True)
        self.box(x, y, w, 44, tint, rounded=True)
        self.box(x, y + 25, w, 18, tint)
        self.line(x, y + 43, x + w, y + 43, stroke)
        self.text(x + 16, y + 30, title, 26, accent, face="serifbold")
        self.box(x + 9, y + h - 34, w - 18, 27, tint, rounded=True)
        self.text(x + w / 2, y + h - 15, question, 17, accent, "center", "bold")


def compose(f):
    x = 17
    f.card(x, 78, 378, 319, 0, "T1  Stripe direction", "Which way do the stripes drift?")
    f.text(x + 17, 143, "Stripe spacing", 16, INK, face="bold")
    f.text(x + 202, 143, "Contrast", 16, INK, face="bold")
    f.line(x + 188, 133, x + 188, 348)
    for shift, key, label in (
        (17, "T1_f2_m1", "2 cycles"),
        (98, "T1_f8_m1", "8 cycles"),
        (202, "classic_T1_f3", "m = .5"),
        (283, "T1_f8_m1", "m = 1"),
    ):
        f.image(key, x + shift, 153, 67, FREQUENCY_CROP)
        f.text(x + shift + 33.5, 238, label, 14.5, MUTED, "center")
    for shift, scene in ((17, "Classic"), (202, "Natural")):
        f.text(x + shift, 260, scene, 15.5, INK, face="bold")
        for i, frame in enumerate((0, 3)):
            f.image(f"{scene.lower()}_T1_f{frame}", x + shift + i * 81, 267, 67, FREQUENCY_CROP)
        f.next_frame(x + shift + 64, 301)
        f.text(x + shift + 74, 352, "Frames 0 and 3 shown", 14, MUTED, "center")

    x = 411
    f.card(x, 78, 378, 319, 1, "T2  Brightening", "Which side brightened?")
    f.text(x + 135, 143, "Before", 16, MUTED, "center", "bold")
    f.text(x + 289, 143, "Event midpoint", 16, MUTED, "center", "bold")
    for y, scene in ((152, "Classic"), (248, "Natural")):
        f.text(x + 11, y + 44, scene, 14.5, INK, face="bold")
        for i, frame in enumerate((132, 148)):
            f.image(f"{scene.lower()}_T2_f{frame}", x + 70 + 154 * i, y, 130, PAIR_CROP)
        f.next_frame(x + 203, y + 39)
    f.text(x + 189, 351, "Luminance increment  ΔY = 0.35", 15.5, MUTED, "center")

    x = 805
    f.card(x, 78, 378, 319, 2, "T3  Orientation", "Which object is closer to vertical?")
    f.text(x + 18, 143, "Classic", 16, INK, face="bold")
    for shift, key, angle in ((18, "T3_small", "2°"), (197, "classic_T3_f118", "32°")):
        f.image(key, x + shift, 153, 162, PAIR_CROP)
        f.text(x + shift + 81, 266, angle, 17, MUTED, "center", "bold")
    f.text(x + 18, 289, "Natural", 16, INK, face="bold")
    f.image("natural_T3_f118", x + 18, 296, 104, PAIR_CROP)
    f.text(x + 137, 317, "32° example", 16, INK, face="bold")
    f.text(x + 137, 340, "Frames at the event midpoint", 14, MUTED)

    x = 17
    f.card(x, 412, 573, 348, 3, "T4  Group motion", "Which way does the group move?")
    f.text(x + 17, 477, "Classic · 25% (2 of 8 move right)", 16, INK, face="bold")
    f.text(x + 298, 477, "Classic · 100% (8 of 8 move right)", 16, INK, face="bold")
    f.line(x + 285, 468, x + 285, 590)
    conditions = f.receipt["conditions"]
    for col, coherence in enumerate((25, 100)):
        trial = conditions[f"T4_n8_c{coherence}"]
        expected = trial["parameters"]
        assert expected["N"] == 8 and expected["coherence"] == coherence / 100
        assert expected["signal_count"] == 8 * coherence // 100 and trial["truth"] == "RIGHT"
        for i, (key, frame) in enumerate(zip(trial["static_keys"], (0, 179), strict=True)):
            left = x + 17 + 280 * col + 138 * i
            f.image(key, left, 490, 116)
            f.text(left + 58, 585, f"Frame {frame}", 14.5, MUTED, "center")
            if coherence == 25:
                for dot in expected["signal_point_ids"]:
                    j = expected["point_ids"].index(dot)
                    velocity = np.array(expected["velocity_xy_px_s"][j])
                    assert np.array_equal(velocity, [32, 0])
                    center = np.array(expected["positions_xy_px"][j]) + velocity * frame / 60
                    cx, cy = center * (116 / 768)
                    f.ring(left + cx, 490 + cy, 5.8, "#ffda67")
        f.next_frame(x + 135 + 280 * col, 529)

    item = conditions["T4_natural_bright"]
    assert item["parameters"]["N"] == 8 and item["parameters"]["coherence"] == 0.125
    assert item["truth"] == "RIGHT"
    f.text(x + 17, 615, "Natural · 1 of 8 moves right", 16, INK, face="bold")
    for i, (key, frame, position) in enumerate(zip(
        item["static_keys"], (0, 179), item["signal_positions_xy_px"], strict=True,
    )):
        left = x + 17 + 138 * i
        f.image(key, left, 624, 116)
        f.text(left + 58, 718, f"Frame {frame}", 14, MUTED, "center")
        cx, cy = np.array(position) * (116 / 768)
        f.ring(left + cx, 624 + cy, 7, INK)
        f.ring(left + cx, 624 + cy, 5.8, "#ffda67")
    f.next_frame(x + 135, 663)
    f.text(x + 298, 649, "All eight objects move.", 17, INK, face="bold")
    f.text(x + 298, 676, "Coherence is the fraction moving", 15.5, MUTED)
    f.text(x + 298, 698, "in the same direction.", 15.5, MUTED)

    x = 610
    f.card(x, 412, 573, 348, 4, "T5  Target matching",
           "Does any candidate have the same orientation as the reference?")
    for i, label in enumerate(("Reference", "Event 1", "Event 2")):
        f.text(x + 139 + i * 158, 477, label, 16, MUTED, "center", "bold")
    f.line(x + 216, 484, x + 216, 696)
    for y, scene in ((490, "Classic"), (600, "Natural")):
        item = f.receipt["examples"][f"{scene.lower()}_T5"]
        assert [event["angle_deg"] for event in item["events"]] == [0, 90]
        assert item["truth"] == "YES"
        f.text(x + 12, y + 47, scene, 15, INK, face="bold")
        for i, key in enumerate(item["static_keys"]):
            left = x + 77 + i * 158
            f.image(key, left, y, 124)
            if i == 1:
                f.box(left - 2, y - 2, 128, 124 * 2 / 3 + 4, stroke="#a84c67", width=2.6)
    f.text(x + 287, 712, "One or two candidates · outlined candidate matches", 15, MUTED, "center")


def rgb(color):
    return tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)) if color else None


def encoded(image, fmt="PNG"):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **({"quality": 82, "optimize": True} if fmt == "JPEG" else {}))
    return buffer.getvalue()


def write_pdf(f, path):
    families = {"sans": "texgyreheros-regular", "bold": "texgyreheros-bold", "serifbold": "texgyretermes-bold"}
    doc = pymupdf.open()
    page = doc.new_page(width=WIDTH * SCALE, height=HEIGHT * SCALE)
    aliases = {}
    for i, (name, font) in enumerate(families.items()):
        key = f"F{i}"
        fontfile = subprocess.check_output(
            ["/Library/TeX/texbin/kpsewhich", font + ".otf"], text=True,
        ).strip()
        page.insert_font(fontname=key, fontfile=fontfile)
        aliases[name] = key, pymupdf.Font(fontfile=fontfile)
    def rect(x, y, w, h):
        return pymupdf.Rect(x * SCALE, y * SCALE, (x + w) * SCALE, (y + h) * SCALE)
    streams = {}
    for kind, payload in f.elements:
        if kind == "text":
            x, y, value, size, color, align, face = payload
            alias, font = aliases[face]
            size *= SCALE
            offset = font.text_length(value, fontsize=size) / 2 if align == "center" else 0
            page.insert_text((x * SCALE - offset, y * SCALE), value, fontsize=size,
                             fontname=alias, color=rgb(color))
        elif kind == "box":
            x, y, w, h, fill, stroke, width, rounded = payload
            radius = (8 / w, 8 / h) if rounded else None
            page.draw_rect(rect(x, y, w, h), fill=rgb(fill), color=rgb(stroke),
                           width=width * SCALE, radius=radius)
        elif kind == "line":
            x0, y0, x1, y1, color, width = payload
            page.draw_line((x0 * SCALE, y0 * SCALE), (x1 * SCALE, y1 * SCALE),
                           color=rgb(color), width=width * SCALE)
        elif kind == "ring":
            x, y, w, h, color = payload
            page.draw_oval(rect(x, y, w, h), color=rgb(color), width=1.6 * SCALE)
        else:
            token, x, y, w, h = payload
            R = rect(x, y, w, h)
            if token not in streams:
                streams[token] = page.insert_image(R, stream=encoded(f.images[token]))
            else:
                page.insert_image(R, xref=streams[token])
    doc.set_metadata({"title": "Video psychophysics", "subject": "Five perceptual tasks"})
    doc.save(path, garbage=4, deflate=True)
    doc.close()


def write_drawio(f, path, preview=False):
    model = ET.Element("mxGraphModel", {
        "page": "1", "pageWidth": str(WIDTH), "pageHeight": str(HEIGHT),
        "grid": "0", "pageScale": "1", "background": WHITE, "adaptiveColors": "none",
    })
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
    metrics = {face: pymupdf.Font(name) for face, name in (("sans", "helv"), ("bold", "hebo"), ("serifbold", "tibo"))}
    def vertex(i, value, style, x, y, w, h):
        node = ET.SubElement(root, "mxCell", {"id": str(i), "parent": "1", "vertex": "1", "style": style, "value": value})
        ET.SubElement(node, "mxGeometry", {"as": "geometry", **dict(zip(
            ("x", "y", "width", "height"), map(lambda n: f"{n:.4f}", (x, y, w, h)), strict=True,
        ))})
    for i, (kind, args) in enumerate(f.elements, 2):
        if kind == "text":
            x, y, value, size, color, align, face = args
            family = "Times New Roman" if face == "serifbold" else "Arial"
            bold = int(face != "sans")
            w = metrics[face].text_length(value, fontsize=size) + 12
            vertex(i, value, f"text;html=1;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;"
                   f"spacing=0;fontSize={size};fontColor={color};fontFamily={family};fontStyle={bold};",
                   x - w / 2 if align == "center" else x, y - .87 * size, w, 1.05 * size)
        elif kind == "box":
            x, y, w, h, fill, stroke, width, rounded = args
            vertex(i, "", f"rounded={int(rounded)};absoluteArcSize=1;arcSize=16;html=1;fillColor={fill or 'none'};"
                   f"strokeColor={stroke or 'none'};strokeWidth={width};", x, y, w, h)
        elif kind == "line":
            x0, y0, x1, y1, color, width = args
            edge = ET.SubElement(root, "mxCell", {"id": str(i), "parent": "1", "edge": "1",
                "style": f"endArrow=none;html=1;strokeColor={color};strokeWidth={width};", "value": ""})
            geom = ET.SubElement(edge, "mxGeometry", {"relative": "1", "as": "geometry"})
            ET.SubElement(geom, "mxPoint", {"x": str(x0), "y": str(y0), "as": "sourcePoint"})
            ET.SubElement(geom, "mxPoint", {"x": str(x1), "y": str(y1), "as": "targetPoint"})
        elif kind == "ring":
            x, y, w, h, color = args
            vertex(i, "", f"ellipse;html=1;fillColor=none;strokeColor={color};strokeWidth=1.6;", x, y, w, h)
        else:
            token, x, y, w, h = args
            asset = f.images[token]
            if preview:
                asset = asset.copy()
                asset.thumbnail((int(w), int(h)), Image.Resampling.LANCZOS)
            mime, fmt = ("jpeg", "JPEG") if preview else ("png", "PNG")
            data = base64.b64encode(encoded(asset, fmt)).decode("ascii")
            vertex(i, "", f"shape=image;html=1;imageAspect=0;image=data:image/{mime},{data};", x, y, w, h)
    if preview:
        ET.ElementTree(model).write(path, encoding="utf-8")
    else:
        wrapper = ET.Element("mxfile", {"host": "app.diagrams.net"})
        ET.SubElement(wrapper, "diagram", {"name": "Figure 1", "id": "figure1"}).append(model)
        ET.ElementTree(wrapper).write(path, encoding="utf-8")


def write_sources(f, path):
    with ZipFile(path, "w", ZIP_DEFLATED) as bundle:
        for key in sorted(f.sources):
            bundle.writestr(f"Stimulus frames/{key}.png", encoded(Image.fromarray(f.arrays[key])))
        bundle.writestr("About the images.txt", "These are the uncropped 768 × 512 source frames used in Figure 1. "
                        "The figure crops some panels to show the relevant objects. Rings in T4 and outlines in T5 "
                        "are separate annotations in the draw.io file.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "task_overview")
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--preview-xml", type=Path)
    parser.add_argument("--source-zip", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.inputs.read_text())
    with np.load(args.frames) as cache:
        f = Figure(cache, receipt)
        compose(f)
        write_pdf(f, args.output.with_suffix(".pdf"))
        write_drawio(f, args.output.with_suffix(".drawio"))
        if args.preview_xml:
            write_drawio(f, args.preview_xml, preview=True)
        if args.source_zip:
            write_sources(f, args.source_zip)
        print(f"Built Figure 1 from {len(f.sources)} verified source frames.")


if __name__ == "__main__":
    main()
