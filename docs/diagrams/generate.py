#!/usr/bin/env python3
"""Generate the README architecture diagrams as themed SVG pairs.

Each diagram is described once (nodes + edges); the script emits a light and a
dark variant per diagram, tuned to GitHub's page backgrounds. The README embeds
them with <picture> so GitHub serves the right theme.

Node categories carry consistent colors across every diagram:
  RAM stores  - blue   (#2a78d6 / #3987e5)
  disk/warm   - aqua   (#1baf7a / #199e70)
  external    - orange (#eb6834 / #d95926)
  processing  - neutral surface + hairline
Labels are always ink-colored; color marks identity, text carries meaning.

Run:  python3 docs/diagrams/generate.py
"""

import os

THEMES = {
    "light": {
        "ink": "#1f2328",
        "muted": "#59636e",
        "hairline": "#d1d9e0",
        "process_fill": "#f6f8fa",
        "arrow": "#59636e",
        "ram": "#2a78d6",
        "disk": "#1baf7a",
        "ext": "#eb6834",
        "tint": 0.08,
    },
    "dark": {
        "ink": "#e6edf3",
        "muted": "#9198a1",
        "hairline": "#3d444d",
        "process_fill": "#161b22",
        "arrow": "#9198a1",
        "ram": "#3987e5",
        "disk": "#199e70",
        "ext": "#d95926",
        "tint": 0.14,
    },
}

FONT = "system-ui,-apple-system,'Segoe UI',sans-serif"


class D:
    def __init__(self, name, width, height):
        self.name = name
        self.w = width
        self.h = height
        self.items = []  # callables(theme) -> svg fragment

    def _register(self, fn):
        self.items.append(fn)

    # -- nodes ---------------------------------------------------------------

    def node(self, x, y, w, h, lines, kind="process"):
        """kind: process | ram | disk | ext | actor"""

        def render(t):
            out = []
            if kind == "process":
                fill, stroke, extra = t["process_fill"], t["hairline"], ""
            elif kind == "actor":
                fill, stroke, extra = "none", t["muted"], ""
            else:
                color = t[kind]
                fill, stroke = color, color
                extra = f' fill-opacity="{t["tint"]}"'
            if kind in ("ram", "disk"):
                out.append(_cylinder(x, y, w, h, fill, stroke, t["tint"]))
            else:
                out.append(
                    f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7"'
                    f' fill="{fill}"{extra} stroke="{stroke}" stroke-width="1.5"/>'
                )
            cap = 7 if kind in ("ram", "disk") else 0
            out.append(_label(x + w / 2, y + cap + h / 2, lines, t["ink"]))
            return "".join(out)

        self._register(render)
        return (x, y, w, h)

    def group(self, x, y, w, h, title):
        def render(t):
            return (
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10"'
                f' fill="none" stroke="{t["hairline"]}" stroke-width="1.5"'
                f' stroke-dasharray="6 5"/>'
                f'<text x="{x + 12}" y="{y + 20}" font-family="{FONT}"'
                f' font-size="12.5" font-weight="600" fill="{t["muted"]}">{title}</text>'
            )

        self._register(render)
        return (x, y, w, h)

    # -- edges ---------------------------------------------------------------

    def edge(self, points, label=None, dashed=False, label_at=0.5, label_dy=-7):
        def render(t):
            pts = " ".join(f"{px},{py}" for px, py in points)
            dash = ' stroke-dasharray="5 5"' if dashed else ""
            out = [
                f'<polyline points="{pts}" fill="none" stroke="{t["arrow"]}"'
                f' stroke-width="1.5"{dash} marker-end="url(#arrow)"/>'
            ]
            if label:
                lx, ly = _along(points, label_at)
                out.append(
                    f'<text x="{lx}" y="{ly + label_dy}" text-anchor="middle"'
                    f' font-family="{FONT}" font-size="12" fill="{t["muted"]}"'
                    f' paint-order="stroke">{label}</text>'
                )
            return "".join(out)

        self._register(render)

    # -- output --------------------------------------------------------------

    def svg(self, t):
        body = "".join(item(t) for item in self.items)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}"'
            f' width="{self.w}" role="img" font-family="{FONT}">'
            f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"'
            f' markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 1 L 9 5 L 0 9 z" fill="{t["arrow"]}"/></marker></defs>'
            f"{body}</svg>"
        )


def _cylinder(x, y, w, h, fill, stroke, tint):
    ry = 8
    return (
        f'<path d="M {x} {y + ry} A {w / 2} {ry} 0 0 1 {x + w} {y + ry}'
        f' V {y + h - ry} A {w / 2} {ry} 0 0 1 {x} {y + h - ry} Z"'
        f' fill="{fill}" fill-opacity="{tint}" stroke="{stroke}" stroke-width="1.5"/>'
        f'<path d="M {x} {y + ry} A {w / 2} {ry} 0 0 0 {x + w} {y + ry}"'
        f' fill="none" stroke="{stroke}" stroke-width="1.5"/>'
    )


def _label(cx, cy, lines, ink):
    if isinstance(lines, str):
        lines = [lines]
    lh = 17
    y0 = cy - (len(lines) - 1) * lh / 2
    out = []
    for i, line in enumerate(lines):
        weight = "600" if i == 0 else "400"
        size = "14.5" if i == 0 else "12.5"
        out.append(
            f'<text x="{cx}" y="{y0 + i * lh}" text-anchor="middle"'
            f' dominant-baseline="middle" font-family="{FONT}" font-size="{size}"'
            f' font-weight="{weight}" fill="{ink}">{line}</text>'
        )
    return "".join(out)


def _along(points, frac):
    segs = []
    total = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        d = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        segs.append((d, (x1, y1), (x2, y2)))
        total += d
    target = total * frac
    run = 0.0
    for d, (x1, y1), (x2, y2) in segs:
        if run + d >= target and d > 0:
            f = (target - run) / d
            return (x1 + (x2 - x1) * f, y1 + (y2 - y1) * f)
        run += d
    return points[-1]


def build():
    diagrams = []

    # 1 — capture: Camera -> Keyframe Segmenter -> Hot Buffer
    d = D("01-capture", 660, 110)
    d.node(16, 34, 104, 42, "Camera", kind="actor")
    d.node(240, 34, 180, 42, "Keyframe Segmenter")
    d.node(524, 22, 122, 66, ["Hot Buffer", "(10 min, RAM)"], kind="ram")
    d.edge([(120, 55), (240, 55)], label="RTSP")
    d.edge([(420, 55), (524, 55)])
    diagrams.append(d)

    # 2 — fan-out from the hot buffer
    d = D("02-fanout", 700, 344)
    d.node(16, 148, 122, 66, ["Hot Buffer", "(10 min, RAM)"], kind="ram")
    d.node(548, 24, 122, 42, "Clients", kind="actor")
    d.node(250, 160, 158, 42, "Motion Analyzer")
    d.node(534, 128, 150, 60, ["Motion Store", "(RAM)"], kind="ram")
    d.node(250, 252, 200, 42, "Detection Worker")
    d.node(534, 244, 150, 60, ["Detection Store", "(RAM)"], kind="ram")
    d.node(16, 252, 158, 60, ["Warm Writer", "(storage)"], kind="disk")
    d.edge([(77, 148), (77, 45), (548, 45)], label="HLS", label_at=0.6)
    d.edge([(138, 181), (250, 181)], label="keyframes")
    d.edge([(408, 181), (534, 166)], label_at=0.5)
    d.edge([(329, 202), (329, 252)], label="crop jobs", label_at=0.5, label_dy=4)
    d.edge([(450, 273), (534, 273)])
    d.edge([(250, 195), (95, 235), (95, 252)], label="finished events", label_at=0.35)
    d.edge([(250, 283), (174, 283)], label="post-hoc upgrades", dashed=True, label_dy=16)
    diagrams.append(d)

    # 3 — inside the motion analyzer (no auto-tuning; it was removed)
    d = D("03-analyzer", 700, 252)
    d.node(16, 22, 122, 66, ["Hot Buffer", "(10 min, RAM)"], kind="ram")
    d.node(250, 34, 190, 42, ["Decode 320×240", "(grayscale)"])
    d.node(534, 22, 150, 60, ["Motion Store", "(RAM)"], kind="ram")
    d.group(60, 128, 580, 100, "Motion Analyzer")
    d.node(84, 160, 158, 44, ["Background", "Subtraction (MOG2)"])
    d.node(272, 160, 168, 44, ["Morphological", "Opening"])
    d.node(470, 160, 150, 44, ["Component", "Filtering"])
    d.edge([(138, 55), (250, 55)], label="keyframes")
    d.edge([(345, 76), (345, 128)])
    d.edge([(242, 182), (272, 182)])
    d.edge([(440, 182), (470, 182)])
    d.edge([(545, 160), (545, 120), (609, 120), (609, 82)])
    diagrams.append(d)

    # 4 — object detection path
    d = D("04-detection", 700, 330)
    d.node(16, 22, 122, 66, ["Hot Buffer", "(10 min, RAM)"], kind="ram")
    d.node(16, 122, 150, 60, ["Motion Store", "(RAM)"], kind="ram")
    d.node(250, 60, 178, 44, "Subsample 4 frames")
    d.node(510, 60, 130, 44, "Crop + JPEG")
    d.node(250, 186, 220, 44, ["Detection Worker", "(global, serial)"])
    d.node(560, 186, 110, 44, "Ollama", kind="ext")
    d.node(16, 250, 150, 60, ["Detection Store", "(RAM)"], kind="ram")
    d.edge([(138, 55), (194, 55), (194, 71), (250, 71)], label="motion event", label_at=0.9)
    d.edge([(166, 152), (194, 152), (194, 93), (250, 93)], label="bounding boxes", label_at=0.12, label_dy=24)
    d.edge([(428, 82), (510, 82)])
    d.edge([(575, 104), (575, 145), (360, 145), (360, 186)], label="bounded queue", label_at=0.55)
    d.edge([(470, 208), (560, 208)], label="one in flight", label_dy=-9)
    d.edge([(250, 219), (194, 240), (166, 268)], label_at=0.5)
    d.edge([(360, 230), (360, 264)], label="upgrade event", dashed=True, label_at=0.5, label_dy=4)
    d.node(280, 264, 160, 50, "Warm Writer", kind="disk")
    diagrams.append(d)

    # 5 — event persistence and warm storage
    d = D("05-storage", 700, 344)
    d.node(16, 16, 122, 66, ["Hot Buffer", "(10 min, RAM)"], kind="ram")
    d.node(16, 112, 150, 60, ["Motion Store", "(RAM)"], kind="ram")
    d.node(16, 202, 150, 60, ["Detection Store", "(RAM)"], kind="ram")
    d.node(250, 120, 158, 44, "Motion Analyzer")
    d.node(250, 220, 158, 44, "Warm Writer")
    d.group(452, 88, 232, 240, "Warm storage")
    d.node(468, 120, 92, 54, "Movements", kind="disk")
    d.node(576, 120, 92, 54, "Objects", kind="disk")
    d.node(468, 192, 92, 54, "Metadata", kind="disk")
    d.node(576, 192, 92, 54, "Thumbnails", kind="disk")
    d.node(468, 264, 200, 42, ["disk or stathost"], kind="actor")
    d.node(548, 16, 122, 40, "Clients", kind="actor")
    d.node(268, 300, 122, 36, "Prune", kind="actor")
    d.edge([(138, 49), (329, 49), (329, 120)], label="segments", label_at=0.35)
    d.edge([(166, 142), (250, 142)])
    d.edge([(166, 232), (208, 232), (208, 152), (250, 152)])
    d.edge([(329, 164), (329, 220)], label="event end", label_at=0.5, label_dy=4)
    d.edge([(408, 242), (452, 242)])
    d.edge([(600, 88), (600, 56)], label="HLS", label_at=0.5, label_dy=4)
    d.edge([(390, 318), (430, 318), (430, 300)], dashed=True)
    diagrams.append(d)

    return diagrams


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    for d in build():
        for mode, theme in THEMES.items():
            path = os.path.join(out_dir, f"{d.name}-{mode}.svg")
            with open(path, "w") as f:
                f.write(d.svg(theme))
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
