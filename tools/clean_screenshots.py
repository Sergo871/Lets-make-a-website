#!/usr/bin/env python3
"""Curăță capturi de ecran de pe Instagram pentru a le folosi într-un site.

Taie interfața aplicației (fundal, butoane, săgeți, bara de răspuns) și
păstrează doar conținutul: pagina de meniu sau fotografia postării.

Folosire:
    python3 tools/clean_screenshots.py <folder-capturi> <folder-iesire>

Optiuni:
    --mode card|photo|auto   card  = pagină de meniu pe fundal (implicit: auto)
                             photo = fotografie care umple ecranul
    --width N                lățimea maximă a imaginii salvate (implicit 1400)
    --contact-sheet          salvează și o planșă cu toate rezultatele

Are nevoie de: pip install opencv-python-headless pillow
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Fâșiile pe care Instagram le suprapune peste o fotografie care umple ecranul:
# săgeata de carusel în dreapta, punctele de sub poză.
PHOTO_TRIM_RIGHT = 62
PHOTO_TRIM_BOTTOM = 32


def find_card(image):
    """Găsește dreptunghiul paginii de meniu așezate peste fundalul aplicației.

    Întoarce (x, y, w, h) sau None dacă nu se distinge un card.
    """
    h, w = image.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 30, 90), np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        # Prea mic ca să fie cardul, sau atât de mare încât e tot ecranul.
        if area < 0.3 * w * h or cw > 0.99 * w or ch > 0.99 * h:
            continue
        # Cardul are colțuri drepte: conturul trebuie să umple dreptunghiul său.
        rect = cv2.minAreaRect(contour)
        rect_area = rect[1][0] * rect[1][1]
        if rect_area == 0 or cv2.contourArea(cv2.convexHull(contour)) < 0.85 * rect_area:
            continue
        if best is None or area > best[0]:
            best = (area, (x, y, cw, ch))
    return best[1] if best else None


def clean(path, mode, width):
    """Întoarce (imagine curățată, cum a fost tăiată) pentru o captură."""
    image = cv2.imread(str(path))
    if image is None:
        return None, "nu s-a putut citi"

    box = find_card(image) if mode in ("auto", "card") else None
    if box:
        x, y, cw, ch = box
        # Un pixel în plus, ca să nu rămână linia de contur a cardului.
        cropped = image[y + 2:y + ch - 2, x + 2:x + cw - 2]
        kind = "card"
    else:
        if mode == "card":
            return None, "nu s-a găsit cardul"
        h, w = image.shape[:2]
        cropped = image[0:h - PHOTO_TRIM_BOTTOM, 0:w - PHOTO_TRIM_RIGHT]
        kind = "foto"

    out = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    if out.width < width:
        scale = width / out.width
        out = out.resize((width, round(out.height * scale)), Image.LANCZOS)
        out = out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))
    elif out.width > width:
        out.thumbnail((width, width * 4), Image.LANCZOS)

    if kind == "card":
        # Colțuri rotunjite, ca pagina să arate a meniu tipărit, nu a captură.
        mask = Image.new("L", out.size, 0)
        radius = max(12, round(out.width * 0.025))
        ImageDraw.Draw(mask).rounded_rectangle(
            (0, 0, out.width - 1, out.height - 1), radius=radius, fill=255)
        out.putalpha(mask)
    return out, kind


def contact_sheet(images, path, columns=6, cell=320):
    thumbs = []
    for image in images:
        thumb = image.convert("RGB").copy()
        thumb.thumbnail((cell, cell))
        thumbs.append(thumb)
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * cell), "white")
    for i, thumb in enumerate(thumbs):
        x = (i % columns) * cell + (cell - thumb.width) // 2
        y = (i // columns) * cell + (cell - thumb.height) // 2
        sheet.paste(thumb, (x, y))
    sheet.save(path, quality=85)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="folderul cu capturi")
    parser.add_argument("dest", type=Path, help="unde se salvează rezultatele")
    parser.add_argument("--mode", choices=["auto", "card", "photo"], default="auto")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--contact-sheet", action="store_true")
    args = parser.parse_args()

    files = sorted(p for p in args.source.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not files:
        sys.exit(f"Nicio imagine în {args.source}")

    args.dest.mkdir(parents=True, exist_ok=True)
    done, skipped = [], []
    for path in files:
        image, kind = clean(path, args.mode, args.width)
        if image is None:
            skipped.append((path.name, kind))
            continue
        out_path = args.dest / f"{len(done) + 1:02d}-{path.stem.replace(' ', '-')}.webp"
        image.save(out_path, "WEBP", quality=88, method=6)
        done.append(image)
        print(f"{kind:5} {image.width}x{image.height}  {out_path.name}")

    if args.contact_sheet and done:
        sheet_path = args.dest / "_toate.jpg"
        contact_sheet(done, sheet_path)
        print(f"\nPlanșă: {sheet_path}")

    print(f"\nGata: {len(done)} imagini în {args.dest}")
    for name, reason in skipped:
        print(f"  sărit: {name} ({reason})")


if __name__ == "__main__":
    main()
