#!/usr/bin/env python3
"""Generate interactive HTML reports from analysis output."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape


def discover_replicates(root: Path) -> List[Dict]:
    """Find replicate folders containing detailed_results.json."""
    replicates = []
    for json_file in root.rglob("detailed_results.json"):
        rep_dir = (
            json_file.parent.parent
            if json_file.parent.name == "results"
            else json_file.parent
        )
        condition = rep_dir.parent.name
        replicates.append(
            {
                "dir": rep_dir,
                "json_path": json_file,
                "name": rep_dir.name,
                "condition": condition,
            }
        )
    return replicates


def load_colony_data(json_path: Path) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # try typical keys
        for key in ["colonies", "data"]:
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def collect_images(rep_dir: Path) -> Dict[str, str]:
    vis_dir = rep_dir / "visualizations"
    front = back = None
    if vis_dir.exists():
        for img in vis_dir.iterdir():
            name = img.name.lower()
            if "front" in name and not front:
                front = img.name
            elif "back" in name and not back:
                back = img.name
    return {"front_img": front, "back_img": back}


def copy_assets(template_dir: Path, out_root: Path) -> None:
    assets_src = template_dir / "assets"
    assets_dst = out_root / "assets"
    if assets_dst.exists():
        shutil.rmtree(assets_dst)
    shutil.copytree(assets_src, assets_dst)


def generate_reports(
    replicates: List[Dict], template_dir: Path, out_root: Path
) -> None:
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    rep_tpl = env.get_template("replicate.html.j2")
    index_tpl = env.get_template("index.html.j2")
    cond_tpl = env.get_template("condition.html.j2")

    grouped: Dict[str, List[Dict]] = {}
    for rep in replicates:
        grouped.setdefault(rep["condition"], []).append(rep)

    for rep in replicates:
        colonies = load_colony_data(rep["json_path"])
        headers = sorted({k for col in colonies for k in col.keys()})
        areas = []
        for col in colonies:
            area = col.get("area")
            if area is None:
                area = col.get("basic_info", {}).get("area")
            if isinstance(area, (int, float)):
                areas.append(area)
        imgs = collect_images(rep["dir"])
        rep.update(imgs)
        html = rep_tpl.render(
            replicate=rep, headers=headers, rows=colonies, areas_json=json.dumps(areas)
        )
        out_path = rep["dir"] / "report.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        rep_rel = out_path.relative_to(out_root)
        rep["page"] = str(rep_rel)
        avg_area = sum(areas) / len(areas) if areas else 0
        rep["stats"] = {"count": len(colonies), "avg_area": avg_area}

    # generate condition summary pages
    cond_dir = out_root / "conditions"
    cond_dir.mkdir(exist_ok=True)
    for cond, reps in grouped.items():
        labels = [r["name"] for r in reps]
        avg_areas = [r["stats"]["avg_area"] for r in reps]
        cond_html = cond_tpl.render(
            condition=cond,
            replicates=reps,
            labels_json=json.dumps(labels),
            avg_areas_json=json.dumps(avg_areas),
        )
        with open(cond_dir / f"{cond}.html", "w", encoding="utf-8") as f:
            f.write(cond_html)

    index_html = index_tpl.render(conditions=grouped)
    with open(out_root / "index.html", "w", encoding="utf-8") as f:
        f.write(index_html)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HTML reports from analysis output"
    )
    parser.add_argument("-i", "--input-dir", required=True, help="analysis output root")
    parser.add_argument(
        "-o", "--output-dir", help="directory to place reports (default: input dir)"
    )
    args = parser.parse_args()
    root = Path(args.input_dir)
    out_root = Path(args.output_dir) if args.output_dir else root
    template_dir = Path(__file__).parent / "report_templates"

    replicates = discover_replicates(root)
    if not replicates:
        print("No replicate data found.")
        return

    copy_assets(template_dir, out_root)
    generate_reports(replicates, template_dir, out_root)
    print(f'Report generated at {out_root / "index.html"}')


if __name__ == "__main__":
    main()
