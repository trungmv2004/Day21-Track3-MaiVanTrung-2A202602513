#!/usr/bin/env python3
"""Write a measured GPU report after NB5; preserve the existing CPU report."""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labkit.submission import load_bundle, render_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="Mai Van Trung (theo tên workspace)")
    parser.add_argument("--student-id", default="2A202602513 (theo tên workspace)")
    parser.add_argument("--reflection", type=pathlib.Path,
                        help="File phản tư do người học cung cấp; không tự tạo trải nghiệm cá nhân")
    parser.add_argument("--packaged-results", action="store_true",
                        help="Kiểm tra bản export Option A: chỉ bắt buộc file adapter correct")
    args = parser.parse_args()
    try:
        bundle = load_bundle(ROOT, require_all_adapters=not args.packaged_results)
        reflection = args.reflection.read_text(encoding="utf-8") if args.reflection else ""
        body = render_report(bundle, name=args.name, student_id=args.student_id, reflection=reflection)
    except (ValueError, KeyError, OSError) as exc:
        print(f"Chưa tạo report GPU: {exc}", file=sys.stderr)
        return 1
    report = ROOT / "submission/REPORT.md"
    backup = ROOT / "submission/REPORT-CPU.md"
    if report.exists() and not backup.exists():
        backup.write_bytes(report.read_bytes())
    report.write_text(body, encoding="utf-8")
    print("Wrote submission/REPORT.md from measured GPU artifacts.")
    if not reflection.strip():
        print("Còn thiếu phản tư cá nhân của người học; phần này chưa đủ rubric 4.4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
