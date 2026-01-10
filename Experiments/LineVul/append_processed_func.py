#!/usr/bin/env python3
"""
CSV 파일에 소스코드를 processed_func 열로 추가합니다.

사용법:
    python append_datasetjasper.py \
        --csv_path /app/RealVul/Dataset/test/jasper_dataset.csv \
        --src_path /app/RealVul/Dataset/test/source_code

입력:
    - --csv_path: 데이터셋 CSV 파일 절대 경로 (file_name 컬럼 필수)
    - --src_path: 소스코드 디렉토리 절대 경로

출력:
    - {csv_path}_append_processed_func.csv
    - 예: jasper_dataset.csv → jasper_dataset_append_processed_func.csv
"""

import csv
import argparse
from pathlib import Path
from typing import Dict, Optional

EMPTY_PLACEHOLDER = " "


def build_source_mapping(src_dir: Path) -> Dict[str, Path]:
    """
    소스 디렉토리의 파일들을 인덱싱하여 {파일명(확장자 제외): Path} 매핑 반환.
    
    - 확장자가 .c인 파일: stem으로 매핑
    - 확장자가 없는 파일: name으로 매핑
    - 기타 확장자: stem으로 매핑
    """
    mapping: Dict[str, Path] = {}

    if not src_dir.exists():
        print(f"  [ERROR] Source directory does not exist: {src_dir}")
        return mapping

    for path in src_dir.iterdir():
        if not path.is_file():
            continue
        
        # 파일명(확장자 제외)을 key로 사용
        # stem: foo.c → foo, foo → foo (확장자 없으면 name과 동일)
        key = path.stem
        
        # 중복 체크 (같은 stem의 파일이 여러 개 있을 경우)
        if key in mapping:
            print(f"  [WARNING] Duplicate stem '{key}': {mapping[key]} vs {path}")
        
        mapping[key] = path

    return mapping


def read_source_text(path: Path) -> str:
    """UTF-8 → ISO-8859-1 순으로 읽기."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def get_output_path(csv_path: Path) -> Path:
    """
    출력 파일 경로 생성.
    예: jasper_dataset.csv → jasper_dataset_append_processed_func.csv
    """
    stem = csv_path.stem  # jasper_dataset
    suffix = csv_path.suffix  # .csv
    new_name = f"{stem}_append_processed_func{suffix}"
    return csv_path.parent / new_name


def main():
    parser = argparse.ArgumentParser(
        description="CSV 파일에 소스코드를 processed_func 열로 추가"
    )
    parser.add_argument(
        "--csv_path",
        required=True,
        help="데이터셋 CSV 파일 절대 경로 (file_name 컬럼 필수)",
    )
    parser.add_argument(
        "--src_path",
        required=True,
        help="소스코드 디렉토리 절대 경로",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    src_path = Path(args.src_path)
    output_path = get_output_path(csv_path)

    print("=" * 60)
    print("Append Dataset - Source Code Merger")
    print("=" * 60)
    print(f"\nPaths:")
    print(f"  - Input CSV   : {csv_path}")
    print(f"  - Source Dir  : {src_path}")
    print(f"  - Output CSV  : {output_path}")

    # 입력 파일 확인
    if not csv_path.exists():
        print(f"\n[ERROR] Input CSV does not exist: {csv_path}")
        return

    # 소스 파일 인덱싱
    print(f"\nLoading source code mapping...")
    mapping = build_source_mapping(src_path)
    print(f"  - Indexed {len(mapping)} source files")

    if len(mapping) == 0:
        print("\n[ERROR] No source files found. Check source directory.")
        return

    # CSV 처리
    print(f"\nProcessing CSV...")

    total_rows = 0
    processed = 0
    missing = 0

    with csv_path.open(newline="", encoding="utf-8") as input_fp, \
         output_path.open("w", newline="", encoding="utf-8") as output_fp:

        reader = csv.DictReader(input_fp)
        if not reader.fieldnames:
            raise RuntimeError("Input CSV has no header (fieldnames).")

        # file_name 컬럼 확인
        if "file_name" not in reader.fieldnames:
            print(f"\n[ERROR] 'file_name' column not found in CSV.")
            print(f"  Available columns: {reader.fieldnames}")
            return

        # processed_func 열 추가
        new_fieldnames = list(reader.fieldnames) + ["processed_func"]
        writer = csv.DictWriter(output_fp, fieldnames=new_fieldnames)
        writer.writeheader()

        for row in reader:
            total_rows += 1

            # file_name에서 확장자 제거하여 key 생성
            file_name_raw = str(row.get("file_name", "")).strip()
            # Path를 사용하여 확장자 제거
            file_name_key = Path(file_name_raw).stem if file_name_raw else ""

            # 소스 파일 찾기
            source_path = mapping.get(file_name_key)

            if source_path is None:
                missing += 1
                row["processed_func"] = EMPTY_PLACEHOLDER
                if missing <= 5:  # 처음 5개만 경고 출력
                    print(f"  [WARN] Missing source for file_name='{file_name_raw}' (key='{file_name_key}')")
            else:
                source_text = read_source_text(source_path)
                row["processed_func"] = source_text if source_text else EMPTY_PLACEHOLDER
                processed += 1

            writer.writerow(row)

    # 결과 출력
    print(f"\n{'=' * 60}")
    print("Completed!")
    print(f"{'=' * 60}")
    print(f"  - Total rows           : {total_rows}")
    print(f"  - Source merged        : {processed}")
    print(f"  - Missing source       : {missing}")
    print(f"  - Output               : {output_path}")


if __name__ == "__main__":
    main()
