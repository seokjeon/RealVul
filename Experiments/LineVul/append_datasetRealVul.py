#!/usr/bin/env python3
"""
Real_Vul_data.csv에 processed_func 컬럼을 추가해
Real_Vul_data_append_processed_func.csv를 생성합니다.
processed_func가 비었을 경우 공백 한 칸(" ")을 넣어 pandas에서 NaN으로
해석되지 않도록 합니다.
"""

import csv
from pathlib import Path
from typing import Dict

real_vul_csv = / "app" / "Dataset" / "Real_Vul_data.csv"
source_dir = / "app" / "Dataset" / "all_source_code"
output_csv = / "app" / "Dataset" / "Real_Vul_data_append_processed_func.csv"
EMPTY_PLACEHOLDER = " "  # 빈 문자열 대신 넣을 값 (pandas keep_default_na=True에서도 NaN 아님)

def load_source_mapping(source_root: Path) -> Dict[str, Path]:
    """file_name(확장자 제외) -> 실제 파일 Path 매핑."""
    mapping: Dict[str, Path] = {}
    for path in source_root.iterdir():
        if path.is_file():
            mapping[path.stem] = path
    return mapping

def read_source_text(path: Path) -> str:
    """원본 파일을 UTF-8 → ISO-8859-1 순으로 읽는다."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")

def main():
    mapping = load_source_mapping(source_dir)
    print(f"Indexed {len(mapping)} source files from {source_dir}")

    with real_vul_csv.open(newline="", encoding="utf-8") as input_fp, \
         output_csv.open("w", newline="", encoding="utf-8") as output_fp:

        reader = csv.DictReader(input_fp)
        fieldnames = reader.fieldnames + ["processed_func"]
        writer = csv.DictWriter(output_fp, fieldnames=fieldnames)
        writer.writeheader()

        missing = 0
        for row in reader:
            file_id = str(row.get("file_name", ""))
            path = mapping.get(file_id)

            if path is None:
                candidates = list(source_dir.glob(f"{file_id}.*"))
                if candidates:
                    path = candidates[0]
                    mapping[file_id] = path  # 캐시
            if path is None:
                missing += 1
                row["processed_func"] = EMPTY_PLACEHOLDER
                writer.writerow(row)
                continue

            row["processed_func"] = read_source_text(path) or EMPTY_PLACEHOLDER
            writer.writerow(row)

    if missing:
        print(f"Warning: {missing} entries had no matching source file "
              f"-> processed_func set to a single space.")
    print(f"Saved {output_csv}")

if __name__ == "__main__":
    main()
