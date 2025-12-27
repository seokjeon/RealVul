#!/usr/bin/env python3
"""
Real_Vul_data.csv에서 project가 'jasper'인 행을 찾으면 즉시
unique_id로 소스코드를 매핑하여 jasper_data_append_processed_func.csv를 생성합니다.
"""

import csv
from pathlib import Path
from typing import Dict

real_vul_csv = Path("/app/RealVul/Dataset/Real_Vul_data.csv")
source_dir = Path("/app/RealVul/Dataset/all_source_code")
output_csv = Path("/app/RealVul/Dataset/jasper_data_append_processed_func.csv")
EMPTY_PLACEHOLDER = " "

def load_source_mapping(source_root: Path) -> Dict[str, Path]:
    """unique_id(확장자 제외) -> 실제 파일 Path 매핑."""
    mapping: Dict[str, Path] = {}
    for path in source_root.iterdir():
        if path.is_file():
            mapping[path.stem] = path
    return mapping

def read_source_text(path: Path) -> str:
    """UTF-8 → ISO-8859-1 순으로 읽기."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")

def main():
    print("Loading source code mapping...")
    mapping = load_source_mapping(source_dir)
    print(f"  - Indexed {len(mapping)} source files")
    
    print("\nProcessing Real_Vul_data.csv (filtering jasper & merging source)...")
    
    processed = 0
    missing = 0
    
    with real_vul_csv.open(newline="", encoding="utf-8") as input_fp, \
         output_csv.open("w", newline="", encoding="utf-8") as output_fp:
        
        reader = csv.DictReader(input_fp)
        new_fieldnames = reader.fieldnames + ["processed_func"]
        writer = csv.DictWriter(output_fp, fieldnames=new_fieldnames)
        writer.writeheader()
        
        for row in reader:
            # jasper가 아니면 스킵
            if row.get("project", "").lower() != "jasper":
                continue
            
            # 즉시 소스코드 매핑
            unique_id = str(row.get("unique_id", ""))
            path = mapping.get(unique_id)
            
            if path is None:
                missing += 1
                row["processed_func"] = EMPTY_PLACEHOLDER
            else:
                row["processed_func"] = read_source_text(path) or EMPTY_PLACEHOLDER
                processed += 1
            
            writer.writerow(row)
    
    print(f"\nCompleted:")
    print(f"  - Jasper rows processed: {processed + missing}")
    print(f"  - Source merged: {processed}")
    print(f"  - Missing source: {missing}")
    print(f"  - Output: {output_csv}")

if __name__ == "__main__":
    main()
