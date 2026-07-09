import os
import csv
from pathlib import Path

def validate_dataset():
    data_dir = Path("data/raw/pdnc/data")
    if not data_dir.exists():
        print("Data directory not found!")
        return

    novels = [d for d in data_dir.iterdir() if d.is_dir()]
    print(f"Found {len(novels)} novels.")
    
    total_quotes = 0
    total_characters = 0
    errors = []
    
    for novel in novels:
        char_file = novel / "character_info.csv"
        quote_file = novel / "quotation_info.csv"
        text_file = novel / "novel_text.txt"
        
        for f in [char_file, quote_file, text_file]:
            if not f.exists():
                errors.append(f"Missing file: {f}")
            elif f.stat().st_size == 0:
                errors.append(f"Empty file: {f}")
                
        # Try parsing CSVs
        if char_file.exists():
            try:
                with open(char_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    total_characters += len(rows) - 1 if len(rows) > 0 else 0
            except Exception as e:
                errors.append(f"Error parsing {char_file}: {e}")
                
        if quote_file.exists():
            try:
                with open(quote_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    total_quotes += len(rows) - 1 if len(rows) > 0 else 0
            except Exception as e:
                errors.append(f"Error parsing {quote_file}: {e}")
                
        if text_file.exists():
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                errors.append(f"Error parsing {text_file}: {e}")
                
    report_path = Path("results/dataset_validation_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("# PDNC Dataset Validation Report\n\n")
        f.write(f"- **Novels Found:** {len(novels)}\n")
        f.write(f"- **Total Quotes Found:** {total_quotes}\n")
        f.write(f"- **Total Characters Found:** {total_characters}\n\n")
        
        if errors:
            f.write("## Errors Detected\n")
            for e in errors:
                f.write(f"- {e}\n")
        else:
            f.write("## Errors Detected\n")
            f.write("No errors detected. Dataset is structurally valid and properly encoded (UTF-8).\n")
            
    print(f"Validation complete. Report saved to {report_path}")

if __name__ == "__main__":
    validate_dataset()
