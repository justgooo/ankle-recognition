import re
import csv
import sys

input_file = "c:/Users/xxx/Desktop/足踝/backlog.md"
output_file = "c:/Users/xxx/Desktop/足踝/results.tsv"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

records = []
for line in lines:
    line = line.strip()
    if line.startswith("- [x]"):
        # Handle special case with multiple records separated by |
        parts = line.split(" | ") if " | " in line else [line]
        
        for part in parts:
            if not part.startswith("- [x]"):
                part = "- [x] " + part.strip()
            
            # Find the first -> or →
            match = re.split(r'→|->', part, maxsplit=1)
            if len(match) > 1:
                desc_part = match[0].replace("- [x]", "").strip()
                result_part = match[1].strip()
                
                # Extract commit: look for 7-char lowercase hex
                commit_match = re.search(r'`([a-f0-9]{7})`', desc_part)
                if not commit_match:
                    commit_match = re.search(r'\b([a-f0-9]{7})\b', desc_part)
                commit = commit_match.group(1) if commit_match else "NA"
                
                # Acc
                acc_match = re.search(r'(?:no_miss_val_acc|acc)=([\d\.]+)', part)
                if acc_match:
                    val_acc = acc_match.group(1)
                else:
                    # sometimes like: AP-FF-01 → 0.574 discard
                    acc_simple = re.search(r'→\s*([\d\.]+)', part)
                    val_acc = acc_simple.group(1) if acc_simple else "NA"
                
                # AUC
                auc_match = re.search(r'(?:val_AUC|val_auc|AUC|auc)=([\d\.]+)', part)
                val_auc = auc_match.group(1) if auc_match else "NA"
                
                # Status
                status = "discard"
                if "keep" in part.lower() or "✅" in part:
                    status = "keep"

                # Config
                config_match = re.search(r'[\(（](.*?)[\)）]', desc_part)
                config = config_match.group(1) if config_match else "NA"
                
                description = desc_part.replace('**', '').strip()

                records.append({
                    "commit": commit,
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "val_f1": "NA",
                    "memory_gb": "NA",
                    "status": status,
                    "config": config,
                    "description": description
                })

# Write to results.tsv
with open(output_file, "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(["commit", "val_acc", "val_auc", "val_f1", "memory_gb", "status", "config", "description"])
    for r in records:
        writer.writerow([r["commit"], r["val_acc"], r["val_auc"], r["val_f1"], r["memory_gb"], r["status"], r["config"], r["description"]])

print(f"Recovered {len(records)} records from backlog.md.")
