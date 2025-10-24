import json
from pathlib import Path

root_dir = Path.cwd()

with open(root_dir / "assets/Sentiment/de_raw.json", "r") as f:
    data = json.load(f)
    
print(len(data["sentences"]["positive"]))
print(len(data["sentences"]["neutral"]))
print(len(data["sentences"]["negative"]))