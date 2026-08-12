import os
import json

def append_end_marker(dataset_dir, folders):
    end_marker = {
        "type": "Q",
        "text": "|end|",
        "cluster": -2
    }
    
    for folder in folders:
        folder_path = os.path.join(dataset_dir, folder)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r+', encoding='utf-8') as f:
                data = json.load(f)

                for section in data.get('list', []):
                    audio = section.get('audio', [])

                    if audio[-1].get("type") == "Q":
                        audio.pop()

                    audio.append(end_marker)

                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=4)
                f.truncate()

