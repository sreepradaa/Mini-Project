import os

# Define class mappings
helmet_class_map = {
    '0': '0',  # motorcycle -> motorcycle
    '1': '2',  # helmet -> helmet
    '2': '4'   # person -> person (if present)
}
seatbelt_class_map = {
    '0': '3',  # seatbelt -> seatbelt
    '1': '3'   # no-seatbelt -> treat as seatbelt (or remove)
}

# Paths to label folders
label_dirs = ["C:/MiniP/dataset/labels/train", "C:/MiniP/dataset/labels/val"]

for label_dir in label_dirs:
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            # Determine which dataset this file likely came from (simplified)
            # You may need to adjust based on filename or other metadata
            if "helmet" in label_file.lower():  # Adjust based on your naming
                class_id = parts[0]
                if class_id in helmet_class_map:
                    parts[0] = helmet_class_map[class_id]
                    new_lines.append(' '.join(parts) + '\n')
            else:  # Assume seatbelt dataset
                class_id = parts[0]
                if class_id in seatbelt_class_map and class_id == '0':  # Only keep seatbelt
                    parts[0] = seatbelt_class_map[class_id]
                    new_lines.append(' '.join(parts) + '\n')
        
        # Write updated labels back to file
        with open(label_path, 'w') as f:
            f.writelines(new_lines)

print("Label files updated!")
