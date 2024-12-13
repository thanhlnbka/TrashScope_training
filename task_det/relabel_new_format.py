import os

# LABEL TRAINING FOR TASK DETECT_CLS SINGLE SHOT
label_dir = "/workspace/obj_det_cls/data_training/labels"  
output_label_dir = "/workspace/obj_det_cls/data_training/new_format_labels"  
os.makedirs(output_label_dir, exist_ok=True)

# CLASS ORG LABELED
classes_det = {
    0: 'Overload_Clean',
    1: 'Overload_Not_clean',
    2: 'Overload_Partial_mix',
    3: 'Overload_Total_mix',
    4: 'Not_overload_Clean',
    5: 'Not_overload_Not_clean',
    6: 'Not_overload_Partial_mix',
    7: 'Not_overload_Total_mix',
    8: 'Not_overload_Empty',
    9: 'Not_overload_Unknown',
    10: 'Overload_Unknown',
    11: 'Unknown_Unknown',
}

# STATUS BIN
classes_det_status_expect = {
    "Overload": {0: [0, 1, 2, 3, 10]},
    "Not_overload": {1: [4, 5, 6, 7, 9]},
    "Unknown": {2: [11]},
    "Empty": {3: [8]}
}


old_to_new_class_map = {}
for group_name, group_data in classes_det_status_expect.items():
    new_class_id = list(group_data.keys())[0]
    old_classes = group_data[new_class_id]
    for old_class in old_classes:
        old_to_new_class_map[old_class] = new_class_id


def convert_labels():
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            input_path = os.path.join(label_dir, label_file)
            output_path = os.path.join(output_label_dir, label_file)

            with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
                for line in f_in:
                    
                    parts = line.strip().split()
                    old_class_id = int(parts[0])  
                    if old_class_id in old_to_new_class_map:
                        new_class_id = old_to_new_class_map[old_class_id]
                        
                        f_out.write(f"{new_class_id} {' '.join(parts[1:])}\n")
                    else:
                        print(f"Warning: Class ID {old_class_id} not mapped, skipping...")


#Convert labels to 4cls status
convert_labels()
print("Conversion completed! New YOLO labels saved.")
