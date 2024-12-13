import xml.etree.ElementTree as ET
from pathlib import Path
import shutil


class_mapping = {
    ("true", "Clean"): 0,
    ("true", "Not_clean"): 1,
    ("true", "Partial_mix"): 2,
    ("true", "Total_mix"): 3,
    ("false", "Clean"): 4,
    ("false", "Not_clean"): 5,
    ("false", "Partial_mix"): 6,
    ("false", "Total_mix"): 7,
    ("false", "Empty"): 8,  
    ("false", "Unknown"): 9,
    ("true", "Unknown"): 10,
    ("unknown", "Unknown"): 11
}

def convert_to_yolo_format(xml_file, batch_dir, output_dir):
    
    output_path = Path(output_dir)
    images_path = output_path / "images"
    labels_path = output_path / "labels"
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)

    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    content_file_train_txt = ""
    
    for image in root.findall('image'):
        img_width = float(image.get('width'))
        img_height = float(image.get('height'))
        img_name = image.get('name')

        
        src_image_path = Path(batch_dir) / f"{Path(img_name).stem}.jpg" 
        
        dst_image_path = images_path / f"{Path(img_name).stem}.jpg"
        content_file_train_txt += f"./images/{Path(img_name).stem}.jpg\n"
        
        print(src_image_path)
        if src_image_path.exists():
            shutil.copy(src_image_path, dst_image_path)

        
        label_file = labels_path / f"{Path(img_name).stem}.txt"
        with label_file.open("w") as f:
            for box in image.findall('box'):  
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))

                
                x_center = (xtl + xbr) / 2 / img_width
                y_center = (ytl + ybr) / 2 / img_height
                width = (xbr - xtl) / img_width
                height = (ybr - ytl) / img_height

                
                overload = None
                type_value = None
                for attribute in box.findall('attribute'):
                    if attribute.get('name') == "Overload":
                        overload = attribute.text.lower()
                    elif attribute.get('name') == "Type":
                        type_value = attribute.text

                
                if type_value == "Empty":
                    class_id = 8  
                else:
                    class_id = class_mapping.get((overload, type_value))

                if class_id is not None:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    with open(f"{output_dir}/train.txt", "a") as f:
        f.write(content_file_train_txt)
    f.close()






# PART 1 - BATCH 123
# xml_file = "/home/thanhln/Desktop/TrashScope/obj_det_cls/in_cvat_labels/frames_10_10_batches/annotations_batch123.xml"
# batch_dir = "/home/thanhln/Desktop/TrashScope/data_sources/frames_10_10_batches/batch_123"
# output_dir = "/home/thanhln/Desktop/TrashScope/obj_det_cls/data_format/input3_part1_batch123"

# # PART 2 - BATCH 1
# xml_file = "/home/thanhln/Desktop/TrashScope/obj_det_cls/in_cvat_labels/frames_10_11_batches/annotations_batch1.xml"
# batch_dir = "/home/thanhln/Desktop/TrashScope/data_sources/frames_10_11_batches/batch_1"
# output_dir = "/home/thanhln/Desktop/TrashScope/obj_det_cls/data_format/input3_part2_batch_1/"

# PART 2 - BATCH 2
# xml_file = "/home/thanhln/Desktop/TrashScope/obj_det_cls/in_cvat_labels/frames_10_11_batches/annotations_batch2.xml"
# batch_dir = "/home/thanhln/Desktop/TrashScope/data_sources/frames_10_11_batches/batch_2"
# output_dir = "/home/thanhln/Desktop/TrashScope/obj_det_cls/data_format/input3_part2_batch_2/"


# # PART 2 - BATCH 3
# xml_file = "/home/thanhln/Desktop/TrashScope/obj_det_cls/in_cvat_labels/frames_10_11_batches/annotations_batch3.xml"
# batch_dir = "/home/thanhln/Desktop/TrashScope/data_sources/frames_10_11_batches/batch_3"
# output_dir = "/home/thanhln/Desktop/TrashScope/obj_det_cls/data_format/input3_part2_batch_3/"


# # PART 4 - Unknow
xml_file = "/home/thanhln/Desktop/TrashScope/obj_det_cls/in_cvat_labels/input3_part4_unknow/annotations.xml"
batch_dir = "/home/thanhln/Desktop/TrashScope/data_sources/input3_part4_unknow"
output_dir = "/home/thanhln/Desktop/TrashScope/obj_det_cls/data_format/input3_part4_unknow/"

# Convert  label for task sigle shot
convert_to_yolo_format(xml_file, batch_dir, output_dir)






















xml_file = "/home/thanhln/Desktop/TrashScope/obj_det_cls/in_cvat_labels/input3_part4_unknow/annotations.xml"
batch_dir = "/home/thanhln/Desktop/TrashScope/data_sources/input3_part4_unknow"
output_dir = "/home/thanhln/Desktop/TrashScope/obj_det_cls/data_format/input3_part4_unknow/"


convert_to_yolo_format(xml_file, batch_dir, output_dir)
