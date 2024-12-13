import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
from ultralytics import YOLO 


reverse_class_mapping = {
    0: ("true", "Clean"),
    1: ("true", "Not_clean"),
    2: ("true", "Partial_mix"),
    3: ("true", "Total_mix"),
    4: ("false", "Clean"),
    5: ("false", "Not_clean"),
    6: ("false", "Partial_mix"),
    7: ("false", "Total_mix"),
    8: ("false", "Empty"),
    9: ("false", "Unknown"),
    10: ("true", "Unknown"),
    11: ("unknown", "Unknown")
    
}

def detect_and_update_xml(model_path, xml_file, batch2_dir, output_xml_file):
    
    model = YOLO(model_path)

    
    tree = ET.parse(xml_file)
    root = tree.getroot()

    
    for image in root.findall('image'):
        
        img_name = image.get('name')
        img_path = Path(batch2_dir) / f"{Path(img_name).stem}.jpg"
        print(img_path)
        
        if not img_path.exists() or len(image.findall('box')) > 0:
            continue

        
        img = cv2.imread(str(img_path))
        results = model(img, conf=0.7)

        for result in results:
            for box in result.boxes:
                
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

                
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])  
                overload, type_value = reverse_class_mapping[class_id]

                
                ET.SubElement(
                    image,
                    "box",
                    {
                        "label": "dumpster",
                        "source": "auto",
                        "occluded": "0",
                        "xtl": f"{x1:.2f}",
                        "ytl": f"{y1:.2f}",
                        "xbr": f"{x2:.2f}",
                        "ybr": f"{y2:.2f}",
                        "z_order": "0",
                    },
                )
                
                box_elem = image.findall('box')[-1]
                ET.SubElement(box_elem, "attribute", {"name": "Overload"}).text = overload
                ET.SubElement(box_elem, "attribute", {"name": "Type"}).text = type_value
                ET.SubElement(box_elem, "attribute", {"name": "Covered"}).text = "Uncorved"

    
    tree.write(output_xml_file, encoding="utf-8", xml_declaration=True)


model_path = "./yolo_training/yolov8s_512_experiment_input3_part1_b123_part2_b12_12cls2/weights/best.pt"
xml_file = "./input3_part3_filter_clean.xml"
batch_dir = "../data_sources/input_part3_filter_clean"
output_xml_file = "./input3_part3_filter_clean_update.xml"


detect_and_update_xml(model_path, xml_file, batch_dir, output_xml_file)
