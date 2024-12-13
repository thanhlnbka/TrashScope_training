import os
import xml.etree.ElementTree as ET
from PIL import Image

# xml_file = "/home/thanhln/Desktop/TrashScope_newreq/input3_part4.xml" 
# input_folder = "/home/thanhln/Desktop/TrashScope/data_sources/input3_part4_unknow"  
# xml_file = "/home/thanhln/Desktop/TrashScope_newreq/input3_part2_batch123.xml" 
# input_folder = "/home/thanhln/Desktop/TrashScope/data_sources/frames_10_10_batches/batch_123" 

xml_file = "/home/thanhln/Desktop/TrashScope_newreq/input3_part3_batch3.xml"  # source labeled
input_folder = "/home/thanhln/Desktop/TrashScope/data_sources/frames_10_11_batches/batch_3" #source images
output_folder = "test_images"  

os.makedirs(output_folder, exist_ok=True)

tree = ET.parse(xml_file)
root = tree.getroot()

for image in root.findall("image"):
    image_name = image.get("name")  
    image_name_only = os.path.basename(image_name) 
    
    image_path = os.path.join(input_folder, image_name_only)

    if not os.path.exists(image_path):
        print(f"Image {image_name} not found in {input_folder}. Skipping...")
        continue

    img = Image.open(image_path)

    for box in image.findall("box"):
        label = box.get("label")  
        xtl, ytl = float(box.get("xtl")), float(box.get("ytl")) 
        xbr, ybr = float(box.get("xbr")), float(box.get("ybr")) 

        attributes = {attr.get("name"): attr.text for attr in box.findall("attribute")}
        box_type = attributes.get("Type", "NA")
        covered = attributes.get("Covered", "Uncorved")

        cropped_img = img.crop((xtl, ytl, xbr, ybr))

        type_folder = os.path.join(output_folder, box_type)
        os.makedirs(type_folder, exist_ok=True)

        base_name, ext = os.path.splitext(image_name_only)
        output_name = f"{base_name}_{covered}.jpg"
        output_path = os.path.join(type_folder, output_name)

        cropped_img.save(output_path)
        print(f"Saved: {output_path}")
