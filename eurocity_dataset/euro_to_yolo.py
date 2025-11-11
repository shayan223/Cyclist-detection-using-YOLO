"""
Convert EuroCity Persons dataset from JSON format to YOLO format.
This script processes the ECP dataset and converts it to YOLOv8/YOLOv11 format.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# Class mapping: EuroCity identity -> YOLO class ID
# Class 0: cyclist, Class 1: pedestrian
CLASS_MAPPING = {
    'rider': 0,  # cyclist
    'rider+vehicle-group-far-away': 0,  # cyclist (far away)
    'rider+vehicle': 0,  # cyclist
    'person-group-far-away': 1,  # pedestrian
    'person': 1,  # pedestrian
    'person-group': 1,  # pedestrian group
}

CLASS_NAMES = ['cyclist', 'pedestrian']


def convert_bbox_to_yolo(x0: float, y0: float, x1: float, y1: float, 
                         img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from (x0, y0, x1, y1) format to YOLO format.
    YOLO format: normalized (x_center, y_center, width, height)
    
    Args:
        x0, y0: Top-left corner coordinates
        x1, y1: Bottom-right corner coordinates
        img_width: Image width
        img_height: Image height
    
    Returns:
        Tuple of (x_center, y_center, width, height) normalized to [0, 1]
    """
    # Calculate center coordinates
    x_center = (x0 + x1) / 2.0
    y_center = (y0 + y1) / 2.0
    
    # Calculate width and height
    width = x1 - x0
    height = y1 - y0
    
    # Normalize by image dimensions
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    # Ensure values are within [0, 1]
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))
    
    return x_center_norm, y_center_norm, width_norm, height_norm


def extract_annotations(json_data: Dict, img_width: int, img_height: int) -> List[Tuple[int, float, float, float, float]]:
    """
    Extract annotations from EuroCity JSON format.
    
    Args:
        json_data: Parsed JSON data
        img_width: Image width
        img_height: Image height
    
    Returns:
        List of tuples: (class_id, x_center, y_center, width, height)
    """
    annotations = []
    
    def process_children(children: List[Dict]):
        """Recursively process children to find all objects."""
        for child in children:
            identity = child.get('identity', '')
            
            # Check if this identity should be included
            if identity in CLASS_MAPPING:
                class_id = CLASS_MAPPING[identity]
                x0 = child.get('x0', 0)
                y0 = child.get('y0', 0)
                x1 = child.get('x1', 0)
                y1 = child.get('y1', 0)
                
                # Validate bounding box
                if x1 > x0 and y1 > y0:
                    x_center, y_center, width, height = convert_bbox_to_yolo(
                        x0, y0, x1, y1, img_width, img_height
                    )
                    annotations.append((class_id, x_center, y_center, width, height))
            
            # Recursively process nested children
            if 'children' in child and child['children']:
                process_children(child['children'])
    
    # Process the root children
    if 'children' in json_data:
        process_children(json_data['children'])
    
    return annotations


def convert_dataset(split: str = 'train'):
    """
    Convert EuroCity dataset split (train or val) to YOLO format.
    
    Args:
        split: 'train' or 'val'
    """
    # Define paths
    script_dir = Path(__file__).parent
    img_base_dir = script_dir / f'ECP_day_img_{split}' / 'ECP' / 'day' / 'img' / split
    label_base_dir = script_dir / f'ECP_day_labels_{split}' / 'ECP' / 'day' / 'labels' / split
    output_dir = script_dir / 'eurocity_yolo' / split
    
    # Create output directories
    output_img_dir = output_dir / 'images'
    output_label_dir = output_dir / 'labels'
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all city directories
    if not img_base_dir.exists():
        print(f"Warning: Image directory {img_base_dir} does not exist!")
        return
    
    city_dirs = [d for d in img_base_dir.iterdir() if d.is_dir()]
    
    total_images = 0
    total_annotations = 0
    skipped_images = 0
    
    print(f"\nProcessing {split} split...")
    print(f"Found {len(city_dirs)} cities")
    
    for city_dir in city_dirs:
        city_name = city_dir.name
        print(f"  Processing {city_name}...")
        
        # Get corresponding label directory
        label_city_dir = label_base_dir / city_name
        
        if not label_city_dir.exists():
            print(f"    Warning: Label directory {label_city_dir} does not exist, skipping {city_name}")
            continue
        
        # Get all image files
        image_files = list(city_dir.glob('*.png'))
        
        for img_file in image_files:
            # Find corresponding label file
            label_file = label_city_dir / f"{img_file.stem}.json"
            
            if not label_file.exists():
                skipped_images += 1
                continue
            
            try:
                # Read JSON label
                with open(label_file, 'r') as f:
                    json_data = json.load(f)
                
                # Get image dimensions
                img_width = json_data.get('imagewidth', 1920)
                img_height = json_data.get('imageheight', 1024)
                
                # Extract annotations
                annotations = extract_annotations(json_data, img_width, img_height)
                
                # Skip images with no valid annotations
                if not annotations:
                    skipped_images += 1
                    continue
                
                # Copy image to output directory
                output_img_path = output_img_dir / img_file.name
                shutil.copy2(img_file, output_img_path)
                
                # Write YOLO format label file
                output_label_path = output_label_dir / f"{img_file.stem}.txt"
                with open(output_label_path, 'w') as f:
                    for class_id, x_center, y_center, width, height in annotations:
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                total_images += 1
                total_annotations += len(annotations)
                
            except Exception as e:
                print(f"    Error processing {img_file.name}: {e}")
                skipped_images += 1
                continue
        
        print(f"    {city_name}: {len(image_files)} images processed")
    
    print(f"\n{split.upper()} split conversion complete:")
    print(f"  Total images: {total_images}")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Skipped images: {skipped_images}")
    print(f"  Average annotations per image: {total_annotations/total_images if total_images > 0 else 0:.2f}")


def create_data_yaml():
    """Create data.yaml file for YOLO training."""
    script_dir = Path(__file__).parent
    output_dir = script_dir / 'eurocity_yolo'
    
    yaml_content = f"""# EuroCity Persons Dataset converted to YOLO format
# Dataset path (relative to this file)
path: {output_dir.absolute()}
train: train/images
val: val/images

# Classes
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated data.yaml at {yaml_path}")


def main():
    """Main function to convert the entire dataset."""
    print("=" * 60)
    print("EuroCity Persons to YOLO Format Converter")
    print("=" * 60)
    
    # Convert train and val splits
    convert_dataset('train')
    convert_dataset('val')
    
    # Create data.yaml
    create_data_yaml()
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"\nOutput directory: {Path(__file__).parent / 'eurocity_yolo'}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Class mapping: {CLASS_MAPPING}")


if __name__ == '__main__':
    main()

