#!/usr/bin/env python3
"""
Example usage script for combining YOLO datasets with class filtering

This script demonstrates how to use the combine_datasets.py script
to merge multiple YOLO format datasets with class filtering capabilities.
"""

import subprocess
import sys
from pathlib import Path


def run_combine_datasets(input_dirs, output_dir, class_mapping=None, filter_classes=None):
    """
    Run the combine_datasets.py script with the given parameters.
    
    Args:
        input_dirs: List of input dataset directories
        output_dir: Output directory for merged dataset
        class_mapping: Optional path to class mapping JSON file
        filter_classes: Optional list of class names to filter for
    """
    cmd = [
        sys.executable, "combine_datasets.py",
        "--input-dirs"] + input_dirs + [
        "--output-dir", output_dir
    ]
    
    if class_mapping:
        cmd.extend(["--class-mapping", class_mapping])
    
    if filter_classes:
        cmd.extend(["--filter-classes"] + filter_classes)
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        return False


def main():
    """Example usage of the dataset combiner with class filtering."""
    
    # Example 1: Filter for only cyclist class
    print("="*60)
    print("EXAMPLE 1: Filter for Cyclist Class Only")
    print("="*60)
    
    input_dirs = [
        "cyclist_training_data",
        "Dataset2_cyclist_detection"
    ]
    output_dir = "Cyclist_Only_Dataset"
    filter_classes = ["cyclist", "0"]  # Include both "cyclist" and "0" (which represents cyclist in dataset2)
    
    success = run_combine_datasets(input_dirs, output_dir, filter_classes=filter_classes)
    
    if success:
        print(f"\n✅ Successfully created cyclist-only dataset in '{output_dir}'")
        
        # Check the results
        output_path = Path(output_dir)
        if output_path.exists():
            print(f"\nOutput directory structure:")
            for split in ["train", "valid", "test"]:
                split_dir = output_path / split
                if split_dir.exists():
                    images_count = len(list((split_dir / "images").glob("*")))
                    labels_count = len(list((split_dir / "labels").glob("*.txt")))
                    print(f"  {split}/: {images_count} images, {labels_count} labels")
            
            # Check data.yaml
            data_yaml = output_path / "data.yaml"
            if data_yaml.exists():
                print(f"\n✅ Generated data.yaml file")
                with open(data_yaml, 'r') as f:
                    content = f.read()
                    print("Content preview:")
                    print(content[:500] + "..." if len(content) > 500 else content)
    else:
        print(f"\n❌ Failed to create cyclist-only dataset")
    
    # Example 2: Filter for multiple specific classes
    print("\n" + "="*60)
    print("EXAMPLE 2: Filter for Multiple Classes")
    print("="*60)
    
    output_dir_multi = "Multi_Class_Dataset"
    filter_classes_multi = ["cyclist", "person", "0"]  # Include cyclist, person, and class "0"
    
    success = run_combine_datasets(input_dirs, output_dir_multi, filter_classes=filter_classes_multi)
    
    if success:
        print(f"\n✅ Successfully created multi-class filtered dataset in '{output_dir_multi}'")
    else:
        print(f"\n❌ Failed to create multi-class filtered dataset")
    
    # Example 3: Create a class mapping file for better class names
    print("\n" + "="*60)
    print("EXAMPLE 3: Class Mapping with Filtering")
    print("="*60)
    
    class_mapping = {
        "cyclist_training_data": {
            "cyclist": "cyclist",
            "person": "person"
        },
        "Dataset2_cyclist_detection": {
            "0": "cyclist"  # Map class "0" to "cyclist"
        }
    }
    
    mapping_file = "class_mapping_filtered.json"
    import json
    with open(mapping_file, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print(f"Created class mapping file: {mapping_file}")
    print("Content:")
    print(json.dumps(class_mapping, indent=2))
    
    # Example 4: Using mapping with filtering
    print("\n" + "="*60)
    print("EXAMPLE 4: Using Class Mapping with Filtering")
    print("="*60)
    
    output_dir_mapped = "Mapped_Filtered_Dataset"
    filter_classes_mapped = ["cyclist", "person"]  # Only these classes after mapping
    
    success = run_combine_datasets(input_dirs, output_dir_mapped, mapping_file, filter_classes_mapped)
    
    if success:
        print(f"\n✅ Successfully created mapped and filtered dataset in '{output_dir_mapped}'")
    else:
        print(f"\n❌ Failed to create mapped and filtered dataset")
    
    # Example 5: Show available classes in datasets
    print("\n" + "="*60)
    print("EXAMPLE 5: Available Classes in Datasets")
    print("="*60)
    
    import yaml
    
    for dataset_dir in input_dirs:
        data_yaml_path = Path(dataset_dir) / "data.yaml"
        if data_yaml_path.exists():
            with open(data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            print(f"\n{dataset_dir}:")
            print(f"  Classes: {data.get('names', [])}")
            print(f"  Number of classes: {data.get('nc', 0)}")


if __name__ == "__main__":
    main()
