#!/usr/bin/env python3
"""
YOLO Dataset Combiner Script

This script combines multiple YOLO format datasets into a single merged dataset.
It handles class name mapping, file copying, and generates a unified data.yaml file.
It also supports filtering datasets to include only specific classes.

Usage:
    # Basic combination
    python combine_datasets.py --input-dirs dataset1 dataset2 dataset3 --output-dir merged_dataset
    
    # With class mapping
    python combine_datasets.py --input-dirs dataset1 dataset2 --output-dir merged_dataset --class-mapping mapping.json
    
    # Filter for specific classes only
    python combine_datasets.py --input-dirs dataset1 dataset2 --output-dir merged_dataset --filter-classes cyclist person
    
    # Combine filtering and mapping
    python combine_datasets.py --input-dirs dataset1 dataset2 --output-dir merged_dataset --filter-classes cyclist --class-mapping mapping.json
"""

import os
import shutil
import argparse
import yaml
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
import hashlib
from collections import defaultdict
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle


class YOLODatasetCombiner:
    def __init__(self, input_dirs: List[str], output_dir: str, class_mapping: Dict = None, 
                 filter_classes: List[str] = None, show_visualization: bool = True):
        """
        Initialize the YOLO dataset combiner.
        
        Args:
            input_dirs: List of input dataset directories
            output_dir: Output directory for merged dataset
            class_mapping: Optional class mapping dictionary
            filter_classes: Optional list of class names to include (filters out others)
            show_visualization: Whether to display random samples with bounding boxes
        """
        self.input_dirs = [Path(d) for d in input_dirs]
        self.output_dir = Path(output_dir)
        self.class_mapping = class_mapping or {}
        self.filter_classes = filter_classes
        self.show_visualization = show_visualization
        
        # Validate input directories
        self._validate_input_dirs()
        
        # Create output directory structure
        self._create_output_structure()
        
        # Track merged classes and statistics
        self.merged_classes = {}
        self.class_counter = 0
        self.file_stats = defaultdict(int)
        self.duplicate_files = []
        self.filtered_files = defaultdict(int)  # Track filtered files
        
    def _validate_input_dirs(self):
        """Validate that all input directories exist and contain YOLO format data."""
        for input_dir in self.input_dirs:
            if not input_dir.exists():
                raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
            
            # Check for data.yaml file
            data_yaml = input_dir / "data.yaml"
            if not data_yaml.exists():
                raise FileNotFoundError(f"data.yaml not found in {input_dir}")
            
            # Check for train/valid/test directories
            required_dirs = ["train", "valid"]
            for split in required_dirs:
                split_dir = input_dir / split
                if not split_dir.exists():
                    print(f"Warning: {split} directory not found in {input_dir}")
                else:
                    # Check for images and labels subdirectories
                    images_dir = split_dir / "images"
                    labels_dir = split_dir / "labels"
                    if not images_dir.exists() or not labels_dir.exists():
                        print(f"Warning: images or labels directory missing in {split_dir}")
    
    def _create_output_structure(self):
        """Create the output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each split
        for split in ["train", "valid", "test"]:
            split_dir = self.output_dir / split
            split_dir.mkdir(exist_ok=True)
            (split_dir / "images").mkdir(exist_ok=True)
            (split_dir / "labels").mkdir(exist_ok=True)
    
    def _load_dataset_info(self, dataset_dir: Path) -> Dict:
        """Load dataset information from data.yaml file."""
        data_yaml_path = dataset_dir / "data.yaml"
        
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return data
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _merge_classes(self, dataset_classes: List[str], dataset_name: str) -> Dict[int, int]:
        """
        Merge class names from different datasets, handling duplicates and conflicts.
        If filter_classes is specified, include only classes that map to the filtered set
        after applying alias normalization (e.g., 'person' -> 'pedestrian').
        
        Args:
            dataset_classes: List of class names from current dataset
            dataset_name: Name of the current dataset
            
        Returns:
            Dictionary mapping old class indices to new class indices
        """
        class_mapping = {}
        
        # Normalize filter set once for comparisons
        normalized_filter: Set[str] = set(c.lower() for c in self.filter_classes) if self.filter_classes else None
        
        for old_idx, class_name in enumerate(dataset_classes):
            # Determine the target class name (apply alias normalization first)
            target_class_name = self._get_target_class_name(class_name)
            
            # If we are filtering and the target class is not in the requested set, skip
            if normalized_filter is not None and (target_class_name is None or target_class_name.lower() not in normalized_filter):
                continue
            
            # Check if target class already exists in merged classes
            if target_class_name in self.merged_classes.values():
                # Find the existing class index
                new_idx = next(idx for idx, name in self.merged_classes.items() if name == target_class_name)
            else:
                # Add new class
                new_idx = self.class_counter
                self.merged_classes[new_idx] = target_class_name
                self.class_counter += 1
            
            class_mapping[old_idx] = new_idx
        
        return class_mapping
    
    def _get_target_class_name(self, original_class_name: str) -> str:
        """
        Determine the target class name for mapping.
        Maps cyclist and pedestrian related classes to canonical names.
        
        Args:
            original_class_name: Original class name from source dataset
            
        Returns:
            Target class name for the merged dataset, or None if excluded by filtering
        """
        name_lower = original_class_name.lower()
        
        cyclist_aliases = {
            'cyclist', 'bike', 'bicycle', 'biker', 'cycling', 'cyclists', 'person_on_bike', 'person_on_bicycle'
        }
        pedestrian_aliases = {
            'pedestrian', 'pedestrians', 'person', 'persons', 'people', 'walker', 'walkers', 'human', 'humans', 'Persona', 'Person'
        }
        
        # Canonicalize known aliases
        if name_lower in cyclist_aliases:
            return 'cyclist'
        if name_lower in pedestrian_aliases:
            return 'pedestrian'
        
        # If filter is provided, only include classes that match the filter exactly after aliasing
        if self.filter_classes:
            normalized_filter: Set[str] = set(c.lower() for c in self.filter_classes)
            # If the original name is directly requested (no alias), keep as-is
            if name_lower in normalized_filter:
                return original_class_name
            # Otherwise, exclude by returning None
            return None
        
        # No filter: keep original class name
        return original_class_name
    
    def _copy_and_update_labels(self, src_labels_dir: Path, dst_labels_dir: Path, 
                               class_mapping: Dict[int, int], dataset_name: str):
        """Copy label files and update class indices, filtering out files without target classes."""
        if not src_labels_dir.exists():
            return
        
        copied_labels = 0
        filtered_labels = 0
        
        for label_file in src_labels_dir.glob("*.txt"):
            try:
                # Read and update label file
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                updated_lines = []
                has_target_classes = False
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # class_id x y w h
                        old_class_id = int(parts[0])
                        if old_class_id in class_mapping:
                            new_class_id = class_mapping[old_class_id]
                            parts[0] = str(new_class_id)
                            updated_lines.append(' '.join(parts) + '\n')
                            has_target_classes = True
                
                # Only copy the label file if it contains target classes
                if has_target_classes and updated_lines:
                    dst_label_file = dst_labels_dir / f"{dataset_name}_{label_file.name}"
                    with open(dst_label_file, 'w') as f:
                        f.writelines(updated_lines)
                    copied_labels += 1
                else:
                    filtered_labels += 1
                    
            except Exception as e:
                print(f"    Warning: Failed to process label file {label_file.name}: {e}")
                continue
        
        # Track statistics
        self.filtered_files[f"{dataset_name}_labels"] = filtered_labels
        return copied_labels
    
    def _copy_images(self, src_images_dir: Path, dst_images_dir: Path, 
                    dataset_name: str, file_hashes: Dict[str, str], 
                    valid_image_names: Set[str] = None) -> int:
        """Copy image files, handling duplicates and filtering by valid image names."""
        if not src_images_dir.exists():
            return 0
        
        copied_count = 0
        filtered_count = 0
        
        for img_file in src_images_dir.glob("*"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                try:
                    # If we have a list of valid image names, check if this image should be copied
                    if valid_image_names is not None:
                        # Remove the dataset prefix to match with original filename
                        base_name = img_file.stem
                        if base_name not in valid_image_names:
                            filtered_count += 1
                            continue
                    
                    # Calculate file hash to detect duplicates
                    file_hash = self._get_file_hash(img_file)
                    
                    if file_hash in file_hashes:
                        # Duplicate file found
                        self.duplicate_files.append({
                            'file': img_file.name,
                            'dataset': dataset_name,
                            'duplicate_of': file_hashes[file_hash]
                        })
                        continue
                    
                    # Copy file with dataset prefix
                    dst_img_file = dst_images_dir / f"{dataset_name}_{img_file.name}"
                    
                    # Ensure destination directory exists
                    dst_img_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(img_file, dst_img_file)
                    
                    file_hashes[file_hash] = f"{dataset_name}_{img_file.name}"
                    copied_count += 1
                    
                except Exception as e:
                    print(f"    Warning: Failed to copy {img_file.name}: {e}")
                    continue
        
        # Track filtered images
        if valid_image_names is not None:
            self.filtered_files[f"{dataset_name}_images"] = filtered_count
        
        return copied_count
    
    def _get_valid_image_names(self, src_labels_dir: Path, class_mapping: Dict[int, int]) -> Set[str]:
        """Get set of image names that have labels with target classes."""
        valid_names = set()
        
        if not src_labels_dir.exists():
            return valid_names
        
        for label_file in src_labels_dir.glob("*.txt"):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                has_target_classes = False
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # class_id x y w h
                        old_class_id = int(parts[0])
                        if old_class_id in class_mapping:
                            has_target_classes = True
                            break
                
                if has_target_classes:
                    # Add the base name (without extension) to valid names
                    valid_names.add(label_file.stem)
                    
            except Exception as e:
                print(f"    Warning: Failed to process label file {label_file.name}: {e}")
                continue
        
        return valid_names
    
    def combine_datasets(self):
        """Main method to combine all datasets."""
        print(f"Starting dataset combination...")
        print(f"Input directories: {[str(d) for d in self.input_dirs]}")
        print(f"Output directory: {self.output_dir}")
        
        if self.filter_classes:
            print(f"Filtering for classes: {self.filter_classes}")
        
        # Track file hashes to detect duplicates
        file_hashes = {}
        
        # Process each dataset
        for dataset_dir in self.input_dirs:
            dataset_name = dataset_dir.name
            print(f"\nProcessing dataset: {dataset_name}")
            
            # Load dataset information
            dataset_info = self._load_dataset_info(dataset_dir)
            
            # Merge classes
            dataset_classes = dataset_info.get('names', [])
            class_mapping = self._merge_classes(dataset_classes, dataset_name)
            
            print(f"  Classes: {dataset_classes}")
            print(f"  Class mapping: {class_mapping}")
            
            # Show class mapping details
            if class_mapping:
                print(f"  Class mapping details:")
                for old_idx, new_idx in class_mapping.items():
                    old_class = dataset_classes[old_idx]
                    new_class = self.merged_classes[new_idx]
                    print(f"    {old_class} (index {old_idx}) -> {new_class} (index {new_idx})")
            
            if not class_mapping:
                print(f"  Warning: No classes from {dataset_name} match the filter criteria")
                continue
            
            # Process each split (train, valid, test)
            for split in ["train", "valid", "test"]:
                split_dir = dataset_dir / split
                if not split_dir.exists():
                    continue
                
                print(f"  Processing {split} split...")
                
                # First, process labels to get valid image names
                src_labels_dir = split_dir / "labels"
                dst_labels_dir = self.output_dir / split / "labels"
                copied_labels = self._copy_and_update_labels(src_labels_dir, dst_labels_dir, 
                                                           class_mapping, dataset_name)
                
                # Get valid image names (images that have corresponding labels with target classes)
                valid_image_names = self._get_valid_image_names(src_labels_dir, class_mapping)
                
                # Copy images (only those with valid labels)
                src_images_dir = split_dir / "images"
                dst_images_dir = self.output_dir / split / "images"
                copied_images = self._copy_images(src_images_dir, dst_images_dir, 
                                                dataset_name, file_hashes, valid_image_names)
                
                self.file_stats[f"{dataset_name}_{split}"] = copied_images
                print(f"    Copied {copied_images} images, {copied_labels} labels")
                
                # Show filtering statistics
                if self.filter_classes:
                    filtered_images = self.filtered_files.get(f"{dataset_name}_images", 0)
                    filtered_labels = self.filtered_files.get(f"{dataset_name}_labels", 0)
                    if filtered_images > 0 or filtered_labels > 0:
                        print(f"    Filtered out: {filtered_images} images, {filtered_labels} labels")
        
        # Generate merged data.yaml
        self._generate_merged_yaml()
        
        # Print summary
        self._print_summary()
        
        # Visualize random samples if requested
        if self.show_visualization:
            self._visualize_random_samples()
    
    def _generate_merged_yaml(self):
        """Generate the merged data.yaml file."""
        # Sort classes by index to ensure consistent ordering
        sorted_classes = sorted(self.merged_classes.items())
        class_names = [name for _, name in sorted_classes]
        
        merged_data = {
            'train': '../train/images',
            'val': '../valid/images',
            'test': '../test/images',
            'nc': len(self.merged_classes),
            'names': class_names
        }
        
        # Add roboflow info if available
        merged_data['roboflow'] = {
            'workspace': 'merged_dataset',
            'project': 'combined_yolo_dataset',
            'version': 1,
            'license': 'CC BY 4.0',
            'url': 'https://universe.roboflow.com/merged_dataset/combined_yolo_dataset/dataset/1'
        }
        
        # Write merged data.yaml
        output_yaml_path = self.output_dir / "data.yaml"
        with open(output_yaml_path, 'w') as f:
            yaml.dump(merged_data, f, default_flow_style=False)
        
        print(f"\nGenerated merged data.yaml with {len(self.merged_classes)} classes")
        print(f"Classes: {class_names}")
        
        # Also create a dataset.yaml file (alternative naming convention)
        dataset_yaml_path = self.output_dir / "dataset.yaml"
        with open(dataset_yaml_path, 'w') as f:
            yaml.dump(merged_data, f, default_flow_style=False)
        
        print(f"Also created dataset.yaml file")
    
    def _visualize_random_samples(self, num_samples: int = 6):
        """Display random samples from the merged dataset with bounding boxes."""
        print(f"\n" + "="*60)
        print("VISUALIZING RANDOM SAMPLES FROM MERGED DATASET")
        print("="*60)
        
        # Collect all available images
        all_images = []
        for split in ["train", "valid", "test"]:
            images_dir = self.output_dir / split / "images"
            if images_dir.exists():
                for img_file in images_dir.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        all_images.append((split, img_file))
        
        if not all_images:
            print("No images found in the merged dataset!")
            return
        
        # Randomly sample images
        sample_images = random.sample(all_images, min(num_samples, len(all_images)))
        
        # Create subplot layout
        cols = 3
        rows = (len(sample_images) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for idx, (split, img_path) in enumerate(sample_images):
            if idx >= len(axes):
                break
                
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            # Load corresponding label file
            label_path = self.output_dir / split / "labels" / f"{img_path.stem}.txt"
            bboxes = []
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            bbox_width = float(parts[3])
                            bbox_height = float(parts[4])
                            
                            # Convert YOLO format to pixel coordinates
                            x1 = int((x_center - bbox_width / 2) * width)
                            y1 = int((y_center - bbox_height / 2) * height)
                            x2 = int((x_center + bbox_width / 2) * width)
                            y2 = int((y_center + bbox_height / 2) * height)
                            
                            bboxes.append((x1, y1, x2, y2, class_id))
            
            # Display image
            axes[idx].imshow(image)
            # Count classes for display
            class_counts = {}
            for _, _, _, _, cid in bboxes:
                cls_name = self.merged_classes.get(cid, f"class_{cid}")
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            counts_str = ", ".join(f"{v} {k}(s)" for k, v in class_counts.items()) if class_counts else "0 objects"
            axes[idx].set_title(f"{split}: {img_path.name}\n{counts_str}", fontsize=10)
            axes[idx].axis('off')
            
            # Draw bounding boxes
            for x1, y1, x2, y2, class_id in bboxes:
                rect = Rectangle((x1, y1), x2 - x1, y2 - y1, 
                               linewidth=2, edgecolor='red', facecolor='none')
                axes[idx].add_patch(rect)
                
                # Add class label
                class_name = self.merged_classes.get(class_id, f"class_{class_id}")
                axes[idx].text(x1, y1 - 5, class_name, color='red', fontsize=8, 
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        # Hide unused subplots
        for idx in range(len(sample_images), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f"Random Samples from Merged Dataset ({len(all_images)} total images)", 
                    fontsize=14, y=0.98)
        
        # Save the visualization
        viz_path = self.output_dir / "sample_visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"Sample visualization saved to: {viz_path}")
        
        # Display the plot
        plt.show()
        
        print(f"\nDisplayed {len(sample_images)} random samples from the merged dataset")
        print(f"Red boxes show detected object bounding boxes")
    
    def _print_summary(self):
        """Print a summary of the combination process."""
        print("\n" + "="*60)
        print("DATASET COMBINATION SUMMARY")
        print("="*60)
        
        print(f"Output directory: {self.output_dir}")
        print(f"Total classes: {len(self.merged_classes)}")
        print(f"Class names: {list(self.merged_classes.values())}")
        
        if self.filter_classes:
            print(f"Filtered for classes: {self.filter_classes}")
        
        print("\nFile statistics:")
        for key, count in self.file_stats.items():
            print(f"  {key}: {count} images")
        
        if self.filtered_files:
            print("\nFiltered files:")
            for key, count in self.filtered_files.items():
                print(f"  {key}: {count} files filtered out")
        
        if self.duplicate_files:
            print(f"\nDuplicate files found: {len(self.duplicate_files)}")
            print("Duplicates:")
            for dup in self.duplicate_files[:10]:  # Show first 10
                print(f"  {dup['file']} from {dup['dataset']} (duplicate of {dup['duplicate_of']})")
            if len(self.duplicate_files) > 10:
                print(f"  ... and {len(self.duplicate_files) - 10} more")
        
        print("\nDirectory structure:")
        for split in ["train", "valid", "test"]:
            split_dir = self.output_dir / split
            if split_dir.exists():
                images_count = len(list((split_dir / "images").glob("*")))
                labels_count = len(list((split_dir / "labels").glob("*.txt")))
                print(f"  {split}/: {images_count} images, {labels_count} labels")


def load_class_mapping(mapping_file: str) -> Dict:
    """Load class mapping from JSON file."""
    with open(mapping_file, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Combine multiple YOLO datasets into one")
    parser.add_argument("--input-dirs", nargs="+", required=True,
                       help="List of input dataset directories")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for merged dataset")
    parser.add_argument("--class-mapping", 
                       help="JSON file with class mapping rules")
    parser.add_argument("--filter-classes", nargs="+",
                       help="List of class names to include (filters out others)")
    
    args = parser.parse_args()
    
    # Load class mapping if provided
    class_mapping = None
    if args.class_mapping:
        class_mapping = load_class_mapping(args.class_mapping)
    
    # Create combiner and run
    combiner = YOLODatasetCombiner(args.input_dirs, args.output_dir, class_mapping, args.filter_classes)
    combiner.combine_datasets()


if __name__ == "__main__":
    main()
