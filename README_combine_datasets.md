# YOLO Dataset Combiner

A Python script to combine multiple YOLO format datasets into a single merged dataset with optional class filtering.

## Features

- **Automatic Class Merging**: Intelligently merges class names from different datasets
- **Class Filtering**: Filter datasets to include only specific classes
- **Alias Normalization**: Maps common synonyms to canonical classes (e.g., person/pedestrian → pedestrian; bike/bicycle/biker → cyclist)
- **Duplicate Detection**: Detects and reports duplicate images using MD5 hashing
- **File Organization**: Maintains proper YOLO directory structure
- **Label Updates**: Automatically updates class indices in label files
- **Comprehensive Reporting**: Provides detailed statistics and summary

## Requirements

- Python 3.6+
- PyYAML
- Standard library modules (os, shutil, pathlib, hashlib, etc.)

## Installation

```bash
pip install PyYAML
```

## Usage

### Basic Usage

```bash
python combine_datasets.py --input-dirs dataset1 dataset2 dataset3 --output-dir merged_dataset
```

### With Class Mapping

```bash
python combine_datasets.py --input-dirs dataset1 dataset2 --output-dir merged_dataset --class-mapping mapping.json
```

### With Class Filtering

```bash
python combine_datasets.py --input-dirs dataset1 dataset2 --output-dir merged_dataset --filter-classes cyclist pedestrian
```

### Combined Usage

```bash
python combine_datasets.py --input-dirs dataset1 dataset2 --output-dir merged_dataset --filter-classes cyclist --class-mapping mapping.json
```

### Example

```bash
# Combine your existing datasets
python combine_datasets.py --input-dirs cyclist_training_data Dataset2_cyclist_detection --output-dir Combined_Dataset

# Filter for only cyclist and pedestrian (two-class dataset)
# Synonyms are supported automatically (e.g., person/persons/people → pedestrian)
python combine_datasets.py --input-dirs cyclist_training_data Dataset2_cyclist_detection --output-dir Cyclist_Pedestrian_Dataset --filter-classes cyclist pedestrian
```

## Input Dataset Structure

Each input dataset should follow the standard YOLO format:

```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Output Structure

The merged dataset will have the same structure:

```
merged_dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Class Filtering

Use the `--filter-classes` flag to include only specific classes in the merged dataset:

```bash
# Two-class: cyclist and pedestrian (person/persons/people are normalized to pedestrian)
python combine_datasets.py --input-dirs dataset1 dataset2 --output-dir filtered_dataset --filter-classes cyclist pedestrian

# If you truly want a single numeric class that already matches your sources
python combine_datasets.py --input-dirs dataset1 dataset2 --output-dir filtered_dataset --filter-classes 0
```

### How Class Filtering Works

1. **Label Filtering**: Only label files containing the specified classes are copied
2. **Image Filtering**: Only images with corresponding valid label files are copied
3. **Class Mapping & Aliases**: Class indices are updated and common synonyms are canonicalized (e.g., person → pedestrian, bicycle → cyclist)
4. **Statistics**: Reports how many files were filtered out

## Class Mapping (Optional)

Create a JSON file to specify how classes should be mapped:

```json
{
  "dataset1": {
    "old_class_name": "new_class_name",
    "cyclist": "cyclist",
    "bike": "cyclist"
  },
  "dataset2": {
    "0": "cyclist",
    "1": "person"
  }
}
```

## Features Explained

### Class Merging
- Automatically detects duplicate class names across datasets
- Assigns consistent class indices in the merged dataset
- Updates all label files with new class indices

### Duplicate Detection
- Uses MD5 hashing to detect identical image files
- Reports duplicates but doesn't copy them twice
- Maintains a log of all duplicate files found

### File Naming
- Prefixes all files with the source dataset name
- Prevents filename conflicts between datasets
- Maintains original file extensions

### Data.yaml Generation
- Creates a unified data.yaml file
- Combines all class names from input datasets
- Sets correct paths for train/valid/test splits

## Example Output

### Basic Combination

```
Starting dataset combination...
Input directories: ['cyclist_training_data', 'Dataset2_cyclist_detection']
Output directory: Combined_Dataset

Processing dataset: cyclist_training_data
  Classes: ['cyclist', 'e-scooter', 'person', 'vehicles']
  Class mapping: {0: 0, 1: 1, 2: 2, 3: 3}
  Processing train split...
    Copied 1234 images, 1234 labels
  Processing valid split...
    Copied 123 images, 123 labels

Processing dataset: Dataset2_cyclist_detection
  Classes: ['0']
  Class mapping: {0: 4}
  Processing train split...
    Copied 25284 images, 25284 labels
  Processing valid split...
    Copied 2432 images, 2432 labels

Generated merged data.yaml with 5 classes

============================================================
DATASET COMBINATION SUMMARY
============================================================
Output directory: Combined_Dataset
Total classes: 5
Class names: ['cyclist', 'e-scooter', 'person', 'vehicles', '0']

File statistics:
  cyclist_training_data_train: 1234 images
  cyclist_training_data_valid: 123 images
  Dataset2_cyclist_detection_train: 25284 images
  Dataset2_cyclist_detection_valid: 2432 images

Duplicate files found: 0

Directory structure:
  train/: 26518 images, 26518 labels
  valid/: 2555 images, 2555 labels
```

### With Class Filtering

```
Starting dataset combination...
Input directories: ['cyclist_training_data', 'Dataset2_cyclist_detection']
Output directory: Cyclist_Only_Dataset
Filtering for classes: ['cyclist', '0']

Processing dataset: cyclist_training_data
  Classes: ['cyclist', 'e-scooter', 'person', 'vehicles']
  Class mapping: {0: 0}
  Processing train split...
    Copied 4335 images, 4335 labels
    Filtered out: 5703 images, 5703 labels
  Processing valid split...
    Copied 538 images, 538 labels
    Filtered out: 729 images, 729 labels

Processing dataset: Dataset2_cyclist_detection
  Classes: ['0']
  Class mapping: {0: 1}
  Processing train split...
    Copied 25254 images, 25260 labels
    Filtered out: 24 images, 24 labels

Generated merged data.yaml with 2 classes

============================================================
DATASET COMBINATION SUMMARY
============================================================
Output directory: Cyclist_Only_Dataset
Total classes: 2
Class names: ['cyclist', '0']
Filtered for classes: ['cyclist', '0']

File statistics:
  cyclist_training_data_train: 4335 images
  cyclist_training_data_valid: 538 images
  Dataset2_cyclist_detection_train: 25254 images
  Dataset2_cyclist_detection_valid: 2432 images

Filtered files:
  cyclist_training_data_labels: 121 files filtered out
  cyclist_training_data_images: 121 files filtered out
  Dataset2_cyclist_detection_labels: 0 files filtered out
  Dataset2_cyclist_detection_images: 0 files filtered out

Directory structure:
  train/: 29589 images, 29595 labels
  valid/: 2970 images, 2970 labels
```

## Error Handling

The script includes comprehensive error handling for:
- Missing input directories
- Invalid data.yaml files
- Missing train/valid/test splits
- File permission issues
- Invalid class mappings

## Running the Example

```bash
python example_combine_datasets.py
```

This will demonstrate the basic usage and create example outputs.
