#!/usr/bin/env python3
"""
Script to extract FoundationPose dataset zip files into a comprehensive structured format.
"""

import os
import zipfile
import argparse
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import re


def create_directory_structure(output_dir, dataset_name):
    """Create the complete directory structure for the dataset."""
    base_dirs = [
        f"images/{dataset_name}",
        f"masks/{dataset_name}",
        f"depth/{dataset_name}",
        f"camera_params/{dataset_name}",
        f"annotations/instance_mappings/{dataset_name}",
        f"annotations/bounding_box_paths/{dataset_name}",
        f"scene_data/{dataset_name}",
        f"metadata/{dataset_name}",
        f"bounding_boxes/{dataset_name}/loose",
    ]

    # GSO-specific directories
    if dataset_name == "gso":
        base_dirs.extend([
            f"occlusion/{dataset_name}",
            f"bounding_boxes/{dataset_name}/tight",
        ])

    for dir_path in base_dirs:
        (Path(output_dir) / dir_path).mkdir(parents=True, exist_ok=True)


def extract_and_organize_zip(args):
    """
    Extract a single zip file and organize into structured format.

    Args:
        args: Tuple of (zip_path, output_dir, dataset_name)
    """
    zip_path, output_dir, dataset_name = args

    try:
        zip_name = Path(zip_path).stem

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get all file paths in the zip
            file_list = zip_ref.namelist()

            # Organize files by type
            organized_files = organize_file_paths(file_list, zip_name, dataset_name)

            # Extract each file to its proper location
            for file_info in organized_files:
                extract_file_to_location(zip_ref, file_info, output_dir)

        return True, zip_path

    except Exception as e:
        return False, f"{zip_path}: {str(e)}"


def organize_file_paths(file_list, zip_name, dataset_name):
    """Organize file paths by data type and generate target paths."""
    organized = []

    for file_path in file_list:
        if file_path.endswith('/'):
            continue  # Skip directories

        # Parse the file path structure
        parts = file_path.split('/')
        if len(parts) < 2:
            continue

        if dataset_name == "gso":
            # GSO structure: object_id/scene_xxxx/scene-hash/[RenderProduct_xxx/]file
            object_id = parts[0]
            if len(parts) >= 3 and parts[1].startswith('scene_'):
                scene_num = parts[1]
                scene_hash = parts[2] if len(parts) > 2 else ""

                # Generate base name
                base_name = f"{object_id}_{scene_num}"

                # Handle files in RenderProduct folders
                if len(parts) >= 5 and parts[3].startswith('RenderProduct'):
                    camera_id = 0 if parts[3] == "RenderProduct_Replicator" else 1
                    full_name = f"{base_name}_cam_{camera_id}"

                    file_type = parts[4]  # e.g., 'rgb', 'depth', etc.
                    filename = parts[5] if len(parts) > 5 else ""

                    target_info = get_target_path_info(file_type, filename, full_name, dataset_name)
                    if target_info:
                        organized.append({
                            'source_path': file_path,
                            'target_path': target_info['path'],
                            'target_dir': target_info['dir']
                        })

                # Handle scene-level files
                elif len(parts) >= 3:
                    filename = parts[-1]
                    if filename == 'states.json':
                        organized.append({
                            'source_path': file_path,
                            'target_path': f"scene_data/{dataset_name}/{base_name}_states.json",
                            'target_dir': 'scene_data'
                        })
                    elif filename == 'scene.usd':
                        organized.append({
                            'source_path': file_path,
                            'target_path': f"scene_data/{dataset_name}/{base_name}_scene.usd",
                            'target_dir': 'scene_data'
                        })
                    elif filename == 'metadata.txt':
                        organized.append({
                            'source_path': file_path,
                            'target_path': f"metadata/{dataset_name}/{base_name}_{scene_hash}_metadata.txt",
                            'target_dir': 'metadata'
                        })
                    elif filename == 'render_stamp.yaml':
                        organized.append({
                            'source_path': file_path,
                            'target_path': f"metadata/{dataset_name}/{base_name}_{scene_hash}_render_stamp.yaml",
                            'target_dir': 'metadata'
                        })

        else:  # objaverse_path_tracing
            # Objaverse structure: zip_id/object_id/[scene-hash/][RenderProduct_xxx/]file
            zip_id = parts[0]
            if len(parts) >= 2:
                object_id = parts[1]

                # Check for drop_objects.yaml at object level
                if len(parts) == 3 and parts[2] == 'drop_objects.yaml':
                    organized.append({
                        'source_path': file_path,
                        'target_path': f"metadata/{dataset_name}/{zip_id}_{object_id}_drop_objects.yaml",
                        'target_dir': 'metadata'
                    })
                    continue

                if len(parts) >= 4 and parts[2].startswith('scene-'):
                    scene_hash = parts[2]
                    base_name = f"{zip_id}_{object_id}_{scene_hash}"

                    # Handle files in RenderProduct folders
                    if len(parts) >= 6 and parts[3].startswith('RenderProduct'):
                        camera_id = 0 if parts[3] == "RenderProduct_Replicator" else 1
                        full_name = f"{base_name}_cam_{camera_id}"

                        file_type = parts[4]
                        filename = parts[5] if len(parts) > 5 else ""

                        target_info = get_target_path_info(file_type, filename, full_name, dataset_name)
                        if target_info:
                            organized.append({
                                'source_path': file_path,
                                'target_path': target_info['path'],
                                'target_dir': target_info['dir']
                            })

                    # Handle scene-level files
                    elif len(parts) >= 4:
                        filename = parts[-1]
                        if filename == 'metadata.txt':
                            organized.append({
                                'source_path': file_path,
                                'target_path': f"metadata/{dataset_name}/{base_name}_metadata.txt",
                                'target_dir': 'metadata'
                            })
                        elif filename == 'render_stamp.yaml':
                            organized.append({
                                'source_path': file_path,
                                'target_path': f"metadata/{dataset_name}/{base_name}_render_stamp.yaml",
                                'target_dir': 'metadata'
                            })
                        elif filename == 'scene.usd':
                            organized.append({
                                'source_path': file_path,
                                'target_path': f"scene_data/{dataset_name}/{base_name}_scene.usd",
                                'target_dir': 'scene_data'
                            })

    return organized


def get_target_path_info(file_type, filename, base_name, dataset_name):
    """Get target path information based on file type."""

    if file_type == 'rgb' and filename.endswith('.png'):
        return {
            'path': f"images/{dataset_name}/{base_name}.png",
            'dir': 'images'
        }
    elif file_type == 'distance_to_image_plane' and filename.endswith('.npy'):
        return {
            'path': f"depth/{dataset_name}/{base_name}.npy",
            'dir': 'depth'
        }
    elif file_type == 'instance_segmentation':
        if filename.endswith('.png'):
            return {
                'path': f"masks/{dataset_name}/{base_name}.png",
                'dir': 'masks'
            }
        elif 'mapping' in filename and filename.endswith('.json'):
            suffix = '_semantics' if 'semantics' in filename else '_mapping'
            return {
                'path': f"annotations/instance_mappings/{dataset_name}/{base_name}{suffix}.json",
                'dir': 'annotations'
            }
    elif file_type == 'camera_params' and filename.endswith('.json'):
        return {
            'path': f"camera_params/{dataset_name}/{base_name}.json",
            'dir': 'camera_params'
        }
    elif file_type == 'occlusion' and filename.endswith('.npy') and dataset_name == 'gso':
        return {
            'path': f"occlusion/{dataset_name}/{base_name}.npy",
            'dir': 'occlusion'
        }
    elif file_type == 'bounding_box_2d_loose':
        if filename.endswith('.npy'):
            return {
                'path': f"bounding_boxes/{dataset_name}/loose/{base_name}.npy",
                'dir': 'bounding_boxes'
            }
        elif 'labels' in filename and filename.endswith('.json'):
            return {
                'path': f"annotations/bounding_box_paths/{dataset_name}/{base_name}_labels.json",
                'dir': 'annotations'
            }
        elif 'prim_paths' in filename and filename.endswith('.json'):
            return {
                'path': f"annotations/bounding_box_paths/{dataset_name}/{base_name}_paths.json",
                'dir': 'annotations'
            }
    elif file_type == 'bounding_box_2d_tight' and dataset_name == 'gso':
        if filename.endswith('.npy'):
            return {
                'path': f"bounding_boxes/{dataset_name}/tight/{base_name}.npy",
                'dir': 'bounding_boxes'
            }
        elif 'labels' in filename and filename.endswith('.json'):
            return {
                'path': f"annotations/bounding_box_paths/{dataset_name}/{base_name}_tight_labels.json",
                'dir': 'annotations'
            }
        elif 'prim_paths' in filename and filename.endswith('.json'):
            return {
                'path': f"annotations/bounding_box_paths/{dataset_name}/{base_name}_tight_paths.json",
                'dir': 'annotations'
            }

    return None


def extract_file_to_location(zip_ref, file_info, output_dir):
    """Extract a file from zip to its target location."""
    source_path = file_info['source_path']
    target_path = Path(output_dir) / file_info['target_path']

    # Create target directory if it doesn't exist
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract the file
    with zip_ref.open(source_path) as source_file:
        with open(target_path, 'wb') as target_file:
            shutil.copyfileobj(source_file, target_file)


def process_dataset(source_dir, output_dir, dataset_type, num_workers=4):
    """Process all zip files for a dataset type."""

    # Map dataset type to directory name
    dataset_name = "gso" if dataset_type == "gso" else "objaverse"
    zip_dir = Path(source_dir) / dataset_type

    # Create directory structure
    create_directory_structure(output_dir, dataset_name)

    # Get all zip files
    zip_files = sorted(zip_dir.glob("*.zip"))

    if not zip_files:
        print(f"No zip files found in {zip_dir}")
        return

    print(f"Found {len(zip_files)} zip files in {dataset_type}")
    print(f"Organizing to structured format in: {output_dir}")

    # Prepare arguments for parallel processing
    args_list = [(str(zip_file), str(output_dir), dataset_name) for zip_file in zip_files]

    # Process with progress bar
    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(extract_and_organize_zip, args_list),
                total=len(zip_files),
                desc=f"Processing {dataset_type}"
            ))
    else:
        results = [extract_and_organize_zip(args) for args in tqdm(args_list, desc=f"Processing {dataset_type}")]

    # Report results
    successes = sum(1 for success, _ in results if success)
    failures = [(path, msg) for success, msg in results if not success]

    print(f"\nProcessing complete for {dataset_type}:")
    print(f"  Success: {successes}/{len(zip_files)}")

    if failures:
        print(f"  Failed: {len(failures)}")
        for path, msg in failures:
            print(f"    - {msg}")


def create_dataset_info(output_dir):
    """Create dataset info file with statistics."""
    info = {
        "dataset": "FoundationPose",
        "description": "6D Object Pose Estimation Dataset",
        "structure_version": "1.0",
        "datasets": {
            "gso": {
                "source": "Google Scanned Objects",
                "features": ["rgb", "depth", "2d_bbox_loose", "2d_bbox_tight", "instance_masks", "6d_pose", "occlusion"]
            },
            "objaverse": {
                "source": "Objaverse Path Tracing",
                "features": ["rgb", "depth", "2d_bbox_loose", "instance_masks", "6d_pose"]
            }
        },
        "image_resolution": [640, 480],
        "cameras_per_scene": 2
    }

    info_path = Path(output_dir) / "metadata" / "dataset_info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)

    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Extract FoundationPose dataset into comprehensive structured format"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source directory containing gso and objaverse_path_tracing folders"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for structured data"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gso", "objaverse_path_tracing", "both"],
        default="both",
        help="Which dataset to extract"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )

    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    if not source_dir.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return

    print(f"FoundationPose Dataset Extraction to Structured Format")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.workers}")
    print("-" * 70)

    # Create base output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ["gso", "both"]:
        process_dataset(source_dir, output_dir, "gso", num_workers=args.workers)

    if args.dataset in ["objaverse_path_tracing", "both"]:
        process_dataset(source_dir, output_dir, "objaverse_path_tracing", num_workers=args.workers)

    # Create dataset info file
    create_dataset_info(output_dir)

    print(f"\n🎉 All extractions complete!")
    print(f"📁 Structured dataset available at: {output_dir}")
    print(f"📊 Dataset info: {output_dir}/metadata/dataset_info.json")


if __name__ == "__main__":
    main()
