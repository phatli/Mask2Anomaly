import os
import json
import shutil
import argparse

def convert_and_move_files(source_dir, dataset_dir, new_subset_name=None):
    # Create destination directories if they don't exist (they will be created once the original_subset_name is known)
    
    # Process each file in the source directory
    for file in os.listdir(source_dir):
        if file.endswith("_leftImg8bit.png"):
            # Extract original_subset_name from the file name
            original_subset_name = file.split('_')[0]
            dest_images_dir = os.path.join(dataset_dir, "leftImg8bit", "train", new_subset_name or original_subset_name)
            dest_labels_dir = os.path.join(dataset_dir, "gtFine", "train", new_subset_name or original_subset_name)
            os.makedirs(dest_images_dir, exist_ok=True)
            os.makedirs(dest_labels_dir, exist_ok=True)
            
            # Construct new file name if subset name is changed
            new_file_name = file if not new_subset_name else file.replace(original_subset_name, new_subset_name)
            source_file_path = os.path.join(source_dir, file)
            dest_file_path = os.path.join(dest_images_dir, new_file_name)

            # Move the image
            shutil.copy(source_file_path, dest_file_path)
            print(f"Moved image: {source_file_path} to {dest_file_path}")

            # Corresponding label file operations
            label_file = file.replace(".png", ".json")
            new_label_file = new_file_name.replace("_leftImg8bit.png", "_gtFine_polygons.json")
            with open(os.path.join(source_dir, label_file), 'r') as f:
                label_data = json.load(f)

            # Convert label format
            new_label_data = {
                "imgHeight": label_data["imageHeight"],
                "imgWidth": label_data["imageWidth"],
                "objects": [{"label": obj["label"], "polygon": obj["points"]} for obj in label_data["shapes"]]
            }

            # Write the new label data
            with open(os.path.join(dest_labels_dir, new_label_file), 'w') as f:
                json.dump(new_label_data, f, indent=4)
            print(f"Converted and moved label file: {label_file} to {new_label_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert and move semantic segmentation images and labels.')
    parser.add_argument('source_dir', type=str, help='Source directory containing images and labels.')
    parser.add_argument('dataset_dir', type=str, help='Destination dataset directory.')
    parser.add_argument('--new_subset_name', type=str, default=None, help='New name for the training subset (optional).')

    args = parser.parse_args()

    convert_and_move_files(args.source_dir, args.dataset_dir, args.new_subset_name)

if __name__ == "__main__":
    main()
