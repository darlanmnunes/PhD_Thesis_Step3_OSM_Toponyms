# convert_SvtDataset.py

"""
What does this script do?
- Converts the SVT XML files to YOLO format.
- Keeps 10% of the original training set as the new test set.
- Splits the original test set into 70% for training and 20% for validation.
- Organises all the files into separate folders for training, validation and testing.
"""

import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import cv2
import shutil
import random

# === 1. Conversion of labels to YOLO ===
def convert_to_yolo_format(xml_file, img_dir, output_img_dir, output_label_dir):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    print(f"Output directories created: {output_img_dir}, {output_label_dir}")

    tree = ET.parse(xml_file)
    root = tree.getroot()

    images = []

    for image in tqdm(root.findall('image'), desc=f"Converting {os.path.basename(xml_file)}", unit="image"):
        img_name = os.path.basename(image.find('imageName').text)
        img_path = os.path.join(img_dir, 'img', img_name)

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {img_path}")
            continue

        # Save the converted image
        img_output_path = os.path.join(output_img_dir, img_name)
        if not os.path.exists(img_output_path):
            cv2.imwrite(img_output_path, img)

        # Create labels file to YOLO
        image_height, image_width = img.shape[:2]
        annotation_file = os.path.join(output_label_dir, img_name.replace('.jpg', '.txt'))

        with open(annotation_file, 'w') as f:
            for tag in image.findall('.//taggedRectangle'):
                class_id = 0
                x_min, y_min = float(tag.get('x')), float(tag.get('y'))
                width, height = float(tag.get('width')), float(tag.get('height'))

                # Conversion to YOLO format
                x_center = (x_min + width / 2) / image_width
                y_center = (y_min + height / 2) / image_height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width/image_width:.6f} {height/image_height:.6f}\n")

        images.append(img_output_path)

    return images

# === 2. Split into Training, Validation and Testing Datasets ===
def split_dataset(all_images, label_dir, output_dir='svt_dataset/svt2yolo_final'):
    random.shuffle(all_images)

    total_images = len(all_images)
    n_train, n_val = int(0.70 * total_images), int(0.20 * total_images)
    n_test = total_images - n_train - n_val

    final_train = all_images[:n_train]
    final_val = all_images[n_train:n_train + n_val]
    final_test = all_images[n_train + n_val:]

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    def copy_data(images, split):
        for img_path in tqdm(images, desc=f"Copying {split} data", unit="file"):
            img_name = os.path.basename(img_path)
            label_name = img_name.replace('.jpg', '.txt')

            dest_img = os.path.join(output_dir, split, 'images', img_name)
            dest_label = os.path.join(output_dir, split, 'labels', label_name)

            shutil.copy2(img_path, dest_img)
            label_path = os.path.join(label_dir, label_name)
            if os.path.exists(label_path):
                shutil.copy2(label_path, dest_label)

    copy_data(final_train, 'train')
    copy_data(final_val, 'val')
    copy_data(final_test, 'test')

    return {'train': len(final_train), 'val': len(final_val), 'test': len(final_test)}

# === 3. Main function ===
if __name__ == "__main__":
    img_dir = 'svt_dataset/svt11'
    train_xml = 'svt_dataset/svt11/train.xml'
    test_xml = 'svt_dataset/svt11/test.xml'

    temp_img_dir = 'svt_dataset/temp/images'
    temp_label_dir = 'svt_dataset/temp/labels'

    # 1️- Dataset conversion to YOLO format
    train_images = convert_to_yolo_format(train_xml, img_dir, temp_img_dir, temp_label_dir)
    test_images = convert_to_yolo_format(test_xml, img_dir, temp_img_dir, temp_label_dir)

    # 2️ - Combining the Datasets and Final Division
    all_images = train_images + test_images
    results = split_dataset(all_images, temp_label_dir)

    # 3️ - Final resume
    print("\nDataset reorganisation completed successfully!")
    print(f"- Training set: {results['train']} images")
    print(f"- Validation set: {results['val']} images")
    print(f"- Test set: {results['test']} images")