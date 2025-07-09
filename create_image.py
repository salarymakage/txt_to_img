import os
import cv2
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from khmernltk import word_tokenize
import re
import uuid # Import the uuid module
from sklearn.model_selection import train_test_split # Import for train/val split

# ========== CONFIGURATION ==========
# Define the folder containing your text files
text_input_folder = "more_important" # Make sure this folder exists and contains your .txt files
output_root_folder = "V4_genimage" # Root folder for all generated data
train_output_folder = os.path.join(output_root_folder, "train") # Folder for training images
val_output_folder = os.path.join(output_root_folder, "val")   # Folder for validation images
labels_folder = os.path.join(output_root_folder, "labels") # Folder for label files (CSV, TXT, Dict)

# Create necessary directories if they don't exist
os.makedirs(train_output_folder, exist_ok=True)
os.makedirs(val_output_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)
os.makedirs(text_input_folder, exist_ok=True) # Ensure input text folder exists

# List of font paths (UPDATE THIS WITH YOUR ACTUAL FONT FILES)
# Ensure these font files exist in the specified "Font" directory
font_paths = [
    "Font/khmerOS.ttf",
    "Font/Battambang-Regular.ttf",
    "Font/Moul-Regular.ttf",
    # Add more Khmer font paths here if you have them   
]

# --- Font Validation (Added from previous suggestion) ---
validated_font_paths = []
for f_path in font_paths:
    try:
        ImageFont.truetype(f_path, 10) # Test load with a small size
        validated_font_paths.append(f_path)
    except Exception as e:
        print(f"Warning: Could not load font '{f_path}'. Skipping. Error: {e}")

font_paths = validated_font_paths
if not font_paths:
    print("Error: No valid font paths found after validation. Please check your 'Font' directory and file names.")
    exit()
# --- End Font Validation ---

font_size_range = (28, 36) # Range for random font sizes
image_size = (600, 80) # Width, Height of the generated images

# ========== FUNCTION TO READ TEXT FROM MULTIPLE FILES ==========
def read_khmer_texts_from_folder(folder_path):
    """
    Reads all text content from .txt files within a specified folder.
    """
    all_khmer_text = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_khmer_text.append(f.read())
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    return "\n".join(all_khmer_text)

# ========== Read all Khmer text from the specified folder ==========
print(f"Reading text from: {text_input_folder}")
khmer_text = read_khmer_texts_from_folder(text_input_folder)

if not khmer_text.strip():
    print(f"Error: No Khmer text found in '{text_input_folder}'. Please add .txt files with content to this folder.")
    exit() # Exit if no text is found

# ========== WORD TOKENIZE WITH KHMERNLTK ==========
print("Tokenizing Khmer text...")
raw_tokens = word_tokenize(khmer_text, return_tokens=True)
cleaned_tokens = [t for t in raw_tokens if t.strip() != '']
print(f"Total cleaned tokens after tokenization: {len(cleaned_tokens)}")

# ========== DYNAMIC SPLIT FUNCTION ==========
def split_khmer_paragraph_dynamic(tokens, min_words=3, max_words=8):
    """
    Splits a list of tokens into chunks of random length (words), forming lines.
    """
    chunks = []
    i = 0
    while i < len(tokens):
        group_size = random.randint(min_words, max_words)
        chunk = tokens[i:i + group_size]
        if chunk:
            chunks.append(' '.join(chunk))
        i += group_size
    return chunks

# ========== SPLIT TO LINES USING DYNAMIC FUNCTION ==========
lines = split_khmer_paragraph_dynamic(cleaned_tokens, min_words=3, max_words=4)
print(f"Generated {len(lines)} lines for image generation with dynamic lengths (3-4 words per line).")

# The sanitize_filename_text function is no longer strictly needed for the main part of the filename
# as we are using UUIDs, but keeping it as a placeholder if future adjustments require it.
def sanitize_text_for_label(text):
    """
    Sanitizes text to remove unwanted characters for labels.
    """
    # Remove characters that are not alphanumeric, Khmer Unicode, or spaces
    sanitized_text = re.sub(r'[^\w\s\u1780-\u17FF]', '', text)
    sanitized_text = re.sub(r'\s+', ' ', sanitized_text).strip() # Replace multiple spaces with single space
    return sanitized_text

# ========== GENERATE IMAGES FUNCTION (Modified to accept output_dir) ==========
def generate_text_image(text, filename, output_dir, variant="clean"):
    """
    Generates an image from text with specified filename and applies chosen variant effects.
    """
    # Create a new RGB image with white background
    img = Image.new('RGB', image_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Randomly select a font and font size
    selected_font_path = random.choice(font_paths) # Use validated_font_paths now
    selected_font_size = random.randint(font_size_range[0], font_size_range[1])

    try:
        font = ImageFont.truetype(selected_font_path, selected_font_size)
    except Exception as e:
        print(f"Error loading font '{selected_font_path}' with size {selected_font_size}. "
              f"Falling back to first valid font. Error: {e}")
        font = ImageFont.truetype(font_paths[0], font_size_range[0])


    # Calculate text bounding box to center the text
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # Handle text overflow (new logic from previous suggestion)
    original_w = w
    original_font_size = selected_font_size
    if w > image_size[0]:
        # Attempt to reduce font size to fit
        while w > image_size[0] and selected_font_size > 10: # Don't go below 10px
            selected_font_size -= 1
            font = ImageFont.truetype(selected_font_path, selected_font_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
        if selected_font_size <= 10 and w > image_size[0]:
            print(f"Warning: Text '{text}' (original w={original_w}) still overflows image even after reducing font size to {selected_font_size}. Skipping this line.")
            return False # Indicate that image was not generated

    # Calculate position to center the text (recalculate if font size changed)
    x = (image_size[0] - w) / 2
    y = (image_size[1] - h) / 2

    # Draw the text on the image
    draw.text((x, y), text, fill=(0, 0, 0), font=font) # Black text

    # Define the full path where the image will be saved
    img_path = os.path.join(output_dir, filename)

    # Apply image processing variants and save
    if variant == "clean":
        img.save(img_path)
    elif variant == "blurred":
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) # Convert PIL Image to OpenCV format
        img_cv = cv2.GaussianBlur(img_cv, (5,5), 1) # Apply Gaussian blur
        cv2.imwrite(img_path, img_cv)
    elif variant == "noisy":
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        noise = np.random.randint(0, 50, (img_cv.shape[0], img_cv.shape[1], 3), dtype='uint8') # Generate random noise
        img_cv = cv2.add(img_cv, noise) # Add noise to image
        cv2.imwrite(img_path, img_cv)
    elif variant == "noise_blur":
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        noise = np.random.randint(0, 50, (img_cv.shape[0], img_cv.shape[1], 3), dtype='uint8')
        img_cv = cv2.add(img_cv, noise)
        img_cv = cv2.GaussianBlur(img_cv, (5,5), 1)
        cv2.imwrite(img_path, img_cv)
    else:
        print(f"Warning: Unknown variant '{variant}'. Saving as clean image.")
        img.save(img_path)

    return True # Indicate that image was successfully generated


# ========== IMAGE GENERATION LOOP (Updated to prepare for split) ==========
print("Starting initial image data collection...")
all_generated_data = [] # List to store data for the CSV file BEFORE splitting
variants = ['clean', 'blurred', 'noisy', 'noise_blur'] # Types of image variations to generate

for idx, line in enumerate(lines):
    sanitized_line = sanitize_text_for_label(line) # Sanitize the text before generation and using as label
    if not sanitized_line: # Skip if line becomes empty after sanitization
        continue

    # Generate a unique ID for each line of text.
    unique_id = uuid.uuid4().hex # Generates a random 32-character hexadecimal string

    for variant in variants:
        filename = f"khmer_img_{unique_id}_{variant}.png" # Filename for both train/val

        # Append details to the data list for CSV creation.
        # We don't generate images here yet, just collect metadata
        all_generated_data.append({
            'filename': filename,
            'label': sanitized_line,
            'variant': variant
        })

print(f"Collected metadata for {len(all_generated_data)} potential images.")

# ========== SAVE LABELS TO CSV (Initial, full dataset) ==========
df_full = pd.DataFrame(all_generated_data)
labels_csv_path = os.path.join(labels_folder, "labels_full_dataset.csv") # Save in the new labels_folder
df_full.to_csv(labels_csv_path, index=False, encoding='utf-8-sig')
print(f"Full dataset CSV labels saved to: {labels_csv_path}")


# ========== SPLIT DATA AND GENERATE PADDLEOCR-COMPATIBLE TXT FILES ==========
print("\nSplitting data into training and validation sets for PaddleOCR...")

# Use the full dataframe for splitting
df_rec = df_full

# Calculate number of unique labels for stratification
num_unique_labels = len(df_rec['label'].unique())
total_samples = len(df_rec)

print(f"Total samples for recognition training (before splitting): {total_samples}")
print(f"Number of unique text labels: {num_unique_labels}")

# Determine appropriate test_size for train/val split
current_test_size_ratio = 0.15 # Your desired validation set ratio (15%)

# Calculate the minimum test_size ratio required to ensure each unique label can appear in the test set
min_test_samples_needed = num_unique_labels # At least one sample per unique label for stratification
min_test_size_ratio_required = min_test_samples_needed / total_samples if total_samples > 0 else 0

actual_test_size = max(current_test_size_ratio, min_test_size_ratio_required)

# Adjust actual_test_size if it's too close to 1.0, which means an empty train set
if actual_test_size >= 0.9:
    actual_test_size = 0.8 # A sensible maximum for test set, leaving at least 20% for train.

# Perform the train-test split
if num_unique_labels > 1 and total_samples >= num_unique_labels:
    try:
        train_df, val_df = train_test_split(df_rec, test_size=actual_test_size, random_state=42, stratify=df_rec['label'])
        print(f"Stratified split applied. Test size ratio: {actual_test_size:.2f}")
    except ValueError as e:
        print(f"Warning: Stratification failed ({e}). Falling back to non-stratified split.")
        train_df, val_df = train_test_split(df_rec, test_size=actual_test_size, random_state=42, stratify=None)
else:
    print("Only one unique label or insufficient data for stratification. Stratification disabled.")
    train_df, val_df = train_test_split(df_rec, test_size=actual_test_size, random_state=42, stratify=None)

print(f"Train set size: {len(train_df)} samples")
print(f"Validation set size: {len(val_df)} samples")

# ========== GENERATE IMAGES FOR TRAIN SET ==========
print(f"\nGenerating images for TRAINING set in '{train_output_folder}'...")
generated_train_count = 0
for index, row in train_df.iterrows():
    success = generate_text_image(row['label'], row['filename'], train_output_folder, row['variant'])
    if success:
        generated_train_count += 1
print(f"Generated {generated_train_count} images for training.")

# ========== GENERATE IMAGES FOR VAL SET ==========
print(f"\nGenerating images for VALIDATION set in '{val_output_folder}'...")
generated_val_count = 0
for index, row in val_df.iterrows():
    success = generate_text_image(row['label'], row['filename'], val_output_folder, row['variant'])
    if success:
        generated_val_count += 1
print(f"Generated {generated_val_count} images for validation.")


# Define the paths for the PaddleOCR-compatible label files
train_rec_path = os.path.join(labels_folder, "train_rec.txt") # Save in labels_folder
val_rec_path = os.path.join(labels_folder, "val_rec.txt")     # Save in labels_folder

# Write training labels to train_rec.txt
print(f"\nWriting training labels to {train_rec_path}...")
with open(train_rec_path, 'w', encoding='utf-8') as f:
    for index, row in train_df.iterrows():
        # Path relative to the output_root_folder (e.g., train/khmer_img_...png)
        image_relative_path = os.path.join(os.path.basename(train_output_folder), row['filename'])
        f.write(f"{image_relative_path}\t{row['label']}\n")
print(f"Generated {len(train_df)} training samples entries.")

# Write validation labels to val_rec.txt
print(f"Writing validation labels to {val_rec_path}...")
with open(val_rec_path, 'w', encoding='utf-8') as f:
    for index, row in val_df.iterrows():
        # Path relative to the output_root_folder (e.g., val/khmer_img_...png)
        image_relative_path = os.path.join(os.path.basename(val_output_folder), row['filename'])
        f.write(f"{image_relative_path}\t{row['label']}\n")
print(f"Generated {len(val_df)} validation samples entries.")

# ========== GENERATE CHARACTER DICTIONARY FILE ==========
print("\nGenerating character dictionary...")
# Get all unique characters from all labels in the recognition dataset
char_set = sorted(list(set(''.join(df_rec['label'].tolist()))))
dict_path = os.path.join(labels_folder, "khmer_char_dict.txt") # Save in labels_folder
with open(dict_path, 'w', encoding='utf-8') as f:
    for char in char_set:
        f.write(char + '\n')
print(f"Generated character dictionary with {len(char_set)} unique characters in {dict_path}")
print("\n All operations completed successfully!")