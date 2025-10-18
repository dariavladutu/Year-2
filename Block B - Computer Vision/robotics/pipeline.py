`import numpy as np
import time
import os
import cv2
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from patchify import patchify, unpatchify
from skimage.morphology import skeletonize

from ot2_gym_wrapper_2 import OT2Env 
from pid_controller import PIDController
from sim_class import Simulation

# --- Helper Functions ---
def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives + K.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives + K.epsilon())
        return precision
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def padder(image, patch_size):
    h, w = image.shape[:2]
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w
    top_padding = int(height_padding / 2)
    bottom_padding = height_padding - top_padding
    left_padding = int(width_padding / 2)
    right_padding = width_padding - left_padding
    padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=0)
    return padded_image, (top_padding, bottom_padding, left_padding, right_padding)

def unpadder(image, padding):
    top_padding, bottom_padding, left_padding, right_padding = padding
    return image[top_padding:image.shape[0] - bottom_padding, left_padding:image.shape[1] - right_padding]

def cropper(image):
    original_shape = image.shape
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, {"original_shape": original_shape, "used_crop": False}
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    padding = 10
    x += padding
    y += padding
    w -= 2 * padding
    h -= 2 * padding
    size = max(w, h)
    x_center, y_center = x + w // 2, y + h // 2
    x_start = max(0, x_center - size // 2)
    y_start = max(0, y_center - size // 2)
    cropped = image[y_start:y_start + size, x_start:x_start + size]
    crop_info = {"original_shape": original_shape, "used_crop": True, "x_start": x_start, "y_start": y_start, "crop_size": size}
    return cropped, crop_info

def uncropper(cropped_img, crop_info):
    original_shape = crop_info["original_shape"]
    canvas = np.zeros(original_shape[:2], dtype=cropped_img.dtype)
    if not crop_info.get("used_crop", False):
        return cropped_img
    x_start, y_start, size = crop_info["x_start"], crop_info["y_start"], crop_info["crop_size"]
    canvas[y_start:y_start + size, x_start:x_start + size] = cropped_img
    return canvas

# --- Main CV Pipeline Function ---
def run_cv_pipeline(image_path, model, patch_size):
    """
    Runs the full computer vision pipeline on a given image.
    """
    print("CV Pipeline: Processing image...")
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        print(f"Could not read image: {image_path}. Skipping.")
        return []

    # Crop, pad, and patchify the image
    cropped_gray, crop_info = cropper(original_image)
    padded_image, padding_info = padder(cropped_gray, patch_size)
    patches = patchify(padded_image, (patch_size, patch_size), step=patch_size)
    
    num_rows, num_cols, h, w = patches.shape
    model_input = patches.reshape(-1, h, w)
    model_input = np.expand_dims(model_input, axis=-1) / 255.0

    # Predict using the model
    predicted_patches = model.predict(model_input, verbose=0)
    
    # Reconstruct the mask from patches
    predicted_patches_reshaped = predicted_patches.reshape(num_rows, num_cols, h, w)
    predicted_padded_mask = unpatchify(predicted_patches_reshaped, padded_image.shape)
    unpadded_mask = unpadder(predicted_padded_mask, padding_info)
    final_mask = uncropper(unpadded_mask, crop_info)

    # Post-processing the mask to find root tips
    binary_mask = (final_mask > 0.1).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    root_tip_pixels = []
    for contour in contours:
        if contour.shape[0] > 0:
            bottom_most_point = contour[contour[:, :, 1].argmax()][0]
            root_tip_pixels.append(bottom_most_point.tolist())

    print(f"CV Pipeline: Found {len(root_tip_pixels)} potential root tips.")
    return root_tip_pixels

# --- Other Functions ---
def save_inoculation_log(filename, data):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as csvfile:
        fieldnames = data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def convert_pixels_to_robot_coords(pixel_coords):
    PLATE_SIZE_MM = 150.0
    PLATE_SIZE_PIXELS = 1000.0 
    PLATE_ORIGIN_ROBOT = np.array([0.10775, 0.088 - 0.026, 0.057])
    conversion_factor = PLATE_SIZE_MM / PLATE_SIZE_PIXELS
    robot_coords = []
    for px, py in pixel_coords:
        x_mm = px * conversion_factor
        y_mm = -py * conversion_factor
        x_m, y_m, z_m = x_mm / 1000.0, y_mm / 1000.0, 0.0
        target_pos = PLATE_ORIGIN_ROBOT + np.array([x_m, y_m, z_m])
        robot_coords.append(target_pos)
    return robot_coords

def main():
    """Main function to run the inoculation task with the PID controller."""
    
    # --- Load CV Model ---
    model_path = r"C:\Users\dari\Documents\GitHub\2024-25b-fai2-adsai-dariavladutu236578\datalab_tasks\task5\dariavladutu_236578_unet_model2_256px.h5"
    try:
        cv_model = load_model(model_path, custom_objects={"f1": f1})
        print("CV Model loaded successfully.")
    except Exception as e:
        print(f"Error loading CV model: {e}")
        return

    # --- Initialization ---
    env = OT2Env(render=True)
    pid = PIDController(kp=5.0, ki=0.5, kd=2.0)
    
    image_path = env.sim.get_plate_image()
    pixel_coordinates = run_cv_pipeline(image_path, cv_model, patch_size=256)
    
    # Sort the detected root tips by their x-coordinate
    sorted_pixel_coordinates = sorted(pixel_coordinates, key=lambda coord: coord[0])
    print("\nRoot tips sorted by horizontal position.")
    
    inoculation_targets = convert_pixels_to_robot_coords(sorted_pixel_coordinates)
    
    print(f"\nConverted {len(inoculation_targets)} pixel coordinates to robot coordinates.")
    
    obs, _ = env.reset()
    
    for i, target in enumerate(inoculation_targets):
        print(f"\n--- Moving to Root Tip {i+1}/{len(inoculation_targets)} at {np.round(target, 4)} ---")
        pid.set_target(target)
        
        while True:
            current_position = obs[:3]
            error = np.linalg.norm(target - current_position)
            
            if error < 0.001:
                print("Target reached. Inoculating...")
                final_position = current_position
                
                inoculate_action = np.array([0, 0, 0, 1])
                obs, _, _, _, _ = env.step(inoculate_action)
                time.sleep(1)
                
                log_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "root_index": i + 1,
                    "target_x": target[0], "target_y": target[1], "target_z": target[2],
                    "reached_x": final_position[0], "reached_y": final_position[1], "reached_z": final_position[2],
                    "error_mm": error * 1000
                }
                save_inoculation_log("inoculation_log.csv", log_data)
                break

            action_3d = pid.update(current_position)
            action_4d = np.append(action_3d, 0)
            obs, _, _, _, _ = env.step(action_4d)
            time.sleep(1./240.)

    print("\n--- All root tips inoculated. Task complete! ---")
    print(f"Log saved to: {os.path.abspath('inoculation_log.csv')}")
    env.close()

if __name__ == '__main__':
    main()
