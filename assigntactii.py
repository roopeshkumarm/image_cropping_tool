import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path, color_range=30, blur_kernel_size=(5, 5), min_contour_area=1000):
    try:
        # Read image
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Unable to read the image")

        # Convert image to RGB color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Gaussian blur to reduce noise
        image_rgb = cv2.GaussianBlur(image_rgb, blur_kernel_size, 0)

        # Calculate the average color of the image
        average_color = np.mean(image_rgb, axis=(0, 1))

        # Define a color range around the average color
        lower_bound = average_color - color_range
        upper_bound = average_color + color_range

        # Create a mask for the region with colors within the specified range
        mask = cv2.inRange(image_rgb, lower_bound, upper_bound)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if contours are found
        if not contours:
            raise ValueError("No contours found.")

        # Filter out small contours
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        # Check if valid contours are found
        if not valid_contours:
            raise ValueError("No valid contours found.")

        # Get the bounding box of the largest valid contour
        largest_contour = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the image
        cropped_image = image[y:y+h, x:x+w]

        return image, cropped_image

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def display_images(original_image, cropped_image):
    plt.figure(figsize=(10, 5))

    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")

    # Display the cropped image
    plt.subplot(1, 2, 2)
    plt.imshow(cropped_image)
    plt.title("Cropped Image")

    plt.show()

def main():
    image_path = "img2.jpg"
    original_image, cropped_image = process_image(image_path)

    if original_image is not None and cropped_image is not None:
        display_images(original_image, cropped_image)
        print("Images displayed successfully.")
    else:
        print("Failed to process the image")

if __name__ == "__main__":
    main()
