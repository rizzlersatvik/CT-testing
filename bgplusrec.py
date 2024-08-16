import requests
from rembg import remove
from PIL import Image
from io import BytesIO
import os
import cv2
import numpy as np

# Define the output folder
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

# Define input thumbnail
thumbnail_path = "thumbnail.png"

# Load the thumbnail image
thumbnail = Image.open(thumbnail_path)

input_data = [
    {
        "roi": [
            {
                "type": "face",
                "top_left_x": 274,
                "top_left_y": 123,
                "width": 204,
                "height": 218
            }
        ],
        "id": "e170aa97-left-e2ef907b-e5b2-46d1-8842-9202a922efc9.png",
        "url": "https://rizzle-api-gateway-prod.s3.amazonaws.com/image/ml-podcast-thumbnails/e2ef907b-e5b2-46d1-8842-9202a922efc9.png",
        "faceCoordinates": {
            "top_left_x": 382,
            "top_left_y": 141,
            "width": 180,
            "height": 278
        },
        "media_preference": "LEFT",
        "size": {
            "width": 1280,
            "height": 720
        },
        "cutout_type": "PERSON"
    }
]

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face_coordinates(image):
    open_cv_image = np.array(image.convert('RGB'))
    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Taking the first detected face
        return {"top_left_x": x, "top_left_y": y, "width": w, "height": h}
    else:
        return None  # Return None if no face is detected

# Process each image
for item in input_data:
    if item.get("cutout_type") == "PERSON":
        # Download the image from the provided URL
        response = requests.get(item["url"])
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
        else:
            raise Exception(f"Failed to download image from {item['url']}")

        # Remove the background
        image_no_bg = remove(image)

        # Get face coordinates, ROI, and media preference
        face_coords = item.get("faceCoordinates") or detect_face_coordinates(image)

        roi = item["roi"][0]

        if face_coords:
            # Calculate the scaling factor to ensure the height matches the ROI height
            scale_factor = roi["height"] / face_coords["height"]

            # Resize the image while maintaining aspect ratio
            new_size = (int(image_no_bg.width * scale_factor), int(image_no_bg.height * scale_factor))
            image_no_bg = image_no_bg.resize(new_size, Image.LANCZOS)

            # Calculate the new position of the face in the resized image
            new_face_width = int(face_coords["width"] * scale_factor)
            new_face_height = int(face_coords["height"] * scale_factor)
            new_face_x = int(face_coords["top_left_x"] * scale_factor)
            new_face_y = int(face_coords["top_left_y"] * scale_factor)

            # Calculate the center of the face coordinates
            face_center_x = new_face_x + new_face_width // 2
            face_center_y = new_face_y + new_face_height // 2

            # Calculate the center of the ROI
            roi_center_x = roi["top_left_x"] + roi["width"] // 2
            roi_center_y = roi["top_left_y"] + roi["height"] // 2

            # Calculate the offset to center the face coordinates within the ROI
            offset_x = roi_center_x - face_center_x
            offset_y = roi["top_left_y"] + roi["height"] - new_face_y - new_face_height
        else:
            # If no face is detected, center the image in the ROI
            offset_x = roi["top_left_x"] + (roi["width"] - image_no_bg.width) // 2
            offset_y = roi["top_left_y"] + (roi["height"] - image_no_bg.height) // 2

        # Create a new blank image with the desired size
        output_image = Image.new("RGBA", (item["size"]["width"], item["size"]["height"]))

        # Paste the background-removed image onto the blank image at the calculated offset
        output_image.paste(image_no_bg, (offset_x, offset_y), image_no_bg)

        # Save the individual output image
        individual_output_image_path = os.path.join(output_folder, f"{item['media_preference'].lower()}.png")
        output_image.save(individual_output_image_path)
        print(f"Individual output image saved at {individual_output_image_path}")

        # Overlay the processed image onto the thumbnail
        thumbnail.paste(output_image, (0, 0), output_image) if item["media_preference"] == "LEFT" else \
            thumbnail.paste(output_image, (thumbnail.width - output_image.width, 0), output_image)

# Save the final composite image
final_output_image_path = os.path.join(output_folder, "thumbnail_output.png")
thumbnail.save(final_output_image_path)
print(f"Final composite image saved at {final_output_image_path}")
