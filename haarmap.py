import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt

def detect_face(image):
    # Load the pre-trained face detection model from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Return the first detected face coordinates or None if no face is detected
    if len(faces) > 0:
        return faces[0]
    else:
        return None

def draw_roi_on_images(input_data_list, template_path):
    # Read the template image
    template_image = cv2.imread(template_path)
    template_with_roi = template_image.copy()

    for input_data in input_data_list:
        # Extract relevant information from input_data
        url = input_data['url']
        rois = input_data['roi']

        # Read the image from URL
        with urllib.request.urlopen(url) as response:
            image_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Detect face in the image
        face_coordinates = detect_face(img)

        if face_coordinates is None:
            print("No face detected in the image.")
            continue

        # Draw rectangle around detected face on the original image
        x1, y1, width, height = face_coordinates
        img_with_face_coordinates = img.copy()
        cv2.rectangle(img_with_face_coordinates, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 2)

        for roi in rois:
            # Draw ROI rectangle on the template image
            roi_x1 = roi['top_left_x']
            roi_y1 = roi['top_left_y']
            roi_width = roi['width']
            roi_height = roi['height']
            cv2.rectangle(template_with_roi, (roi_x1, roi_y1), (roi_x1 + roi_width, roi_y1 + roi_height), (255, 0, 0), 2)

            # Calculate the center of the ROI
            roi_center_x = roi_x1 + roi_width // 2
            roi_center_y = roi_y1 + roi_height // 2

            # Calculate the face center in the original image
            face_center_x = x1 + width // 2
            face_center_y = y1 + height // 2

            # Calculate the required scale factor to ensure no gaps
            scale_x = template_image.shape[1] / img.shape[1]
            scale_y = template_image.shape[0] / img.shape[0]
            scale = max(scale_x, scale_y)  # Use the maximum scale to ensure no gaps

            # Resize the original image
            new_width = int(img.shape[1] * scale)
            new_height = int(img.shape[0] * scale)
            resized_img = cv2.resize(img, (new_width, new_height))

            # Calculate new face center coordinates after resizing
            resized_face_center_x = int(face_center_x * scale)
            resized_face_center_y = int(face_center_y * scale)

            # Calculate the top-left corner for overlaying the resized image on the template
            overlay_x = roi_center_x - resized_face_center_x
            overlay_y = roi_center_y - resized_face_center_y

            # Ensure the overlay image is centered within the template
            overlay_x = max(min(overlay_x, 0), template_image.shape[1] - new_width)
            overlay_y = max(min(overlay_y, 0), template_image.shape[0] - new_height)

            # Create a new blank image for the overlay
            overlay_image = template_image.copy()

            # Calculate the crop and overlay area
            start_y = max(0, -overlay_y)
            end_y = min(new_height, overlay_image.shape[0] - overlay_y)
            start_x = max(0, -overlay_x)
            end_x = min(new_width, overlay_image.shape[1] - overlay_x)

            cropped_resized_img = resized_img[start_y:end_y, start_x:end_x]
            crop_height, crop_width = cropped_resized_img.shape[:2]

            overlay_start_y = max(0, overlay_y)
            overlay_end_y = overlay_start_y + crop_height
            overlay_start_x = max(0, overlay_x)
            overlay_end_x = overlay_start_x + crop_width

            # Fill the overlay image
            overlay_image[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x] = cropped_resized_img

            # Adjust overlay to center face within the ROI
            roi_center_x = overlay_start_x + crop_width // 2
            roi_center_y = overlay_start_y + crop_height // 2

            # Calculate the shift needed to align face center to ROI center
            shift_x = (roi_x1 + roi_width // 2) - roi_center_x
            shift_y = (roi_y1 + roi_height // 2) - roi_center_y

            # Apply shift to the overlay image
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            overlay_image = cv2.warpAffine(overlay_image, M, (overlay_image.shape[1], overlay_image.shape[0]))

            # Print dimensions
            print(f"Template dimensions: {template_image.shape[1]} x {template_image.shape[0]}")
            print(f"Resized original image dimensions: {resized_img.shape[1]} x {resized_img.shape[0]}")

            # Display the images using matplotlib
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 3, 1)
            plt.title('Face Coordinates')
            plt.imshow(cv2.cvtColor(img_with_face_coordinates, cv2.COLOR_BGR2RGB))
            plt.subplot(1, 3, 2)
            plt.title('Template with ROI')
            plt.imshow(cv2.cvtColor(template_with_roi, cv2.COLOR_BGR2RGB))
            plt.subplot(1, 3, 3)
            plt.title('Overlay Image')
            plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
            plt.show()

            # Save the images if needed
            cv2.imwrite(f'{input_data["id"]}_face_coordinates.png', img_with_face_coordinates)
            cv2.imwrite(f'{input_data["id"]}_ROI_on_template.png', template_with_roi)
            cv2.imwrite(f'{input_data["id"]}_Overlay_image.png', overlay_image)

# Example input data list
input_data_list = [
    {
        "roi": [
            {
                "type": "face",
                "top_left_x": 205,
                "top_left_y": 109,
                "width": 181,
                "height": 208
            }
        ],
        "id": "a1169305-left-ec7169f4-7076-4095-b1ce-6fd0aefe1a69.png",
        "url": "https://rizzle-api-gateway.s3.amazonaws.com/image/ml-podcast-thumbnails/ec7169f4-7076-4095-b1ce-6fd0aefe1a69.png",
        "faceCoordinates": {
            "top_left_x": 411,
            "top_left_y": 112,
            "width": 164,
            "height": 223
        },
        "media_preference": "LEFT",
        "size": {
            "width": 1280,
            "height": 720
        },
        "cutout_type": "PERSON",
        "enhance_image": True
    },
    {
        "roi": [
            {
                "type": "face",
                "top_left_x": 896,
                "top_left_y": 94,
                "width": 211,
                "height": 220
            }
        ],
        "id": "09815063-right-aa172b44-f065-4352-a488-bfea1e16cf6a.png",
        "url": "https://rizzle-api-gateway.s3.amazonaws.com/image/ml-podcast-thumbnails/aa172b44-f065-4352-a488-bfea1e16cf6a.png",
        "faceCoordinates": {
            "top_left_x": 615,
            "top_left_y": 130,
            "width": 160,
            "height": 200
        },
        "media_preference": "RIGHT",
        "size": {
            "width": 1280,
            "height": 720
        },
        "cutout_type": "PERSON",
        "enhance_image": True
    }          
]

template_path = 'thumbnail.png'  # Change this to your template image path
draw_roi_on_images(input_data_list, template_path)
