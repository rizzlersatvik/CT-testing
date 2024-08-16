import requests
from PIL import Image, ImageDraw
import numpy as np
from io import BytesIO

def draw_roi_on_image(input_data, template_path):
    # Extract relevant information from input_data
    url = input_data['url']
    face_coordinates = input_data['faceCoordinates']
    roi = input_data['roi'][0]  # Assuming only one ROI for simplicity

    # Read the image from URL using requests
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGBA")

    # Draw rectangle around face coordinates on the original image
    x1 = face_coordinates['top_left_x']
    y1 = face_coordinates['top_left_y']
    width = face_coordinates['width']
    height = face_coordinates['height']
    img_with_face_coordinates = image.copy()
    draw = ImageDraw.Draw(img_with_face_coordinates)
    draw.rectangle([x1, y1, x1 + width, y1 + height], outline="green", width=2)

    # Read the template image
    template_image = Image.open(template_path).convert("RGBA")

    # Draw ROI rectangle on the template image
    roi_x1 = roi['top_left_x']
    roi_y1 = roi['top_left_y']
    roi_width = roi['width']
    roi_height = roi['height']
    template_with_roi = template_image.copy()
    draw = ImageDraw.Draw(template_with_roi)
    draw.rectangle([roi_x1, roi_y1, roi_x1 + roi_width, roi_y1 + roi_height], outline="blue", width=2)

    # Calculate the dimensions of the template
    template_width = template_image.width
    template_height = template_image.height

    # Calculate the face center in the original image
    face_center_x = face_coordinates['top_left_x'] + face_coordinates['width'] // 2
    face_center_y = face_coordinates['top_left_y'] + face_coordinates['height'] // 2

    # Calculate the center of the ROI
    roi_center_x = roi_x1 + roi_width // 2
    roi_center_y = roi_y1 + roi_height // 2

    # Calculate the required scale factor to fit the face into the ROI
    scale_x = roi_width / face_coordinates['width']
    scale_y = roi_height / face_coordinates['height']
    scale = max(scale_x, scale_y)  # Use the maximum scale to ensure the face fits

    # Resize the original image
    new_width = int(image.width * scale)
    new_height = int(image.height * scale)
    resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Calculate new face center coordinates after resizing
    resized_face_center_x = int(face_center_x * scale)
    resized_face_center_y = int(face_center_y * scale)

    # Calculate the top-left corner for overlaying the resized image on the template
    overlay_x = roi_center_x - resized_face_center_x
    overlay_y = roi_center_y - resized_face_center_y

    # Create a new blank image for the overlay
    overlay_image = template_image.copy()

    # Calculate the crop and overlay area
    start_y = max(0, -overlay_y)
    end_y = min(resized_img.height, overlay_image.height - overlay_y)
    start_x = max(0, -overlay_x)
    end_x = min(resized_img.width, overlay_image.width - overlay_x)

    cropped_resized_img = resized_img.crop((start_x, start_y, end_x, end_y))

    # Overlay the resized image onto the template
    overlay_image.paste(cropped_resized_img, (max(0, overlay_x), max(0, overlay_y)), cropped_resized_img)

    # Print dimensions
    print(f"Template dimensions: {template_width} x {template_height}")
    print(f"Resized original image dimensions: {new_width} x {new_height}")

    # Display the images using PIL
    img_with_face_coordinates.show(title="Face Coordinates")
    template_with_roi.show(title="ROI on Template")
    overlay_image.show(title="Overlay Image")

    # Save the images if needed
    #img_with_face_coordinates.save('face_coordinates.png')
    #template_with_roi.save('ROI_on_template.png')
    #overlay_image.save('Overlay_image.png')

input_data =             {
                "roi": [
                    {
                        "type": "face",
                        "top_left_x": 824,
                        "top_left_y": 132,
                        "width": 204,
                        "height": 218
                    }
                ],
                "id": "ac269563-left-beea56e4-9be5-42d0-9cf9-854b4b506ee5.png",
                "url": "https://rizzle-api-gateway-prod.s3.amazonaws.com/image/ml-podcast-thumbnails/beea56e4-9be5-42d0-9cf9-854b4b506ee5.png",
                "faceCoordinates": {
                    "top_left_x": 217,
                    "top_left_y": 163,
                    "width": 177,
                    "height": 232
                },
                "media_preference": "LEFT",
                "size": {
                    "width": 1280,
                    "height": 720
                }
    }
template_path = "thumbnail.png"
draw_roi_on_image(input_data, template_path)
