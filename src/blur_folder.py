# author: Asmaa Mirkhan ~ 2019
import os
import cv2
from DetectorAPI import Detector

# Update your blurBoxes method if necessary.

def process_images(input_folder, output_folder, model_path, threshold=0.1):
    """
    Processes all images in the input_folder by detecting and blurring faces,
    then saves them in the output_folder.
    
    Parameters:
    - input_folder: str, path to the folder containing input images.
    - output_folder: str, path to the folder where processed images will be saved.
    - model_path: str, path to .pb model for face detection.
    - threshold: float, face detection confidence threshold. Default is 0.7.
    """
    
    # Ensure output directory exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create detection object
    detector = Detector(model_path=model_path, name="detection")

    # Loop through each file in the input directory
    for filename in os.listdir(input_folder):
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Full path to the current image file
            image_path = os.path.join(input_folder, filename)
            # Read the image
            image = cv2.imread(image_path)

            # Perform face detection
            faces = detector.detect_objects(image, threshold=threshold)
            
            # Blur the detected faces
            image = blurBoxes(image, faces)

            # Construct a path to save the blurred image
            output_image_path = os.path.join(output_folder, filename)
            
            # Save the blurred image
            cv2.imwrite(output_image_path, image)
            print(f'Processed and saved: {output_image_path}')
        else:
            print(f'Skipped non-image file: {filename}')




def blurBoxes(image, boxes):
    """
    Argument:
    image -- the image that will be edited as a matrix
    boxes -- list of boxes that will be blurred each element must be a dictionary that has [id, score, x1, y1, x2, y2] keys

    Returns:
    image -- the blurred image as a matrix
    """

    for box in boxes:
        # unpack each box
        x1, y1 = box["x1"], box["y1"]
        x2, y2 = box["x2"], box["y2"]

        # crop the image due to the current box
        sub = image[y1:y2, x1:x2]

        # apply GaussianBlur on cropped area
        blur = cv2.blur(sub, (25, 25))

        # paste blurred image on the original image
        image[y1:y2, x1:x2] = blur

    return image


def main():
    input_folder = "C:\\pyproj\\BlurryFaces\\images"
    output_folder = "C:\\pyproj\\BlurryFaces\\processed_img"
    model_path = "C:\\pyproj\\BlurryFaces\\face_model\\face.pb"

    process_images(input_folder, output_folder, model_path)


if __name__ == "__main__":
    main()
