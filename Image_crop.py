import cv2
import os

# Directory containing the images
image_directory = 'D:/Img_crop/selfie_images/'

# Load the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_frontalface_default.xml')

# Create the output directory if it doesn't exist
output_directory = 'D:/Img_crop/cropped/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# List all files in the image directory
image_files = os.listdir(image_directory)

for image_file in image_files:
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        # Load the image
        image_path = os.path.join(image_directory, image_file)
        a = cv2.imread(image_path)

        # Detect faces
        faces = face_cascade.detectMultiScale(a, scaleFactor=1.03, minNeighbors=5)

        if len(faces) > 0:
            # Find the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            face = a[y:y + h, x:x + w]

            # Resize the cropped face to 150x150 pixels
            resized_face = cv2.resize(face, (150, 150))
            output_file = os.path.join(output_directory, f"{os.path.splitext(image_file)[0]}_face.jpg")
            cv2.imwrite(output_file, resized_face)
        else:
            print(f"No faces detected in {image_file}")
    else:
        print(f"Skipping {image_file} - Unsupported file format")

print("Processing completed")
