import cv2
import os
from tqdm import tqdm

# Function to create video from frames
def create_video_from_frames(frames_dir, output_video_path):
    # Get list of image files
    images = [img for img in os.listdir(frames_dir) if img.endswith('.jpg')]
    images.sort()  # Sort images by name

    # Read the first image to get dimensions
    first_image_path = os.path.join(frames_dir, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

    # Iterate over images and write to video
    for image in tqdm(images, desc=f'Creating {output_video_path}'):  
        image_path = os.path.join(frames_dir, image)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    video_writer.release()

# Main function to process all datasets
if __name__ == '__main__':
    base_dir = 'datasets/MOT17'
    for dataset_type in ['train', 'test']:
        dataset_path = os.path.join(base_dir, dataset_type)
        for dataset_name in os.listdir(dataset_path):
            img1_dir = os.path.join(dataset_path, dataset_name, 'img1')
            output_video_path = os.path.join(dataset_path, dataset_name, 'video.mp4')
            if os.path.exists(img1_dir):
                create_video_from_frames(img1_dir, output_video_path)