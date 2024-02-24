import os
import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips
import numpy as np
import shutil

input_video_folder = '/home/soyrl/videos_folder/'
split_frames_folder = '/home/soyrl/frames_to_inpaint/' 
sd_generated_folder = '/home/soyrl/inpainted_imgs_deepfake/'

resized_video_folder = '/home/soyrl/videos_resized/'
temp_video_path = '/home/soyrl/temp_deepfake_video.mp4'

final_video_path = '/home/soyrl/final_deepfake_video.mp4'
final_video_folder = '/home/soyrl/videos/'
cropped_video_path = '/home/soyrl/cropped_video.mp4'

last_path='/home/soyrl/deepfaked_video.mp4'

# #Resize video to 1920*1080 - Needed for inpainting script and for changing dimensions of other videos
def resize_video(input_path, output_path, new_width, new_height):
    # Load the video clip
    clip = VideoFileClip(input_path)

    # Resize the video
    resized_clip = clip.resize((new_width, new_height))

    # Write the resized video with audio to the output file
    resized_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

    print(f"Video resized and saved to {output_path}")

# Example usage
new_width = 1920
new_height = 1080

if not os.path.exists(resized_video_folder):
    os.makedirs(resized_video_folder)

for video_file in os.listdir(input_video_folder):
    input_video_path = input_video_folder + video_file
    resized_video_path = resized_video_folder + video_file.split('.mp4')[0]+'_resized.mp4'
    resize_video(input_video_path, resized_video_path, new_width, new_height)


# # Prompt: split video into frames, process them, and then recreate video

# #Step 1 - Split video into frames
# Takes ~19h in i7-6820HQ 2.70GHz for 3min video - Much faster in i7-10750H 2.60GHz - Just a few minutes

# if not os.path.exists(split_frames_folder):
#     os.makedirs(split_frames_folder)

# cap = cv2.VideoCapture(resized_video_path)
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# for frame_number in range(frame_count):
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame_filename = f'{split_frames_folder}frame_{frame_number:04d}.jpg'
#     cv2.imwrite(frame_filename, frame)

# cap.release()


#Step 2 - Extract masks from frames - Not needed anymore


# # Step 3 - Recreate video

# # List all image files in the folder
# image_files = sorted([f for f in os.listdir(sd_generated_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])

# # Read the first image to get dimensions
# first_image = cv2.imread(os.path.join(sd_generated_folder, image_files[0]))
# height, width, layers = first_image.shape

# # Create a VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = 30  # Desired frame rate
# out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

# # Write each image to the video with the specified frame rate
# for image_file in image_files:
#     image_path = os.path.join(sd_generated_folder, image_file)
#     frame = cv2.imread(image_path)
#     out.write(frame)

# out.release()

# # Load the video clip with the added images
# video_clip = VideoFileClip(temp_video_path)

# # Load the video clip with audio
# audio_clip = VideoFileClip(input_video_path).audio

# # Set the audio of the processed video to the loaded audio
# video_clip = video_clip.set_audio(audio_clip)

# # Write the final video with sound
# video_clip.write_videofile(final_video_path, codec='libx264', audio_codec='aac')

# # Close the video clips
# video_clip.close()


# # Step 4 - Crop other videos to the same size
target_width = 1080
target_height = 1080

def crop_frame(frame, target_width, target_height):
    # Calculate cropping coordinates
    img_width, img_height = frame.shape[1], frame.shape[0]
    left = (img_width - target_width) // 2
    upper = (img_height - target_height) // 2
    right = left + target_width
    lower = upper + target_height

    # Crop the frame
    cropped_frame = frame[upper:lower, left:right]

    return cropped_frame

def process_video(input_path, output_path, target_width, target_height):
    video_clip = VideoFileClip(input_path)

    def process_frame(frame):
        # Crop the processed frame
        cropped_frame = crop_frame(frame, target_width, target_height)

        return cropped_frame

    # Apply processing to each frame of the video clip
    processed_clip = video_clip.fl_image(process_frame)

    # Write the processed frames to a new video file
    processed_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')


video_files=os.listdir(resized_video_folder)

if not os.path.exists(final_video_folder):
    os.makedirs(final_video_folder)

for video_file in video_files:
    original_video_path = resized_video_folder + video_file
    cropped_video_path = final_video_folder + video_file.split('.mp4')[0]+'_cropped.mp4'
    process_video(original_video_path, cropped_video_path, target_width, target_height)


#Depending on file names, order will be decided. Make sure you name them in the order you want to concatenate them.
#'final_deepfake_video.mp4' is the stable diffusion video name
def combine_and_resize_videos(input_folder, output_file, target_resolution):
    video_clips = []

    # Iterate through files in the input folder
    for filename in np.sort(os.listdir(input_folder)):
        if filename.endswith(".mp4"):  # Change the extension accordingly
            file_path = os.path.join(input_folder, filename)
            
            # Load the video clip
            video_clip = VideoFileClip(file_path)
            
            # Resize the video if necessary
            if video_clip.size != target_resolution:
                video_clip = video_clip.resize(target_resolution) 
            
            video_clips.append(video_clip)
            print("Added", filename, "to the list of video clips")

    # Concatenate video clips
    final_clip = concatenate_videoclips(video_clips)

    # Write the concatenated video to the output file
    final_clip.write_videofile(output_file, codec="libx264")

target_resolution = (512,512)  # Change to your desired resolution

# Copy final_video_path file to final_video_folder
shutil.copy2(final_video_path, final_video_folder)

combine_and_resize_videos(final_video_folder, last_path, target_resolution)