# import streamlit as st
# import zipfile
# import os
# from pathlib import Path
# import tempfile
# from natwest_hackathon import deepfake_detection  # Ensure this imports your detection logic

# # Title of the Streamlit app
# st.title("Deepfake Detection App")

# # Section to upload a zipped folder containing base files
# st.header("Upload Zipped Base Folder")
# zip_file = st.file_uploader("Upload a zipped folder containing base files", type=["zip"])

# image_file, audio_file, text_file = None, None, None

# if zip_file is not None:
#     # Create a temporary directory to extract the contents
#     with tempfile.TemporaryDirectory() as temp_dir:
#         zip_path = Path(temp_dir) / "base_folder.zip"
        
#         # Save the uploaded zip file to a temporary path
#         with open(zip_path, mode="wb") as f:
#             f.write(zip_file.read())
        
#         # Extract the zip file
#         with zipfile.ZipFile(zip_path, "r") as zip_ref:
#             zip_ref.extractall(temp_dir)
        
#         # Scan the extracted folder for relevant files (image, audio, instruction)
#         for root, dirs, files in os.walk(temp_dir):
#             for file in files:
#                 if file.endswith(('.png', '.jpg', '.jpeg')):
#                     image_file = file
#                 elif file.endswith(('.mp3', '.wav')):
#                     audio_file = file
#                 elif file.endswith('.txt'):
#                     text_file = file

#         # Check if the required files were found
#         if image_file and audio_file and text_file:
#             st.success("Image, audio, and instruction (text) files retrieved successfully.")
#         else:
#             st.warning("Some required files are missing (image, audio, or instruction).")

# # Section to upload a video file
# st.header("Upload Video for Deepfake Detection")
# video_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

# if video_file is not None:
#     st.success(f"Video uploaded: {video_file.name}")
    
#     # Save the uploaded video to a temporary location for processing
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
#         temp_video.write(video_file.read())
#         video_path = temp_video.name
    
#     # Display the video
#     with open(video_path, 'rb') as video_data:
#         st.video(video_data.read())

#     # Start deepfake detection once both inputs are ready
#     if image_file and video_file and zip_file:
#         # Prepare file paths for the deepfake detection
#         image_path = os.path.join(temp_dir, image_file)
#         audio_path = os.path.join(temp_dir, audio_file)
#         instruction_path = os.path.join(temp_dir, text_file)

#         # Perform deepfake detection
#         deepfake_detection_object = deepfake_detection()
#         final_label = deepfake_detection_object.check_deepfake(temp_dir)

#         # Display the result
#         st.write("Deepfake Detection Result:", final_label)
#     else:
#         st.info("All required files (image, audio, and video) are needed for deepfake detection.")


# ----------------------------------------------------------------------------------------------------------

import streamlit as st
import zipfile
import os
from pathlib import Path
from natwest_hackathon import deepfake_detection  # Ensure this imports your detection logic

# Title of the Streamlit app
st.title("Deepfake Detection App")

# Section to upload a zipped folder containing base files
st.header("Upload Zipped Base Folder")
zip_file = st.file_uploader("Upload a zipped folder containing base files", type=["zip"])

image_file, audio_file, text_file = None, None, None

if zip_file is not None:
    # Create a local directory to extract the contents
	base_dir = Path.cwd() / "base_folder"
	base_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    # Save the uploaded zip file to a local path
	zip_path = base_dir / "base_folder.zip"
	with open(zip_path, mode="wb") as f:
	    f.write(zip_file.read())

	# Extract the zip file
	with zipfile.ZipFile(zip_path, "r") as zip_ref:
	    zip_ref.extractall(base_dir)

	# Scan the extracted folder for relevant files (image, audio, instruction)
	for root, dirs, files in os.walk(base_dir):
	    for file in files:
	        if file.endswith(('.png', '.jpg', '.jpeg')):
	            image_file = Path(root) / file
	            print(f"Image file saved at: {image_file}")
	        elif file.endswith(('.mp3', '.wav')):
	            audio_file = Path(root) / file
	            print(f"Audio file saved at: {audio_file}")
	        elif file.endswith('.txt'):
	            text_file = Path(root) / file
	            print(f"Instruction file saved at: {text_file}")

	# Check if the required files were found
	if image_file and audio_file and text_file:
	    st.success("Image, audio, and instruction (text) files retrieved successfully.")
	else:
	    st.warning("Some required files are missing (image, audio, or instruction).")

# Section to upload a video file
st.header("Upload Video for Deepfake Detection")
video_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

if video_file is not None:
    st.success(f"Video uploaded: {video_file.name}")

    # Save the uploaded video to a local location for processing
    video_path = base_dir / video_file.name
    with open(video_path, 'wb') as f:
        f.write(video_file.read())

    # Display the video
    with open(video_path, 'rb') as video_data:
        st.video(video_data.read())

    # Start deepfake detection once both inputs are ready
    if image_file and video_file and zip_file:
        # Prepare file paths for the deepfake detection
        image_path = base_dir / image_file
        audio_path = base_dir / audio_file
        instruction_path = base_dir / text_file

        # Perform deepfake detection
        deepfake_detection_object = deepfake_detection()
        final_label = deepfake_detection_object.check_deepfake(base_dir)

        # Display the result
        st.write("Deepfake Detection Result:", final_label)
        if final_label.split("_")[0] == "OV":
        	st.write("The video is authentic")
        elif final_label.split("_")[0] == "DFV":
        	st.write("The video is deepfake")
        
        if final_label.split("_")[1] == "OA":
        	st.write("The audio in the video is authentic")
        elif final_label.split("_")[1] == "DFA":
        	st.write("The audio in the video is deepfake")

        if final_label.split("_")[2] == "ST":
        	st.write("The person is speaking the correct text instructions")
        elif final_label.split("_")[2] == "DT":
        	st.write("The person is speaking the wrong text instructions")
    else:
        st.info("All required files (image, audio, and video) are needed for deepfake detection.")
