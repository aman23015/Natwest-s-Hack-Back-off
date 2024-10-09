# import os
# import subprocess
# import sys

# # Function to install packages from requirements.txt
# def install_requirements():
#     try:
#         with open('requirements.txt') as f:
#             packages = f.read().splitlines()
#         for package in packages:
#             subprocess.check_call([sys.executable, "-m", "pip", "install", package])
#     except Exception as e:
#         print(f"An error occurred: {e}")

# # Check and install requirements
# install_requirements()

import os
import cv2
import torch
import random
import torchaudio
import ffmpeg
from deepface import DeepFace
from speechbrain.inference import SpeakerRecognition
from sentence_transformers import SentenceTransformer, util
import subprocess

class ffmpegProcessor:
    def __init__(self):
        self.cmd = 'ffmpeg'
        self.check_ffmpeg_installed()

    def check_ffmpeg_installed(self):
        try:
            subprocess.run([self.cmd, '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            raise EnvironmentError("FFmpeg is not installed or not found in the system path.")

    def extract_audio(self, video_path, audio_output_path):
        if os.path.exists(audio_output_path):
            os.remove(audio_output_path)

        (ffmpeg
         .input(video_path)
         .output(audio_output_path, format='mp3')
         .run(cmd=self.cmd, capture_stdout=True, capture_stderr=True)
         )

# Voice to Text
class Voice_To_Text:
    def __init__(self,
                 device,
                 model='silero_stt',
                 model_repo='snakers4/silero-models',
                 language='en',
                 utt_start_token='',
                 utt_end_token='',
                 **kwargs):
        self.device = device
        self.utt_start_token = utt_start_token
        self.utt_end_token = utt_end_token

        self.stt_model, self.stt_decoder, _ = torch.hub.load(repo_or_dir=model_repo,
                                                             model=model,
                                                             language=language,
                                                             device=device,
                                                             trust_repo=True)
        self.stt_model.to(self.device)

    def recognize_speech_of_audio(self, audio_file_path):
        speech, rate = torchaudio.load(audio_file_path)
        speech = torchaudio.functional.resample(speech, rate, 16000)

        out = self.stt_model(speech.to(self.device))
        text = self.stt_decoder(out[0].cpu())

        text = self.utt_start_token + text + self.utt_end_token
        return text

def create_objects():
    # Create an instance of Voice_To_Text
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    speech_to_text = Voice_To_Text(device=device)

    # Speaker Verification Model
    # Load the pretrained model, changing the savedir to a writable directory
    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="SpeakerVerification"  # Use a relative or accessible path
    )
    
    return speech_to_text, verification

# Face Verification
def Face_Verification(image_path, video_path, frame_to_select=15):
    img1 = cv2.imread(image_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    select = min(frame_to_select, total_frames)
    frame_numbers = sorted(random.sample(range(total_frames), select))
    same_face_found = False
    for frame_number in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            result = DeepFace.verify(img1, frame, model_name='VGG-Face')
            if result["verified"]:
                same_face_found = True
                break
        except Exception as e:
            continue
    cap.release()
    return same_face_found

# Instruction Matching
def compute_similarity(input_text, label_text, threshold=90):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    input_embedding = model.encode(input_text, convert_to_tensor=True)
    label_embedding = model.encode(label_text, convert_to_tensor=True)
    cosine_similarity = util.cos_sim(input_embedding, label_embedding).item()
    similarity_percentage = round(cosine_similarity * 100, 2)
    return similarity_percentage >= threshold

# Combined pipeline
class deepfake_detection():
    def __init__(self, Frame_to_select=15, Threshold_text=90):
        self.Frame_to_select = Frame_to_select
        self.Threshold_text = Threshold_text
        self.speech_to_text, self.verification = create_objects()

    def check_deepfake(self, Folder_path):
        # ########### Paths ###########
        Image_path = os.path.join(Folder_path, "image.jpg")
        Video_path = os.path.join(Folder_path, "video.mp4")
        Audio_path = os.path.join(Folder_path, "audio.mp3")
        Voice_sample_path = os.path.join(Folder_path, "voice_sample.mp3")
        Instruction_path = os.path.join(Folder_path, "instruction.txt")

        # Debugging: Print the paths being checked
        print(f"Checking the following paths:")
        print(f"Image Path: {Image_path}")
        print(f"Video Path: {Video_path}")
        print(f"Audio Path: {Audio_path}")
        print(f"Voice Sample Path: {Voice_sample_path}")
        print(f"Instruction Path: {Instruction_path}")

        # Ensure paths are valid
        if not os.path.exists(Image_path):
            raise FileNotFoundError(f"Image file not found: {Image_path}")
        if not os.path.exists(Video_path):
            raise FileNotFoundError(f"Video file not found: {Video_path}")
        if not os.path.exists(Voice_sample_path):
            raise FileNotFoundError(f"Voice sample file not found: {Voice_sample_path}")
        if not os.path.exists(Instruction_path):
            raise FileNotFoundError(f"Instruction file not found: {Instruction_path}")

        # Verify Face
        same_face_found = Face_Verification(image_path=Image_path,
                                            video_path=Video_path,
                                            frame_to_select=self.Frame_to_select)

        # Extract Voice from Video
        Voice_from_video = ffmpegProcessor()
        Voice_from_video.extract_audio(Video_path, Audio_path)

        # Verify Voice
        score, prediction = self.verification.verify_files(Audio_path, Voice_sample_path)

        # Extract Text from Voice
        text = self.speech_to_text.recognize_speech_of_audio(audio_file_path=Audio_path)

        # Read Text from instruction.txt
        with open(Instruction_path, 'r') as file:
            instruction_text = file.read()

        # Match the Text
        instruction = compute_similarity(input_text=text,
                                         label_text=instruction_text,
                                         threshold=self.Threshold_text)

        # Determine labels based on conditions
        same_face_label = "OV" if same_face_found else "DFV"
        prediction_label = "OA" if prediction else "DFA"
        instruction_label = "ST" if instruction else "DT"

        # Format the final label
        final_label = f"{same_face_label}_{prediction_label}_{instruction_label}"
        return final_label

        
if __name__ == "__main__":
    deepfake_detection_object = deepfake_detection()
    deepfake_detection_object.check_deepfake('/content/Test_Folder')
