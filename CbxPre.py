from CbxDemucsWrapper import load_demucs_model
from CbxDemucsWrapper import demucs_audio
import torch
import os
import re
import shutil

class CbxPre:
    def __init__(self):
        self.modelDemucs = load_demucs_model()
        self.SAMPLING_RATE_PROCESSING = 96000
        self.SAMPLING_RATE_PRE = 16000

    def process(self,recording_path:str):
        processed_path = recording_path
        to_be_deleted = []

        out_path = re.sub(r'[.](mp3|wav)$',".speech.wav",recording_path)
        cmd = ("ffmpeg -y -i \""+processed_path+"\""
                + " -af \"speechnorm=e=3:p=0.7:r=0.005\""
                + " -c:a pcm_s16le -ar "+str(self.SAMPLING_RATE_PROCESSING)
                + " \""+out_path+"\" > \""+out_path+".log\" 2>&1")
        print("CMD: "+cmd)
        os.system(cmd)
        processed_path = out_path
        to_be_deleted.append(processed_path)
        to_be_deleted.append(processed_path+".log")

        out_path = re.sub(r'[.](mp3|wav)$',".bandpass.wav",recording_path)
        cmd = ("ffmpeg -y -i \""+processed_path+"\""
                + " -af \"highpass=f=100,lowpass=f=5000\""
                + " -c:a pcm_s16le -ar "+str(self.SAMPLING_RATE_PROCESSING)
                + " \""+out_path+"\" > \""+out_path+".log\" 2>&1")
        print("CMD: "+cmd)
        os.system(cmd)
        processed_path = out_path
        to_be_deleted.append(processed_path)
        to_be_deleted.append(processed_path+".log")
        
        demucs_audio(pathIn=processed_path,model=self.modelDemucs,device="cuda:0")
        processed_path = re.sub(r'[.](mp3|wav)$',".vocals.wav",processed_path)
        to_be_deleted.append(processed_path)

        out_path = re.sub(r'[.](mp3|wav)$',".loud.wav",recording_path)
        cmd = ("ffmpeg -y -i \""+processed_path+"\""
                + " -filter:a loudnorm"
                + " -c:a pcm_s16le -ar "+str(self.SAMPLING_RATE_PRE)
                + " \""+out_path+"\" > \""+out_path+".log\" 2>&1")
        print("CMD: "+cmd)
        os.system(cmd)
        processed_path = out_path
        to_be_deleted.append(processed_path)
        to_be_deleted.append(processed_path+".log")

        out_path = re.sub(r'[.](mp3|wav)$',".pre.wav",recording_path)
        shutil.copy2(processed_path,out_path)

        for p in to_be_deleted:
            print("Deleting: "+p)
            os.remove(p)

        print("PRE DONE")

        return out_path
