from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.tokenizer import _LANGUAGE_CODES, Tokenizer
import ctranslate2
from CbxAligner import CbxAligner
from CbxTokenizer import CbxToken
from CbxUtils import format_time_ms
import logging
import torch
import os
import re

class CbxSTT():
    def __init__(self,language="en"):
        self.language = language
        #large-v2 provides with better results than large-v1,large-v3 or large-v3-turbo
        model = "large-v2" 
        compute_type="float16"
        ctranslate2.set_log_level(logging.INFO)
        print("LOADING: "+model+"/"+compute_type)
        self.model = WhisperModel(model, device="cuda", compute_type=compute_type)
        #Batched model doesn't offer the same options
        #self.batched_model = BatchedInferencePipeline(model=self.model,language=language)

    #Transcribe a pre-processed recording
    def process(self,initial_prompt:str,recording_path:str):
        tokenizer = self.model.hf_tokenizer
        #Prompt size must be less than 224 tokens
        initial_prompt_tokens = tokenizer.encode(initial_prompt)
        print("PROMPT SIZE="+str(len(initial_prompt_tokens.tokens)))

        processed_path = re.sub(r'[.](mp3|wav)$',".pre.wav",recording_path)
        print("STT on: "+processed_path)
        segments, info = self.model.transcribe(processed_path
                ,language=self.language
                ,initial_prompt=initial_prompt
                ,word_timestamps = True
                #beam_size only use with temperature=0, but best_off is ignore inthis case
                # ,beam_size=5
                #Do not remove parts
                ,no_speech_threshold = 1.0
                #Stop only on high log_prob
                ,log_prob_threshold = -0.01
                #A bit of repetition_penalty, but a higher value often provides with bad results
                ,repetition_penalty=1.1
                #Bad recordings often provide with some text ignored, thus enforce long sentences
                ,length_penalty = 0.3
                #Detect only strong redudancies
                ,compression_ratio_threshold = 5
                #Only >0 to get several propositions per values, 
                # and don't stop on suppose good single one in fact not good
                ,temperature = [
                    0.0001,
                    0.1,
                    0.2,
                    0.4,
                    0.7,
                    1.0,
                ]
                ,vad_filter = True
                #High values causes some speech parts ignored, 
                # while too small values don't cut properly between sentences
                ,vad_parameters={
                    "onset":0.01,
                    "offset":0.005,
                    "min_silence_duration_ms":800}

                )

        #XML-like output of segment time information (used by align() parser below)
        transcribed = "\n".join([f"<SEG BE='{format_time_ms(segment.start)} + {format_time_ms(segment.end - segment.start,True)} = {format_time_ms(segment.end)}'>{segment.text}" for segment in segments]).strip()
        # for segment in segments:
        #     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        # print(transcribed)

        with open(re.sub(r'[.](mp3|wav)$','.cbx.txt',recording_path), 'w', encoding='utf-8') as f:
            f.write(transcribed)
    
    #Comparison between Whisper4LQR transcription (.cbx.txt) and a previous one (.txt)
    def align(self,recording_path:str):
        html_path = re.sub(r'[.](mp3|wav)$','.html',recording_path)
        print("Building: "+html_path+" ...")
        with open(re.sub(r'[.](mp3|wav)$','.cbx.txt',recording_path), 'r', encoding='utf-8') as f:
            transcribed = f.read()
        with open(re.sub(r'[.](mp3|wav)$','.txt',recording_path), 'r', encoding='utf-8') as f:
            orig = f.read()

        orig = re.sub(r'((^|\n)(Agent|Client)[() 0-9:]+)+\n',"\n",orig).strip()

        aligner = CbxAligner()
        pairs = aligner.alignXml(orig,transcribed)

        #Build HTML trace merging identical parts
        altrace = []
        alhtml = []
        wait1 = []
        wait2 = []
        state = statePrev = 0
        for p in pairs:
            t1 = ""
            if p[0] is not None:
                t1 = p[0].token
            t2 = ""
            if p[1] is not None:
                t2 = p[1].token

            if re.sub(r'[.,\'↲-]+',"",t1).strip() == "" and re.sub(r'[.,\'↲-]+',"",t2).strip() == "":
                wait1.append(t1)
                wait2.append(t2)
                continue
            
            if t1.lower().strip() == t2.lower().strip():
                state = 0
            else:
                state = 1
                
            
            if statePrev == state:
                wait1.append(t1)
                wait2.append(t2)
                continue

            if len(wait1) > 0 or len(wait2) > 0:
                w1 = "".join(wait1)
                w2 = "".join(wait2)
                w1 = re.sub(r'\'>',"]</span>",w1)
                w2 = re.sub(r'\'>',"]</span>",w2)
                w1 = re.sub(r'<SEG BE=\'',"<span style='color: blue'>[",w1)
                w2 = re.sub(r'<SEG BE=\'',"<span style='color: blue'>[",w2)
                altrace.append(f"{w1}\t{w2}")
                if statePrev == 0:
                    color = "black"
                else:
                    color = "red"
                alhtml.append(f"<tr style='color: {color};'><td>[{w1}]</td><td>[{w2}]</td></tr>")
                wait1 = []
                wait2 = []
            
            wait1.append(t1)
            wait2.append(t2)
            statePrev = state

        if len(wait1) > 0 or len(wait2) > 0:
            w1 = "".join(wait1)
            w2 = "".join(wait2)
            w1 = re.sub(r'\'>',"]</span>",w1)
            w2 = re.sub(r'\'>',"]</span>",w2)
            w1 = re.sub(r'<SEG BE=\'',"<span style='color: blue'>[",w1)
            w2 = re.sub(r'<SEG BE=\'',"<span style='color: blue'>[",w2)
            altrace.append(f"{w1}\t{w2}")
            if statePrev == 0:
                color = "black"
            else:
                color = "red"
            alhtml.append(f"<tr style='color: {color};'><td>[{w1}]</td><td>[{w2}]</td></tr>")
            wait1 = []
            wait2 = []

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write("<!DOCTYPE html>\n<html>\n<head>\n<meta charset='UTF-8'>\n</head>\n<body>\n"
                    +"<table>\n")
            f.write("\n".join(alhtml))
            f.write("\n</table>\n</body>\n</html>\n")
