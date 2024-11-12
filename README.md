## Whisper for Low-Quality Recordings

Whisper4LQR is an adaptation of [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper).

Whisper4LQR's purpose is to provides better recognition capabilities on low-quality recordings, even if it may take significantly more computation time than Faster-Whisper.

Transcription optimisation:
- Pre-processing to extract, clean, and enhance voices with minimal noise.
- VAD (Voice Activity Detection) used to segment sentences as effectively as possible.
- Prompts applied to each segment to ensure a better vocabulary recognition across the entire recording.
- Intensive generation of hypotheses selected based on word count and compression (redundancy) criteria.
- Combined parameters, balanced for the whole process optimisation.

To have a look at Faster-Whisper modifications, [search for #CBX](https://github.com/search?q=repo%3AEtienneAb3d%2FWhisper4LQR%20%23CBX&type=code) in the repository.

*Developed with the help of our partner [Feedae](https://www.feedae.com/)*

## Installation

**Check ffmpeg version >=4.4**
```sh
ffmpeg -version

Output should be:
=================
ffmpeg version 4.4.3-0ubuntu1~20.04.sav2 Copyright (c) 2000-2022 the FFmpeg developers
[...]

Install latest:
===============
sudo add-apt-repository -y ppa:savoury1/ffmpeg4
sudo apt-get -qq install -y ffmpeg

```

**Whisper4LQR installation**

```sh
git clone https://github.com/EtienneAb3d/Whisper4LQR.git
cd Whisper4LQR
pip install -r requirements.txt
```

## File pre-processing

```python
from CbxPre import CbxPre

recording_paths = [...path list...]

cbxPre = CbxPre()

for recording_path in recording_paths:
    #Pre-processing (output a ".cbx.wav" file)
    cbxPre.process(recording_path=recording_path)

```

## Transcription

```python
from CbxSTT import CbxSTT

initial_prompt=(""
            # Insert your vocabulary as short expressions, using comma and points.
            # Max 224 tokens, counted while running.
            # Example:
            +" Run the software, save the file, a secure configuration."
            +" Linux, Mac OS X, Windows."
            # Keep these two last lines to ensure a small distance between 
            # the first future transcribed words and the above prompt words
            +" Please be patient."
            +" Beh Hein Ouais Heu Hmm Hum Ok..."
            )

recording_paths = [...path list...]

cbxSTT = CbxSTT(language="en")

for recording_path in recording_paths:
    #Transcribe (needs a ".cbx.wav" file)
    cbxSTT.process(initial_prompt=initial_prompt,recording_path=recording_path)
    #Compare with a previous transcription (".txt" vs ".cbx.txt")
    cbxSTT.align(recording_path=recording_path)

```


<hr>
This tool is a demonstration of our know-how.<br/>
If you are interested in a commercial/industrial AI linguistic project, contact us:<br/>
https://cubaix.com
