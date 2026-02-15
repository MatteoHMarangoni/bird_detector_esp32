# Bird Detector ESP32  
Low-power embedded bird detector for ESP32 Arduino framework. 
Developed within the project Chorusing Symbionts. 

---

## Summary

This repository documents a complete pipeline to build a **low-power, real-time bird detector** running entirely on an ESP32 microcontroller using Arduino code.

The system:

- Continuously captures environmental audio
- Runs embedded machine learning inference locally
- Classifies sound as **bird** or **non-bird**
- Operates without any cloud connection
- Is optimized for low latency and low power consumption

The classifier is trained on a custom ecoacoustic dataset and deployed using **Edge Impulse**.

A complete step-by-step DIY guide (hardware, dataset preparation, training, deployment, evaluation) will be added here soon.

---

## Context: Chorusing Symbionts

This bird detector was developed as part of the sound art installation **Chorusing Symbionts**.
https://matteomarangoni.com/Chorusing-Symbionts-page

In this project, artificial creatures generate music in public parks in response to each other and the environment. 
One project goal is to enable these artificial creatrures to detect bird vocalisations and respond through conflict-avoidant behaviours.

Project constraints require:

- Fully on-device inference (no remote server)
- ESP32 microcontroller 
- Very low power consumption (battery + solar)
- Low latency for real-time musical interaction
- Recognition of common bird species at the exhibition site

At the time of development (2024–2025), no off-the-shelf solution met these constraints.

---

## Edge Impulse Project

Public project:

https://studio.edgeimpulse.com/public/806211/live

Details:

- Labels: `bird`, `noise`
- Audio: 16bit, 16 kHz wav
- Dataset size: ~27 hours
- 102,000+ samples
- MFCC feature extraction
- Lightweight neural network classifier
- Optimised for embedded deployment

The dataset was curated to remove intelligible human speech for privacy protection.

---

## Repository Branches

This repository contains two main hardware configurations:

### `main`
For use with **custom electronics**:
- ESP32-S3 (8MB PSRAM)
- ES8388 audio codec
- CMC-4015-25T electret microphone
- Full differential microphone circuit
- Adjustable preamp gain

This configuration provides significantly improved performance compared to common MEMS microphone breakouts.

### `INMP441`
For use with:
- ESP32 board
- INMP441 I2S MEMS microphone breakout

This is suitable for rapid prototyping and workshops.

---

## Pipeline Overview

The complete workflow consists of four stages:

1. **Data Collection**
2. **Data Preparation**
3. **Model Training**
4. **Deployment**

Full documentation will be added soon. Below is a high-level overview.

---

## 1. Data Collection

Data is collected using:

- Raspberry Pi 4B
- Clippy EM272Z1 microphone
- Rode AI Micro interface
- BirdNET-Pi
- Custom parallel sampling script

BirdNET-Pi automatically collects bird vocalisations.  
A custom script collects non-bird environmental audio when no bird is detected.

Collection typically runs for 4–8 weeks per exhibition site.

Raw dataset format:

- Bird: 3 sec, 16bit, 48kHz WAV
- Non-bird: 15 sec, 416bit, 48kHz WAV

---

## 2. Data Preparation

Processing is performed locally using Python 3.11 inside a virtual environment.

Steps:

- Segment 3s / 15s recordings into 1s chunks
- Remove silence
- Filter low-confidence BirdNET detections
- Remove species mismatches
- Remove bird audio from non-bird category
- Downsample 48kHz → 16kHz
- Remove intelligible human speech (privacy filtering)
- Balance bird vs. non-bird categories

Target output:

- 1 sec mono
- 16 kHz, 16bit
- Clean bird vs noise folders
- Approximately balanced classes

Python dependencies include:


---

## 3. Model Training (Edge Impulse)

- MFCC feature extraction
- Lightweight neural network
- EON compiler
- Quantized INT8 deployment

Free Edge Impulse account is sufficient.

---

## 4. Deployment

Hardware options:

### Option A: MEMS (INMP441)
- Simple wiring
- Lower performance

### Option B: Custom Board (ES8388 + electret mic)
- Higher gain 
- Better distant detection
- Improved real-world accuracy

Firmware is built using:

- PlatformIO
- Espressif 32 platform
- WEMOS LOLIN S3 PRO board profile

Deployment steps:

1. Export model from Edge Impulse as Arduino Library (EON, int8)
2. Place library folder in `/lib`
3. Include the generated header in `main.cpp`
4. Flash firmware

---

## Measuring Accuracy in Real Conditions

Edge Impulse test accuracy does not equal field performance.

A hardware-in-the-loop evaluation system is included:

- Play labeled audio via speakers
- Capture with ESP32 + mic
- Run inference
- Record results
- Compare predictions vs ground truth

Evaluation script:

`auto_play_record_test.py`

Used to measure:
- True positive rate
- False positive rate
- Confidence margins
- Latency
- Real-world performance difference between microphones

---

## License

### Dataset & Trained Model
Creative Commons Attribution 4.0 International (CC BY 4.0)

You may share and adapt the dataset and trained model with appropriate attribution.

Full license:
https://creativecommons.org/licenses/by/4.0/

### Code / Firmware
3-Clause BSD License  
(Edge Impulse generated inference code + project firmware)

See `LICENSE` file for details.

Third-party libraries (e.g., TensorFlow Lite) remain under their respective open-source licenses.

---

## Credits

This project is developed by Matteo Maragoni with support from the Chorusing Symbionts project team:
- Ahnjili Zhuparris, project advisor for machine learning, development data collection script  
- Stephan Olde, IoT developer, development deployment code  
- Matthijs Munnik, custom electronics design  
- Niels Gräber, project intern, data preparation  

The publication of this DIY Bird Detector is supported by V2, Rotterdam, Microdosing A.I. residency.

Chorusing Symbionts is produced in collaboration with:
- Crossing Parallels, Delft  
- Instrument inventors initiative, The Hague  
- Zuiderparktheatre, The Hague  
- Marres, Maastricht  
- Zone2Source, Amsterdam  
- STUK, Leuven  

Scientific advice:
- TU Delft, Crossing Parallels  
- TU Eindhoven, Innovation Space, Eindhoven Artificial Intelligence Systems Institute  
- Naturalis Biodiversity Center, Arise project, Leiden  

Funding:
- Creative Industries fund NL  
- Performing Arts Fund NL  
- Municipality of The Hague


## Documentation

A complete DIY guide covering:

- Hardware schematics
- Data collection setup
- Python data processing scripts
- Model training configuration
- Deployment instructions
- Evaluation methodology

will be added soon.

