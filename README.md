# Bird Detector ESP32

This branch provides a bird detector running localy on an ESP32 S3 Pro MCU with a hook to a python script to be run on a computer. 
It is based on a audio classifier trained with Edge Impulse on a custom dataset of bird and environmental audio. 
It runs the classifier using a microphone connected to an es8388 board connected to an ESP32 S3 Pro. It uses the [Arduino Audio Tools] (https://github.com/pschatzmann/arduino-audio-tools) library created by Phil Schatzmann as audio driver/interface in the code for the ESP32.

The ESP32 code can be used together with the python script audio_receiver_and_playback.py to run interferences on the microphone as well as storing the recorded files in a recordings folder in the project directory for analysis later. There will be a latest_recording.wav file with the latest audio sample for easy playing of the latest audio sample on the computer as well as all audio files being stored with timestamps. All audio files include the clasification from the esp32 as well as the confidence of that inference. As an example: bird_0p960 means that the classification was bird with a score of 0.960. by default, scores >0.5 are classified bird and scored <=0.5 are classified as nobird.



How to use:

0.1 Have vscode with platformio installed.
0.2 install a python environment and install the libraries from python_script/requirements.txt.
0.2.1 If an error occurs that serial is not recognized, the library pyserial should be installed via pip install pyserial.

1. compile and upload code to esp32.
1.1 detection threshold can be changed by changing line 24 in src/main.cpp: float score_threshold = 0.5f; Anything over will be labeled bird anything under or equal will be labeled no bird.
1.2 optionally, change line 22 in src/main.cpp: int gain = 10; to adjust the microphone input gain if it is too loud or quiet.
2. Change the port in line 12 of python_script/audio_receiver_playback.py to match the port of the esp32 on your device. example: DEFAULT_PORT = "COM9"
3. run python_script/audio_receiver_and_playback.py
4. send r in the serial monitor through the python script to record new samples and store the recordings in the recordings folder in the github project folder.
4.1 the command p, plays back the last sample on the esp32 via the speaker/headphones on the es8388. The volume/gain of this has been lowered by a lot so this function is not completely functional.
5. See the serial monitor output for details on the classification and check the recordings folder for the audio files.

Details in the serial monitor:

After sending the "r" command you will get the inference result and the timing of preprocessing and inferences on the ESP32. Preporcessing is the time for feature extraction which is the conversion of the saved audio file in memory on the ESP32 to the proper format for the edge impulse inferences. There also is the time the inference took on the ESP32. Also, the python script will calculate the dominant frequency in the received audio clip from the ESP32 to be able to do microphone frequency tests.

An example of the serial monitor output:

Sent recording command to ESP32
ESP32: Starting recording...
ESP32: Inference result: NO_BIRD (score: 0.406) | Feature extraction: 6.85 ms | Inference: 47.27 ms
Recording started. Sample rate: 16000Hz, Expected samples: 16000
Received 16000 samples (32000 bytes) in 0.12 seconds
ESP32: Sent 16000 samples (32000 bytes) recorded over 1201 ms
ESP32: Recording saved in memory. Send 'p' to play it back.
Audio saved to \recordings\nobird_0p406_124637.wav
Audio also saved as \recordings\latest_recording_nobird_0p406.wav      
Dominant frequency: 50.00 Hz




Update 18/09/2025

Some notes:
The audio driver library and edge impulse both use signal_t as a type. This has been solved by replacing all signal_t references in the edge impulse library by signal_t_ei through a simple replace all within that folder, make sure that match whole word is enabled in the search (this is the ab with a line under it in the search box in vscode).







Update 16/09/2025 

The dataset was collected in the garden of the Museum Marres in Maastricht in July 2025 and containes the following. 

Bird category:

Blackbird
Blackcap
Chaffinch
Magpie
Pigeon / Dove
Robin
Swift
Dunnock
Grey Spotted Woodpecker
Green Woodpecker

Noise category includes:

people talking, dinner, large gathering, public speech
children playing in school yard
music event
rain, wind
cars, airplanes
construction work
church bells

The dataset contains a total of 6h 22m of audio, of which 2h52m bird audio and 2h28m noise, and is subdivided further as 84% training and 16% testing. 
Audio is formatted 16khz, 16 bits mono raw. 
Audio was gathered using as hardware a Raspberry PI attached with a Rode AI-micro audio interface and a Clippy EM272Z1 Mono Microphone and as software BirdNet-PI in parallel with a custom script that collects environmental audio when no bird detections are occurring an an amplitude threshold is crossed. 
Bird audio was further screened using Birdnet Analyser, purging all 3s chunks with confidence < 0.9. 
Noise audio was also screened using Birdnet Analyser, purging chunks with confidence > 0.1. 

The classifier is based on MFCC (see below for parameters)
Edge Impulse rates the classifier at 87% accuracy, with estimated 8 ms inference time 12.5k peak ram usage, 45.7k flash usage, when running the 8 bit quantized model with the EON compiler on an ESP32. 

Accuracy was confirmed when running the model on the MCU over the entire testing set, however latency was an order of magnitude higher.


Training settings for the classifer on Edge Impulse: 

Create impulse:
Windows Size: 1000ms
Window Increase: 500ms
Frequency: 16000hz
Zero-pad data: enabled

Audio MFCC block
Classification block
Output features 2 (bird, noise)

For the MFCC tab parameters: 
Number of coefficients: 13
Frame length: 0.02
Frame stride: 0.02
Filter number: 32
FFT Length: 256
Nomalization window size: 101
Low frequency: 0
High frequency : not set
Coefficient 0.98

Classifier tab:
Number of training cycles: 100
Learned optimizer: disabled
Learing rate 0.005
Training processor: 0.005
Advanced training settings; None set
Data augmentation: off


- Running this model on the MCU on the entire testing set loaded on an SD card (1770 bird  samples of 1 sec and 1770 no_bird samples of 1 sec) gave the following results:

Bird samples: 1770, correctly identified: 1583 (89.44%)
No-bird samples: 1770, correctly identified: 1514 (85.54%)
Overall accuracy: 87.49% (3097 of 3540 correct)
Inference timing - Avg: 0.047 s, Min: 0.047 s, Max: 0.051 s
Loading timing - Avg: 0.253 s, Min: 0.159 s, Max: 5.210 s


HOW TO USE:

Note that the dataset was removed from this public repo for privacy reasons (as it contains fragments of speech recorded in public space), to use you need to add your own data. 

- Create the following two folders in the root of an SD card:
/bird
/no_bird

Format data to 1 sec samples, mono, 16bit, 16khz raw .wav

- copy your data into the two folders

- insert sd card in your microcontroller's micro sd slot (Lolin S3 Pro) 
(contacts facing down!)

- build the main sketch

- upload the main sketch

- set terminal baud rate to 115200

- watch terminal output for accuracy 

//////////////////////////////////////////////////////////////

16/05/2025

Main.cpp runs an audio inference on the esp32 (regular) with continous inferences on 2 seconds of audio with a moving window of 0.5 seconds. If the last 4 detections are >90% a bird then a bird is detected and the code will pause for 1 second. This can be customized by changing the following parameters on top of the file;
#define DETECTION_THRESHOLD 0.9          // Minimum confidence for "bird" detection
#define REQUIRED_WINDOWS_OVER_THRESHOLD 4 // How many slices in window must be over threshold
#define COOLDOWN_DELAY_MS 1000  

This code is also found in microphone_inference
ei_or_tinychip file runs inferences on sd card files based on line 9 between edge impulse and tinychirp. The model parameters for tinychirp have been updated.

27/04/2025
Comment line 9 out to use the tinychirp port, uncomment it to use the edge impulse infereneces.
//#define USE_EDGEIMPULSE 

18/04/2025
There are two codes to try with an esp32 wrover (lolin d32) and an sd card (on lolin d32 pro the sd card slot is included) these codes are main.cpp and main_edgeImpulse_sdcard. Both load files from an sd card, the files from the sd card folder. Main.cpp uses the direct port from tinychirp and does not seem to classify bird correctly, unknown if the model they included was working to begin with but it is fast, stats are down below. As an alternative I uploaded our trained model transformer_time_quantized_20250317_140229_dp1000.tflite to edge impulse which takes a trained tflite model (does not support .pt but it can be converted to onnx which is supported). Edge impulse optimizes the trained network and deploys it in a neat arduino library which is easy to use. The inferences run a bit slower but do have a way better prediction statistic, which means that our device is capable and we just need to make sure we have a proper model to use in the transformer time model which I will work on later. To test the edge impulse case you have to rename main_edgeimulse_sdcard to main.cpp and then you can upload. The platformio.ini file was also changed to support the psram and the esp32 wrover.


Edge impulse
--- Current Statistics ---
Bird samples: 20, correctly identified: 19 (95.00%)
No-bird samples: 20, correctly identified: 11 (55.00%)
Overall accuracy: 75.00% (30 of 40 correct) 


Tinychirp transformer time

----- Current Statistics -----
Bird samples: 100, correctly identified: 4 (4.00%)
No-bird samples: 100, correctly identified: 100 (100.00%)
Overall accuracy: 52.00% (104 of 200 correct)


- I kept the files of the rest of the project Komorebi project included but for now they are not called in main.cpp which runs all the code. In main.cpp audio is read from the microphone and then processed by the machine learning algorithm. For now, just 1 second is processed because of memory limitations which does not allow for 3 seconds inference which would require a workaround. Also, we dont aim to do 3 seconds so work on getting that to work might not be the most efficient use of time. 

- For now I just used the wm8960 library from sparkfun as it was easier to tweak than the mozzi compatible one. This handles the i2c configuration of the wm8960 but otherwise uses the general i2s functions. I needed to boost the input gain on my mic to be able to get inputs (  codec.setLMICBOOST(WM8960_MIC_BOOST_GAIN_29DB); // Boost if needed). Also the wm8960 supports two mics so we need to just use one, I chose the left one but if you are using your setup you might need to change it to the right one if the mic is not registering. 

- When uploaded the serial monitor will prompt to press r to start recording and will return some audio data and the inference.

- To just run a test inference on a random sample, simple_test_code_input can be renamed to main.cpp or copied into it and uploaded to do the test. 

- I was not able to get a positive result for bird detection yet but the basics are there now so we can built further on that.

todo:
- filter out noise from microphone
- 3 seconds inference instead of 1

