# Bird detector ESP32

This repo is used to infer bird recordings localy on an S3 Pro 

Update 16/09/2025 

- The code was edited and cleaned up by Stephan to use it to run inference on an ESP32 Lolin S3 Pro, using files on an sd card inserted into the microcontroller

- older models didn't seem to run correctly, always having a very low accuracy. 
Thus, Stephan made a new Edge Impulse Model with the following settings:

Create impulse:
Windows Size: 1000ms
Window Increase: 500ms
Frequency: 16000hz
Zero-pad data: enabled

Audio MFCC block
Classification block
Output features 2 (bird, noise)

For the MFCC tab parameters I used the autotune parameters button and ran it with the following:
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


- Running this model on the validation data of the training set (1770 bird  samples of 1 sec and 1770 no_bird samples of 1 sec) gave the following results:
===== FINAL STATISTICS =====
Bird samples: 1770, correctly identified: 1583 (89.44%)
No-bird samples: 1770, correctly identified: 1514 (85.54%)
Overall accuracy: 87.49% (3097 of 3540 correct)
Inference timing - Avg: 0.047 s, Min: 0.047 s, Max: 0.051 s
Loading timing - Avg: 0.253 s, Min: 0.159 s, Max: 5.210 s

- Niels also added an .rtf file with the output of the serial monitor, this is not the entire thing, as he was not aware the memory would run out after a 1000 lines 

- Niels is working on documenting how to use the model 

HOW TO USE:

- download the correct edge impulse model

- put the folder with the correct model in the /lib folder 

- copy your bird recordings into the /sdcard/bird folder
(it is important that your recordings are seperated into 1 sec, mono, 16bit, 16khz raw segments)

- copy your noise/non-bird recordings into the sdcard/no_bird folder

- copy the both folders into an empty sd card

- insert sd card in your microcontroller's micro sd slot (Lolin S3 Pro) 
(contacts facing down!)

- build the main sketch

- upload the main sketch

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

