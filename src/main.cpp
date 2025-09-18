#include <Arduino.h>
#include <driver/i2s.h>
#include <math.h>

// Define PI if not already defined
#define PI 3.14159265359

// Setup ES8388 microphone (input) using AudioTools library
#include <Wire.h>
#include "AudioTools.h"
#include "AudioTools/AudioLibs/I2SCodecStream.h"
#include "AudioTools/AudioLibs/MozziStream.h"
#define MOZZI_AUDIO_MODE MOZZI_OUTPUT_EXTERNAL_CUSTOM
#include <Oscil.h>               // oscillator template

#include <tables/sin2048_int8.h> // sine table for oscillator

// Edge Impulse inference includes and globals
#include <marres_clone_inferencing.h>
static float *features = NULL;

int gain = 10; // change if too loud or silent

float score_threshold = 0.5f; // Threshold for bird detection

// Forward declarations

void setupES8388Microphone();
void recordAndSendAudio();
void playbackAudio();

void printAverageAmplitude();
void playTestTone();

void readES8388Data(); // For ES8388 input

void runInferenceOnRecordedAudio(); // Run Edge Impulse inference
bool extractFeaturesFromBuffer();
int raw_feature_get_data();


// Configuration constants
#define SAMPLE_RATE 16000 // 16kHz sample rate
#define BUFFER_SIZE 1024
#define RECORD_TIME_MS 1000 // 1-second recording

// ES8388 Codec pins (from reference code)
#define ES8388_I2C_SDA 4
#define ES8388_I2C_SCL 5
#define ES8388_I2C_ADDR 0x10
#define ES8388_I2C_SPEED 100000
#define ES8388_I2S_MCLK 0
#define ES8388_I2S_BCLK 15
#define ES8388_I2S_LRCK 9
#define ES8388_I2S_DOUT 6 // Data Out from ES8388 to MCU
#define ES8388_I2S_DIN 10 // Data In from MCU to ES8388

    // Variables for audio processing
    int sumAmplitude = 0;
int sampleCount = 0;
int16_t minAmplitude = INT16_MAX;
int16_t maxAmplitude = INT16_MIN;
unsigned long lastPrintTime = 0;
const int printInterval = 1000; // 1 second

AudioInfo es8388_info(SAMPLE_RATE, 1, 16);
DriverPins es8388_pins;
AudioBoard es8388_board(AudioDriverES8388, es8388_pins);
I2SCodecStream es8388_i2s(es8388_board);
TwoWire es8388_wire = TwoWire(0);
MozziStream mozzi; // audio source

StreamCopy copier(es8388_i2s, mozzi); // copy source to sink

Oscil<SIN2048_NUM_CELLS, SAMPLE_RATE> aSin(SIN2048_DATA);

// Buffer to store recorded audio for playback
int16_t *recordedAudio = NULL;
int recordedSamples = 0;

bool playTone = false;
bool playRecording = false;

unsigned long testToneStartTime = 0;
const int frequency = 1000; // 1 kHz tone
const int duration_s = 1;   // 1 second duration
// test tone
int samples_count = 0;

// recording playback
int recorded_samples = 0;

// Command flags for main loop
volatile bool flag_record = false;
volatile bool flag_playback = false;
volatile bool flag_testtone = false;




void setup()
{
  Serial.begin(115200);
  while (!Serial)
  {
    delay(10);
  }
  Serial.println("ESP32 Audio System");

  // setup mozzi
  auto cfg = mozzi.defaultConfig();
  cfg.control_rate = CONTROL_RATE;
  cfg.copyFrom(es8388_info);
  mozzi.begin(cfg);

  Serial.println("Configuring ES8388 codec (input)...");
  es8388_pins.addI2C(PinFunction::CODEC, ES8388_I2C_SCL, ES8388_I2C_SDA, ES8388_I2C_ADDR, ES8388_I2C_SPEED, es8388_wire);
  es8388_pins.addI2S(PinFunction::CODEC, ES8388_I2S_MCLK, ES8388_I2S_BCLK, ES8388_I2S_LRCK, ES8388_I2S_DOUT, ES8388_I2S_DIN);
  es8388_pins.begin();

  CodecConfig cfg_i2s;
  cfg_i2s.output_device = DAC_OUTPUT_ALL;
  cfg_i2s.input_device = ADC_INPUT_LINE1; // or ADC_INPUT_ALL for all inputs
  es8388_board.begin(cfg_i2s);

  auto io = es8388_i2s.defaultConfig(RXTX_MODE);
  io.copyFrom(es8388_info);
  io.output_device = DAC_OUTPUT_ALL;
  io.input_device = ADC_INPUT_LINE1; // Use only LINE1 (mono)
  io.buffer_size = BUFFER_SIZE;
  io.channels = 1; // Force mono mode
  es8388_i2s.begin(io);
  es8388_i2s.setVolume(1.0f);
  es8388_i2s.setInputVolume(0.6f);
  Serial.println("ES8388 codec configured for mono input.");

  // setup mozzi sine
  aSin.setFreq(1000); // set the frequency

  delay(500);

  // Play a test tone to verify the speaker is working
  playTestTone();
}

void loop()
{
  copier.copy();

  // Serial command parsing moved here from updateControl()
  if (!playTone && !playRecording)
  {
    if (Serial.available() > 0)
    {
      char incomingByte = Serial.read();
      if (incomingByte == 'r')
      {
        flag_record = true;
      }
      else if (incomingByte == 'p')
      {
        flag_playback = true;
      }
      else if (incomingByte == 't')
      {
        flag_testtone = true;
      }
    }
  }

  // Handle serial command flags in main loop
  if (flag_record)
  {
    flag_record = false;
    Serial.println("Starting recording...");
    recordAndSendAudio();
  }
  if (flag_playback)
  {
    flag_playback = false;
    Serial.println("Playing back recorded audio...");
    playbackAudio();
  }
  if (flag_testtone)
  {
    flag_testtone = false;
    Serial.println("Playing test tone...");
    playTestTone();
  }
}

AudioOutputMozzi updateAudio()
{

  if (playTone)
  {
    samples_count++;
    return (aSin.next() * 100) >>
           8; // shift back to STANDARD audio range, like /256 but faster
  }
  else if (playRecording)
  {
    recorded_samples++;
    return (recordedAudio[recorded_samples - 1]) / 100;
  }
  else
  {
    return 0;
  }
}

void updateControl()
{
  if (playTone)
  {
    if (samples_count >= duration_s * SAMPLE_RATE)
    {
      playTone = false;
      Serial.println("Test tone complete.");
    }
  }
  if (playRecording)
  {
    if (recorded_samples >= (SAMPLE_RATE * (RECORD_TIME_MS / 1000)))
    {
      playRecording = false;
      Serial.println("Playback complete.");
    }
  }

  if (!playTone && !playRecording)
  {
    // Optional: continue monitoring audio levels
    readES8388Data();
    // printAverageAmplitude();
  }
}

// Function to play a test tone to verify speaker operation
void playTestTone()
{

  Serial.printf("Playing %d Hz test tone for %d ms...\n", frequency, duration_s * 1000);

  // Generate and play the sine wave in chunks
  samples_count = 0;

  playTone = true;
}

// Function to record audio and send it over serial

void recordAndSendAudio()
{
  const int total_samples = SAMPLE_RATE * (RECORD_TIME_MS / 1000);

  // Free any previous recording
  if (recordedAudio != NULL)
  {
    free(recordedAudio);
    recordedAudio = NULL;
  }

  // Allocate memory for new recording
  recordedAudio = (int16_t *)malloc(total_samples * sizeof(int16_t));

  if (recordedAudio == NULL)
  {
    Serial.println("Failed to allocate memory for audio buffer");
    return;
  }

  size_t bytesRead = 0;
  recordedSamples = 0;

  unsigned long startTime = millis();
  unsigned long lastSampleTime = micros();
  const unsigned long sampleInterval = 1000000 / SAMPLE_RATE; // microseconds per sample

  // Read data until we have enough samples or timeout
  while (recordedSamples < total_samples && (millis() - startTime) < RECORD_TIME_MS + 500)
  {
    // ES8388 input using AudioTools
    size_t bytesReadThisTime = es8388_i2s.readBytes((uint8_t *)&recordedAudio[recordedSamples],
                            min((total_samples - recordedSamples) * sizeof(int16_t), BUFFER_SIZE * sizeof(int16_t)));
    if (bytesReadThisTime > 0)
    {
      recordedSamples += bytesReadThisTime / sizeof(int16_t);
      bytesRead += bytesReadThisTime;
    }
  }

  // Run inference on the recorded audio
  runInferenceOnRecordedAudio();

  // Send header information as text
  Serial.println("BEGIN_AUDIO");
  Serial.println(SAMPLE_RATE);
  Serial.println(recordedSamples);

  // Send a marker to indicate binary data start
  Serial.println("BEGIN_BINARY");

  // Send the audio data as binary
  Serial.write((uint8_t *)recordedAudio, recordedSamples * sizeof(int16_t));

  // Send a text marker to indicate end
  Serial.println();
  Serial.println("END_AUDIO");
  Serial.printf("Sent %d samples (%d bytes) recorded over %d ms\n",
          recordedSamples, bytesRead, (millis() - startTime));

  Serial.println("Recording saved in memory. Send 'p' to play it back.");

  
}

void playbackAudio()
{
  if (recordedAudio == NULL)
  {
    Serial.println("ERROR: recordedAudio buffer is NULL");
    return;
  }

  if (recordedSamples <= 0)
  {
    Serial.printf("ERROR: Invalid recordedSamples: %d\n", recordedSamples);
    return;
  }

  Serial.printf("Playing back %d samples from buffer at %p...\n", recordedSamples, recordedAudio);
  recorded_samples = 0;
  playRecording = true;

  //   Serial.printf("Played %d samples (%d bytes) in %d ms\n",
  //            samplesPlayed, bytesWritten, duration);
}

void readES8388Data()
{

  int16_t rawBuffer[BUFFER_SIZE];
  size_t bytesRead = es8388_i2s.readBytes((uint8_t *)rawBuffer, BUFFER_SIZE * sizeof(int16_t));
  if (bytesRead == 0)
    return;
  int numSamples = bytesRead / sizeof(int16_t);
  for (int i = 0; i < numSamples; i++)
  {
    int16_t sample = rawBuffer[i];
    int absSample = abs(sample);
    sumAmplitude += absSample;
    sampleCount++;
    if (sample < minAmplitude)
      minAmplitude = sample;
    if (sample > maxAmplitude)
      maxAmplitude = sample;
  }
}

void printAverageAmplitude()
{
  if (millis() - lastPrintTime >= printInterval)
  {
    if (sampleCount > 0)
    {
      int avgAmplitude = sumAmplitude / sampleCount;

      Serial.printf("📊 Average Amplitude: %d | Min: %d | Max: %d\n",
                    avgAmplitude, minAmplitude, maxAmplitude);

      // Reset counters
      sumAmplitude = 0;
      sampleCount = 0;
      minAmplitude = INT16_MAX;
      maxAmplitude = INT16_MIN;
    }

    lastPrintTime = millis();
  }
}

// Edge Impulse: required function to provide data to the classifier
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr)
{
  memcpy(out_ptr, features + offset, length * sizeof(float));
  return 0;
}

// Convert int16_t audio buffer to float features with min-max normalization
bool extractFeaturesFromBuffer(const int16_t *audio, size_t audio_len, float *features, size_t feature_size)
{
  // Find min and max
  int16_t min_val = INT16_MAX;
  int16_t max_val = INT16_MIN;
  for (size_t i = 0; i < audio_len; i++)
  {
    if (audio[i] < min_val)
      min_val = audio[i];
    if (audio[i] > max_val)
      max_val = audio[i];
  }
  if (min_val == max_val)
  {
    min_val = -1;
    max_val = 1;
  }
  // Normalize and copy to features
  size_t i = 0;
  for (; i < audio_len && i < feature_size; i++)
  {
    float sample = audio[i];
    // Min-max normalization to [-1, 1]
    float norm = (sample - min_val) / (float)(max_val - min_val) * 2.0f - 1.0f;
    features[i] = norm;
  }
  // Zero-pad if needed
  for (; i < feature_size; i++)
  {
    features[i] = 0.0f;
  }
  return true;
}

// Run inference on the recorded audio buffer
void runInferenceOnRecordedAudio()
{
  if (!recordedAudio || recordedSamples <= 0)
  {
    Serial.println("No audio recorded for inference.");
    return;
  }
  if (!features)
  {
    features = (float *)malloc(EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE * sizeof(float));
    if (!features)
    {
      Serial.println("Failed to allocate features buffer");
      return;
    }
  }
  // Timing: feature extraction
  unsigned long t_feat_start = micros();
  bool feat_ok = extractFeaturesFromBuffer(recordedAudio, recordedSamples, features, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
  unsigned long t_feat_end = micros();
  float feat_time_ms = (t_feat_end - t_feat_start) / 1000.0f;
  if (!feat_ok)
  {
    Serial.println("Feature extraction failed");
    return;
  }
  // Timing: inference
  signal_t_ei signal;
  signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
  signal.get_data = [](size_t offset, size_t length, float *out_ptr) {
    return raw_feature_get_data(offset, length, out_ptr);
  };
  ei_impulse_result_t result = {0};
  unsigned long t_inf_start = micros();
  EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);
  unsigned long t_inf_end = micros();
  float inf_time_ms = (t_inf_end - t_inf_start) / 1000.0f;
  if (res != EI_IMPULSE_OK)
  {
    Serial.printf("ERR: Failed to run classifier (%d)\n", res);
    return;
  }
  // Find bird class index
  float bird_score = 0.0f;
  int bird_index = -1;
  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++)
  {
    if (strcmp(ei_classifier_inferencing_categories[ix], "bird") == 0)
    {
      bird_index = ix;
      break;
    }
  }
  if (bird_index >= 0)
  {
    bird_score = result.classification[bird_index].value;
  }
  else
  {
    bird_score = result.classification[0].value;
    Serial.println("Warning: No 'bird' class found, using first class as bird");
  }
  Serial.printf("Inference result: %s (score: %.3f) | Feature extraction: %.2f ms | Inference: %.2f ms\n",
    bird_score > score_threshold ? "BIRD" : "NO_BIRD", bird_score, feat_time_ms, inf_time_ms);
}