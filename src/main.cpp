// INMP441 I2S microphone input and serial streaming

#include <Arduino.h>
#include <driver/i2s.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <Bird_Detector_ESP32_inferencing.h>

// Choose one mode:
#define CLASSIFIER_MODE// enable to have continous classification in regular serial monitor
//#define COMMUNICATION_MODE // enable to stream audio and classifications to host via a python file

//#define LED_PIN 15 // If defined an led will turned on (check the pin) when the inference returns a  positive classification for visual feedback.

// --- Configuration constants ---
#define SAMPLE_RATE 16000        // 16 kHz
#define RECORD_TIME_MS 1000      // 1 second
#define BUFFER_SIZE 1024         // samples per chunk
// Software gain to apply to samples (in dB). Set to 0.0 for no change.
#define AUDIO_GAIN_DB 10.0f
// Warm-up time in milliseconds to let the ADC/I2S settle before capturing
#define WARMUP_MS 1000

// --- INMP441 I2S pins ---
#define I2S_IN_SCK 6 // Serial Clock
#define I2S_IN_WS 5  // Word Select (Left/Right Clock)
#define I2S_IN_SD 7  // Serial Data

#define I2S_PORT I2S_NUM_0

// --- I2S RX init (MCU master providing BCLK/LRCK) ---
static void i2s_init_rx() {

   const i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = 16000,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT, // Use left channel
      .communication_format = I2S_COMM_FORMAT_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = 1024,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0};

  const i2s_pin_config_t pin_config = {
      .bck_io_num = I2S_IN_SCK,
      .ws_io_num = I2S_IN_WS,
      .data_out_num = I2S_PIN_NO_CHANGE, // We're only reading
      .data_in_num = I2S_IN_SD};

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);

}

// --- Edge Impulse: file-local inference helpers ---
static float *features = NULL;
static float inference_score_threshold = 0.5f;

// --- Static DRAM audio buffer (reused across recordings) ---
static int16_t *audio_buf = nullptr;
static size_t audio_buf_bytes = 0;

// --- Simple level monitoring (like in inmp441.cpp) ---
static int sumAmplitude = 0;
static int sampleCount = 0;
static int16_t minAmplitude = INT16_MAX;
static int16_t maxAmplitude = INT16_MIN;
static unsigned long lastPrintTime = 0;
static const int printInterval = 1000; // ms

void readI2SData()
{

  int16_t rawBuffer[BUFFER_SIZE];
  size_t bytesRead;

  esp_err_t result = i2s_read(I2S_NUM_0, &rawBuffer, sizeof(rawBuffer), &bytesRead, 100);

  if (result != ESP_OK || bytesRead == 0)
  {
    return;
  }

  int numSamples = bytesRead / sizeof(int16_t);

  for (int i = 0; i < numSamples; i++)
  {
    int16_t sample = (int16_t)(rawBuffer[i]);
    if (i < 20)
    {
      // Serial.println(sample);
    }
    int absSample = abs(sample);

    sumAmplitude += absSample;
    sampleCount++;

    if (sample < minAmplitude)
      minAmplitude = sample;
    if (sample > maxAmplitude)
      maxAmplitude = sample;
  }

}

static int raw_feature_get_data(size_t offset, size_t length, float *out_ptr)
{
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}

static bool extractFeaturesFromBuffer(const int16_t *audio, size_t audio_len, float *out_features, size_t feature_size)
{
    // Find min and max
    int16_t min_val = INT16_MAX;
    int16_t max_val = INT16_MIN;
    for (size_t i = 0; i < audio_len; i++)
    {
        if (audio[i] < min_val) min_val = audio[i];
        if (audio[i] > max_val) max_val = audio[i];
    }
    if (min_val == max_val)
    {
        min_val = -1;
        max_val = 1;
    }
    // Normalize and copy to features (min-max to [-1,1])
    size_t i = 0;
    for (; i < audio_len && i < feature_size; i++)
    {
        float sample = audio[i];
        float norm = (sample - min_val) / (float)(max_val - min_val) * 2.0f - 1.0f;
        out_features[i] = norm;
    }
    for (; i < feature_size; i++)
    {
        out_features[i] = 0.0f;
    }
    return true;
}

static void runInferenceOnRecordedAudio(const int16_t *audio, size_t audio_len)
{
    if (!audio || audio_len == 0)
    {
        Serial.println("No audio provided for inference.");
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

    unsigned long t_feat_start = micros();
    bool feat_ok = extractFeaturesFromBuffer(audio, audio_len, features, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    unsigned long t_feat_end = micros();
    float feat_time_ms = (t_feat_end - t_feat_start) / 1000.0f;
    if (!feat_ok)
    {
        Serial.println("Feature extraction failed");
        return;
    }

    // Prepare EI signal - use plain function pointer
    signal_t signal;
    signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
    signal.get_data = raw_feature_get_data;

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

    // Find "bird" class index
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

    Serial.printf("Inference result: %s (score: %.3f) | Feature ext: %.2f ms | Inference: %.2f ms\n",
                  bird_score > inference_score_threshold ? "BIRD" : "NO_BIRD",
                  bird_score, feat_time_ms, inf_time_ms);

    #ifdef LED_PIN
       if (bird_score > inference_score_threshold) {
           digitalWrite(LED_PIN, HIGH);
       } else {
           digitalWrite(LED_PIN, LOW);
       }
    #endif
}

// --- Recording and serial streaming ---
static void recordAndSendAudio() {
    const int total_samples = (SAMPLE_RATE * RECORD_TIME_MS) / 1000;
    const size_t total_bytes = (size_t)total_samples * sizeof(int16_t);

    // Use a single static DRAM buffer (reused across recordings)
    if (total_bytes > 0 && audio_buf_bytes < total_bytes) {
        if (audio_buf) {
            free(audio_buf);
            audio_buf = nullptr;
            audio_buf_bytes = 0;
        }
        audio_buf = (int16_t *)malloc(total_bytes);
        if (audio_buf) {
            audio_buf_bytes = total_bytes;
            Serial.println("Using static DRAM buffer for full capture");
        } else {
            Serial.println("Failed to allocate memory for audio buffer");
            return;
        }
    }

    // Zero DMA buffer before capture
    i2s_zero_dma_buffer(I2S_PORT);
    const unsigned long startTime = millis();

    // NOTE: For the full-buffer path we will print the inference result first,
    // then print the BEGIN_AUDIO header so the Python receiver can capture the
    // "Inference result:" line in its pre-BEGIN_AUDIO parsing loop.

    {
        int16_t *buf = audio_buf;
        // compute gain factor from dB
        const float gain_factor = powf(10.0f, (AUDIO_GAIN_DB) / 20.0f);
        // Read entire recording into buffer
        size_t samples_filled = 0;
        while (samples_filled < (size_t)total_samples && (millis() - startTime) < (RECORD_TIME_MS + 500)) {
            size_t to_read = min((size_t)BUFFER_SIZE, (size_t)total_samples - samples_filled);
            size_t bytes_read = 0;
            esp_err_t ok = i2s_read(I2S_PORT,
                                    buf + samples_filled,
                                    to_read * sizeof(int16_t),
                                    &bytes_read,
                                    100);
            if (ok != ESP_OK) {
                Serial.printf("I2S Read error: %d\n", ok);
                break;
            }
            if (bytes_read > 0) {
                samples_filled += (bytes_read / sizeof(int16_t));
            }
        }

        // Apply software gain (in-place) with clipping
        if (AUDIO_GAIN_DB != 0.0f) {
            for (size_t i = 0; i < samples_filled; ++i) {
                int32_t v = (int32_t)((float)buf[i] * gain_factor);
                if (v > 32767) v = 32767;
                else if (v < -32768) v = -32768;
                buf[i] = (int16_t)v;
            }
        }

        // Run inference on the captured buffer
        runInferenceOnRecordedAudio(buf, samples_filled);

#if defined(COMMUNICATION_MODE)
        // Announce header for Python receiver AFTER inference so the Python script
        // can capture the "Inference result:" line before BEGIN_AUDIO.
        Serial.println("BEGIN_AUDIO");
        Serial.println(SAMPLE_RATE);
        Serial.println(samples_filled);
        Serial.println("BEGIN_BINARY");

        // Send buffer in chunks to avoid huge Serial.write calls
        const size_t send_chunk_bytes = 4096;
        size_t bytes_sent = 0;
        const size_t bytes_to_send = samples_filled * sizeof(int16_t);
        while (bytes_sent < bytes_to_send) {
            size_t chunk = min(send_chunk_bytes, bytes_to_send - bytes_sent);
            Serial.write(((uint8_t *)buf) + bytes_sent, chunk);
            bytes_sent += chunk;
            delay(1);
        }
        Serial.flush();
#endif
    }

#if defined(COMMUNICATION_MODE)
    Serial.println();
    Serial.println("END_AUDIO");
#endif

}

void setup() {
    // Use a higher baud for faster transfers (ensure host matches this)
    Serial.begin(921600);
    while (!Serial) {
        delay(10);
    }
    Serial.println("INMP441 I2S capture (I2S RX) ready");
#if defined(CLASSIFIER_MODE) && !defined(COMMUNICATION_MODE)
    Serial.println("Mode: CLASSIFIER_MODE (inference-only, no audio streaming)");
#elif defined(COMMUNICATION_MODE)
    Serial.println("Mode: COMMUNICATION_MODE (streaming + inference)");
#endif

    // I2S RX
    i2s_init_rx();


    #ifdef LED_PIN
        pinMode(LED_PIN, OUTPUT);
        digitalWrite(LED_PIN, LOW);
    #endif
}

void loop() {
#if defined(CLASSIFIER_MODE) && !defined(COMMUNICATION_MODE)
    // In classifier-only mode, record 1s clips, run inference, and report.
    recordAndSendAudio();
    delay(10);
    return;
#else
    readI2SData();
    if (Serial.available() > 0) {
        char cmd = Serial.read();
        if (cmd == 'r') {
            Serial.println("Starting recording...");
            recordAndSendAudio();
        } else if (cmd == 'c') {
            // Start continuous mode: record-send-record-send...
            Serial.println("Starting continuous recording (send 's' to stop)...");
            while (true) {
                // If stop command received, break
                if (Serial.available() > 0) {
                    char in = Serial.read();
                    if (in == 's') {
                        Serial.println("Stopping continuous recording");
                        break;
                    }
                }
                recordAndSendAudio();
                // small pause to allow host to process and to check for stop
                delay(10);
            }
        } else if (cmd == 's') {
            // Stop command - useful if continuous was started from host
            Serial.println("Stop command received (not in continuous mode)");
        } else if (cmd == 'p') {
            Serial.println("Playback not implemented in ADC-only mode");
        }
    }
#endif
}

