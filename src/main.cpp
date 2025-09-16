/*
 * Audio Inference for ESP32 with SD Card Support
 *
 * This code loads WAV files from an SD card, applies min-max scaling,
 * and runs neural network inference with statistics tracking.
 */

/* Combined Audio Inference with Mode Selection */


#include <Arduino.h>
#include <SD.h>
#include <SPI.h>
#include <vector>

#include <marres_clone_inferencing.h> // Edge Impulse library

// Audio buffer stored in PSRAM
static float *input_data = NULL;

// PSRAM configuration
#if CONFIG_SPIRAM_SUPPORT || defined(CONFIG_SPIRAM)
#define USE_PSRAM 1
#else
#define USE_PSRAM 0
#endif

// SD Card settings for Lolin D32 Pro (built-in SD card)
//#define SD_CS 4

// SD Card settings for Lolin S3 pro
#define SD_MOSI 11 // GPIO11
#define SD_MISO 13 // GPIO13
#define SD_CLK 12  // GPIO12
#define SD_CS 46   // GPIO46, changed this from pin 10 as this was giving issues! (pin 46 is printed on the S3 itself)

#define BIRD_FOLDER "/bird"
#define NO_BIRD_FOLDER "/no_bird"
#define BIRD_INDEX_FILE "/bird_index.txt"
#define NO_BIRD_INDEX_FILE "/no_bird_index.txt"

// WAV file constants
#define WAV_HEADER_SIZE 44
#define SAMPLE_RATE 16000  // Expected sample rate
#define BITS_PER_SAMPLE 16 // Expected bits per sample

// Statistics tracking
typedef struct
{
  int total_bird_samples;
  int total_nobird_samples;
  int correct_bird_predictions;
  int correct_nobird_predictions;
  unsigned long total_inference_time;
  unsigned long max_inference_time;
  unsigned long min_inference_time;
  unsigned long total_loading_time;
  unsigned long max_loading_time;
  unsigned long min_loading_time;
  bool first_measurement; // Add flag for first measurement
} Statistics;

Statistics stats = {0, 0, 0, 0, 0, 0, ULONG_MAX, 0, 0, ULONG_MAX, true};


// Edge Impulse configuration
static float *features = NULL;
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr)
{
  memcpy(out_ptr, features + offset, length * sizeof(float));
  return 0;
}

// Function to extract features from WAV file with min-max scaling
bool extractFeaturesFromWav(const char *filename, float *features, size_t feature_size)
{
  File wavFile = SD.open(filename);
  if (!wavFile)
  {
    Serial.printf("Failed to open file: %s\n", filename);
    return false;
  }

  // Skip WAV header (44 bytes)
  wavFile.seek(44);

  // Buffer for reading
  const int buffer_size = 512;
  int16_t buffer[buffer_size];

  // First pass: find min and max values for scaling
  int16_t min_val = INT16_MAX;
  int16_t max_val = INT16_MIN;
  size_t total_samples = 0;

  while (wavFile.available() && total_samples < feature_size)
  {
    int bytesRead = wavFile.read((uint8_t *)buffer, min(buffer_size * sizeof(int16_t), (feature_size - total_samples) * sizeof(int16_t)));
    int samples_read = bytesRead / sizeof(int16_t);
    for (int i = 0; i < samples_read; i++)
    {
      if (buffer[i] < min_val)
        min_val = buffer[i];
      if (buffer[i] > max_val)
        max_val = buffer[i];
    }
    total_samples += samples_read;
  }

  // Avoid division by zero if the audio is silent
  if (min_val == max_val)
  {
    min_val = -1;
    max_val = 1;
  }

  // Reset file position to beginning of audio data
  wavFile.seek(44);

  // Second pass: apply min-max scaling to normalize the audio
  size_t feature_idx = 0;
  while (wavFile.available() && feature_idx < feature_size)
  {
    int bytesRead = wavFile.read((uint8_t *)buffer, min(buffer_size * sizeof(int16_t), (feature_size - feature_idx) * sizeof(int16_t)));
    int samples_read = bytesRead / sizeof(int16_t);
    for (int i = 0; i < samples_read && feature_idx < feature_size; i++)
    {
      float sample = buffer[i];
      // Normalization can be added here if needed
      features[feature_idx++] = sample;
    }
  }
  // Fill remaining features with zeros if file is shorter than needed
  while (feature_idx < feature_size)
  {
    features[feature_idx++] = 0.0f;
  }

  wavFile.close();
  return true;
}


// --- Persistent Index Functions ---

// Save a vector of file paths as a text index file
bool saveIndexFile(const char *indexPath, const std::vector<String> &files)
{
  File idx = SD.open(indexPath, FILE_WRITE);
  if (!idx)
  {
    Serial.printf("Failed to open index file for writing: %s\n", indexPath);
    return false;
  }
  for (const auto &f : files)
  {
    idx.println(f);
  }
  idx.close();
  return true;
}

// Load a vector of file paths from a text index file
std::vector<String> loadIndexFile(const char *indexPath)
{
  std::vector<String> files;
  File idx = SD.open(indexPath, FILE_READ);
  if (!idx)
  {
    Serial.printf("Failed to open index file for reading: %s\n", indexPath);
    return files;
  }
  while (idx.available())
  {
    String line = idx.readStringUntil('\n');
    line.trim();
    if (line.length() > 0)
      files.push_back(line);
  }
  idx.close();
  return files;
}

// Scan SD directory for .wav files
std::vector<String> getFilesInDir(const char *dirname)
{
  std::vector<String> files;
  File root = SD.open(dirname);
  if (!root || !root.isDirectory())
  {
    Serial.printf("Failed to open directory: %s\n", dirname);
    return files;
  }
  File file = root.openNextFile();
  while (file)
  {
    if (!file.isDirectory() && String(file.name()).endsWith(".wav"))
    {
      files.push_back(String(dirname) + "/" + String(file.name()));
    }
    file = root.openNextFile();
  }
  root.close();
  Serial.printf("Found %d WAV files in %s folder\n", files.size(), dirname);
  return files;
}

// Build or load index, depending on forceRebuild and file existence
std::vector<String> getIndexedFiles(const char *folder, const char *indexFile, bool forceRebuild)
{
  std::vector<String> files;
  if (!forceRebuild && SD.exists(indexFile))
  {
    Serial.printf("Loading index file: %s\n", indexFile);
    files = loadIndexFile(indexFile);
  }
  if (forceRebuild || files.empty())
  {
    Serial.printf("Building index for folder: %s\n", folder);
    files = getFilesInDir(folder);
    if (!saveIndexFile(indexFile, files))
    {
      Serial.printf("Warning: Failed to save index file: %s\n", indexFile);
    }
  }
  return files;
}

// --- WAV Loading and Inference Functions ---

bool loadWavFile(const char *filename, float *buffer, int maxSamples)
{
  File wavFile = SD.open(filename);
  if (!wavFile)
  {
    Serial.printf("Failed to open file: %s\n", filename);
    return false;
  }

  // Read WAV header (44 bytes)
  uint8_t header[WAV_HEADER_SIZE];
  wavFile.read(header, WAV_HEADER_SIZE);

  // Quick check if this is a valid WAV file
  if (header[0] != 'R' || header[1] != 'I' || header[2] != 'F' || header[3] != 'F' ||
      header[8] != 'W' || header[9] != 'A' || header[10] != 'V' || header[11] != 'E')
  {
    Serial.printf("Not a valid WAV file: %s\n", filename);
    wavFile.close();
    return false;
  }

  // Parse number of channels from header
  uint16_t numChannels = header[22] | (header[23] << 8);

  // Parse sample rate from header
  uint32_t sampleRate = header[24] | (header[25] << 8) |
                        (header[26] << 16) | (header[27] << 24);

  // Parse bits per sample from header
  uint16_t bitsPerSample = header[34] | (header[35] << 8);

  Serial.printf("File: %s, Channels: %d, Sample Rate: %d, Bits/Sample: %d\n",
                filename, numChannels, sampleRate, bitsPerSample);
  // Verify format is compatible
  if (numChannels != 1)
  {
    Serial.println("Only mono WAV files are supported");
    wavFile.close();
    return false;
  }

  // Read audio data
  int bytesPerSample = bitsPerSample / 8;
  int samplesRead = 0;
  uint8_t sampleBytes[4]; // Max 32-bit samples

  // First pass: find min and max values for scaling
  int32_t min_val = INT32_MAX;
  int32_t max_val = INT32_MIN;
  int32_t sample;

  while (wavFile.available() && samplesRead < maxSamples)
  {
    wavFile.read(sampleBytes, bytesPerSample);
    // Convert bytes to sample value based on bit depth
    if (bitsPerSample == 8)
    {
      sample = sampleBytes[0] - 128;
    }
    else if (bitsPerSample == 16)
    {
      sample = (int16_t)(sampleBytes[0] | (sampleBytes[1] << 8));
    }
    else if (bitsPerSample == 24)
    {
      sample = (sampleBytes[0] | (sampleBytes[1] << 8) | (sampleBytes[2] << 16));
      if (sample & 0x800000)
        sample |= 0xFF000000; // Sign extend
    }
    else if (bitsPerSample == 32)
    {
      sample = (sampleBytes[0] | (sampleBytes[1] << 8) |
                (sampleBytes[2] << 16) | (sampleBytes[3] << 24));
    }
    else
    {
      Serial.printf("Unsupported bits per sample: %d\n", bitsPerSample);
      wavFile.close();
      return false;
    }
    // Update min/max
    if (sample < min_val)
      min_val = sample;
    if (sample > max_val)
      max_val = sample;
    samplesRead++;
  }

  // Avoid division by zero if the audio is silent
  if (min_val == max_val)
  {
    Serial.println("Warning: Audio file has no variation (possibly silent)");
    min_val = -1;
    max_val = 1;
  }

  // Reset file position to beginning of audio data
  wavFile.seek(WAV_HEADER_SIZE);

  // Second pass: read and normalize samples
  samplesRead = 0;
  while (wavFile.available() && samplesRead < maxSamples)
  {
    wavFile.read(sampleBytes, bytesPerSample);
    // Convert bytes to sample value based on bit depth
    if (bitsPerSample == 8)
    {
      sample = sampleBytes[0] - 128;
    }
    else if (bitsPerSample == 16)
    {
      sample = (int16_t)(sampleBytes[0] | (sampleBytes[1] << 8));
    }
    else if (bitsPerSample == 24)
    {
      sample = (sampleBytes[0] | (sampleBytes[1] << 8) | (sampleBytes[2] << 16));
      if (sample & 0x800000)
        sample |= 0xFF000000; // Sign extend
    }
    else if (bitsPerSample == 32)
    {
      sample = (sampleBytes[0] | (sampleBytes[1] << 8) |
                (sampleBytes[2] << 16) | (sampleBytes[3] << 24));
    }
    // Normalize sample to [-1, 1] based on min/max values
    float normalized_value;
    if (sample < 0)
    {
      normalized_value = (min_val != 0) ? (float)sample / abs(min_val) : 0;
    }
    else
    {
      normalized_value = (max_val != 0) ? (float)sample / max_val : 0;
    }
    // Ensure value is within [-1, 1]
    normalized_value = constrain(normalized_value, -1.0f, 1.0f);
    buffer[samplesRead++] = normalized_value;
  }
  // Fill remaining buffer with zeros if file is shorter than maxSamples
  while (samplesRead < maxSamples)
  {
    buffer[samplesRead++] = 0.0f;
  }
  wavFile.close();
  return true;
}

void printStatistics()
{
  Serial.println("\n----- Current Statistics -----");
  int total_samples = stats.total_bird_samples + stats.total_nobird_samples;
  int total_correct = stats.correct_bird_predictions + stats.correct_nobird_predictions;
  Serial.printf("Bird samples: %d, correctly identified: %d (%.2f%%)\n",
                stats.total_bird_samples,
                stats.correct_bird_predictions,
                stats.total_bird_samples > 0 ? (float)stats.correct_bird_predictions * 100 / stats.total_bird_samples : 0);
  Serial.printf("No-bird samples: %d, correctly identified: %d (%.2f%%)\n",
                stats.total_nobird_samples,
                stats.correct_nobird_predictions,
                stats.total_nobird_samples > 0 ? (float)stats.correct_nobird_predictions * 100 / stats.total_nobird_samples : 0);
  Serial.printf("Overall accuracy: %.2f%% (%d of %d correct)\n",
                total_samples > 0 ? (float)total_correct * 100 / total_samples : 0,
                total_correct,
                total_samples);
  if (total_samples > 0)
  {
    Serial.printf("Inference timing - Avg: %.3f s, Min: %.3f s, Max: %.3f s\n",
                  (stats.total_inference_time / total_samples) / 1000000.0f,
                  stats.min_inference_time / 1000000.0f,
                  stats.max_inference_time / 1000000.0f);
    Serial.printf("Loading timing - Avg: %.3f s, Min: %.3f s, Max: %.3f s\n",
                  (stats.total_loading_time / total_samples) / 1000000.0f,
                  stats.min_loading_time / 1000000.0f,
                  stats.max_loading_time / 1000000.0f);
  }
  Serial.println("------------------------------\n");
}

bool runInferenceOnFile(const char *filename, bool is_bird_sample)
{
  Serial.printf("Processing file: %s\n", filename);
  unsigned long loading_start_time = micros();
  unsigned long loading_time = 0;
  unsigned long inference_time = 0;
  bool predicted_as_bird = false;
  bool is_correct = false;


  // Edge Impulse inference
  if (!extractFeaturesFromWav(filename, features, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE))
  {
    return false;
  }
  loading_time = micros() - loading_start_time;
  // Set up signal for inferencing
  signal_t signal;
  signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
  signal.get_data = &raw_feature_get_data;
  // Run the impulse
  unsigned long inference_start_time = micros();
  ei_impulse_result_t result = {0};
  EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);
  if (res != EI_IMPULSE_OK)
  {
    Serial.printf("ERR: Failed to run classifier (%d)\n", res);
    return false;
  }
  inference_time = micros() - inference_start_time;
  // Get prediction (assuming binary classification with "bird" class)
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
  predicted_as_bird = (bird_score > 0.5); // Adjust threshold if needed
  is_correct = (predicted_as_bird == is_bird_sample);


  // Update loading time statistics
  stats.total_loading_time += loading_time;
  if (stats.first_measurement || loading_time < stats.min_loading_time)
  {
    stats.min_loading_time = loading_time;
  }
  if (stats.first_measurement || loading_time > stats.max_loading_time)
  {
    stats.max_loading_time = loading_time;
  }
  // Update inference time statistics
  stats.total_inference_time += inference_time;
  if (stats.first_measurement || inference_time < stats.min_inference_time)
  {
    stats.min_inference_time = inference_time;
  }
  if (stats.first_measurement || inference_time > stats.max_inference_time)
  {
    stats.max_inference_time = inference_time;
  }
  // Clear first measurement flag after first update
  stats.first_measurement = false;
  // Update classification statistics
  if (is_bird_sample)
  {
    stats.total_bird_samples++;
    if (is_correct)
      stats.correct_bird_predictions++;
  }
  else
  {
    stats.total_nobird_samples++;
    if (is_correct)
      stats.correct_nobird_predictions++;
  }
  // Print prediction results
  Serial.printf("Prediction: %s\n", predicted_as_bird ? "BIRD" : "NO_BIRD");
  Serial.printf("Ground truth: %s, Prediction %s\n",
                is_bird_sample ? "BIRD" : "NO_BIRD",
                is_correct ? "CORRECT" : "INCORRECT");
  Serial.printf("Loading time: %.3f s\n", loading_time / 1000000.0f);
  Serial.printf("Inference time: %.3f s\n", inference_time / 1000000.0f);
  return true;
}

void setup()
{
  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial)
  {
    ;
  }


  Serial.println("Starting Audio Inference using Edge Impulse model");


  // Check PSRAM availability
  Serial.print("Total heap: ");
  Serial.println(ESP.getHeapSize());
  Serial.print("Free heap: ");
  Serial.println(ESP.getFreeHeap());
  Serial.print("Total PSRAM: ");
  Serial.println(ESP.getPsramSize());
  Serial.print("Free PSRAM: ");
  Serial.println(ESP.getFreePsram());
  if (ESP.getPsramSize() == 0)
  {
    Serial.println("PSRAM not available or not enabled in IDE!");
    while (1)
      ;
  }

  // Initialize SD card
  if (!SD.begin(SD_CS))
  {
    Serial.println("SD Card initialization failed!");
    while (1)
      ;
  }
  Serial.println("SD Card initialized successfully");


  features = (float *)ps_malloc(EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE * sizeof(float));
  if (features == NULL)
  {
    Serial.println("Failed to allocate memory in PSRAM for features");
    while (1)
      ;
  }
  Serial.println("Successfully allocated features array in PSRAM");


  // --- Persistent index logic ---
  bool forceRebuild = false; // Set true to force index rebuild on next boot
  std::vector<String> birdFiles = getIndexedFiles(BIRD_FOLDER, BIRD_INDEX_FILE, forceRebuild);
  std::vector<String> noBirdFiles = getIndexedFiles(NO_BIRD_FOLDER, NO_BIRD_INDEX_FILE, forceRebuild);

  // Determine how many files to process
  size_t max_files = min(birdFiles.size(), noBirdFiles.size());
  if (max_files == 0)
  {
    Serial.println("No files found in one or both folders");
  }
  else
  {
    Serial.printf("Will process %d files from each folder\n", max_files);
    for (size_t i = 0; i < max_files; i++)
    {
      runInferenceOnFile(birdFiles[i].c_str(), true);
      runInferenceOnFile(noBirdFiles[i].c_str(), false);
      if ((i + 1) % 5 == 0 || i == max_files - 1)
      {
        printStatistics();
      }
    }
  }
  Serial.println("\n===== FINAL STATISTICS =====");
  printStatistics();


  // Free input data buffer
  free(input_data);


  Serial.println("Testing complete");
}

void loop()
{
  delay(1000);
}
