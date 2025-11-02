// Barebones ES8388 microphone (ADC) differential MIC1 input and serial streaming

#include <Arduino.h>
#include <driver/i2s.h>
#include <Wire.h>
#include <esp_heap_caps.h>
#include <math.h>

// --- Configuration constants ---
#define SAMPLE_RATE 16000        // 16 kHz
#define RECORD_TIME_MS 3000      // 3 seconds
#define BUFFER_SIZE 1024         // samples per chunk
// Software gain to apply to samples (in dB). Set to 0.0 for no change.
#define AUDIO_GAIN_DB 10.0f
// Warm-up time in milliseconds to let the ADC/I2S settle before capturing
#define WARMUP_MS 1000

// --- ES8388 pins (keep your board mapping) ---
#define ES8388_I2C_SDA 8
#define ES8388_I2C_SCL 9
#define ES8388_I2C_SPEED 100000
#define ES8388_ADDR 0x10

#define ES8388_I2S_MCLK 14
#define ES8388_I2S_BCLK 13
#define ES8388_I2S_LRCK 11
#define ES8388_I2S_DOUT 10  // ES8388 -> MCU (ADC data)

#define I2S_PORT I2S_NUM_0

// --- I2C helpers ---
static TwoWire codecWire = TwoWire(0);

static void es_write(uint8_t reg, uint8_t val) {
    codecWire.beginTransmission(ES8388_ADDR);
    codecWire.write(reg);
    codecWire.write(val);
    codecWire.endTransmission();
}

static void es_reset() {
    es_write(0x00, 0x80); // reset
    delay(50);
    es_write(0x00, 0x00);
}

// --- ES8388 ADC bring-up: MIC1 differential, I2S slave, 16-bit ---
static void es_adc_init_mic1_diff() {
    // Stop digital blocks
    es_write(0x02, 0xF3); // CHIPPOWER: stop STM, DLL, digital

    // I2S slave
    es_write(0x08, 0x00); // MASTERMODE: slave mode

    // Share LRCK for ADC/DAC
    es_write(0x2B, 0x80); // DACCONTROL11: same LRCK for ADC/DAC

    // Startup references
    es_write(0x00, 0x05); // CONTROL1: startup reference
    es_write(0x01, 0x40); // CONTROL2: power-up paths

    // Power on ADC and LIN/RIN input
    es_write(0x03, 0x00); // ADCPOWER: power up

    // Mic gain / boost (choose ONE line below)
    // es_write(0x09, 0x00); // ADCCONTROL1:  +0 dB PGA (L=0,R=0)
    // es_write(0x09, 0x11); // ADCCONTROL1:  +3 dB PGA (L=1,R=1)
    // es_write(0x09, 0x22); // ADCCONTROL1:  +6 dB PGA (L=2,R=2)
    // es_write(0x09, 0x33); // ADCCONTROL1:  +9 dB PGA (L=3,R=3)
    // es_write(0x09, 0x44); // ADCCONTROL1: +12 dB PGA (L=4,R=4)
    // es_write(0x09, 0x55); // ADCCONTROL1: +15 dB PGA (L=5,R=5)
    // es_write(0x09, 0x66); // ADCCONTROL1: +18 dB PGA (L=6,R=6)
    //es_write(0x09, 0x77); // ADCCONTROL1: +21 dB PGA (L=7,R=7) [selected]
     es_write(0x09, 0x88); // ADCCONTROL1: +24 dB PGA (L=8,R=8) [max]

    // Differential input enable
    es_write(0x0A, 0xF0); // ADCCONTROL2: differential

    // Select LIN1/RIN1 as differential pair for MIC1
    es_write(0x0B, 0x02); // ADCCONTROL3: LIN1 & RIN1

    // I2S format and word length: 16-bit I2S standard, L/R data
    // 0x0D used by reference driver for 16-bit I2S normal
    es_write(0x0C, 0x0D); // ADCCONTROL4

    // Clocking ratio: MCLK/LRCK = 256r
    es_write(0x0D, 0x02); // ADCCONTROL5

    // ADC digital volumes 0 dB
    es_write(0x10, 0x00); // ADCCONTROL8: LADC vol
    es_write(0x11, 0x00); // ADCCONTROL9: RADC vol

    // ALC disabled for now
    es_write(0x12, 0x00); // ADCCONTROL10: ALC disabled
    // Optional ALC configuration (examples)
    // es_write(0x12, 0xE2); // ADCCONTROL10: ALC enable, PGA max/min
    // es_write(0x13, 0xA0); // ADCCONTROL11: ALC target, hold
    // es_write(0x14, 0x12); // ADCCONTROL12: decay/attack
    // es_write(0x15, 0x06); // ADCCONTROL13: ALC mode
    // es_write(0x16, 0xC3); // ADCCONTROL14: noise gate

    // Start digital blocks
    es_write(0x02, 0x55); // CHIPPOWER: start STM, DLL, digital
}

// --- I2S RX init (MCU master providing BCLK/LRCK/MCLK) ---
static void i2s_init_rx() {
    i2s_config_t cfg = {
            .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
            .sample_rate = SAMPLE_RATE,
            .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
            .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT, // capture mono from left
            .communication_format = I2S_COMM_FORMAT_STAND_I2S,
            .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
            // increase DMA buffering to avoid underruns while serial is sending
            .dma_buf_count = 12,
            .dma_buf_len = 1024,
            // use APLL where available for stable clock generation
            .use_apll = true,
            .tx_desc_auto_clear = false,
            .fixed_mclk = 0};

    i2s_driver_install(I2S_PORT, &cfg, 0, nullptr);

    i2s_pin_config_t pins = {
            .mck_io_num = ES8388_I2S_MCLK,
            .bck_io_num = ES8388_I2S_BCLK,
            .ws_io_num = ES8388_I2S_LRCK,
            .data_out_num = -1,            // we don't transmit
            .data_in_num = ES8388_I2S_DOUT // from ES8388 ADC
    };

    i2s_set_pin(I2S_PORT, &pins);
    i2s_set_sample_rates(I2S_PORT, SAMPLE_RATE);
}

// --- Recording and serial streaming ---
static void recordAndSendAudio() {
    const int total_samples = (SAMPLE_RATE * RECORD_TIME_MS) / 1000;

    const size_t total_bytes = (size_t)total_samples * sizeof(int16_t);

    // Try to allocate a contiguous buffer in PSRAM for the entire recording.
    // If PSRAM isn't available or allocation fails, fall back to streaming mode.
    int16_t *psram_buf = nullptr;
    if (total_bytes > 0) {
        psram_buf = (int16_t *)heap_caps_malloc(total_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (psram_buf) {
            Serial.println("Using PSRAM buffer for full capture");
        } else {
            // Serial.println("PSRAM buffer unavailable, using streaming mode");
        }
    }

    // Zero DMA buffer and perform a short warm-up read to let ADC/I2S settle
    i2s_zero_dma_buffer(I2S_PORT);
    const size_t warmup_samples = (size_t)((SAMPLE_RATE * WARMUP_MS) / 1000);
    if (warmup_samples > 0) {
        int16_t warmbuf[BUFFER_SIZE];
        size_t warmed = 0;
        while (warmed < warmup_samples) {
            size_t to_read = min((size_t)BUFFER_SIZE, warmup_samples - warmed);
            size_t bytes_read = 0;
            esp_err_t ok = i2s_read(I2S_PORT, warmbuf, to_read * sizeof(int16_t), &bytes_read, pdMS_TO_TICKS(100));
            if (ok != ESP_OK || bytes_read == 0) {
                // If no data yet, wait a bit
                delay(5);
            } else {
                warmed += (bytes_read / sizeof(int16_t));
            }
        }
        // small pause
        delay(5);
    }

    // Announce header for Python receiver (after warm-up so data starts immediately)
    Serial.println("BEGIN_AUDIO");
    Serial.println(SAMPLE_RATE);
    Serial.println(total_samples);
    Serial.println("BEGIN_BINARY");

    if (psram_buf) {
        // compute gain factor from dB
        const float gain_factor = powf(10.0f, (AUDIO_GAIN_DB) / 20.0f);
        // Read entire recording into PSRAM buffer, then send it in chunks.
        size_t samples_filled = 0;
        while (samples_filled < (size_t)total_samples) {
            size_t to_read = min((size_t)BUFFER_SIZE, (size_t)total_samples - samples_filled);
            size_t bytes_read = 0;
            esp_err_t ok = i2s_read(I2S_PORT, psram_buf + samples_filled, to_read * sizeof(int16_t), &bytes_read, portMAX_DELAY);
            if (ok != ESP_OK || bytes_read == 0) {
                // pad with zeros
                memset(psram_buf + samples_filled, 0, to_read * sizeof(int16_t));
                bytes_read = to_read * sizeof(int16_t);
            }
            samples_filled += (bytes_read / sizeof(int16_t));
        }

        // Apply software gain (in-place) with clipping
        if (AUDIO_GAIN_DB != 0.0f) {
            for (size_t i = 0; i < (size_t)total_samples; ++i) {
                int32_t v = (int32_t)((float)psram_buf[i] * gain_factor);
                if (v > 32767) v = 32767;
                else if (v < -32768) v = -32768;
                psram_buf[i] = (int16_t)v;
            }
        }

        // Send header then send buffer in chunks to avoid huge Serial.write calls
        const size_t send_chunk_bytes = 4096;
        size_t bytes_sent = 0;
        while (bytes_sent < total_bytes) {
            size_t chunk = min(send_chunk_bytes, total_bytes - bytes_sent);
            Serial.write(((uint8_t *)psram_buf) + bytes_sent, chunk);
            bytes_sent += chunk;
            // small yield to allow UART to flush
            delay(1);
        }
        // Ensure all UART bytes are transmitted before signalling end
        Serial.flush();
        // free PSRAM buffer
        heap_caps_free(psram_buf);
    } else {
        // Stream samples in chunks directly to serial to avoid big mallocs
        int samples_sent = 0;
        int16_t buf[BUFFER_SIZE];
        while (samples_sent < total_samples) {
            size_t to_read = min(BUFFER_SIZE, total_samples - samples_sent);
            size_t bytes_read = 0;
            // Read mono 16-bit samples
            esp_err_t ok = i2s_read(I2S_PORT, buf, to_read * sizeof(int16_t), &bytes_read, portMAX_DELAY);
            if (ok != ESP_OK || bytes_read == 0) {
                // if read fails, pad with zeros to keep framing
                memset(buf, 0, to_read * sizeof(int16_t));
                bytes_read = to_read * sizeof(int16_t);
            }
            // Apply software gain per-sample before sending
            if (AUDIO_GAIN_DB != 0.0f) {
                const float gain_factor_stream = powf(10.0f, (AUDIO_GAIN_DB) / 20.0f);
                size_t samples_read = bytes_read / sizeof(int16_t);
                for (size_t si = 0; si < samples_read; ++si) {
                    int32_t vv = (int32_t)((float)buf[si] * gain_factor_stream);
                    if (vv > 32767) vv = 32767;
                    else if (vv < -32768) vv = -32768;
                    buf[si] = (int16_t)vv;
                }
            }
            Serial.write((uint8_t *)buf, bytes_read);
            samples_sent += (bytes_read / sizeof(int16_t));
        }
        // Ensure UART transmits remaining bytes before printing end marker
        Serial.flush();
    }

    Serial.println();
    Serial.println("END_AUDIO");
}

void setup() {
    // Use a higher baud for faster transfers (ensure host matches this)
    Serial.begin(921600);
    while (!Serial) {
        delay(10);
    }
    Serial.println("ES8388 MIC1 differential capture (I2S RX) ready");

    // I2C and ES8388 init
    codecWire.begin(ES8388_I2C_SDA, ES8388_I2C_SCL, ES8388_I2C_SPEED);
    es_reset();
    es_adc_init_mic1_diff();

    // I2S RX
    i2s_init_rx();
}

void loop() {
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
}

