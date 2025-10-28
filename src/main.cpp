// Barebones ES8388 microphone (ADC) differential MIC1 input and serial streaming

#include <Arduino.h>
#include <driver/i2s.h>
#include <Wire.h>

// --- Configuration constants ---
#define SAMPLE_RATE 16000        // 16 kHz
#define RECORD_TIME_MS 2000      // 2 seconds
#define BUFFER_SIZE 1024         // samples per chunk

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
            .dma_buf_count = 8,
            .dma_buf_len = 256,
            .use_apll = false,
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
    const int total_samples = SAMPLE_RATE * (RECORD_TIME_MS / 1000);

    // Announce header for Python receiver
    Serial.println("BEGIN_AUDIO");
    Serial.println(SAMPLE_RATE);
    Serial.println(total_samples);
    Serial.println("BEGIN_BINARY");

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
        Serial.write((uint8_t *)buf, bytes_read);
        samples_sent += (bytes_read / sizeof(int16_t));
    }

    Serial.println();
    Serial.println("END_AUDIO");
}

void setup() {
    Serial.begin(115200);
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
        } else if (cmd == 'p') {
            Serial.println("Playback not implemented in ADC-only mode");
        }
    }
}

