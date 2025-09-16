#ifndef TRANSFORMER_MODEL_H
#define TRANSFORMER_MODEL_H

#include <Arduino.h>
#include "model_weights.h"

class TransformerModel {
private:
    // Model dimensions
    static const int INPUT_LENGTH = 48000;
    static const int CONV1_OUT_CHANNELS = 32;
    static const int CONV1_KERNEL_SIZE = 5;
    static const int CONV2_OUT_CHANNELS = 64;
    static const int CONV2_KERNEL_SIZE = 3;
    static const int TOKEN_DIM = 16;
    static const int NUM_TOKENS = 8;
    static const int NUM_HEADS = 2;
    static const int FF_DIM = 64;
    
    // Memory-efficient scratch buffer
    float* buffer;
    size_t buffer_size;
    
    // Memory management
    float* allocateBuffer(size_t size) {
        if (size > buffer_size) {
            Serial.println("Warning: Buffer overflow!");
            return buffer;
        }
        return buffer;
    }
    
    // Normalization
    void normalizeSample(const uint8_t* input, float* output, size_t length) {
        uint8_t min_val = 255;
        uint8_t max_val = 0;
        
        // Find min and max
        for (size_t i = 0; i < length; i++) {
            if (input[i] < min_val) min_val = input[i];
            if (input[i] > max_val) max_val = input[i];
        }
        
        // Normalize to [0, 1]
        float range = static_cast<float>(max_val - min_val);
        if (range < 1e-9f) range = 1e-9f; // Avoid division by zero
        
        for (size_t i = 0; i < length; i++) {
            output[i] = static_cast<float>(input[i] - min_val) / range;
        }
    }
    
    // 1D Convolution (optimized for ESP32)
    void conv1d(const float* input, float* output, int input_length, int in_channels, 
                int out_channels, int kernel_size, const float* weights, const float* bias) {
        const int padding = kernel_size / 2;
        
        // Initialize output with zeros
        for (int oc = 0; oc < out_channels; oc++) {
            for (int i = 0; i < input_length; i++) {
                float sum = 0.0f;
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int k = 0; k < kernel_size; k++) {
                        const int idx = i + k - padding;
                        if (idx >= 0 && idx < input_length) {
                            sum += input[ic * input_length + idx] * 
                                  weights[(oc * in_channels + ic) * kernel_size + k];
                        }
                    }
                }
                output[oc * input_length + i] = sum + bias[oc];
            }
        }
    }
    
    // Batch Normalization + ReLU (fused for efficiency)
    void batchNormRelu(float* x, int length, int channels, 
                      const float* weight, const float* bias, 
                      const float* running_mean, const float* running_var) {
        const int feature_size = length / channels;
        const float epsilon = 1e-5f;
        
        for (int c = 0; c < channels; c++) {
            const float inv_std = 1.0f / sqrtf(running_var[c] + epsilon);
            const float scale = weight[c] * inv_std;
            const float shift = bias[c] - running_mean[c] * scale;
            
            for (int i = 0; i < feature_size; i++) {
                const int idx = c * feature_size + i;
                x[idx] = scale * x[idx] + shift;
                // Fused ReLU
                if (x[idx] < 0) x[idx] = 0;
            }
        }
    }
    
    // Max Pooling
    void maxPool1d(const float* input, float* output, int input_length, int channels, int stride) {
        const int output_length = input_length / stride;
        
        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < output_length; i++) {
                float max_val = -INFINITY;
                for (int j = 0; j < stride; j++) {
                    const int idx = c * input_length + i * stride + j;
                    if (input[idx] > max_val) {
                        max_val = input[idx];
                    }
                }
                output[c * output_length + i] = max_val;
            }
        }
    }
    
    // Global Average Pooling
    void globalAvgPool1d(const float* input, float* output, int input_length, int channels) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (int i = 0; i < input_length; i++) {
                sum += input[c * input_length + i];
            }
            output[c] = sum / input_length;
        }
    }
    
    // Linear Layer
    void linear(const float* input, float* output, int input_size, int output_size, 
               const float* weight, const float* bias) {
        for (int i = 0; i < output_size; i++) {
            float sum = 0.0f;
            for (int j = 0; j < input_size; j++) {
                sum += input[j] * weight[i * input_size + j];
            }
            output[i] = sum + bias[i];
        }
    }
    
    // ReLU Activation
    void relu(float* x, int size) {
        for (int i = 0; i < size; i++) {
            if (x[i] < 0) x[i] = 0;
        }
    }
    
    // Layer Normalization (optimized for ESP32)
    void layerNorm(float* x, int batch_size, int seq_length, int embed_dim, 
                  const float* weight, const float* bias) {
        const float epsilon = 1e-5f;
        
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < seq_length; i++) {
                float mean = 0.0f;
                float var = 0.0f;
                
                // Calculate mean
                for (int j = 0; j < embed_dim; j++) {
                    mean += x[(b * seq_length + i) * embed_dim + j];
                }
                mean /= embed_dim;
                
                // Calculate variance
                for (int j = 0; j < embed_dim; j++) {
                    const float diff = x[(b * seq_length + i) * embed_dim + j] - mean;
                    var += diff * diff;
                }
                var /= embed_dim;
                
                // Normalize and scale
                const float inv_std = 1.0f / sqrtf(var + epsilon);
                for (int j = 0; j < embed_dim; j++) {
                    x[(b * seq_length + i) * embed_dim + j] = 
                        (x[(b * seq_length + i) * embed_dim + j] - mean) * 
                        inv_std * weight[j] + bias[j];
                }
            }
        }
    }
    
    // Multi-head attention (optimized implementation)
    void multiHeadAttention(float* x, float* tmp_buffer, int seq_length, int embed_dim, int num_heads,
                           const float* in_proj_weight, const float* in_proj_bias,
                           const float* out_proj_weight, const float* out_proj_bias) {
        const int head_dim = embed_dim / num_heads;
        
        // Save residual
        float* residual = tmp_buffer;
        float* qkv = tmp_buffer + seq_length * embed_dim;
        float* attn_output = tmp_buffer + seq_length * embed_dim * 4;
        
        // Copy for residual connection
        memcpy(residual, x, seq_length * embed_dim * sizeof(float));
        
        // Linear projection for Q, K, V (fused)
        for (int i = 0; i < seq_length; i++) {
            for (int j = 0; j < 3 * embed_dim; j++) {
                float sum = 0.0f;
                for (int k = 0; k < embed_dim; k++) {
                    sum += x[i * embed_dim + k] * in_proj_weight[j * embed_dim + k];
                }
                qkv[i * 3 * embed_dim + j] = sum + in_proj_bias[j];
            }
        }
        
        // Split into heads and compute scaled dot-product attention
        // This is a memory-efficient implementation for ESP32
        // We compute one head at a time to save memory
        
        for (int h = 0; h < num_heads; h++) {
            // Extract Q, K, V for this head
            float* q = tmp_buffer + seq_length * embed_dim * 5;
            float* k = q + seq_length * head_dim;
            float* v = k + seq_length * head_dim;
            float* scores = v + seq_length * head_dim;
            
            // Extract Q, K, V for current head
            for (int i = 0; i < seq_length; i++) {
                for (int j = 0; j < head_dim; j++) {
                    q[i * head_dim + j] = qkv[i * 3 * embed_dim + h * head_dim + j];
                    k[i * head_dim + j] = qkv[i * 3 * embed_dim + embed_dim + h * head_dim + j];
                    v[i * head_dim + j] = qkv[i * 3 * embed_dim + 2 * embed_dim + h * head_dim + j];
                }
            }
            
            // Compute attention scores
            for (int i = 0; i < seq_length; i++) {
                for (int j = 0; j < seq_length; j++) {
                    float sum = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        sum += q[i * head_dim + d] * k[j * head_dim + d];
                    }
                    scores[i * seq_length + j] = sum / sqrtf(head_dim);
                }
            }
            
            // Softmax
            for (int i = 0; i < seq_length; i++) {
                // Find max for numerical stability
                float max_val = scores[i * seq_length];
                for (int j = 1; j < seq_length; j++) {
                    if (scores[i * seq_length + j] > max_val) {
                        max_val = scores[i * seq_length + j];
                    }
                }
                
                // Compute exp and sum
                float sum_exp = 0.0f;
                for (int j = 0; j < seq_length; j++) {
                    scores[i * seq_length + j] = expf(scores[i * seq_length + j] - max_val);
                    sum_exp += scores[i * seq_length + j];
                }
                
                // Normalize
                for (int j = 0; j < seq_length; j++) {
                    scores[i * seq_length + j] /= sum_exp;
                }
            }
            
            // Apply attention to values
            for (int i = 0; i < seq_length; i++) {
                for (int d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq_length; j++) {
                        sum += scores[i * seq_length + j] * v[j * head_dim + d];
                    }
                    attn_output[i * embed_dim + h * head_dim + d] = sum;
                }
            }
        }
        
        // Output projection
        for (int i = 0; i < seq_length; i++) {
            for (int j = 0; j < embed_dim; j++) {
                float sum = 0.0f;
                for (int k = 0; k < embed_dim; k++) {
                    sum += attn_output[i * embed_dim + k] * out_proj_weight[j * embed_dim + k];
                }
                x[i * embed_dim + j] = residual[i * embed_dim + j] + sum + out_proj_bias[j];
            }
        }
    }
    
    // Feed Forward Network
    void feedForward(float* x, float* tmp_buffer, int seq_length, int embed_dim, int ff_dim,
                    const float* layer1_weight, const float* layer1_bias,
                    const float* layer2_weight, const float* layer2_bias) {
        // Save residual
        float* residual = tmp_buffer;
        float* ff_output = tmp_buffer + seq_length * embed_dim;
        
        // Copy for residual connection
        memcpy(residual, x, seq_length * embed_dim * sizeof(float));
        
        // First linear layer + ReLU
        for (int i = 0; i < seq_length; i++) {
            for (int j = 0; j < ff_dim; j++) {
                float sum = 0.0f;
                for (int k = 0; k < embed_dim; k++) {
                    sum += x[i * embed_dim + k] * layer1_weight[j * embed_dim + k];
                }
                ff_output[i * ff_dim + j] = sum + layer1_bias[j];
                // ReLU
                if (ff_output[i * ff_dim + j] < 0) ff_output[i * ff_dim + j] = 0;
            }
        }
        
        // Second linear layer
        for (int i = 0; i < seq_length; i++) {
            for (int j = 0; j < embed_dim; j++) {
                float sum = 0.0f;
                for (int k = 0; k < ff_dim; k++) {
                    sum += ff_output[i * ff_dim + k] * layer2_weight[j * ff_dim + k];
                }
                x[i * embed_dim + j] = residual[i * embed_dim + j] + sum + layer2_bias[j];
            }
        }
    }
    
    // Transformer Block
    void transformerBlock(float* x, float* tmp_buffer, int seq_length, int embed_dim, int num_heads, int ff_dim,
                         const float* attn_in_proj_weight, const float* attn_in_proj_bias,
                         const float* attn_out_proj_weight, const float* attn_out_proj_bias,
                         const float* ff_layer1_weight, const float* ff_layer1_bias,
                         const float* ff_layer2_weight, const float* ff_layer2_bias,
                         const float* norm1_weight, const float* norm1_bias,
                         const float* norm2_weight, const float* norm2_bias) {
        // First copy for pre-normalization
        float* norm_x = tmp_buffer + seq_length * embed_dim * 6;
        memcpy(norm_x, x, seq_length * embed_dim * sizeof(float));
        
        // Layer Norm 1
        layerNorm(norm_x, 1, seq_length, embed_dim, norm1_weight, norm1_bias);
        
        // Self-Attention
        multiHeadAttention(norm_x, tmp_buffer, seq_length, embed_dim, num_heads,
                          attn_in_proj_weight, attn_in_proj_bias,
                          attn_out_proj_weight, attn_out_proj_bias);
        
        // Add residual (already done in multiHeadAttention)
        
        // Second copy for pre-normalization
        memcpy(norm_x, norm_x, seq_length * embed_dim * sizeof(float));
        
        // Layer Norm 2
        layerNorm(norm_x, 1, seq_length, embed_dim, norm2_weight, norm2_bias);
        
        // Feed Forward Network
