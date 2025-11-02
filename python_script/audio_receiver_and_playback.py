import serial
import time
import wave
import struct
import argparse
import os
from datetime import datetime

import numpy as np

# Default serial port and baud - change these to match your ESP32 connection
DEFAULT_PORT = "COM3"
DEFAULT_BAUD = 921600


def compute_dominant_frequency(samples, sample_rate):
    """
    Compute the most dominant frequency in the audio samples using FFT.
    samples: 1D numpy array of int16
    sample_rate: sample rate in Hz
    Returns: dominant frequency in Hz
    """
    if len(samples) == 0:
        return None
    # Remove DC offset
    samples = samples - np.mean(samples)
    # FFT
    fft_result = np.fft.rfft(samples)
    fft_magnitude = np.abs(fft_result)
    # Ignore DC (0 Hz)
    fft_magnitude[0] = 0
    # Find the peak frequency
    peak_index = np.argmax(fft_magnitude)
    freqs = np.fft.rfftfreq(len(samples), d=1.0/sample_rate)
    dominant_freq = freqs[peak_index]
    return dominant_freq


# SerialManager class to manage persistent serial connection
class SerialManager:
    def __init__(self, port, baudrate, timeout=30):
        import serial
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None

    def open(self):
        if self.ser is None or not self.ser.is_open:
            self.ser = serial.Serial(
                self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)  # Allow time for connection to stabilize
            print(
                f"[SerialManager] Connected to {self.port} at {self.baudrate} baud")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"[SerialManager] Closed serial connection on {self.port}")

    def write(self, data):
        self.open()
        self.ser.write(data)

    def readline(self):
        self.open()
        return self.ser.readline()

    def read(self, size):
        self.open()
        return self.ser.read(size)

    @property
    def in_waiting(self):
        self.open()
        return self.ser.in_waiting if self.ser else 0


def read_exact(serial_mgr, expected_bytes, overall_timeout_seconds=None):
    """
    Read exactly expected_bytes from serial_mgr, using multiple reads if necessary.
    Returns bytes object (may be shorter if timed out).
    """
    if overall_timeout_seconds is None:
        # base timeout estimate: allow at least serial_mgr.timeout plus a margin
        overall_timeout_seconds = max(10, serial_mgr.timeout)

    deadline = time.time() + overall_timeout_seconds
    data = bytearray()
    while len(data) < expected_bytes and time.time() < deadline:
        to_read = expected_bytes - len(data)
        chunk = serial_mgr.read(to_read)
        if chunk:
            data.extend(chunk)
        else:
            time.sleep(0.01)
    return bytes(data)


def seconds_to_hms(total_seconds):
    """Convert total seconds to H:MM:SS string."""
    total_seconds = int(round(total_seconds))
    hrs = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hrs}:{mins:02d}:{secs:02d}"


def ensure_recordings_dir():
    """Create recordings directory if it doesn't exist"""
    recordings_dir = "recordings"
    if not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir)
        print(
            f"Created recordings directory: {os.path.abspath(recordings_dir)}")
    return recordings_dir


def get_timestamped_filename(recordings_dir):
    """Generate a filename with timestamp in the recordings directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(recordings_dir, f"recording_{timestamp}.wav")
    return filename


def auto_record_and_playback(serial_mgr):
    """
    Automatically record and play back audio once on startup
    """
    print("===== STARTING AUTOMATIC RECORD AND PLAYBACK CYCLE =====")
    output_file = record_audio(serial_mgr, None, playback=True)
    print("===== AUTOMATIC CYCLE COMPLETE =====")
    print("Send 'r' to record or 'p' to play back the last recording")
    return output_file


def record_audio(serial_mgr, output_file=None, playback=False):
    """
    Receive audio data from ESP32 over serial and save as a WAV file with timestamp
    """
    # Create recordings directory and get timestamped filename if not specified
    if output_file is None:
        recordings_dir = ensure_recordings_dir()
        output_file = get_timestamped_filename(recordings_dir)

    try:

        # Send command to start recording
        serial_mgr.write(b'r')
        print("Sent recording command to ESP32")

        # --- Parse ESP32 messages for inference result (before BEGIN_AUDIO) ---
        classification = None
        score = None
        import re
        while True:
            line = serial_mgr.readline().decode('utf-8', errors='replace').strip()
            if not line:
                print("Timeout waiting for recording to start")
                return
            if line.startswith("BEGIN_AUDIO"):
                break
            print(f"ESP32: {line}")
            if line.startswith("Inference result:"):
                m = re.search(
                    r"Inference result: (BIRD|NO_BIRD) \(score: ([0-9.]+)\)", line)
                if m:
                    classification = m.group(1)
                    score = m.group(2)

        # Read header information
        sample_rate = int(serial_mgr.readline().decode('utf-8').strip())
        num_samples = int(serial_mgr.readline().decode('utf-8').strip())

        print(
            f"Recording started. Sample rate: {sample_rate}Hz, Expected samples: {num_samples}")

        # Wait for binary data marker
        line = serial_mgr.readline().decode('utf-8').strip()
        if line != "BEGIN_BINARY":
            raise ValueError(f"Expected 'BEGIN_BINARY', got '{line}'")

        # Read binary data
        start_time = time.time()

        # Read all the binary data (chunked to ensure we get exactly expected bytes)
        # 2 bytes per sample (16-bit)
        expected_bytes = num_samples * 2
        # estimate a reasonable overall timeout based on baudrate
        try:
            bytes_per_sec = serial_mgr.baudrate / 10.0
        except Exception:
            bytes_per_sec = 11520.0
        estimated_time = expected_bytes / bytes_per_sec
        overall_timeout = max(serial_mgr.timeout + 5, estimated_time + 5)
        binary_data = read_exact(serial_mgr, expected_bytes, overall_timeout)
        if len(binary_data) < expected_bytes:
            print(
                f"Warning: expected {expected_bytes} bytes but received {len(binary_data)} bytes")

        # Read the end marker
        serial_mgr.readline()  # Read the newline after binary data
        line = serial_mgr.readline().decode('utf-8', errors='replace').strip()
        if line != "END_AUDIO":
            print(f"Warning: Expected 'END_AUDIO', got '{line}'")

        duration = time.time() - start_time
        print(
            f"Received {len(binary_data)//2} samples ({len(binary_data)} bytes) in {duration:.2f} seconds")

        # --- Also parse any remaining ESP32 messages for inference result (after audio) ---
        status_messages = []
        while serial_mgr.in_waiting:
            msg = serial_mgr.readline().decode('utf-8', errors='replace').strip()
            if msg:
                status_messages.append(msg)
                print(f"ESP32: {msg}")
                if classification is None and msg.startswith("Inference result:"):
                    m = re.search(
                        r"Inference result: (BIRD|NO_BIRD) \(score: ([0-9.]+)\)", msg)
                    if m:
                        classification = m.group(1)
                        score = m.group(2)

        # Fallback if not found in status messages
        if classification is None or score is None:
            classification = "UNKNOWN"
            score = "NA"

        # Sanitize for filenames
        class_str = classification.replace("_", "").lower()
        score_str = score.replace(".", "p")

        # Build new filenames
        recordings_dir = os.path.dirname(output_file)
        timestamp = os.path.splitext(os.path.basename(output_file))[
            0].split('_')[-1]
        archive_name = f"{class_str}_{score_str}_{timestamp}.wav"
        archive_path = os.path.join(recordings_dir, archive_name)
        latest_name = f"latest_recording_{class_str}_{score_str}.wav"
        latest_path = os.path.join(recordings_dir, latest_name)

        # Save archive file
        with wave.open(archive_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(binary_data)
        print(f"Audio saved to {os.path.abspath(archive_path)}")

        # Delete previous latest recording(s)
        for f in os.listdir(recordings_dir):
            if f.startswith("latest_recording_") and f.endswith(".wav"):
                try:
                    os.remove(os.path.join(recordings_dir, f))
                except Exception:
                    pass

        # Save as latest recording
        with wave.open(latest_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(binary_data)
        print(f"Audio also saved as {os.path.abspath(latest_path)}")

        # Compute and print dominant frequency
        samples = np.frombuffer(binary_data, dtype=np.int16)
        dominant_freq = compute_dominant_frequency(samples, sample_rate)
        if dominant_freq is not None:
            print(f"Dominant frequency: {dominant_freq:.2f} Hz")
        else:
            print("Could not determine dominant frequency.")

        # Send playback command if requested
        if playback:
            print("Sending playback command to ESP32...")
            serial_mgr.write(b'p')

            # Read and display any status messages from playback
            timeout = time.time() + 5  # 5 second timeout
            while time.time() < timeout:
                if serial_mgr.in_waiting:
                    msg = serial_mgr.readline().decode('utf-8', errors='replace').strip()
                    if msg:
                        print(f"ESP32: {msg}")
                else:
                    time.sleep(0.1)

        return output_file

    except Exception as e:
        print(f"Error: {e}")

    return None


def receive_audio_once(serial_mgr, output_file=None, compute_dominant=True):
    """
    Receive a single recording sent by the ESP32 without sending the 'r' command.
    This is useful when the ESP is in continuous send mode.
    """
    # Create recordings directory and get timestamped filename if not specified
    if output_file is None:
        recordings_dir = ensure_recordings_dir()
        output_file = get_timestamped_filename(recordings_dir)

    try:
        classification = None
        score = None
        import re

        # Read lines until BEGIN_AUDIO
        while True:
            line = serial_mgr.readline().decode('utf-8', errors='replace').strip()
            if not line:
                print("Timeout waiting for recording to start")
                return
            if line.startswith("BEGIN_AUDIO"):
                break
            print(f"ESP32: {line}")
            if line.startswith("Inference result:"):
                m = re.search(
                    r"Inference result: (BIRD|NO_BIRD) \(score: ([0-9.]+)\)", line)
                if m:
                    classification = m.group(1)
                    score = m.group(2)

        # Read header information
        sample_rate = int(serial_mgr.readline().decode('utf-8').strip())
        num_samples = int(serial_mgr.readline().decode('utf-8').strip())

        print(
            f"Recording started. Sample rate: {sample_rate}Hz, Expected samples: {num_samples}")

        # Wait for binary data marker
        line = serial_mgr.readline().decode('utf-8').strip()
        if line != "BEGIN_BINARY":
            raise ValueError(f"Expected 'BEGIN_BINARY', got '{line}'")

        # Read binary data (chunked)
        expected_bytes = num_samples * 2
        try:
            bytes_per_sec = serial_mgr.baudrate / 10.0
        except Exception:
            bytes_per_sec = 11520.0
        estimated_time = expected_bytes / bytes_per_sec
        overall_timeout = max(serial_mgr.timeout + 5, estimated_time + 5)
        binary_data = read_exact(serial_mgr, expected_bytes, overall_timeout)
        if len(binary_data) < expected_bytes:
            print(
                f"Warning: expected {expected_bytes} bytes but received {len(binary_data)} bytes")

        # Read the end marker (and newline)
        serial_mgr.readline()
        line = serial_mgr.readline().decode('utf-8', errors='replace').strip()
        if line != "END_AUDIO":
            print(f"Warning: Expected 'END_AUDIO', got '{line}'")

        # --- Also parse any remaining ESP32 messages for inference result (after audio) ---
        status_messages = []
        while serial_mgr.in_waiting:
            msg = serial_mgr.readline().decode('utf-8', errors='replace').strip()
            if msg:
                status_messages.append(msg)
                print(f"ESP32: {msg}")
                if classification is None and msg.startswith("Inference result:"):
                    m = re.search(
                        r"Inference result: (BIRD|NO_BIRD) \(score: ([0-9.]+)\)", msg)
                    if m:
                        classification = m.group(1)
                        score = m.group(2)

        if classification is None or score is None:
            classification = "UNKNOWN"
            score = "NA"

        class_str = classification.replace("_", "").lower()
        score_str = score.replace('.', 'p')

        recordings_dir = os.path.dirname(output_file)
        timestamp = os.path.splitext(os.path.basename(output_file))[
            0].split('_')[-1]
        archive_name = f"{class_str}_{score_str}_{timestamp}.wav"
        archive_path = os.path.join(recordings_dir, archive_name)
        latest_name = f"latest_recording_{class_str}_{score_str}.wav"
        latest_path = os.path.join(recordings_dir, latest_name)

        # Save archive
        with wave.open(archive_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(binary_data)
        print(f"Audio saved to {os.path.abspath(archive_path)}")

        # Update latest
        for f in os.listdir(recordings_dir):
            if f.startswith("latest_recording_") and f.endswith('.wav'):
                try:
                    os.remove(os.path.join(recordings_dir, f))
                except Exception:
                    pass
        with wave.open(latest_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(binary_data)
        print(f"Audio also saved as {os.path.abspath(latest_path)}")

        if compute_dominant:
            samples = np.frombuffer(binary_data, dtype=np.int16)
            dominant_freq = compute_dominant_frequency(samples, sample_rate)
            if dominant_freq is not None:
                print(f"Dominant frequency: {dominant_freq:.2f} Hz")
            else:
                print("Could not determine dominant frequency.")
        duration_seconds = float(num_samples) / \
            float(sample_rate) if sample_rate > 0 else 0.0
        return archive_path, duration_seconds

    except Exception as e:
        print(f"Error in receive_audio_once: {e}")
    return None, 0.0


def play_last_recording(serial_mgr):
    """
    Send command to ESP32 to play back the last recording stored in its memory
    """
    try:
        # Send playback command
        serial_mgr.write(b'p')
        print("Sent playback command to ESP32")

        # Read and display any status messages
        timeout = time.time() + 5  # 5 second timeout
        while time.time() < timeout:
            if serial_mgr.in_waiting:
                msg = serial_mgr.readline().decode('utf-8', errors='replace').strip()
                if msg:
                    print(f"ESP32: {msg}")
                    timeout = time.time() + 5  # Reset timeout when receiving messages
            else:
                time.sleep(0.1)

    except Exception as e:
        print(f"Error: {e}")


def interactive_mode(serial_mgr):
    """
    Run an interactive session where user can type 'r' to record or 'p' to play
    """
    print("\nEntering interactive mode. Type:")
    print("  r - to record a new audio sample")
    print("  c - to start continuous recording (press Ctrl+C to stop)")
    print("  p - to play the last recorded audio")
    print("  q - to quit")

    last_file = None

    try:
        while True:
            command = input("> ").strip().lower()

            if command == 'r':
                last_file = record_audio(serial_mgr)
            elif command == 'c':
                print("Starting continuous recording (press Ctrl+C to stop)")
                serial_mgr.write(b'c')
                try:
                    rec_count = 0
                    total_seconds = 0.0
                    while True:
                        path, dur = receive_audio_once(
                            serial_mgr, compute_dominant=False)
                        if path:
                            rec_count += 1
                            total_seconds += dur
                            print(
                                f"Recorded #{rec_count}: {os.path.basename(path)}  clip={seconds_to_hms(dur)} total={seconds_to_hms(total_seconds)}")
                        else:
                            print("Warning: failed to receive recording")
                except KeyboardInterrupt:
                    print("Stopping continuous recording")
                    try:
                        serial_mgr.write(b's')
                    except Exception:
                        pass
            elif command == 'p':
                play_last_recording(serial_mgr)
            elif command == 'q':
                print("Exiting...")
                break
            else:
                print("Unknown command. Use 'r' to record, 'p' to play, or 'q' to quit.")

    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Record and play audio from ESP32 via serial')
    parser.add_argument('--port', default=DEFAULT_PORT,
                        help=f'Serial port (default: {DEFAULT_PORT})')
    parser.add_argument('--baudrate', type=int, default=DEFAULT_BAUD,
                        help=f'Baud rate (default: {DEFAULT_BAUD})')
    parser.add_argument('--output', default=None,
                        help='Output file name (default: auto-generated timestamp)')
    parser.add_argument('--playback', action='store_true',
                        help='Play back the recording on the ESP32 after recording')
    parser.add_argument('--play-only', action='store_true',
                        help='Only play back the last recording (no new recording)')
    parser.add_argument('--auto', action='store_true',
                        help='Run automatic record+playback once on startup')
    parser.add_argument('--continuous', action='store_true',
                        help='Start MCU continuous recording mode and receive recordings until interrupted')

    args = parser.parse_args()

    serial_mgr = SerialManager(args.port, args.baudrate)
    try:
        serial_mgr.open()
        if args.play_only:
            play_last_recording(serial_mgr)
        elif args.continuous:
            # Tell MCU to start continuous mode and receive recordings until interrupted
            print("Starting MCU continuous mode (press Ctrl+C to stop)")
            serial_mgr.write(b'c')
            try:
                rec_count = 0
                total_seconds = 0.0
                while True:
                    path, dur = receive_audio_once(
                        serial_mgr, compute_dominant=False)
                    if path:
                        rec_count += 1
                        total_seconds += dur
                        print(
                            f"Recorded #{rec_count}: {os.path.basename(path)}  clip={seconds_to_hms(dur)} total={seconds_to_hms(total_seconds)}")
                    else:
                        print("Warning: failed to receive recording")
            except KeyboardInterrupt:
                print("Interrupted by user - stopping MCU continuous mode")
                try:
                    serial_mgr.write(b's')
                except Exception:
                    pass
        else:
            # Optionally run an automatic record+playback cycle if requested
            if args.auto:
                auto_record_and_playback(serial_mgr)

            # Then enter interactive mode for commands
            interactive_mode(serial_mgr)
    finally:
        serial_mgr.close()
