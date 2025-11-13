import serial
import time
import wave
import struct
import argparse
import os
from datetime import datetime

import numpy as np
import threading

# Default serial port - change this to match your ESP32 connection
DEFAULT_PORT = "/dev/cu.usbmodem14401"

TEST_FILES_BASE = os.path.join(os.path.dirname(__file__), "../input")
BIRD_TEST_DIR = os.path.join(TEST_FILES_BASE, "bird")
NO_BIRD_TEST_DIR = os.path.join(TEST_FILES_BASE, "no_bird")
RECORDINGS_BASE = "recordings"
RECORDINGS_BIRD = os.path.join(RECORDINGS_BASE, "bird")
RECORDINGS_NO_BIRD = os.path.join(RECORDINGS_BASE, "no_bird")

# State for cycling through test files
_test_files = []
_test_file_index = [0]  # Mutable for session


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
    def __init__(self, port, baudrate, timeout=10):
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


def get_test_files():
    """Return a list of (filepath, label) tuples for all test files, sorted."""
    files = []
    for label, folder in [("bird", BIRD_TEST_DIR), ("no_bird", NO_BIRD_TEST_DIR)]:
        if os.path.isdir(folder):
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith(".wav"):
                    files.append((os.path.join(folder, f), label))
    return files


def ensure_label_recordings_dir(label):
    """Ensure the recordings subfolder for the label exists."""
    target_dir = RECORDINGS_BIRD if label == "bird" else RECORDINGS_NO_BIRD
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return target_dir


def play_wav_file(filepath):
    """Play a wav file using sounddevice (blocking)."""
    try:
        import sounddevice as sd
        import soundfile as sf
        data, fs = sf.read(filepath, dtype='float32')
        print(f"Playing test file: {os.path.basename(filepath)}")
        sd.play(data, fs)
        sd.wait()
        print("Test file playback finished.")
    except Exception as e:
        print(f"Could not play test file: {e}")


def record_audio(serial_mgr, output_file=None, playback=False, label=None):
    """
    Receive audio data from ESP32 over serial and save as a WAV file with timestamp.
    If label is given, save in the corresponding subfolder.
    """
    # Create recordings directory and get timestamped filename if not specified
    if output_file is None:
        if label:
            recordings_dir = ensure_label_recordings_dir(label)
        else:
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

        # Read all the binary data at once
        # 2 bytes per sample (16-bit)
        binary_data = serial_mgr.read(num_samples * 2)

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
        timestamp = os.path.splitext(os.path.basename(output_file))[0].split('_')[-1]
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


def auto_cycle(serial_mgr):
    """
    Automatically cycle through all files in bird, then no_bird, playing and recording each.
    """
    all_files = []
    # Get bird files first, then no_bird files
    if os.path.isdir(BIRD_TEST_DIR):
        for f in sorted(os.listdir(BIRD_TEST_DIR)):
            if f.lower().endswith(".wav"):
                all_files.append((os.path.join(BIRD_TEST_DIR, f), "bird"))
    if os.path.isdir(NO_BIRD_TEST_DIR):
        for f in sorted(os.listdir(NO_BIRD_TEST_DIR)):
            if f.lower().endswith(".wav"):
                all_files.append((os.path.join(NO_BIRD_TEST_DIR, f), "no_bird"))
    if not all_files:
        print("No test files found in bird or no_bird folders under test_files.")
        return

    for test_file, label in all_files:
        print(f"\n--- Playing and recording: {os.path.basename(test_file)} ({label}) ---")
        playback_thread = threading.Thread(target=play_wav_file, args=(test_file,))
        playback_thread.start()
        record_audio(serial_mgr, output_file=None, playback=False, label=label)
        playback_thread.join()
        time.sleep(1)  # 1 second break


def interactive_mode(serial_mgr):
    """
    Run an interactive session where user can type 'r' to record or 'p' to play
    """
    print("\nEntering interactive mode. Type:")
    print("  r - to record a new audio sample (and play next test file)")
    print("  a - to automatically cycle through all test files (bird first, then no_bird)")
    print("  p - to play the last recorded audio")
    print("  q - to quit")

    # Prepare test file list
    global _test_files
    _test_files = get_test_files()
    if not _test_files:
        print("No test files found in bird or no_bird folders under test_files.")
    last_file = None

    try:
        while True:
            command = input("> ").strip().lower()

            if command == 'r':
                if not _test_files:
                    print("No test files to play.")
                    continue
                idx = _test_file_index[0] % len(_test_files)
                test_file, label = _test_files[idx]
                _test_file_index[0] = (idx + 1) % len(_test_files)
                playback_thread = threading.Thread(target=play_wav_file, args=(test_file,))
                playback_thread.start()
                last_file = record_audio(serial_mgr, output_file=None, playback=False, label=label)
                playback_thread.join()
            elif command == 'a':
                auto_cycle(serial_mgr)
            elif command == 'p':
                play_last_recording(serial_mgr)
            elif command == 'q':
                print("Exiting...")
                break
            else:
                print("Unknown command. Use 'r' to record, 'a' to auto-cycle, 'p' to play, or 'q' to quit.")

    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Record and play audio from ESP32 via serial')
    parser.add_argument('--port', default=DEFAULT_PORT,
                        help=f'Serial port (default: {DEFAULT_PORT})')
    parser.add_argument('--baudrate', type=int, default=115200,
                        help='Baud rate (default: 115200)')
    parser.add_argument('--output', default=None,
                        help='Output file name (default: auto-generated timestamp)')
    parser.add_argument('--playback', action='store_true',
                        help='Play back the recording on the ESP32 after recording')
    parser.add_argument('--play-only', action='store_true',
                        help='Only play back the last recording (no new recording)')
    parser.add_argument('--no-auto', action='store_true',
                        help='Skip automatic recording on startup')

    args = parser.parse_args()

    serial_mgr = SerialManager(args.port, args.baudrate)
    try:
        if args.play_only:
            serial_mgr.open()
            play_last_recording(serial_mgr)
        else:
            # First, do an automatic record+playback cycle unless --no-auto is specified
            serial_mgr.open()
            if not args.no_auto:
                auto_record_and_playback(serial_mgr)

            # Then enter interactive mode for commands
            interactive_mode(serial_mgr)
    finally:
        serial_mgr.close()
