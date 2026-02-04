import argparse
import os
import sys
import time
import wave
from datetime import datetime
import threading

try:
    import serial
    from serial.tools import list_ports
except Exception:  # pragma: no cover
    serial = None
    list_ports = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

# Optional playback dependencies (test harness)
try:
    import sounddevice as sd
    import soundfile as sf
except Exception:  # pragma: no cover
    sd = None
    sf = None

# Defaults
DEFAULT_BAUD = 921600
DEFAULT_TIMEOUT = 30

# Test file locations (kept alongside this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILES_BASE = os.path.join(SCRIPT_DIR, "input")
BIRD_TEST_DIR = os.path.join(TEST_FILES_BASE, "bird")
NO_BIRD_TEST_DIR = os.path.join(TEST_FILES_BASE, "no_bird")
RECORDINGS_BASE = os.path.join(SCRIPT_DIR, "recordings")
RECORDINGS_BIRD = os.path.join(RECORDINGS_BASE, "bird")
RECORDINGS_NO_BIRD = os.path.join(RECORDINGS_BASE, "no_bird")

# State for cycling through test files
_test_files = []
_test_file_index = [0]


def set_test_dirs(base_dir):
    global TEST_FILES_BASE, BIRD_TEST_DIR, NO_BIRD_TEST_DIR
    TEST_FILES_BASE = base_dir
    BIRD_TEST_DIR = os.path.join(TEST_FILES_BASE, "bird")
    NO_BIRD_TEST_DIR = os.path.join(TEST_FILES_BASE, "no_bird")


def compute_dominant_frequency(samples, sample_rate):
    """
    Compute the most dominant frequency in the audio samples using FFT.
    Returns dominant frequency in Hz, or None if unavailable.
    """
    if np is None or samples is None or len(samples) == 0:
        return None
    samples = samples - np.mean(samples)
    fft_result = np.fft.rfft(samples)
    fft_magnitude = np.abs(fft_result)
    if len(fft_magnitude) == 0:
        return None
    fft_magnitude[0] = 0
    peak_index = int(np.argmax(fft_magnitude))
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    return float(freqs[peak_index])


class SerialManager:
    def __init__(self, port, baudrate, timeout=DEFAULT_TIMEOUT):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None

    def open(self):
        if serial is None:
            raise RuntimeError("pyserial is required. Install with: pip install pyserial")
        if self.ser is None or not self.ser.is_open:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2)
            print(f"[SerialManager] Connected to {self.port} at {self.baudrate} baud")

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
    if overall_timeout_seconds is None:
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
    total_seconds = int(round(total_seconds))
    hrs = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hrs}:{mins:02d}:{secs:02d}"


def ensure_recordings_dir(base_dir=RECORDINGS_BASE):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created recordings directory: {os.path.abspath(base_dir)}")
    return base_dir


def ensure_label_recordings_dir(label):
    target_dir = RECORDINGS_BIRD if label == "bird" else RECORDINGS_NO_BIRD
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return target_dir


def ensure_input_dirs():
    os.makedirs(BIRD_TEST_DIR, exist_ok=True)
    os.makedirs(NO_BIRD_TEST_DIR, exist_ok=True)
    return TEST_FILES_BASE


def get_timestamped_filename(recordings_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(recordings_dir, f"recording_{timestamp}.wav")


def get_test_files():
    files = []
    for label, folder in [("bird", BIRD_TEST_DIR), ("no_bird", NO_BIRD_TEST_DIR)]:
        if os.path.isdir(folder):
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith(".wav"):
                    files.append((os.path.join(folder, f), label))
    return files


def play_wav_file(filepath):
    if sd is None or sf is None:
        print("Playback dependencies not installed. Skipping test file playback.")
        return
    try:
        data, fs = sf.read(filepath, dtype="float32")
        print(f"Playing test file: {os.path.basename(filepath)}")
        sd.play(data, fs)
        sd.wait()
        print("Test file playback finished.")
    except Exception as e:
        print(f"Could not play test file: {e}")


def record_audio(serial_mgr, output_file=None, playback=False, label=None, compute_dominant=True):
    if output_file is None:
        if label:
            recordings_dir = ensure_label_recordings_dir(label)
        else:
            recordings_dir = ensure_recordings_dir()
        output_file = get_timestamped_filename(recordings_dir)

    try:
        serial_mgr.write(b"r")
        print("Sent recording command to ESP32")

        classification = None
        score = None
        import re

        while True:
            line = serial_mgr.readline().decode("utf-8", errors="replace").strip()
            if not line:
                print("Timeout waiting for recording to start")
                return None
            if line.startswith("BEGIN_AUDIO"):
                break
            print(f"ESP32: {line}")
            if line.startswith("Inference result:"):
                m = re.search(r"Inference result: (BIRD|NO_BIRD) \(score: ([0-9.]+)\)", line)
                if m:
                    classification = m.group(1)
                    score = m.group(2)

        sample_rate = int(serial_mgr.readline().decode("utf-8").strip())
        num_samples = int(serial_mgr.readline().decode("utf-8").strip())
        print(f"Recording started. Sample rate: {sample_rate}Hz, Expected samples: {num_samples}")

        line = serial_mgr.readline().decode("utf-8").strip()
        if line != "BEGIN_BINARY":
            raise ValueError(f"Expected 'BEGIN_BINARY', got '{line}'")

        expected_bytes = num_samples * 2
        try:
            bytes_per_sec = serial_mgr.baudrate / 10.0
        except Exception:
            bytes_per_sec = 11520.0
        estimated_time = expected_bytes / bytes_per_sec
        overall_timeout = max(serial_mgr.timeout + 5, estimated_time + 5)
        binary_data = read_exact(serial_mgr, expected_bytes, overall_timeout)
        if len(binary_data) < expected_bytes:
            print(f"Warning: expected {expected_bytes} bytes but received {len(binary_data)} bytes")

        serial_mgr.readline()
        line = serial_mgr.readline().decode("utf-8", errors="replace").strip()
        if line != "END_AUDIO":
            print(f"Warning: Expected 'END_AUDIO', got '{line}'")

        status_messages = []
        while serial_mgr.in_waiting:
            msg = serial_mgr.readline().decode("utf-8", errors="replace").strip()
            if msg:
                status_messages.append(msg)
                print(f"ESP32: {msg}")
                if classification is None and msg.startswith("Inference result:"):
                    m = re.search(r"Inference result: (BIRD|NO_BIRD) \(score: ([0-9.]+)\)", msg)
                    if m:
                        classification = m.group(1)
                        score = m.group(2)

        if classification is None or score is None:
            classification = "UNKNOWN"
            score = "NA"

        class_str = classification.replace("_", "").lower()
        score_str = score.replace(".", "p")

        recordings_dir = os.path.dirname(output_file)
        timestamp = os.path.splitext(os.path.basename(output_file))[0].split("_")[-1]
        archive_name = f"{class_str}_{score_str}_{timestamp}.wav"
        archive_path = os.path.join(recordings_dir, archive_name)
        latest_name = f"latest_recording_{class_str}_{score_str}.wav"
        latest_path = os.path.join(recordings_dir, latest_name)

        with wave.open(archive_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(binary_data)
        print(f"Audio saved to {os.path.abspath(archive_path)}")

        for f in os.listdir(recordings_dir):
            if f.startswith("latest_recording_") and f.endswith(".wav"):
                try:
                    os.remove(os.path.join(recordings_dir, f))
                except Exception:
                    pass

        with wave.open(latest_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(binary_data)
        print(f"Audio also saved as {os.path.abspath(latest_path)}")

        if compute_dominant and np is not None:
            samples = np.frombuffer(binary_data, dtype=np.int16)
            dominant_freq = compute_dominant_frequency(samples, sample_rate)
            if dominant_freq is not None:
                print(f"Dominant frequency: {dominant_freq:.2f} Hz")
            else:
                print("Could not determine dominant frequency.")

        if playback:
            print("Sending playback command to ESP32...")
            serial_mgr.write(b"p")
            timeout = time.time() + 5
            while time.time() < timeout:
                if serial_mgr.in_waiting:
                    msg = serial_mgr.readline().decode("utf-8", errors="replace").strip()
                    if msg:
                        print(f"ESP32: {msg}")
                else:
                    time.sleep(0.1)

        return archive_path

    except Exception as e:
        print(f"Error: {e}")

    return None


def receive_audio_once(serial_mgr, output_file=None, compute_dominant=True):
    if output_file is None:
        recordings_dir = ensure_recordings_dir()
        output_file = get_timestamped_filename(recordings_dir)

    try:
        classification = None
        score = None
        import re

        while True:
            line = serial_mgr.readline().decode("utf-8", errors="replace").strip()
            if not line:
                print("Timeout waiting for recording to start")
                return None, 0.0
            if line.startswith("BEGIN_AUDIO"):
                break
            print(f"ESP32: {line}")
            if line.startswith("Inference result:"):
                m = re.search(r"Inference result: (BIRD|NO_BIRD) \(score: ([0-9.]+)\)", line)
                if m:
                    classification = m.group(1)
                    score = m.group(2)

        sample_rate = int(serial_mgr.readline().decode("utf-8").strip())
        num_samples = int(serial_mgr.readline().decode("utf-8").strip())
        print(f"Recording started. Sample rate: {sample_rate}Hz, Expected samples: {num_samples}")

        line = serial_mgr.readline().decode("utf-8").strip()
        if line != "BEGIN_BINARY":
            raise ValueError(f"Expected 'BEGIN_BINARY', got '{line}'")

        expected_bytes = num_samples * 2
        try:
            bytes_per_sec = serial_mgr.baudrate / 10.0
        except Exception:
            bytes_per_sec = 11520.0
        estimated_time = expected_bytes / bytes_per_sec
        overall_timeout = max(serial_mgr.timeout + 5, estimated_time + 5)
        binary_data = read_exact(serial_mgr, expected_bytes, overall_timeout)
        if len(binary_data) < expected_bytes:
            print(f"Warning: expected {expected_bytes} bytes but received {len(binary_data)} bytes")

        serial_mgr.readline()
        line = serial_mgr.readline().decode("utf-8", errors="replace").strip()
        if line != "END_AUDIO":
            print(f"Warning: Expected 'END_AUDIO', got '{line}'")

        status_messages = []
        while serial_mgr.in_waiting:
            msg = serial_mgr.readline().decode("utf-8", errors="replace").strip()
            if msg:
                status_messages.append(msg)
                print(f"ESP32: {msg}")
                if classification is None and msg.startswith("Inference result:"):
                    m = re.search(r"Inference result: (BIRD|NO_BIRD) \(score: ([0-9.]+)\)", msg)
                    if m:
                        classification = m.group(1)
                        score = m.group(2)

        if classification is None or score is None:
            classification = "UNKNOWN"
            score = "NA"

        class_str = classification.replace("_", "").lower()
        score_str = score.replace(".", "p")

        recordings_dir = os.path.dirname(output_file)
        timestamp = os.path.splitext(os.path.basename(output_file))[0].split("_")[-1]
        archive_name = f"{class_str}_{score_str}_{timestamp}.wav"
        archive_path = os.path.join(recordings_dir, archive_name)
        latest_name = f"latest_recording_{class_str}_{score_str}.wav"
        latest_path = os.path.join(recordings_dir, latest_name)

        with wave.open(archive_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(binary_data)
        print(f"Audio saved to {os.path.abspath(archive_path)}")

        for f in os.listdir(recordings_dir):
            if f.startswith("latest_recording_") and f.endswith(".wav"):
                try:
                    os.remove(os.path.join(recordings_dir, f))
                except Exception:
                    pass
        with wave.open(latest_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(binary_data)
        print(f"Audio also saved as {os.path.abspath(latest_path)}")

        if compute_dominant and np is not None:
            samples = np.frombuffer(binary_data, dtype=np.int16)
            dominant_freq = compute_dominant_frequency(samples, sample_rate)
            if dominant_freq is not None:
                print(f"Dominant frequency: {dominant_freq:.2f} Hz")
            else:
                print("Could not determine dominant frequency.")

        duration_seconds = float(num_samples) / float(sample_rate) if sample_rate > 0 else 0.0
        return archive_path, duration_seconds

    except Exception as e:
        print(f"Error in receive_audio_once: {e}")
    return None, 0.0


def play_last_recording(serial_mgr):
    try:
        if sd is None or sf is None:
            print("Playback dependencies not installed. Cannot play local audio.")
            return

        if not os.path.isdir(RECORDINGS_BASE):
            print("No recordings folder found.")
            return

        latest_path = None
        latest_mtime = -1.0
        for root, _dirs, files in os.walk(RECORDINGS_BASE):
            for name in files:
                if name.startswith("latest_recording_") and name.endswith(".wav"):
                    path = os.path.join(root, name)
                    try:
                        mtime = os.path.getmtime(path)
                    except Exception:
                        continue
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        latest_path = path

        if not latest_path:
            print("No latest recording found to play.")
            return

        print(f"Playing latest recording: {os.path.basename(latest_path)}")
        data, fs = sf.read(latest_path, dtype="float32")
        sd.play(data, fs)
        sd.wait()
        print("Playback finished.")

    except Exception as e:
        print(f"Error: {e}")


def auto_cycle(serial_mgr):
    all_files = []
    if os.path.isdir(BIRD_TEST_DIR):
        for f in sorted(os.listdir(BIRD_TEST_DIR)):
            if f.lower().endswith(".wav"):
                all_files.append((os.path.join(BIRD_TEST_DIR, f), "bird"))
    if os.path.isdir(NO_BIRD_TEST_DIR):
        for f in sorted(os.listdir(NO_BIRD_TEST_DIR)):
            if f.lower().endswith(".wav"):
                all_files.append((os.path.join(NO_BIRD_TEST_DIR, f), "no_bird"))
    if not all_files:
        print(
            "No test files found in bird or no_bird folders under input.\n"
            "Expected folder tree:\n"
            "  input/\n"
            "    bird/\n"
            "    no_bird/\n"
            f"Base path: {TEST_FILES_BASE}\n"
            "Or pass --test-dir."
        )
        return

    for test_file, label in all_files:
        print(f"\n--- Playing and recording: {os.path.basename(test_file)} ({label}) ---")
        playback_thread = threading.Thread(target=play_wav_file, args=(test_file,))
        playback_thread.start()
        record_audio(serial_mgr, output_file=None, playback=False, label=label)
        playback_thread.join()
        time.sleep(1)


def interactive_mode(serial_mgr):
    print("\nEntering interactive mode. Type:")
    print("  r - record a new audio sample")
    print("  c - start continuous recording (press Ctrl+C to stop)")
    print("  p - play the last recorded audio")
    print("  t - record and play next test file")
    print("  a - auto-cycle all test files (bird, then no_bird)")
    print("  q - quit")

    global _test_files
    _test_files = get_test_files()

    try:
        while True:
            command = input("> ").strip().lower()

            if command == "r":
                record_audio(serial_mgr)
            elif command == "c":
                print("Starting continuous recording (press Ctrl+C to stop)")
                serial_mgr.write(b"c")
                try:
                    rec_count = 0
                    total_seconds = 0.0
                    while True:
                        path, dur = receive_audio_once(serial_mgr, compute_dominant=False)
                        if path:
                            rec_count += 1
                            total_seconds += dur
                            print(
                                f"Recorded #{rec_count}: {os.path.basename(path)}  clip={seconds_to_hms(dur)} total={seconds_to_hms(total_seconds)}"
                            )
                        else:
                            print("Warning: failed to receive recording")
                except KeyboardInterrupt:
                    print("Stopping continuous recording")
                    try:
                        serial_mgr.write(b"s")
                    except Exception:
                        pass
            elif command == "p":
                play_last_recording(serial_mgr)
            elif command == "t":
                if not _test_files:
                    print(
                        "No test files found in bird or no_bird folders under input.\n"
                        "Expected folder tree:\n"
                        "  input/\n"
                        "    bird/\n"
                        "    no_bird/\n"
                        f"Base path: {TEST_FILES_BASE}\n"
                        "Or pass --test-dir."
                    )
                    continue
                idx = _test_file_index[0] % len(_test_files)
                test_file, label = _test_files[idx]
                _test_file_index[0] = (idx + 1) % len(_test_files)
                playback_thread = threading.Thread(target=play_wav_file, args=(test_file,))
                playback_thread.start()
                record_audio(serial_mgr, output_file=None, playback=False, label=label)
                playback_thread.join()
            elif command == "a":
                auto_cycle(serial_mgr)
            elif command == "q":
                print("Exiting...")
                break
            else:
                print("Unknown command. Use r, c, p, t, a, or q.")

    except KeyboardInterrupt:
        print("\nExiting...")


def select_serial_port(preferred=None, interactive=True):
    if preferred:
        return preferred
    if list_ports is None:
        raise RuntimeError("pyserial is required for port discovery.")

    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports detected.")

    if len(ports) == 1:
        return ports[0].device

    for p in ports:
        if "ACM" in p.device or "USB" in p.device:
            return p.device

    if interactive:
        print("Multiple serial ports detected. Select one:")
        for idx, p in enumerate(ports, start=1):
            print(f"  {idx}. {p.device} - {p.description}")
        while True:
            choice = input("Select port number: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(ports):
                return ports[int(choice) - 1].device
            print("Invalid selection. Try again.")

    return ports[0].device


def main():
    parser = argparse.ArgumentParser(description="Record and play audio from ESP32 via serial")
    parser.add_argument("--port", default=None, help="Serial port (auto-detect if omitted)")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUD, help=f"Baud rate (default: {DEFAULT_BAUD})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"Serial timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--output", default=None, help="Output file name (default: auto-generated timestamp)")
    parser.add_argument("--playback", action="store_true", help="Play back the recording on the ESP32 after recording")
    parser.add_argument("--play-only", action="store_true", help="Only play back the last recording (no new recording)")
    parser.add_argument("--auto", action="store_true", help="Run automatic record+playback once on startup")
    parser.add_argument("--continuous", action="store_true", help="Start MCU continuous recording mode and receive recordings until interrupted")
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports and exit")
    parser.add_argument(
        "--test-dir",
        default=None,
        help="Folder containing bird/ and no_bird/ test wavs (default: ./input next to this script)",
    )

    args = parser.parse_args()

    if args.list_ports:
        if list_ports is None:
            print("pyserial is required for port discovery.")
            return 1
        ports = list(list_ports.comports())
        if not ports:
            print("No serial ports detected.")
            return 1
        for p in ports:
            print(f"{p.device} - {p.description}")
        return 0

    if args.test_dir:
        set_test_dirs(os.path.abspath(args.test_dir))

    ensure_input_dirs()
    ensure_recordings_dir()

    port = select_serial_port(args.port, interactive=True)
    serial_mgr = SerialManager(port, args.baudrate, timeout=args.timeout)

    try:
        serial_mgr.open()
        if args.play_only:
            play_last_recording(serial_mgr)
            return 0

        if args.continuous:
            print("Starting MCU continuous mode (press Ctrl+C to stop)")
            serial_mgr.write(b"c")
            try:
                rec_count = 0
                total_seconds = 0.0
                while True:
                    path, dur = receive_audio_once(serial_mgr, compute_dominant=False)
                    if path:
                        rec_count += 1
                        total_seconds += dur
                        print(
                            f"Recorded #{rec_count}: {os.path.basename(path)}  clip={seconds_to_hms(dur)} total={seconds_to_hms(total_seconds)}"
                        )
                    else:
                        print("Warning: failed to receive recording")
            except KeyboardInterrupt:
                print("Interrupted by user - stopping MCU continuous mode")
                try:
                    serial_mgr.write(b"s")
                except Exception:
                    pass
            return 0

        if args.auto:
            print("===== STARTING AUTOMATIC RECORD AND PLAYBACK CYCLE =====")
            record_audio(serial_mgr, args.output, playback=True)
            print("===== AUTOMATIC CYCLE COMPLETE =====")

        interactive_mode(serial_mgr)
        return 0

    finally:
        serial_mgr.close()


if __name__ == "__main__":
    sys.exit(main())
