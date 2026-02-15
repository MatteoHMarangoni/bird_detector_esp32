import os
import time
import threading
import datetime
import numpy as np
import csv
import wave
import shutil
import logging
import argparse
import sounddevice as sd
import psutil
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import re
from collections import deque
import traceback

# Define global variables
stop_event = threading.Event()
processed_csv_files = set()
processed_nonbird_wav_files = set()
processed_bird_wav_files = set()
non_bird_files_info = []
script_start_time = datetime.datetime.now()
file_lock = threading.Lock()
audio_queue = Queue(maxsize=50000)
executor = ThreadPoolExecutor(max_workers=4)

def set_real_time_priority():
    try:
        p = psutil.Process(os.getpid())
        p.nice(-20)  # For Linux (Raspberry Pi), set high priority using nice value
    except Exception as e:
        logging.warning(f"Unable to set real-time priority: {e}")

def setup_logging(log_file_path):
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

def initialize_logging(streamed_audio_folder):
    if not os.path.exists(streamed_audio_folder):
        os.makedirs(streamed_audio_folder)
    log_file_path = os.path.join(streamed_audio_folder,
                                 f'birdnet_log_{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}.txt')
    setup_logging(log_file_path)
    return log_file_path


def log_memory_usage(memory_log_interval):
    while not stop_event.is_set():
        try:
            memory_info = psutil.virtual_memory()
            cpu_usage = psutil.cpu_percent()
            logging.info(f"CPU Usage: {cpu_usage}%")
            logging.info(f"Memory Usage: {memory_info.percent}%")
            print(f"CPU Usage: {cpu_usage}%")
            print(f"Memory Usage: {memory_info.percent}%")
            time.sleep(memory_log_interval)
        except Exception as e:
            logging.error(f"Error in log_memory_usage: {e}")


def save_wav_file(filepath, channels, bit_depth, sampling_rate, data):
    try:
        with wave.open(filepath, 'w') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(bit_depth // 8)
            wf.setframerate(sampling_rate)
            wf.writeframes(data)
    except Exception as e:
        logging.error(f"Error saving WAV file at {filepath}: {e}")
        print(f"Error saving WAV file at {filepath}: {e}")


def parse_datetime_from_filename(filename):
    patterns = [
        r'(\d{4}-\d{2}-\d{2})[^\d]*(\d{2})[:\-](\d{2})[:\-](\d{2})',    # Matches 'YYYY-MM-DD-HH:MM:SS' or 'YYYY-MM-DD-HH-MM-SS'
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            date_str = match.group(1)
            time_parts = match.groups()[1:]
            time_str = ':'.join(time_parts)
            try:
                return datetime.datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue
    raise ValueError(f"Could not parse datetime from filename: {filename}")


def record_audio_stream(rolling_window_length, rolling_window_step, bird_audio_folder, channels, bit_depth, sampling_rate):
    # Calculate buffer sizes
    total_buffer_size = int((rolling_window_length + rolling_window_step) * sampling_rate)
    window_size = int(rolling_window_length * sampling_rate)
    audio_buffer = np.zeros(total_buffer_size, dtype=np.float32)
    buffer_lock = threading.Lock()
    last_save_time = None
    first_audio_packet_time = None
    buffer_write_index = 0

    def audio_callback(indata, frames, time, status):
        nonlocal first_audio_packet_time, buffer_write_index
        if status:
            logging.warning(f'Audio callback status: {status}')
        try:
            if first_audio_packet_time is None:
                first_audio_packet_time = datetime.datetime.now()
            # Copy data directly into the audio buffer
            with buffer_lock:
                num_samples = len(indata[:, 0])
                end_index = buffer_write_index + num_samples
                if end_index <= total_buffer_size:
                    audio_buffer[buffer_write_index:end_index] = indata[:, 0]
                else:
                    # Handle wrap-around
                    first_part = total_buffer_size - buffer_write_index
                    audio_buffer[buffer_write_index:] = indata[:first_part, 0]
                    audio_buffer[:end_index - total_buffer_size] = indata[first_part:, 0]
                buffer_write_index = end_index % total_buffer_size
        except Exception as e:
            logging.error(f"Error in audio_callback: {e}")

    try:
        # Ensure the bird audio folder exists
        os.makedirs(bird_audio_folder, exist_ok=True)

        stream = sd.InputStream(
            channels=channels,
            samplerate=sampling_rate,
            callback=audio_callback,
            blocksize=int(sampling_rate * 0.1)  # Process in 100ms chunks
        )

        with stream:
            stream_start_time = datetime.datetime.now()
            logging.info(f"Started audio stream at {stream_start_time}")
            print(f"Started audio stream at: {stream_start_time}")

            while not stop_event.is_set():
                current_time = datetime.datetime.now()

                # Check if it's time to save a new recording
                if last_save_time is None or (current_time - last_save_time).total_seconds() >= rolling_window_step:
                    if first_audio_packet_time is None:
                        logging.warning("First audio packet time is not set. Skipping this iteration.")
                        continue

                    with buffer_lock:
                        # Calculate the indices for the window
                        end_index = buffer_write_index
                        start_index = (buffer_write_index - window_size) % total_buffer_size
                        if start_index < end_index:
                            frames = audio_buffer[start_index:end_index]
                        else:
                            frames = np.concatenate((audio_buffer[start_index:], audio_buffer[:end_index]))
                        frames = np.clip(frames * 32767, -32768, 32767).astype(np.int16)

                    window_end_time = current_time
                    window_start_time = window_end_time - datetime.timedelta(seconds=rolling_window_length)

                    start_time_str = window_start_time.strftime('%Y-%m-%d-%H:%M:%S')
                    end_time_str = window_end_time.strftime('%Y-%m-%d-%H:%M:%S')
                    wav_filename = f"streamed_start_{start_time_str}_end_{end_time_str}.wav"
                    wav_filepath = os.path.join(bird_audio_folder, wav_filename)

                    try:
                        executor.submit(save_wav_file, wav_filepath, channels, bit_depth, sampling_rate, frames.tobytes())
                        print(f"Saved new recording: {wav_filepath}")
                        logging.info(f"Saved new recording: {wav_filepath}")

                        # Update last save time
                        last_save_time = current_time

                    except Exception as e:
                        logging.error(f"Error saving WAV file: {e}")
                        print(f"Error saving WAV file: {e}")

                time.sleep(0.01)

    except Exception as e:
        logging.error(f"Error in record_audio_stream: {e}")
        print(f"Error in record_audio_stream: {e}")
    finally:
        logging.info("Stopping audio stream")
        print("Stopping audio stream")

        
def process_csv_files_in_order(processed_folder, wav_start_time, wav_end_time):
    csv_files = []
    for root, _, files in os.walk(processed_folder):
        for file in files:
            if file.endswith(".csv") and file not in processed_csv_files:
                csv_path = os.path.join(root, file)

                # Parse date and time from filename
                try:
                    parts = file.split('-')
                    if len(parts) < 5:
                        logging.error(f"Filename '{file}' does not have the expected format. Skipping.")
                        print(f"Filename '{file}' does not have the expected format. Skipping.")
                        continue

                    date_str = '-'.join(parts[:3])  # Gets 'YYYY-MM-DD'
                    time_str = parts[4].split('.')[0]  # Gets 'HH:MM:SS'
                    csv_datetime = datetime.datetime.strptime(f"{date_str}-{time_str}", '%Y-%m-%d-%H:%M:%S')

                    # Skip CSV files with a datetime before the script start time
                    if csv_datetime < script_start_time:
                        continue

                    # Skip CSV files that do not fall within the WAV file's start and end times
                    if not (wav_start_time <= csv_datetime <= wav_end_time):
                        continue
                except (ValueError, IndexError) as e:
                    logging.error(f"Error parsing filename '{file}': {e}")
                    print(f"Error parsing filename '{file}': {e}")
                    continue

                # Check if the CSV file is empty (excluding header)
                try:
                    with open(csv_path, 'r') as csv_file:
                        reader = csv.reader(csv_file)
                        rows = list(reader)
                        if len(rows) <= 1:  # Only header or completely empty
                            logging.info(f"CSV file '{csv_path}' is empty (excluding header). Skipping.")
                            print(f"CSV file '{csv_path}' is empty (excluding header). Skipping.")
                            processed_csv_files.add(file)  # Mark as processed even if empty
                            continue
                except Exception as e:
                    logging.error(f"Error reading CSV file '{file}': {e}")
                    print(f"Error reading CSV file '{file}': {e}")
                    continue

                csv_end_time = csv_datetime + datetime.timedelta(seconds=15)
                if wav_start_time <= csv_datetime <= wav_end_time and csv_end_time <= wav_end_time:
                    logging.info(f"Checking CSV file: {csv_path}")
                    print(f"Checking CSV file: {csv_path}")
                    logging.info(f"Adding CSV file '{csv_path}' for processing")
                    print(f"Adding CSV file '{csv_path}' for processing")
                    csv_files.append((csv_path, os.path.getmtime(csv_path)))

    # Sort files by their modification time (earliest first)
    csv_files.sort(key=lambda x: x[1])
    return [csv_path for csv_path, _ in csv_files]


def save_bird_sounds(processed_folder, bird_audio_folder, non_bird_audio_folder, channels, bit_depth, sampling_rate, bird_to_nonbird_list, rolling_window_length):
    global processed_csv_files, processed_bird_wav_files, non_bird_files_info
    if not os.path.exists(bird_audio_folder):
        os.makedirs(bird_audio_folder)
    if not os.path.exists(non_bird_audio_folder):
        os.makedirs(non_bird_audio_folder)

    while not stop_event.is_set():
        try:
            relevant_wav_files = [
                f for f in os.listdir(bird_audio_folder)
                if f.endswith(".wav") and f.startswith("streamed")
            ]
            relevant_wav_files.sort()

            if not relevant_wav_files:
                time.sleep(rolling_window_length)
                continue

            for wav_file_index, wav_file in enumerate(relevant_wav_files):
                if wav_file in processed_bird_wav_files:
                    continue

                # logging.info(f"Processing WAV file: {wav_file}")
                # print(f"Processing WAV file: {wav_file}")
                
                try:
                    start_time_str, end_time_str = wav_file.split('_')[2], wav_file.split('_')[4]
                    wav_start_time = datetime.datetime.strptime(start_time_str, '%Y-%m-%d-%H:%M:%S')
                    wav_end_time = datetime.datetime.strptime(end_time_str.split('.')[0], '%Y-%m-%d-%H:%M:%S')
                except ValueError as e:
                    logging.error(f"Error parsing WAV filename '{wav_file}': {e}\n{traceback.format_exc()}")
                    print(f"Error parsing WAV filename '{wav_file}': {e}")
                    continue

                # Get the list of corresponding CSV files for the current WAV file
                try:
                    csv_files = process_csv_files_in_order(processed_folder, wav_start_time, wav_end_time)
                except Exception as e:
                    logging.error(f"Error processing CSV files for WAV file '{wav_file}': {e}\n{traceback.format_exc()}")
                    print(f"Error processing CSV files for WAV file '{wav_file}': {e}")
                    continue

                # Load the WAV audio data
                try:
                    with wave.open(os.path.join(bird_audio_folder, wav_file), 'r') as wf:
                        audio_data = wf.readframes(wf.getnframes())
                        audio_data = np.frombuffer(audio_data, dtype=np.int16)
                except Exception as e:
                    logging.error(f"Error loading WAV file '{wav_file}': {e}\n{traceback.format_exc()}")
                    print(f"Error loading WAV file '{wav_file}': {e}")
                    continue

                for csv_path in csv_files:
                    file = os.path.basename(csv_path)
                    if file not in processed_csv_files:
                        try:
                            with open(csv_path, 'r') as csvfile:
                                reader = csv.DictReader(csvfile, delimiter=';')
                                rows = list(reader)
                                if not rows:
                                    processed_csv_files.add(file)
                                    continue

                                # Parse CSV start time correctly
                                parts = file.split('-')
                                date_str = '-'.join(parts[:3])  # Gets 'YYYY-MM-DD'
                                time_str = parts[4].split('.')[0]  # Gets 'HH:MM:SS'
                                csv_datetime = datetime.datetime.strptime(f"{date_str}-{time_str}", '%Y-%m-%d-%H:%M:%S')

                                # Check if the CSV timestamp is within the WAV file's start and end times
                                if not (wav_start_time <= csv_datetime <= wav_end_time):
                                    continue

                                # Calculate the sample range for the full 15-second bird audio
                                wav_offset = (csv_datetime - wav_start_time).total_seconds()
                                full_start_sample = int(wav_offset * sampling_rate)
                                full_end_sample = full_start_sample + int(15 * sampling_rate)

                                # Handle cases where 15-second audio spans multiple WAV files
                                bird_data = None
                                if full_end_sample <= len(audio_data):
                                    # Case 1: The entire 15 seconds is within the current WAV file
                                    bird_data = audio_data[full_start_sample:full_end_sample]
                                else:
                                    # Case 2: The 15-second window spans into the next WAV file
                                    remaining_samples = full_end_sample - len(audio_data)

                                    # Get data from current file
                                    bird_data = audio_data[full_start_sample:]

                                    # Load data from the next WAV file if it exists
                                    if wav_file_index + 1 < len(relevant_wav_files):
                                        next_wav_file = relevant_wav_files[wav_file_index + 1]
                                        with wave.open(os.path.join(bird_audio_folder, next_wav_file), 'r') as next_wf:
                                            next_audio_data = next_wf.readframes(next_wf.getnframes())
                                            next_audio_data = np.frombuffer(next_audio_data, dtype=np.int16)

                                        bird_data = np.concatenate((bird_data, next_audio_data[:remaining_samples]))

                                # Ensure bird data is exactly 15 seconds long
                                if len(bird_data) < 15 * sampling_rate:
                                    bird_data = np.pad(bird_data, (0, 15 * sampling_rate - len(bird_data)), 'constant')

                                # Save the full 15-second bird audio
                                full_save_path = os.path.join(bird_audio_folder, f"{csv_datetime.strftime('%Y-%m-%d-%H:%M:%S')}_bird_full.wav")
                                save_wav_file(full_save_path, channels, bit_depth, sampling_rate, bird_data.tobytes())
                                logging.info(f"Saved full bird sound: {full_save_path}")
                                print(f"Saved full bird sound: {full_save_path}")

                                for row in rows:
                                    try:
                                        start_s = float(row['Start (s)'])
                                        end_s = float(row['End (s)'])
                                        common_name = re.sub(r'[^a-zA-Z0-9]', '_', row['Common name'])

                                        # Calculate the start and end samples based on the difference between wav and csv timestamps
                                        start_sample = int((wav_offset + start_s) * sampling_rate)
                                        end_sample = int((wav_offset + end_s) * sampling_rate)

                                        bird_data = audio_data[start_sample:end_sample]

                                        # Save bird or non-bird sounds based on classification
                                        if np.max(np.abs(bird_data)) > 0:
                                            if any(non_bird.lower() in common_name.lower() for non_bird in bird_to_nonbird_list):
                                                save_path = os.path.join(non_bird_audio_folder, f"{csv_datetime.strftime('%Y-%m-%d-%H:%M:%S')}_{start_s}_{end_s}_nonbird_{common_name}.wav")
                                                save_wav_file(save_path, channels, bit_depth, sampling_rate, bird_data.tobytes())
                                                non_bird_files_info.append({'file_path': save_path, 'start_time': csv_datetime + datetime.timedelta(seconds=start_s), 'end_time': csv_datetime + datetime.timedelta(seconds=end_s)})
                                            else:
                                                save_path = os.path.join(bird_audio_folder, f"{csv_datetime.strftime('%Y-%m-%d-%H:%M:%S')}_{start_s}_{end_s}_bird_{common_name}.wav")
                                                save_wav_file(save_path, channels, bit_depth, sampling_rate, bird_data.tobytes())
                                    except Exception as e:
                                        logging.error(f"Error processing row '{row}' in CSV '{csv_path}': {e}\n{traceback.format_exc()}")
                                        print(f"Error processing row '{row}' in CSV '{csv_path}': {e}")

                        except Exception as e:
                            logging.error(f"Error reading CSV file '{csv_path}': {e}\n{traceback.format_exc()}")
                            print(f"Error reading CSV file '{csv_path}': {e}")
                            continue

                        shutil.copy(csv_path, bird_audio_folder)
                        processed_csv_files.add(file)

                processed_bird_wav_files.add(wav_file)
        except Exception as e:
            logging.error(f"Error in save_bird_sounds: {e}\n{traceback.format_exc()}")
            print(f"Error in save_bird_sounds: {e}")
        time.sleep(1)


import os
import time
import datetime
import numpy as np
import wave
import logging
import threading

def save_non_bird_sounds(bird_audio_folder, non_bird_audio_folder, non_bird_record_duration, non_bird_interval, non_bird_volume_threshold, rolling_window_length, bit_depth, sampling_rate):
    if not os.path.exists(non_bird_audio_folder):
        os.makedirs(non_bird_audio_folder)

    logging.info("Started save_non_bird_sounds thread")
    print("Started save_non_bird_sounds thread")

    last_saved_time = None

    while not stop_event.is_set():
        # Record the start time of processing
        processing_start_time = datetime.datetime.now()

        # Ensure a consistent saving interval
        if last_saved_time is not None:
            elapsed_time = (processing_start_time - last_saved_time).total_seconds()
            if elapsed_time < non_bird_interval:
                time_to_sleep = non_bird_interval - elapsed_time
                logging.info(f"Sleeping for {time_to_sleep} seconds to maintain interval.")
                time.sleep(time_to_sleep)
                processing_start_time = datetime.datetime.now()

        # Update the last saved time
        last_saved_time = processing_start_time

        # Determine the start and end times for the non-bird segment
        segment_end_time = processing_start_time
        segment_start_time = segment_end_time - datetime.timedelta(seconds=non_bird_record_duration)

        logging.info(f"Segment start time: {segment_start_time}, Segment end time: {segment_end_time}")
        print(f"Segment start time: {segment_start_time}, Segment end time: {segment_end_time}")

        # Ensure streamed audio files are saved before processing
        time.sleep(rolling_window_length)

        # Initialize an array for the non-bird data
        total_samples = int(non_bird_record_duration * sampling_rate)
        non_bird_data = np.zeros(total_samples, dtype=np.float32)  # Use float for accumulation, convert to int later

        # Find relevant WAV files
        wav_files = [f for f in os.listdir(bird_audio_folder) if f.endswith('.wav') and f.startswith("streamed")]
        wav_files.sort()
        relevant_files = []

        for wav_file in wav_files:
            try:
                start_time_str = wav_file.split('_')[2]
                end_time_str = wav_file.split('_')[4].split('.')[0]
                wav_start_time = datetime.datetime.strptime(start_time_str, '%Y-%m-%d-%H:%M:%S')
                wav_end_time = datetime.datetime.strptime(end_time_str, '%Y-%m-%d-%H:%M:%S')

                # Check if the WAV file overlaps with the non-bird segment time
                if wav_start_time < segment_end_time and wav_end_time > segment_start_time:
                    relevant_files.append((wav_file, wav_start_time, wav_end_time))
            except ValueError as e:
                logging.error(f"Error parsing WAV file '{wav_file}': {e}")
                continue

        logging.info(f"Found {len(relevant_files)} relevant files for non-bird segment.")
        print(f"Found {len(relevant_files)} relevant files for non-bird segment.")

        if not relevant_files:
            logging.info("No relevant WAV files found for the current non-bird segment.")
            print("No relevant WAV files found for the current non-bird segment.")
            continue

        for wav_file, wav_start_time, wav_end_time in relevant_files:
            wav_filepath = os.path.join(bird_audio_folder, wav_file)
            try:
                with wave.open(wav_filepath, 'r') as wf:
                    audio_data = wf.readframes(wf.getnframes())
                    audio_data = np.frombuffer(audio_data, dtype=np.int16)

                    # Calculate overlap between WAV file and segment
                    overlap_start_time = max(segment_start_time, wav_start_time)
                    overlap_end_time = min(segment_end_time, wav_end_time)

                    # Calculate sample indices in WAV file
                    wav_start_sample = int((overlap_start_time - wav_start_time).total_seconds() * sampling_rate)
                    wav_end_sample = int((overlap_end_time - wav_start_time).total_seconds() * sampling_rate)

                    # Calculate sample indices in non_bird_data
                    segment_start_sample = int((overlap_start_time - segment_start_time).total_seconds() * sampling_rate)
                    segment_end_sample = int((overlap_end_time - segment_start_time).total_seconds() * sampling_rate)

                    # Ensure indices are within array bounds
                    wav_start_sample = max(0, wav_start_sample)
                    wav_end_sample = min(len(audio_data), wav_end_sample)
                    segment_start_sample = max(0, segment_start_sample)
                    segment_end_sample = min(len(non_bird_data), segment_end_sample)

                    # Calculate the number of samples to copy
                    length = min(wav_end_sample - wav_start_sample, segment_end_sample - segment_start_sample)

                    if length > 0:
                        # Accumulate data from WAV file to non_bird_data to avoid overwrites
                        non_bird_data[segment_start_sample:segment_start_sample + length] = audio_data[wav_start_sample:wav_start_sample + length].astype(np.float32)
                    else:
                        logging.warning(f"No overlap in samples for file '{wav_file}'.")
                        print(f"No overlap in samples for file '{wav_file}'")

            except Exception as e:
                logging.error(f"Error processing non-bird sounds in file '{wav_file}': {e}")

        # Check if there is any non-zero audio data to save and if it exceeds the volume threshold
        if np.max(np.abs(non_bird_data)) == 0:
            logging.warning("No non-bird audio data extracted for saving.")
            print("No non-bird audio data extracted for saving.")
        elif np.max(np.abs(non_bird_data)) < non_bird_volume_threshold:
            logging.warning("Non-bird audio data below volume threshold, not saving.")
            print("Non-bird audio data below volume threshold, not saving.")
        else:
            # Convert accumulated float data to int16
            non_bird_data = np.clip(non_bird_data, -32768, 32767).astype(np.int16)

            # Save the non-bird audio
            non_bird_filename = f"{segment_start_time.strftime('%Y-%m-%d-%H:%M:%S')}_nonbird.wav"
            non_bird_filepath = os.path.join(non_bird_audio_folder, non_bird_filename)

            try:
                save_wav_file(non_bird_filepath, 1, bit_depth, sampling_rate, non_bird_data.tobytes())
                logging.info(f"Saved non-bird sound at {non_bird_filepath}")
                print(f"Saved non-bird sound at {non_bird_filepath}")
            except Exception as e:
                logging.error(f"Error saving non-bird sound at {non_bird_filepath}: {e}")
                print(f"Error saving non-bird sound at {non_bird_filepath}: {e}")


def remove_overlapping_non_bird_sounds(non_bird_audio_folder, bird_audio_folder):
    while not stop_event.is_set():
        try:
            non_bird_files = [f for f in os.listdir(non_bird_audio_folder) if f.endswith('.wav') and not f.startswith('streamed')]
            bird_files = [f for f in os.listdir(bird_audio_folder) if f.endswith('.wav') and '_full' in f]
            csv_files = [f for f in os.listdir(bird_audio_folder) if f.endswith('.csv')]

            # Parse CSV files to get the date and start time
            csv_timestamps = set()
            for csv_file in csv_files:
                try:
                    parts = csv_file.split('-')
                    date_str = '-'.join(parts[:3])
                    time_str = parts[4].split('.')[0]
                    csv_datetime = datetime.datetime.strptime(f"{date_str}-{time_str}", "%Y-%m-%d-%H:%M:%S")
                    csv_timestamps.add(csv_datetime)
                except (ValueError, IndexError):
                    logging.error(f"Error parsing CSV filename '{csv_file}'")

            # Remove overlapping non-bird sounds if they overlap with bird sounds
            for non_bird_file in non_bird_files:
                try:
                    non_bird_start_str, non_bird_end_str = non_bird_file.split('_')[0], non_bird_file.split('_')[1].split('.')[0]
                    non_bird_start_time = datetime.datetime.strptime(non_bird_start_str, '%Y-%m-%d-%H:%M:%S')
                    non_bird_end_time = datetime.datetime.strptime(non_bird_end_str, '%Y-%m-%d-%H:%M:%S')
                except (ValueError, IndexError):
                    continue

                for bird_file in bird_files:
                    try:
                        bird_start_str = bird_file.split('_')[0]
                        bird_end_str = bird_file.split('_')[2].split('.')[0]
                        bird_start_time = datetime.datetime.strptime(bird_start_str, '%Y-%m-%d-%H:%M:%S')
                        bird_end_time = datetime.datetime.strptime(bird_end_str, '%Y-%m-%d-%H:%M:%S')
                    except (ValueError, IndexError):
                        continue

                    # Check for overlap
                    if bird_start_time < non_bird_end_time and bird_end_time > non_bird_start_time:
                        non_bird_filepath = os.path.join(non_bird_audio_folder, non_bird_file)
                        try:
                            os.remove(non_bird_filepath)
                            logging.info(f"Removed overlapping non-bird sound: {non_bird_filepath}")
                            print(f"Removed overlapping non-bird sound: {non_bird_filepath}")
                            break
                        except Exception as e:
                            logging.error(f"Error removing non-bird file '{non_bird_filepath}': {e}")

            # Remove bird files that do not have accompanying CSV files
            for bird_file in bird_files:
                try:
                    bird_start_str = bird_file.split('_')[0]
                    bird_datetime = datetime.datetime.strptime(bird_start_str, "%Y-%m-%d-%H:%M:%S")

                    if bird_datetime not in csv_timestamps:
                        bird_filepath = os.path.join(bird_audio_folder, bird_file)
                        os.remove(bird_filepath)
                        logging.info(f"Removed bird sound without CSV: {bird_filepath}")
                        print(f"Removed bird sound without CSV: {bird_filepath}")

                        # Remove other bird files with the same date and start_str
                        for other_bird_file in os.listdir(bird_audio_folder):
                            if other_bird_file.startswith(bird_start_str) and other_bird_file != bird_file:
                                other_bird_filepath = os.path.join(bird_audio_folder, other_bird_file)
                                os.remove(other_bird_filepath)
                                logging.info(f"Removed related bird sound: {other_bird_filepath}")
                                print(f"Removed related bird sound: {other_bird_filepath}")
                except (ValueError, IndexError, FileNotFoundError) as e:
                    logging.error(f"Error processing bird file '{bird_file}': {e}")
        except Exception as e:
            logging.error(f"Error in remove_overlapping_non_bird_sounds: {e}")

        time.sleep(600)



def delete_old_processed_files(delete_interval, bird_audio_folder):
    while not stop_event.is_set():
        current_time = datetime.datetime.now()

        with file_lock:
            # Delete streamed files (existing behavior)
            for wav_file in os.listdir(bird_audio_folder):
                file_path = os.path.join(bird_audio_folder, wav_file)
                # Handle streamed files
                if wav_file.endswith('.wav') and wav_file.startswith('streamed_'):
                    try:
                        end_time_str = wav_file.split('_')[4].split('.')[0]
                        wav_end_time = datetime.datetime.strptime(end_time_str, '%Y-%m-%d-%H:%M:%S')
                        if (current_time - wav_end_time).total_seconds() >= delete_interval:
                            os.remove(file_path)
                            logging.info(f"Deleted old bird file: {wav_file}")
                            print(f"Deleted old bird file: {wav_file}")
                    except Exception as e:
                        logging.error(f"Error deleting file '{wav_file}': {e}")
                # Handle full bird files
                elif wav_file.endswith('_bird_full.wav'):
                    try:
                        start_time_str = wav_file.split('_')[0]
                        wav_start_time = datetime.datetime.strptime(start_time_str, '%Y-%m-%d-%H:%M:%S')
                        if (current_time - wav_start_time).total_seconds() >= delete_interval:
                            os.remove(file_path)
                            logging.info(f"Deleted old full bird file: {wav_file}")
                            print(f"Deleted old full bird file: {wav_file}")
                    except Exception as e:
                        logging.error(f"Error deleting full bird file '{wav_file}': {e}")

            # NEW: Delete old CSV files
            for csv_file in os.listdir(bird_audio_folder):
                if csv_file.endswith('.csv'):
                    csv_path = os.path.join(bird_audio_folder, csv_file)
                    try:
                        # Assumes CSV name: 2025-06-28-birdnet-08:31:30.wav.csv
                        parts = csv_file.split('-')
                        if len(parts) >= 5:
                            date_str = '-'.join(parts[:3])  # '2025-06-28'
                            time_part = parts[4].split('.')[0]  # '08:31:30'
                            csv_datetime = datetime.datetime.strptime(f"{date_str}-{time_part}", "%Y-%m-%d-%H:%M:%S")
                            file_age = (current_time - csv_datetime).total_seconds()
                            if file_age >= 1800:  # 30 minutes
                                os.remove(csv_path)
                                logging.info(f"Deleted old CSV file: {csv_file}")
                                print(f"Deleted old CSV file: {csv_file}")
                    except Exception as e:
                        logging.error(f"Error deleting CSV file '{csv_file}': {e}")
                        print(f"Error deleting CSV file '{csv_file}': {e}")

        time.sleep(delete_interval)




def start_threads(args, processed_folder, bird_audio_folder, non_bird_audio_folder):
    threads = []
    threads.append(threading.Thread(target=record_audio_stream, args=(
        args.rolling_window_length, args.rolling_window_step,
        bird_audio_folder, args.channels, args.bit_depth, args.sampling_rate), daemon=True))
    threads.append(threading.Thread(target=save_non_bird_sounds, args=(
        bird_audio_folder, non_bird_audio_folder,
        args.non_bird_record_duration, args.non_bird_interval, args.non_bird_volume_threshold, args.rolling_window_length,
        args.bit_depth, args.sampling_rate), daemon=True))
    threads.append(threading.Thread(target=save_bird_sounds, args=(
        processed_folder, bird_audio_folder, non_bird_audio_folder,
        args.channels, args.bit_depth, args.sampling_rate, args.bird_to_nonbird_list,
        args.rolling_window_length), daemon=True))
    threads.append(threading.Thread(target=log_memory_usage, args=(args.memory_log_interval,), daemon=True))
    threads.append(threading.Thread(target=remove_overlapping_non_bird_sounds, args=(
        non_bird_audio_folder, bird_audio_folder), daemon=True))
    threads.append(threading.Thread(target=delete_old_processed_files, args=(
        args.delete_interval, bird_audio_folder), daemon=True))

    for t in threads:
        t.start()

    return threads



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bird and non-bird audio recording script.')
    parser.add_argument('--processed_folder', type=str, default='/home/matteo/BirdSongs/Extracted/Processed/', help='Path to the processed folder')
    parser.add_argument('--streamed_audio_folder', type=str, default='/home/matteo/BirdSongs/StreamedAudio/', help='Path to the streamed audio folder')
    parser.add_argument('--sampling_rate', type=int, default=48000, help='Sampling rate in Hz')
    parser.add_argument('--bit_depth', type=int, default=16, help='Bit depth (number of bits per sample)')
    parser.add_argument('--channels', type=int, default=1, help='Number of audio channels (1 for mono)')
    parser.add_argument('--rolling_window_length', type=int, default=120, help='Rolling window length in seconds')
    parser.add_argument('--rolling_window_step', type=int, default=30, help='Rolling window step in seconds')
    parser.add_argument('--non_bird_interval', type=int, default=120, help='Interval in seconds to record non-bird sounds')
    parser.add_argument('--non_bird_record_duration', type=int, default=15, help='Duration in seconds for non-bird recordings')
    parser.add_argument('--non_bird_volume_threshold', type=int, default=1000, help='Minimum volume threshold for non bird sounds')
    parser.add_argument('--memory_log_interval', type=int, default=300, help='Interval in seconds to log memory and CPU usage')
    parser.add_argument('--delete_interval', type=int, default=900, help='Interval in seconds to check when to delete files')
    parser.add_argument('--bird_to_nonbird_list', type=str, nargs='+', default=['Human', 'Engine', 'Environmental'], help='List of keywords to classify birds as non-bird sounds')

    args = parser.parse_args()

    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    # Paths
    processed_folder = args.processed_folder
    streamed_audio_folder = args.streamed_audio_folder
    bird_audio_folder = os.path.join(streamed_audio_folder, 'Birds')
    non_bird_audio_folder = os.path.join(streamed_audio_folder, 'NoBirds')

    # Start Threads
    threads = start_threads(args, processed_folder, bird_audio_folder, non_bird_audio_folder)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        for thread in threads:
            thread.join()
        print("Stopping all threads gracefully.")
        logging.info("Stopping all threads gracefully.")
