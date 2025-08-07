import cv2
import numpy as np
from rich.console import Console
from rich import print as rprint
import time
import sys
import multiprocessing as mp
from multiprocessing import Queue as MPQueue, Process, pool

import threading
from queue import Queue
from line_profiler import profile


# def profile_function(func):
#     def wrapper(*args, **kwargs):
#         profiler = LineProfiler()
#         profiler.add_function(func)
#         profiler.enable()
#         result = func(*args, **kwargs)
#         profiler.disable()
#         profiler.print_stats()
#         return result

#     return wrapper


def array_to_unicode(frame_array, dark_mode=True):

    frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2GRAY)
    if dark_mode:
        unicode_chars = [" ", "░", "▒", "▓", "█"]
    else:

        unicode_chars = ["█", "▓", "▒", "░", " "]

    uni_array = np.full(frame_array.shape, unicode_chars[0], dtype="<U1")

    colour_bands = np.array([51, 102, 153, 204])

    for min_val, max_val, char in zip(
        colour_bands[:-1], colour_bands[1:], unicode_chars[1:]
    ):
        uni_array[(frame_array > min_val) & (frame_array <= max_val)] = char

    uni_array[frame_array > colour_bands[-1]] = unicode_chars[-1]

    frame_string = "\n".join(["".join(row) for row in uni_array])
    return frame_string


def rich_array_to_unicode(video_array, dark_mode=True):
    greyscale_array = cv2.cvtColor(video_array, cv2.COLOR_BGR2GRAY)
    if dark_mode:
        # unicode_chars = [" ", "░", "▒", "▓", "█"]
        unicode_chars = [" ", "ඞ", "ඞ", "ඞ", "ඞ"]
    else:
        unicode_chars = ["█", "▓", "▒", "░", " "]

    r = video_array[:, :, 2]
    g = video_array[:, :, 1]
    b = video_array[:, :, 0]

    uni_array = np.full(greyscale_array.shape, unicode_chars[0], dtype=object)
    colour_bands = np.array([51, 102, 153, 204])

    for min_val, max_val, char in zip(
        colour_bands[:-1], colour_bands[1:], unicode_chars[1:]
    ):
        mask = (greyscale_array > min_val) & (greyscale_array <= max_val)

        uni_array[mask] = np.array(
            [
                f"[rgb({r[i,j]},{g[i,j]},{b[i,j]})]{char}[/rgb({r[i,j]},{g[i,j]},{b[i,j]})]"
                for i, j in zip(*np.where(mask))
            ],
            dtype=object,
        )

    mask = greyscale_array > colour_bands[-1]
    uni_array[mask] = np.array(
        [
            f"[rgb({r[i,j]},{g[i,j]},{b[i,j]})]{unicode_chars[-1]}[/rgb({r[i,j]},{g[i,j]},{b[i,j]})]"
            for i, j in zip(*np.where(mask))
        ],
        dtype=object,
    )

    frame_string = "\n".join(["".join(row) for row in uni_array])
    return frame_string


def faster_rich_array_to_unicode(video_array, dark_mode=True):
    greyscale_array = cv2.cvtColor(video_array, cv2.COLOR_BGR2GRAY)
    
    if dark_mode:
        unicode_chars = [" ", "░", "▒", "▓", "█"]
    else:
        unicode_chars = ["█", "▓", "▒", "░", " "]

    r = video_array[:, :, 2]
    g = video_array[:, :, 1] 
    b = video_array[:, :, 0]

    uni_array = np.full(greyscale_array.shape, unicode_chars[0], dtype=object)
    colour_bands = np.array([51, 102, 153, 204])

    for min_val, max_val, char in zip(
        colour_bands[:-1], colour_bands[1:], unicode_chars[1:]
    ):
        mask = (greyscale_array > min_val) & (greyscale_array <= max_val)
        
        # Fixed: Create proper RGB markup strings
        uni_array[mask] = np.array([
            f"[rgb({r[i,j]},{g[i,j]},{b[i,j]})]{char}[/rgb({r[i,j]},{g[i,j]},{b[i,j]})]"
            for i, j in zip(*np.where(mask))
        ], dtype=object)

    # Handle brightest pixels
    mask = greyscale_array > colour_bands[-1]
    uni_array[mask] = np.array([
        f"[rgb({r[i,j]},{g[i,j]},{b[i,j]})]{unicode_chars[-1]}[/rgb({r[i,j]},{g[i,j]},{b[i,j]})]"
        for i, j in zip(*np.where(mask))
    ], dtype=object)

    frame_string = "\n".join(["".join(row) for row in uni_array])
    return frame_string



def mp_frame_reader(video_path, raw_frame_queue, stop_event):
    """Frame reader for multiprocessing - creates its own VideoCapture"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        stop_event.set()
        return
        
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break
        frame = cv2.resize(frame, (80, 40))
        try:
            raw_frame_queue.put(frame, timeout=0.1)
        except:
            pass
    
    cap.release()


def mp_frame_reader_batch(video_path, raw_frame_queue, stop_event, num_workers):
    """Frame reader that sends frames with IDs for batch processing"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        stop_event.set()
        return
        
    frame_id = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break
        frame = cv2.resize(frame, (200, 100))
        
        # Send the SAME frame to ALL workers
        frame_data = (frame, frame_id)
        for _ in range(num_workers):  # Send to each worker
            try:
                raw_frame_queue.put(frame_data, timeout=0.1)
            except:
                pass
        
        frame_id += 1
    
    cap.release()


def mp_frame_processor(raw_frame_queue, processed_frame_queue, stop_event, dark_mode, use_rich_colors):
    """Frame processor for multiprocessing"""
    while not stop_event.is_set():
        try:
            frame = raw_frame_queue.get(timeout=0.1)
            frame = cv2.flip(frame, 1)

            if use_rich_colors:
                unicode_frame = faster_rich_array_to_unicode(frame, dark_mode)
            else:
                unicode_frame = array_to_unicode(frame, dark_mode)
                
            try:
                processed_frame_queue.put(unicode_frame, timeout=0.1)
            except:
                pass
        except:
            continue
def mp_frame_processor_batch(raw_frame_queue, processed_frame_queue, split_shape, index, stop_event, dark_mode, use_rich_colors):
    """Process a specific section of each frame - for parallel frame processing"""
    rows, cols = split_shape
    
    while not stop_event.is_set():
        try:
            frame_data = raw_frame_queue.get(timeout=0.1)
            if frame_data is None:
                break
                
            frame, frame_id = frame_data
            frame = cv2.flip(frame, 1)
            
            # Calculate this worker's section of the frame
            height, width = frame.shape[:2]
            section_height = height // rows
            section_width = width // cols
            
            # Calculate row and column from index
            row = index // cols
            col = index % cols
            
            # Extract this worker's section
            start_y = row * section_height
            end_y = (row + 1) * section_height if row < rows - 1 else height
            start_x = col * section_width
            end_x = (col + 1) * section_width if col < cols - 1 else width
            
            frame_section = frame[start_y:end_y, start_x:end_x]
            
            # Process the section
            if use_rich_colors:
                unicode_section = faster_rich_array_to_unicode(frame_section, dark_mode)
            else:
                unicode_section = array_to_unicode(frame_section, dark_mode)
            
            # Send back: frame_id, section_index, processed_section, position_info
            result = {
                'frame_id': frame_id,
                'section_index': index,
                'content': unicode_section,
                'position': (start_y, end_y, start_x, end_x),
                'section_shape': (row, col)
            }
            
            try:
                processed_frame_queue.put(result, timeout=0.1)
            except:
                pass
                
        except Exception as e:
            continue

def display_video(cap, fps=30, dark_mode=True, use_rich_colors=False):
    console = Console()

    sys.stdout.write("\033[?25l\033[2J\033[H")
    sys.stdout.flush()

    try:
        while True:
            ret, frame = cap.read() 
            if not ret:
                break
            frame = cv2.resize(frame, (80, 40))
            frame = cv2.flip(frame, 1)

            sys.stdout.write("\033[H")

            if use_rich_colors:
                unicode_frame = faster_rich_array_to_unicode(frame, dark_mode)
                console.print(unicode_frame, markup=True, end="")
            else:
                unicode_frame = array_to_unicode(frame, dark_mode)
                console.print(unicode_frame, end="")

            sys.stdout.flush()

            time.sleep(1 / fps)

    except KeyboardInterrupt:
        pass
    finally:

        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


def display_video_thread(cap, fps=30, dark_mode=True, use_rich_colors=False):
    raw_frame_queue = Queue(maxsize=5)
    proccessed_frame_queue = Queue(maxsize=5)
    console = Console()
    stop_event = threading.Event()

    def frame_reader():
        while not stop_event.is_set():
            ret     , frame = cap.read()
            if not ret:
                stop_event.set()  # Signal that video is done
                break
            frame = cv2.resize(frame, (80, 40))
            try:
                raw_frame_queue.put(frame, timeout=0.1)
            except:
                pass

    def frame_processor():
        while not stop_event.is_set():
            try:
                frame = raw_frame_queue.get(
                    timeout=0.1
                )  # Use timeout instead of checking empty
                frame = cv2.flip(frame, 1)

                if use_rich_colors:
                    unicode_frame = faster_rich_array_to_unicode(frame, dark_mode)
                else:
                    unicode_frame = array_to_unicode(frame, dark_mode)
                try:
                    proccessed_frame_queue.put(unicode_frame, timeout=0.1)
                except:
                    pass
                raw_frame_queue.task_done()
            except:
                continue  # Timeout occurred, check stop_event again

    reader_thread = threading.Thread(target=frame_reader)
    processor_thread = threading.Thread(target=frame_processor)

    reader_thread.start()
    processor_thread.start()

    sys.stdout.write("\033[?25l\033[2J\033[H")
    sys.stdout.flush()
    try:
        while not stop_event.is_set():
            try:
                unicode_frame = proccessed_frame_queue.get(timeout=0.1)  # Use timeout
                sys.stdout.write("\033[H")
                console.print(unicode_frame, markup=True, end="")
                sys.stdout.flush()
                proccessed_frame_queue.task_done()
                time.sleep(1 / fps)
            except:
                continue

    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        reader_thread.join(timeout=1)
        processor_thread.join(timeout=1)
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


def display_video_mp(video_path, fps=30, dark_mode=True, use_rich_colors=False):
    """Multiprocessing version - pass video path instead of VideoCapture object"""
    raw_frame_queue = MPQueue(maxsize=5)
    processed_frame_queue = MPQueue(maxsize=5)
    console = Console()
    stop_event = mp.Event()

    # Use module-level functions for multiprocessing
    reader_process = Process(target=mp_frame_reader, args=(video_path, raw_frame_queue, stop_event))
    processor_process = Process(target=mp_frame_processor, 
                               args=(raw_frame_queue, processed_frame_queue, stop_event, dark_mode, use_rich_colors))
    
    reader_process.start()
    processor_process.start()
    
    sys.stdout.write("\033[?25l\033[2J\033[H")
    sys.stdout.flush()
    
    try:
        while not stop_event.is_set():
            try:
                unicode_frame = processed_frame_queue.get(timeout=0.1)
                sys.stdout.write("\033[H")
                console.print(unicode_frame, markup=use_rich_colors, end="")
                sys.stdout.flush()
                time.sleep(1 / fps)
            except:
                continue
                
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        reader_process.terminate()  # Use terminate for processes
        processor_process.terminate()
        reader_process.join(timeout=1)
        processor_process.join(timeout=1)
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
def display_video_mp_batch(video_path, fps=30, dark_mode=True, use_rich_colors=False, split_shape=(2, 3)):
    """Multiprocessing version with frame splitting - each frame processed by multiple workers"""
    rows, cols = split_shape
    num_workers = rows * cols
    
    raw_frame_queue = MPQueue(maxsize=10)
    processed_frame_queue = MPQueue(maxsize=20)  # Larger queue for multiple sections
    console = Console()
    stop_event = mp.Event()
    
    # Start frame reader
    reader_process = Process(target=mp_frame_reader_batch, args=(video_path, raw_frame_queue, stop_event, num_workers))
    reader_process.start()
    
    # Start multiple processor workers - one for each frame section
    processor_processes = []
    for i in range(num_workers):
        worker = Process(
            target=mp_frame_processor_batch,
            args=(raw_frame_queue, processed_frame_queue, split_shape, i, stop_event, dark_mode, use_rich_colors)
        )
        worker.start()
        processor_processes.append(worker)
    
    sys.stdout.write("\033[?25l\033[2J\033[H")
    sys.stdout.flush()
    
    # Frame assembly logic
    current_frame_id = 0
    frame_sections = {}  # Store sections until complete frame is ready
    
    try:
        while not stop_event.is_set():
            try:
                section_result = processed_frame_queue.get(timeout=0.1)
                frame_id = section_result['frame_id']
                section_index = section_result['section_index']
                content = section_result['content']
                position = section_result['position']
                section_shape = section_result['section_shape']
                
                # Store section
                if frame_id not in frame_sections:
                    frame_sections[frame_id] = {}
                
                frame_sections[frame_id][section_index] = {
                    'content': content,
                    'position': position,
                    'section_shape': section_shape
                }
                
                # Check if we have all sections for ANY complete frame (not just current_frame_id)
                for check_frame_id in sorted(frame_sections.keys()):
                    if len(frame_sections[check_frame_id]) == num_workers:
                        # Assemble the complete frame
                        assembled_frame = assemble_frame_sections(frame_sections[check_frame_id], split_shape)
                        
                        if assembled_frame:  # Check if assembly worked
                            # Display assembled frame
                            sys.stdout.write("\033[H")
                            console.print(assembled_frame, markup=use_rich_colors, end="")
                            sys.stdout.flush()
                            time.sleep(1 / fps)
                        
                        # Clean up old frame data
                        del frame_sections[check_frame_id]
                        current_frame_id = max(current_frame_id, check_frame_id + 1)
                        break  # Only process one frame at a time
                    
            except Exception as e:
                continue
                
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        reader_process.terminate()
        for worker in processor_processes:
            worker.terminate()
        reader_process.join(timeout=1)
        for worker in processor_processes:
            worker.join(timeout=1)
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


def assemble_frame_sections(sections, split_shape):
    """Assemble processed frame sections back into a complete frame"""
    rows, cols = split_shape
    
    # Create 2D grid to hold sections
    section_grid = [[None for _ in range(cols)] for _ in range(rows)]
    
    # Place each section in the correct position
    for section_index, section_data in sections.items():
        row = section_index // cols
        col = section_index % cols
        section_grid[row][col] = section_data['content']
    
    # Check if all sections are present
    for r in range(rows):
        for c in range(cols):
            if section_grid[r][c] is None:
                return None
    
    # Assemble sections row by row
    assembled_lines = []
    for row_idx, row in enumerate(section_grid):
        # Split each section into lines
        section_lines = [section.split('\n') if section else [] for section in row]
        
        # Find max number of lines in this row
        max_lines = max(len(lines) for lines in section_lines) if section_lines else 0
        
        # Combine lines horizontally
        for line_idx in range(max_lines):
            combined_line = ""
            for section_lines_list in section_lines:
                if line_idx < len(section_lines_list):
                    combined_line += section_lines_list[line_idx]
                else:
                    # Calculate section width for proper padding
                    section_width = 80 // cols  # Assuming 80 char width
                    combined_line += " " * section_width
            assembled_lines.append(combined_line)
    
    return '\n'.join(assembled_lines)

def main():
    # Show test image first
    test_image = cv2.imread("test.jpg")
    if test_image is None:
        print(
            "Error: Could not load test.jpg. Please check if the file exists and is a valid image."
        )
        return
    test_image = cv2.resize(test_image, (220, 403))
    unicode_image = rich_array_to_unicode(test_image, dark_mode=True)
    rprint(unicode_image)

    input("Press Enter to start performance comparison...")

    video_path = "test.mp4"

    
    cap_test = cv2.VideoCapture(video_path)
    if not cap_test.isOpened():
        print(
            "Error: Could not open video file. Please check if the file exists and is accessible."
        )
        return
    cap_test.release()

    cap = cv2.VideoCapture(video_path)

    threaded_time = None
    singletime = None
    color = True
    stime = time.time()
    print("Running threaded version...")
    display_video_thread(cap, fps=30, dark_mode=True, use_rich_colors=color)
    threaded_time = time.time() - stime


    
    stime = time.time()
    print("Running multi-threaded version...")
    display_video_mp(video_path, fps=30, dark_mode=True, use_rich_colors=color)  # Pass video path
    multitime = time.time() - stime

    # NEW: Test batch processing with frame splitting
    input()
    stime = time.time()
    print("Running batch multi-processing version (2x3 split)...")
    
    display_video_mp_batch(video_path, fps=30, dark_mode=True, use_rich_colors=color, split_shape=(2, 3))
   
    batchtime = time.time() - stime
    input()


    cap = cv2.VideoCapture(video_path)
    stime = time.time()
    print("Running single-threaded version...")
   
    display_video(cap, fps=30, dark_mode=True, use_rich_colors=color)
    singletime = time.time() - stime
    print(f"\nSingle Time taken: {singletime:.2f} seconds")
    print(f"Thread Time taken: {threaded_time:.2f} seconds")
    print(f"Multi Time taken: {multitime:.2f} seconds")
    print(f"Batch Time taken: {batchtime:.2f} seconds")
    print(f"single to threadSpeedup: {singletime / threaded_time:.2f}x")
    print(f"single to multiSpeedup: {singletime / multitime:.2f}x")
    print(f"single to batchSpeedup: {singletime / batchtime:.2f}x")
    print(f"thread to multiSpeedup: {threaded_time / multitime:.2f}x")
    print(f"multi to batchSpeedup: {multitime / batchtime:.2f}x")

    cap_fps = cv2.VideoCapture(video_path)
    frame_count = int(cap_fps.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_actual = cap_fps.get(cv2.CAP_PROP_FPS)
    cap_fps.release()

    if fps_actual > 0:
        expected_time = frame_count / fps_actual
        print(f"Video has {frame_count} frames at {fps_actual:.2f} fps.")
        print(f"Expected time taken {expected_time:.2f} seconds")
    else:
        print("Could not determine video FPS.")

    cap.release()


if __name__ == "__main__":
    main()
