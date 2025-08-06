import cv2
import numpy as np
from rich.console import Console
from rich import print as rprint
import time
import sys


def array_to_unicode(frame_array, dark_mode=True):
    
    frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2GRAY)
    if dark_mode:
        unicode_chars = [" ", "░", "▒", "▓", "█"] 
    else:
        
        unicode_chars = ["█", "▓", "▒", "░", " "]  
    
    uni_array = np.full(frame_array.shape, unicode_chars[0], dtype="<U1")

    colour_bands = np.array([51, 102, 153, 204])

    
    for min_val, max_val, char in zip(colour_bands[:-1], colour_bands[1:], unicode_chars[1:]):
        uni_array[(frame_array > min_val) & (frame_array <= max_val)] = char

    uni_array[frame_array > colour_bands[-1]] = unicode_chars[-1]

    frame_string = '\n'.join([''.join(row) for row in uni_array])
    return frame_string
def rich_array_to_unicode(video_array, dark_mode=True):
    greyscale_array = cv2.cvtColor(video_array, cv2.COLOR_BGR2GRAY)
    if dark_mode:
        # unicode_chars = [" ", "░", "▒", "▓", "█"] 
        unicode_chars = [" ","ඞ", "ඞ", "ඞ", "ඞ"]
    else:
        unicode_chars = ["█", "▓", "▒", "░", " "]
    
    
    r = video_array[:, :, 2]  
    g = video_array[:, :, 1]  
    b = video_array[:, :, 0]  
    
    
    uni_array = np.full(greyscale_array.shape, unicode_chars[0], dtype=object)
    colour_bands = np.array([51, 102, 153, 204])
    
    for min_val, max_val, char in zip(colour_bands[:-1], colour_bands[1:], unicode_chars[1:]):
        mask = (greyscale_array > min_val) & (greyscale_array <= max_val)
        
        uni_array[mask] = np.array([f"[rgb({r[i,j]},{g[i,j]},{b[i,j]})]{char}[/rgb({r[i,j]},{g[i,j]},{b[i,j]})]" 
                                   for i, j in zip(*np.where(mask))], dtype=object)
    
    
    mask = greyscale_array > colour_bands[-1]
    uni_array[mask] = np.array([f"[rgb({r[i,j]},{g[i,j]},{b[i,j]})]{unicode_chars[-1]}[/rgb({r[i,j]},{g[i,j]},{b[i,j]})]"
                                 for i, j in zip(*np.where(mask))], dtype=object)

    frame_string = '\n'.join([''.join(row) for row in uni_array])
    return frame_string
def display_video(cap, fps=30, dark_mode=True, use_rich_colors=False):
    console = Console()
    
    
    
    sys.stdout.write('\033[?25l\033[2J\033[H')
    sys.stdout.flush()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (80, 40)) 
            frame = cv2.flip(frame, 1) 
            
            
            sys.stdout.write('\033[H')
            
            if use_rich_colors:
                unicode_frame = rich_array_to_unicode(frame, dark_mode)
                console.print(unicode_frame, markup=True, end='')
            else:
                unicode_frame = array_to_unicode(frame, dark_mode)
                console.print(unicode_frame, end='')
            
            sys.stdout.flush()
            time.sleep(1 / fps)
            
    except KeyboardInterrupt:
        pass
    finally:
        
        sys.stdout.write('\033[?25h')
        sys.stdout.flush()
def main():

    
    test_image = cv2.imread("src/test.jpg")
    if test_image is None:
        print("Error: Could not load test.jpg. Please check if the file exists and is a valid image.")
        return
    test_image = cv2.resize(test_image, (220, 403))
    unicode_image = rich_array_to_unicode(test_image, dark_mode=True)
    rprint(unicode_image)

    input("Press Enter to continue...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video file. Please check if the file exists and is accessible.")
        return

    

    
    
    
    display_video(cap, fps=30, dark_mode=True, use_rich_colors=True)
    cap.release()
if __name__ == "__main__":
    main()