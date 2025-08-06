import cv2
import numpy as np
import ncurses


def array_to_unicode(array, dark_mode=True):
    if dark_mode:
        unicode_chars = [" ", "░", "▒", "▓", "█"] 
    else:
        
        unicode_chars = ["█", "▓", "▒", "░", " "]  
    
    uni_array = np.full(array.shape, unicode_chars[0], dtype="<U1")

    colour_bands = np.array([51, 102, 153, 204])

    
    for min_val, max_val, char in zip(colour_bands[:-1], colour_bands[1:], unicode_chars[1:]):
        uni_array[(array > min_val) & (array <= max_val)] = char
    
    uni_array[array > colour_bands[-1]] = unicode_chars[-1]
    
    frame_string = '\n'.join([''.join(row) for row in uni_array])
    return frame_string
