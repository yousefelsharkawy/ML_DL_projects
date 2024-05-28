import string
import pytesseract

# character-number mapping that OCR can confuse
# each key is a character that can be confused with a single digit
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5',
                    'B': '8'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '8': 'B'}




def license_complies_format(license_plate):
    """
    Check if the license plate complies with the format

    Args:
        license_plate (str): The license plate text

    Returns:
        bool: True if the license plate complies with the format, False otherwise
    """
    if len(license_plate) != 7:
        return False
    
    # check the first 2 characters are letters, the next 2 are digits and the last 3 are letters
    if not license_plate[:2].isalpha() or not license_plate[2:4].isdigit() or not license_plate[4:].isalpha():
        return False
    else:
        return True


def format_license(text):
    """
    Format the license plate text, and fix any common OCR errors

    Args:
        license_plate (str): The license plate text

    Returns:
        tuple: Tuple containing the formatted license plate text and the confidence score
    """
    # if length is not 7, return the text as it is (TODO:we can do better cleaning and regex handling)
    if len(text) != 7:
        return text
    else:
        if not text[:2].isalpha():
            if text[0] in dict_int_to_char:
                text = dict_int_to_char[text[0]] + text[1:]
            if text[1] in dict_int_to_char:
                text = text[0] + dict_int_to_char[text[1]] + text[2:]
        if not text[2:4].isdigit():
            if text[2] in dict_char_to_int:
                text = text[:2] + dict_char_to_int[text[2]] + text[3:]
            if text[3] in dict_char_to_int:
                text = text[:3] + dict_char_to_int[text[3]] + text[4:]
        if not text[4:].isalpha():
            if text[4] in dict_int_to_char:
                text = text[:4] + dict_int_to_char[text[4]] + text[5:]
            if text[5] in dict_int_to_char:
                text = text[:5] + dict_int_to_char[text[5]] + text[6:]
            if text[6] in dict_int_to_char:
                text = text[:6] + dict_int_to_char[text[6]]
        return text
    
    

def get_car(license_plate, Vehicle_track_ids):
    """
    Get the vehicle coordinates and id based on the license plate coordinates

    Args:
        license_plate(Tuple): The license plate coordinates (x1, y1, x2, y2, conf)
        Vehicle_track_ids(List): List of vehicle coordinates and ids (x1, y1, x2, y2, track_id)

    Returns:
        Tuple: The vehicle coordinates and id corresponding to the license plate
    """
    x1, y1, x2, y2 = license_plate.xyxy.flatten().tolist()

    for vehicle in Vehicle_track_ids:
        x1_v, y1_v, x2_v, y2_v, track_id = vehicle
        if x1 >= x1_v and y1 >= y1_v and x2 <= x2_v and y2 <= y2_v:
            return x1_v, y1_v, x2_v, y2_v, track_id
    
    return -1, -1, -1, -1, -1



def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the license plate image

    Args:
        license_plate_crop(PIL.Image.Image): The cropped image of the license plate

    Returns:
        tuple: Tuple containing the formatted license plate text and the confidence score

    """
    text = pytesseract.image_to_string(license_plate_crop, lang='eng')
    # remove any non-alphanumeric characters
    text = ''.join(e for e in text if e.isalnum()).upper()
    text = format_license(text)


    if license_complies_format(text):
        return text, 'high'
    elif len(text) == 7:
        return text, 'medium'
    else:
        return text, 'low'
    

def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox', 'car_bbox_confidence',
                                                'license_plate_bbox', 'license_plate_bbox_confidence', 'license_number','license_number_confidence'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                f.write('{},{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                        car_id,
                                                        '[{} {} {} {}]'.format(
                                                            results[frame_nmr][car_id]['car']['bbox'][0],
                                                            results[frame_nmr][car_id]['car']['bbox'][1],
                                                            results[frame_nmr][car_id]['car']['bbox'][2],
                                                            results[frame_nmr][car_id]['car']['bbox'][3]),
                                                        results[frame_nmr][car_id]['car']['bbox_conf'],
                                                        '[{} {} {} {}]'.format(
                                                            results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                            results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                            results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                            results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                        results[frame_nmr][car_id]['license_plate']['bbox_conf'],
                                                        results[frame_nmr][car_id]['license_plate']['text'],
                                                        results[frame_nmr][car_id]['license_plate']['text_conf']
                                                        )
                        )
    f.close()


if __name__ == '__main__':
    # Test the functions
    print(format_license('NAI3NRU'))