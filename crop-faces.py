# https://stackoverflow.com/questions/65105644/how-to-face-extraction-from-images-in-a-folder-with-mtcnn-in-python
from mtcnn import MTCNN
import cv2
import os
import glob
from pathlib import Path




def main():
    destination_folder = 'cropped'
    source_folder = 'data/images'

    # Load all images from the directory
    images_found = list(map(str, glob.iglob(source_folder + '/**/*.jpeg', recursive=True)))

    # Get all the target paths
    target_paths = []
    for image in images_found:
        target_paths.append(copy_tree(image, destination_folder)) 

    list(map(crop_image, images_found, target_paths))

    print('finished!')



def copy_tree(source_file, destination_folder):
    """
        Copy a file path while maintaining its subdirectory structure
        and replace its top-level directory with the destination folder.

        Example:
            copy_tree("Downloads/files/example.txt", "Documents") ==> "Documents/files/example.txt"

    Args:
        source_file (string): file path to copy

        destination_folder (string): the new top-level folder

    Returns:
        a new file path with its top-level folder replaced.
    """
    source_path = Path(source_file)
    name_to_change = source_path.parts[0] # Get the first part of the file path
    target_path = "/".join([part if part != name_to_change else destination_folder
                            for part in source_path.parts])[0:]
    return target_path




def crop_image(image_path, destination_file):
    detector = MTCNN() 
    img=cv2.imread(image_path)
    data=detector.detect_faces(img)
    biggest=0
    if data !=[]:
        for faces in data:
            box=faces['box']            
            # calculate the area in the image
            area = box[3]  * box[2]
            if area>biggest:
                biggest=area
                bbox=box 
        bbox[0]= 0 if bbox[0]<0 else bbox[0]
        bbox[1]= 0 if bbox[1]<0 else bbox[1]
        img=img[bbox[1]: bbox[1]+bbox[3],bbox[0]: bbox[0]+ bbox[2]] 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert from bgr to rgb
        print('saving image:', destination_file)
        os.makedirs(os.path.dirname(destination_file), exist_ok=True)
        cv2.imwrite(destination_file, img)




if __name__ == '__main__':
    main()
