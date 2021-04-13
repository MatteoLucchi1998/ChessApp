# Module built to find the squares of the chess board and crop each square for the dataset
# The distribution inside the different folder is done by using the keyboard

import DetectAllPoints as dap
import cv2
import numpy as np
import math
import io
import base64
from PIL import Image

# Finds all the squares inside the chess board by using a matrix of points
# pointsMatrix -> the matrix containing the chess board's points
# squares_found -> the chess board's squares
def getSquares(pointsMatrix):
    squares_found = []
    for i in range(0, len(pointsMatrix)-1):
        for j in range(0, len(pointsMatrix[i])-1):
            squares_found.append([pointsMatrix[i][j], pointsMatrix[i][j+1], pointsMatrix[i+1][j+1],pointsMatrix[i+1][j]])
    return squares_found

def getSingleImage(cropped_pieces_list, index):
    cropped = cropped_pieces_list[index]
    return cropped


# Lightly color a square
# square_to_fill -> the square to fill
# img_orig -> t6he original image
# out_img -> the image with the colored square
def fillSquare(square_to_fill, img_orig):
    out_img = img_orig.copy()
    cv2.fillPoly(img_orig, pts =np.array([square_to_fill], dtype=np.int32), color=(0,255,0))
    ALPHA = 0.5
    cv2.addWeighted(img_orig, ALPHA, out_img, 1 - ALPHA, 0, out_img)
    return out_img


# Sorts the points based on the Y axis
# sub_li -> the points list
def Sort_Y(sub_li): 
    # reverse = None (Sorts in Ascending order) 
    # key is set to sort using second element of  
    # sublist lambda has been used 
    return(sorted(sub_li, key = lambda x: x[1]))

# Sorts the points based on the X axis
# sub_li -> the points list
def Sort_X(sub_li): 
    return(sorted(sub_li, key = lambda x: x[0]))



# Cropps the squares from the image that will be used to make the dataset 
# file -> the image file we want to crop

def cropPieces(img, matrix):
    

    if matrix is not None:
        squares = getSquares(matrix)

        ratio_h = 1.5   #Definisce il rapporto verticale/orizzontale   scegli tra 1.5 o 2
        ratio_w = 1   #Io lascerei 1
        pieces_cropped = []
        for square in squares:
            img_copy = img.copy()
            out = fillSquare(square, img_copy)
            sort_y = Sort_Y(square)
            # We get the points with trhe lowest y and the two lowest x 
            new_y = sort_y[3][1]
            sort_x = Sort_X(square)
            first_new_x = sort_x[0][0]
            second_new_x = sort_x[3][0]

            bot_left, bot_right = [first_new_x,new_y], [second_new_x,new_y]

            
            base_len = math.dist(bot_left, bot_right)   
            
            start_x, start_y = int(bot_left[0]), int(bot_left[1] - (base_len * ratio_h))
            
            end_x, end_y = int(bot_right[0]), int(bot_right[1])
            
            if start_y < 0:
                start_y = 0
                
            cropped = img[start_y: end_y, start_x: end_x]
            cropped = cv2.resize(cropped,(224, 224))
            cropped = Image.fromarray(cropped)

            byte_array = io.BytesIO()
            cropped.save(byte_array, format='JPEG')
            encoded_image = base64.encodebytes(byte_array.getvalue()).decode('ascii')

            pieces_cropped.append(encoded_image)

        return pieces_cropped
    else:
        return None
        


