# Module built to get a metrix containing oll the points from the chess board
# getMatrixFromImage receive a image file in input (if null takes a picture from the camera)
# It returns a colored image representing the image passed or taken and the matrix representing
# the points of the chess board 

from PIL import Image
import cv2
import base64
import io
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

import DetectionFunctions as df

np.set_printoptions(suppress=True, linewidth=200) # Better printing of arrays
plt.rcParams['image.cmap'] = 'jet' # Default colormap is jet

#return the image and the matrix 
def getMatrixFromImage(byteArray):

    img = Image.open(io.BytesIO(bytes(byteArray)))
    img = np.array(img)
    img_orig = Image.fromarray(img)
    img_width, img_height = img_orig.size



    # Resize
    aspect_ratio = min(500.0/img_width, 500.0/img_height)
    new_width, new_height = ((np.array(img_orig.size) * aspect_ratio)).astype(int)
    img = img_orig.resize((new_width,new_height), resample=Image.BILINEAR)
    img_rgb = img
    img = img.convert('L') # grayscale
    img = np.array(img)
    img_rgb = np.array(img_rgb)
    M, ideal_grid, grid_next, grid_good, spts = df.findChessboard(img)

    #xy-unwarp -> the inner points of the inner chessboard
    #board_outline -> the corners (they are five because the first one is repeated)
    #boarder_points_?? -> the edges (?? edge of board: boarder_points_01 = edge from corner 0 to 1)


    # View
    if M is not None:
        M, _ = df.generateNewBestFit((ideal_grid+8)*32, grid_next, grid_good) # generate mapping for warping image
        img_warp = cv2.warpPerspective(img, M, (17*32, 17*32), flags=cv2.WARP_INVERSE_MAP)

        best_lines_x, best_lines_y = df.getBestLines(img_warp)
        xy_unwarp = df.getUnwarpedPoints(best_lines_x, best_lines_y, M)
        board_outline_unwarp = df.getBoardOutline(best_lines_x, best_lines_y, M)
        
        borders_points_01 = []
        borders_points_12 = []
        borders_points_23 = []
        borders_points_30 = []
        for i in range(0,len(xy_unwarp)):
            if i%7 == 0:
                a,b = df.slope_intercept(xy_unwarp[i,0],xy_unwarp[i,1],xy_unwarp[i+1,0],xy_unwarp[i+1,1])
                x_30, y_30 = df.line_intersection(([np.float32(0),b],[xy_unwarp[i,0],xy_unwarp[i,1]]),([board_outline_unwarp[3,0],board_outline_unwarp[3,1]],[board_outline_unwarp[0,0],board_outline_unwarp[0,1]]))
                x_12, y_12 = df.line_intersection(([-b/a,np.float32(0)],[xy_unwarp[i,0],xy_unwarp[i,1]]),([board_outline_unwarp[1,0],board_outline_unwarp[1,1]],[board_outline_unwarp[2,0],board_outline_unwarp[2,1]]))
                borders_points_30.append([x_30, y_30])
                borders_points_12.append([x_12, y_12])
            
            if i in range(0,7):
                a,b = df.slope_intercept(xy_unwarp[i,0],xy_unwarp[i,1],xy_unwarp[i+7,0],xy_unwarp[i+7,1])
                x_01, y_01 = df.line_intersection(([np.float32(0),b],[xy_unwarp[i,0],xy_unwarp[i,1]]),([board_outline_unwarp[0,0],board_outline_unwarp[0,1]],[board_outline_unwarp[1,0],board_outline_unwarp[1,1]]))
                x_23, y_23 = df.line_intersection(([-b/a,np.float32(0)],[xy_unwarp[i,0],xy_unwarp[i,1]]),([board_outline_unwarp[2,0],board_outline_unwarp[2,1]],[board_outline_unwarp[3,0],board_outline_unwarp[3,1]]))
                borders_points_01.append([x_01, y_01])
                borders_points_23.append([x_23, y_23])

        first_line = np.concatenate(([board_outline_unwarp[0]],borders_points_01,[board_outline_unwarp[1]]),axis=0)
        last_line = np.concatenate(([board_outline_unwarp[3]],borders_points_23,[board_outline_unwarp[2]]),axis=0)
        inner_lines = df.chunks(xy_unwarp, 7)
        for i in range(0, len(borders_points_12)):
            inner_lines[i] = np.concatenate(([borders_points_30[i]],inner_lines[i],[borders_points_12[i]]),axis=0)

        matrix = np.vstack(([first_line], inner_lines, [last_line]))
        clear_image = img_rgb.copy()
        #uncomment to see points on the image
        df.color_points(img_rgb, matrix)
        img_rgb = Image.fromarray(img_rgb)
        img_rgb = img_rgb.resize((img_width, img_height), resample=Image.BILINEAR)
        byte_array = io.BytesIO()
        img_rgb.save(byte_array, format='JPEG')
        encoded_image = base64.encodebytes(byte_array.getvalue()).decode('ascii')
        # cv2.imshow("ImageRGB", img_rgb)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        return clear_image, encoded_image, matrix
        

    else:
        # cv2.imshow("Image", img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        return None, None , None
            


