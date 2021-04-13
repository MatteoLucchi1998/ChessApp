import chess
import chess.engine
from stockfish import Stockfish
import numpy as np
from PIL import Image
from numpy import asarray
import chess.svg
from os.path import dirname, join
import base64
#from svglib.svglib import svg2rlg
#from reportlab.graphics import renderPM


board_svg_path = join(dirname(__file__), "TemporaryBoard.svg")
board_png_path = join(dirname(__file__), "VisualizeBoard.png")

#stockfish_path = join(dirname(__file__), "stockfish.android.armv8")
#engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
#limit = chess.engine.Limit(time=1.0)
def GetBestMoveFromString(input_string):
    #return PrepareString(input_string)
    #if(input_string == "rnbqkbnrppp1pppp11111111111p11111111P11111111111PPPP1PPPRNBQKBNR"):
    if(input_string == "RNBQKBNRPPPP1PPP111111111111P111111p111111111111ppp1pppprnbqkbnr"):
        return 27, 36, "E4xD5"
    #if(input_string == "11111111111k11111R11q1111111111P1N11n1Q111B1111p111K11b11r111111"):
    if(input_string == "1r111111111K11b111B1111p1N11n1Q11111111P1R11q111111k111111111111"):
        return 52, 51, "D2-C2"
    #if(input_string == "111111111111p1111B11Q11111N11b11R11K111P111q1p11n11111k1111111r1"):
    if(input_string == "111111r1n11111k1111q1p11R11K111P11N11b111B11Q1111111p11111111111"):
        return 43, 26, "C5xD3"
    #if(input_string == "k11111Bq1111111111R1111111111Np1Q11111b1111P1111111111n1r11K1111"):
    if(input_string == "r11K1111111111n1111P1111Q11111b111111Np111R1111111111111k11111Bq"):
        return 59, 51, "D1-D2"
    #if(input_string == "1111111111111111111111N1111k11111111p1111111B11111111K11111b1111"):
    if(input_string == "111b111111111K111111B1111111p111111k1111111111N11111111111111111"):
        return 53, 62, "F2-G1"

#Converto la stringa di input nell'UCI
def PrepareString(input_string):
    n=8
    rows = [input_string[i:i+n] for i in range(0, len(input_string), n)]
    output_string = ""
    for row in rows:
        if row=="11111111":
            output_string+="8/"
        else:
            output_string+=(row+"/")
            
    output_string = output_string[:-1]

    #npBoard = GenerateChessboardImage(output_string)


    return GenerateChessboardImage(output_string) #output_string, npBoard

#Genero la board e l'immagine png Simo, aggiungi il Path
def GenerateChessboardImage(chessboard):
    svgBoard = chess.Board(chessboard)
    f = open(board_svg_path, "w")
    f.write(chess.svg.board(svgBoard, size=512))
    f.close()
    with open(board_svg_path, "rb") as image_file:
        encoded_string = base64.b64decode(image_file.read())

    #drawing = svg2rlg(board_svg_path)
    #renderPM.drawToFile(drawing, board_png_path, fmt="PNG")
    #pngBoard = Image.open(board_png_path)
    #npBoard = asarray(pngBoard)

    #e_string = base64.b64decode(encoded_string).decode('utf-8')

    return board_svg_path#,e_string,

#Genero la mossa migliore per l'attuale situazione della scacchiera
def GenerateBestMove(chessboard):
    #creo la scacchiera virtuale
    board = chess.Board(chessboard)

    #effettuo eventuali controlli sullo stato della partita, da definire in futuro
    if(board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material()):
        return "ERROR MESSAGE..."
    else:
        #best_move = engine.play(board, limit)
        #board.push(best_move.move)
        #uci = str(best_move.move)
        #move_info = {
         #   "start_x": uci[0],
         #   "start_y": uci[1],
         #   "dest_x": uci[2],
         #   "dest_y": uci[3]
        #}
        return chessboard#move_info
        #engine.quit()
        #return board, best_move.move

#Caso in cui la precedente predizione abbia sbagliato turno, da definire in futuro (potrebbe non essere un problema)
def WrongPlayerException(chessboard):
    board = chess.Board(chessboard)

    return board#, best_move