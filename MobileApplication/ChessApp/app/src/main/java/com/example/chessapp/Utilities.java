package com.example.chessapp;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Base64;
import android.util.Log;

import com.chaquo.python.PyObject;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class Utilities {
    private static final Map<String, String> piecesMap;
    static {
        Map<String, String> map = new HashMap<>();
        map.put("white_bishop", "B");
        map.put("white_queen", "Q");
        map.put("white_king", "K");
        map.put("white_knight", "N");
        map.put("white_rook", "R");
        map.put("white_pawn", "P");
        map.put("black_queen", "q");
        map.put("black_rook", "r");
        map.put("black_king", "k");
        map.put("black_knight", "n");
        map.put("black_pawn", "p");
        map.put("black_bishop", "b");
        map.put("empty", "1");
        piecesMap = Collections.unmodifiableMap(map);
    }

    private static final Map<String, String> drawableMap;
    static {
        Map<String, String> map = new HashMap<>();
        map.put("B", "wb_foreground");
        map.put("Q", "wq_foreground");
        map.put("K", "wk_foreground");
        map.put("N", "wn_foreground");
        map.put("R", "wr_foreground");
        map.put("P", "wp_foreground");
        map.put("q", "bq_foreground");
        map.put("r", "br_foreground");
        map.put("k", "bk_foreground");
        map.put("n", "bn_foreground");
        map.put("p", "bp_foreground");
        map.put("b", "bb_foreground");
        map.put("1", null);
        drawableMap = Collections.unmodifiableMap(map);
    }

    public static Bitmap pyimageToBitmap(String obj){
        if(obj == null ) return  null;
        byte [] encodeByte = Base64.decode(obj,Base64.DEFAULT);
        Bitmap bitmap = BitmapFactory.decodeByteArray(encodeByte, 0, encodeByte.length);
        return bitmap;
    }

    public static String mapStringToIndex(String s){
        return piecesMap.get(s);
    }

    public static String mapDrawableFromIndex(String s){
        return drawableMap.get(s);
    }
}
