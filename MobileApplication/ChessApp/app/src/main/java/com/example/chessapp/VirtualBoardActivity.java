package com.example.chessapp;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.res.Resources;
import android.graphics.Color;
import android.graphics.drawable.Drawable;
import android.os.Build;
import android.os.Bundle;
import android.text.Layout;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import android.widget.RelativeLayout;
import android.widget.RelativeLayout.LayoutParams;
import android.widget.TextView;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class VirtualBoardActivity extends AppCompatActivity {

    List<Integer> ids = new ArrayList<>();
    Button btn_predict_white, btn_predict_black;
    TextView show_prediction;
    int width;
    int new_line_id = 1;
    int dark_color = Color.rgb(0, 0, 0);
    int light_color = Color.rgb(255, 255, 255);
    int current_color = light_color;
    int switch_current_color = dark_color;
    boolean switch_colors = true;


    RelativeLayout rl;
    @RequiresApi(api = Build.VERSION_CODES.M)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_virtual_board);
        if(!Python.isStarted()){
            Python.start(new AndroidPlatform((this)));
        }
        btn_predict_white = (Button) findViewById(R.id.predict_white);
        btn_predict_black = (Button) findViewById(R.id.predict_black);
        show_prediction = (TextView) findViewById(R.id.predicted_move);
        setScaledButtonWidth();

        rl = (RelativeLayout) findViewById(R.id.chessBoard);

        createGrid();


        final String boardString = getIntent().getExtras().getString("Board");
        String[] boardSplitted = splitAndReversString(boardString);

        for(int i = 0; i < 8; i++){

            for(int j = 0; j < 8 ; j++){

                char piece = boardSplitted[i].charAt(j);
                int index_id = (i * 8) + j ;
                Button btn = (Button) findViewById( ids.get(index_id));
                String drawableString = Utilities.mapDrawableFromIndex(String.valueOf(piece));
                if(drawableString != null){
                    Resources resources = getApplicationContext().getResources();
                    final int resourceId = resources.getIdentifier(drawableString, "drawable", getApplicationContext().getPackageName());
                    btn.setForeground(resources.getDrawable(resourceId));

                }
            }


        }

        btn_predict_white.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                predictMove(boardString);
            }
        });

    }

    private String[] splitAndReversString(String stringBoard){
        int len = stringBoard.length();
        //n determines the variable that divide the string in 'n' equal parts
        int n = 8;
        int temp = 0, chars = len/n;
        //Stores the array of string
        String[] equalStr = new String [n];
        //Check whether a string can be divided into n equal parts
        if(len % n != 0) {
            System.out.println("Sorry this string cannot be divided into "+ n +" equal parts.");
        }
        else {
            for (int i = 0; i < len; i = i + chars) {
                //Dividing string in n equal part using substring()
                String part = stringBoard.substring(i, i + chars);
                equalStr[temp] = part;
                temp++;
            }
        }
        Collections.reverse(Arrays.asList(equalStr));
        return equalStr;
    }



    private void createGrid(){
        for(int i = 1; i <= 8;i ++){
            switch_colors = !switch_colors;
            LayoutParams first_lp = new LayoutParams(width,width);
            Button first_btn = new Button(this);
            if(i != 1) {
                first_lp.addRule(RelativeLayout.BELOW, new_line_id);
                new_line_id +=  8;

            }
            first_btn.setId(new_line_id);
            ids.add(first_btn.getId());
            first_btn.setBackgroundColor(getSquareColor());

            rl.addView(first_btn, first_lp);
            for(int j = 1; j < 8; j++){
                LayoutParams lp = new LayoutParams(width,width);
                Button btn = new Button(this);
                if(i != 1) {
                    lp.addRule(RelativeLayout.BELOW, new_line_id - 8);
                }
                lp.addRule(RelativeLayout.RIGHT_OF, new_line_id + j -1);
                btn.setId(new_line_id + j);
                btn.setBackgroundColor(getSquareColor());
                ids.add(btn.getId());
                
                rl.addView(btn, lp);
            }
        }
    }

    private int getSquareColor(){
        int temp;
        if(switch_colors){
            temp = switch_current_color;
            if(switch_current_color == light_color){
                switch_current_color = dark_color;
            }else{
                switch_current_color = light_color;
            }
        }else{
            temp = current_color;
            if(current_color == light_color){
                current_color = dark_color;
            }else{
                current_color = light_color;
            }
        }
        return temp;
    }

    private void setScaledButtonWidth(){
        DisplayMetrics displayMetrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        width = displayMetrics.widthPixels / 8;
    }


    private void predictMove(String boardString){
        Python py = Python.getInstance();
        final PyObject predictor = py.getModule("ChessMoveGenerator");
        PyObject obj= predictor.callAttr("GetBestMoveFromString",boardString);
        colorPredictedMove(Integer.parseInt(obj.asList().get(0).toString()), Integer.parseInt(obj.asList().get(1).toString()));
        show_prediction.setText(obj.asList().get(2).toString());
    }

    private void colorPredictedMove(int x, int y){
        Button btn_dest = (Button) findViewById( ids.get(x));
        Button btn_piece = (Button) findViewById( ids.get(y));
        btn_piece.setBackgroundColor(Color.GREEN);
        btn_dest.setBackgroundColor(Color.YELLOW);

    }
}