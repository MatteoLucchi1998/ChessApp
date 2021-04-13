package com.example.chessapp;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.content.FileProvider;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;


import java.io.ByteArrayOutputStream;
import java.io.File;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;

public class PredictActivity extends AppCompatActivity {

    private final int TOTAL_SQUARES = 64;
    TextView title;
    ImageView imageView;
    Button gallery;
    Button cancel;
    ConstraintLayout constraintLayout;
    Button camera;
    Button confirm;
    TextView tv;
    Uri mCurrentPhotoPath;
    private PyObject pyobjectSaver;
    private static  String TAG = "GalleryActivity";
    private static final int SELECT_PICTURE_CODE = 1;
    private static final int TAKE_PICTURE_CODE = 11;
    private static final int PERMISSION_CODE = 100;
    private List<String> results;
    private Classifier piecesClassifier;
    private Classifier colorsClassifier;
    private Classifier classifier;
    //private String piecesModelPath = "mobilenetv2_pieces.tflite";
    //private String piecesLabelPath = "pieces_labels.txt";
    //private String colorsModelPath = "mobilenetv2_colors.tflite";
    //private String colorsLabelPath = "color_labels.txt";
    private String labelPath = "full_labels.txt";
    private String modelPath = "mobilenetv2.tflite";
    private int inputSize = 224;
    private Activity activity;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_predict);

        constraintLayout = (ConstraintLayout)findViewById(R.id.gallery_layout);
        gallery = (Button)findViewById(R.id.gallery);
        imageView = (ImageView) findViewById(R.id.imageView);
        tv = (TextView) findViewById(R.id.tv);
        confirm = (Button) findViewById(R.id.confirm);
        cancel = (Button) findViewById(R.id.cancel);
        camera = (Button) findViewById(R.id.camera);
        title = (TextView) findViewById(R.id.gallery_title);
        if(!Python.isStarted()){
            Python.start(new AndroidPlatform((this)));
        }

        activity = this;


        try {
            initClassifier();
        } catch (IOException e) {
            e.printStackTrace();
        }


        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                if(checkGalleryPermission()){
                    pickImageFromGallery();
                }
            }
        });

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(checkCameraPermission()) {
                    takeImageFromCamera();
                }
            }
        });

        cancel.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                setUpView();
            }
        });

        confirm.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                PredictionThread predictionThread = new PredictionThread();
                predictionThread.start();
                findViewById(R.id.loadingPanel).setVisibility(View.VISIBLE);

            }
        });
    }

    private void initClassifier() throws IOException {

        classifier = new Classifier(activity, modelPath, labelPath);
    }



    private String getBoardString(List<String> results){
        StringBuilder s = new StringBuilder();
        for(int i = 0; i < results.size(); i++){
            s.append(Utilities.mapStringToIndex(results.get(i)));
        }
        return s.toString();
    }

// PER TESTARE IL MODELLO DEVI GUARDARE QUESTO METODO E I LOG DI OUTPUT
    private List<String> cropSquares() throws InterruptedException {
        final List<String> results = new ArrayList<>();
        if(pyobjectSaver != null){
            Python py = Python.getInstance();
            final PyObject detector = py.getModule("PiecesDetector");
            PyObject obj_detector = detector.callAttr("cropPieces", pyobjectSaver.asList().get(0), pyobjectSaver.asList().get(2));
            final PyObject single_image = py.getModule("PiecesDetector");

            for(int i = 0; i < TOTAL_SQUARES; i++) {
                PyObject obj_single_piece = single_image.callAttr("getSingleImage", obj_detector, i);

                Bitmap bitmap = Utilities.pyimageToBitmap(obj_single_piece.toString());


                Map<String, Float> result = classifier.makePrediction(bitmap);
                Pair<String, Float> final_result = classifier.getTopProbability(result);
                Log.d(TAG, i + "---> "+final_result.first + " " + final_result.second);
                results.add(final_result.first);
            }
        }
        return results;
    }




    private void setUpView(){
        findViewById(R.id.loadingPanel).setVisibility(View.GONE);
        gallery.setVisibility(View.VISIBLE);
        title.setVisibility(View.VISIBLE);
        confirm.setVisibility(View.GONE);
        cancel.setVisibility(View.GONE);
        camera.setVisibility(View.VISIBLE);
        constraintLayout.setBackgroundResource(R.drawable.gallery_background);
        imageView.setImageResource(0);
    }




    private void takeImageFromCamera() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        // Ensure that there's a camera activity to handle the intent
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            // Create the File where the photo should go
            File photoFile = null;
            try {
                photoFile = createImageFile();
            } catch (Exception e) {
                // Error occurred while creating the File
                Log.println(Log.ERROR,"error camera","error camera");
            }
            // Continue only if the File was successfully created
            if (photoFile != null) {

                Uri photoURI = FileProvider.getUriForFile(getApplicationContext(),
                        "com.example.android.fileprovider",
                        photoFile);
                Log.d("mylog", photoURI.getPath());
                SharedPreferences sharedPref = PredictActivity.this.getPreferences(Context.MODE_PRIVATE);
                SharedPreferences.Editor editor = sharedPref.edit();
                editor.putString("uriImage", photoURI.toString());
                editor.apply();

                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);

                startActivityForResult(takePictureIntent, TAKE_PICTURE_CODE);

            }
        }
    }

    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );
        // Save a file: path for use with ACTION_VIEW intents
        String mCurrentPhotoPath = image.getAbsolutePath();
        Log.d("mylog", "Path: " + mCurrentPhotoPath);
        return image;
    }



    private void pickImageFromGallery() {
        Intent intent = new Intent((Intent.ACTION_PICK));
        intent.setType("image/*");
        startActivityForResult(intent, SELECT_PICTURE_CODE);
    }

    class PredictionThread extends Thread{

        @Override
        public void run() {
            super.run();
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    try {
                        List<String> results = cropSquares();
                        String boardString = getBoardString(results);
                        findViewById(R.id.loadingPanel).setVisibility(View.GONE);
                        Intent intent = new Intent(activity, VirtualBoardActivity.class);
                        Bundle bundle = new Bundle();
                        bundle.putString("Board", boardString);
                        intent.putExtras(bundle);
                        startActivity(intent);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            });

        }
    }

    class MatrixThread extends Thread{
        Bitmap image;

        MatrixThread(Bitmap image){
            this.image = image;
        }

        @Override
        public void run() {
            super.run();
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                    image.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
                    byte[] byteArray = byteArrayOutputStream .toByteArray();
                    Python py = Python.getInstance();
                    final PyObject pyObject = py.getModule("DetectAllPoints");
                    PyObject obj = pyObject.callAttr("getMatrixFromImage", (Object) byteArray);
                    if(obj.asList().size() == 3 && obj.asList().get(1) != null) {

                        Bitmap bitmap = Utilities.pyimageToBitmap(obj.asList().get(1).toString());
                        if(bitmap != null) {
                            pyobjectSaver = obj;

                            imageView.setImageBitmap(bitmap);
                            findViewById(R.id.loadingPanel).setVisibility(View.INVISIBLE);


                            cancel.setVisibility(View.VISIBLE);
                            confirm.setVisibility(View.VISIBLE);
                            constraintLayout.setBackgroundColor(Color.rgb(0,0,0));

                        }else{
                            Toast.makeText(getApplicationContext(), "Nothing found..", Toast.LENGTH_SHORT).show();
                            setUpView();
                        }
                    }else{
                        Toast.makeText(getApplicationContext(), "Not enough return arguments..", Toast.LENGTH_SHORT).show();
                        setUpView();
                    }
                }
            });

        }
    }

    private void hideOnImageShowing(){
        gallery.setVisibility(View.GONE);
        camera.setVisibility(View.GONE);
        title.setVisibility(View.GONE);
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        SharedPreferences sharedPref = PredictActivity.this.getPreferences(Context.MODE_PRIVATE);
        String UriStr = sharedPref.getString("uriImage","empty");

        if (resultCode == RESULT_OK) {
            Log.d("mylog", "GGGGGG");
            Bitmap selectedImage = null;

            if(requestCode == TAKE_PICTURE_CODE){

                try {
                    Uri uri = Uri.parse(UriStr);
                    Log.d("mylog", "uri: "+uri.toString());
                    selectedImage = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                    //selectedImage = rotateImage(selectedImage, 90);
                } catch (IOException e) {
                    Log.println(Log.ERROR,"uri error","uri error");
                }
            }else {
                final Uri imageUri = data.getData();
                final InputStream imageStream;
                try {
                    assert imageUri != null;
                    imageStream = getContentResolver().openInputStream(imageUri);
                    selectedImage = BitmapFactory.decodeStream(imageStream);
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }
            }


            assert selectedImage != null;
            Bitmap bmp32 = selectedImage.copy(Bitmap.Config.RGB_565, true);


            MatrixThread matrixThread = new MatrixThread(bmp32);
            findViewById(R.id.loadingPanel).setVisibility(View.VISIBLE);
            hideOnImageShowing();
            matrixThread.start();
        }
    }

    private Bitmap getRotatedBitmap(Bitmap bitmap, String path){
        ExifInterface ei = null;
        try {
            ei = new ExifInterface(path);
        } catch (IOException e) {
            e.printStackTrace();
        }
        int orientation = ei.getAttributeInt(ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_UNDEFINED);

        Bitmap rotatedBitmap = null;
        switch(orientation) {

            case ExifInterface.ORIENTATION_ROTATE_90:
                rotatedBitmap = rotateImage(bitmap, 90);
                break;

            case ExifInterface.ORIENTATION_ROTATE_180:
                rotatedBitmap = rotateImage(bitmap, 180);
                break;

            case ExifInterface.ORIENTATION_ROTATE_270:
                rotatedBitmap = rotateImage(bitmap, 270);
                break;

            case ExifInterface.ORIENTATION_NORMAL:
            default:
                rotatedBitmap = bitmap;
        }
        return rotatedBitmap;
    }

    private Bitmap rotateImage(Bitmap source, float angle) {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(),
                matrix, true);
    }





    private boolean checkGalleryPermission(){
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
            if(checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED ){
                String[] permissions = {Manifest.permission.READ_EXTERNAL_STORAGE};
                requestPermissions(permissions, PERMISSION_CODE);
                return false;
            }
        }
        return true;
    }

    private boolean checkCameraPermission(){
        boolean flag = true;
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
            if(checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
                String[] permissions = {Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE};
                requestPermissions(permissions, PERMISSION_CODE);
                flag = false;
            }
            if(checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED){
                String[] permissions = {Manifest.permission.WRITE_EXTERNAL_STORAGE};
                requestPermissions(permissions, PERMISSION_CODE);
                flag = false;
            }
        }
        return flag;
    }
}
