package com.example.chessapp;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;
import android.util.Pair;

import androidx.annotation.NonNull;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class Classifier {
    private Interpreter interpreter;
    private List<String> labelList;
    private int imageSizeX;
    private int imageSizeY;
    private static final float IMAGE_MEAN = 0.0f;
    private static final float IMAGE_STD = 255.0f;
    private static final float PROB_MEAN = 0.0f;
    private static final float PROB_STD = 1.0f;
    private final float MAX_RESULTS = 3;
    private final float THRESHOLD = 0.5F;
    private TensorImage inputImageBuffer;
    private Activity activity;

    Classifier(Activity activity, String modelPath, String labelPath) throws IOException {
        this.activity = activity;
        Interpreter.Options options = new Interpreter.Options();
        interpreter = new Interpreter(loadModelFile(activity.getAssets(), modelPath), options);
        labelList = loadLabelList(labelPath);
    }

    public Pair<String, Float> getTopProbability(Map<String, Float> labelProb) {
        // trova la classificazione con il punteggio più alto
        PriorityQueue<Pair<String, Float>> pq =
                new PriorityQueue<>(
                        1,
                        new Comparator<Pair<String, Float>>() {
                            @Override
                            public int compare(Pair<String, Float> lhs, Pair<String, Float> rhs) {
                                return Float.compare(rhs.second, lhs.second);
                            }
                        });




        for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
            //se l'utente ha inserito l'opzione tutte le piante
            pq.add(new Pair<String, Float>(entry.getKey(), entry.getValue()));

        }




        return pq.poll();
    }

    public Map<String, Float> makePrediction(Bitmap bitmap){
        int imageTensorIndex = 0;
        int[] imageShape = interpreter.getInputTensor(imageTensorIndex).shape();
        imageSizeY = imageShape[1];
        imageSizeX = imageShape[2];

        DataType imageDataType = interpreter.getInputTensor(imageTensorIndex).dataType();
        int probabilityTensorIndex = 0;
        int[] probabilityShape = interpreter.getOutputTensor(probabilityTensorIndex).shape();
        DataType probabilityDataType = interpreter.getOutputTensor(probabilityTensorIndex).dataType();
        inputImageBuffer = loadImage(bitmap, imageDataType);
        TensorBuffer outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

        // output buffer for probs
        TensorProcessor probabilityProcessor = new TensorProcessor.Builder()
                .add(new NormalizeOp(PROB_MEAN, PROB_STD))
                .build();

        interpreter.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());

        return new TensorLabel(labelList, probabilityProcessor.process(outputProbabilityBuffer))
                .getMapWithFloatValue();
    }

    private TensorImage loadImage(final Bitmap bitmap, DataType imageDataType) {

        inputImageBuffer = new TensorImage(imageDataType);
        inputImageBuffer.load(bitmap);

        //crete readable image for the interpreter
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(IMAGE_MEAN, IMAGE_STD))
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }


    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException{
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private List<String> loadLabelList(String labelPath) throws IOException{
        return FileUtil.loadLabels(activity, labelPath);
    }





}
