package com.example.myapplication;

import android.app.Activity;
import android.os.Bundle;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;
import android.widget.Button;
import android.widget.Toast;

import androidx.annotation.Nullable;

import com.example.face.core.EigenFaceDetector;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity {

    private static final int W = 112, H = 112;
    private EigenFaceDetector detector;
    private boolean trained = false;

    @Override protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "OpenCV load fail", Toast.LENGTH_LONG).show();
            return;
        }

        detector = new EigenFaceDetector(W, H, 80);

        Button btnTrain   = findViewById(R.id.btnTrain);
        Button btnPredict = findViewById(R.id.btnPredict);

        /* ---- TRAIN ---- */
        btnTrain.setOnClickListener(v -> {
            try {
                trainFromAssets();
                trained = true;
                Toast.makeText(this, "Training done", Toast.LENGTH_SHORT).show();
            } catch (IOException e) {
                Log.e("Train", "Asset read error", e);
                Toast.makeText(this, "Training failed: assets missing", Toast.LENGTH_SHORT).show();
            }
        });

        /* ---- PREDICT ---- */
        btnPredict.setOnClickListener(v -> {
            if (!trained) {
                Toast.makeText(this, "Train first!", Toast.LENGTH_SHORT).show();
                return;
            }
            try {
                Mat test  = loadGrayFromAssets("faces/test.png");
                int label = detector.predict(test);
                Toast.makeText(this, "Label = " + label, Toast.LENGTH_SHORT).show();
            } catch (IOException e) {
                Log.e("Predict", "Test image load error", e);
                Toast.makeText(this, "Test image not found", Toast.LENGTH_SHORT).show();
            }
        });
    }

    /* ------------- 파일 → GRAY Mat ------------- */
    private Mat loadGrayFromAssets(String path) throws IOException {
        try (InputStream is = getAssets().open(path)) {
            Bitmap bmp = BitmapFactory.decodeStream(is);
            Mat rgba = new Mat();
            Utils.bitmapToMat(bmp, rgba);

            Mat gray = new Mat();
            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY);
            Imgproc.resize(gray, gray, new Size(W, H));
            gray.convertTo(gray, CvType.CV_8U);
            return gray;
        }
    }

    /* ------------- 학습 루틴 ------------- */
    private void trainFromAssets() throws IOException {
        String[] files = getAssets().list("faces");
        if (files == null || files.length == 0)
            throw new IOException("faces folder empty");

        List<Mat> images = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        for (String f : files) {
            Mat img = loadGrayFromAssets("faces/" + f);
            images.add(img);
            labels.add(labelFromName(f));        // 파일명 규칙 → 라벨
        }
        detector.train(images, labels.stream().mapToInt(i -> i).toArray());
    }

    private int labelFromName(String filename) {
        if (filename.startsWith("alice")) return 0;
        if (filename.startsWith("bob"))   return 1;
        return -1;
    }
}
