// File: app/src/main/java/com/example/myapplication/MainActivity.java
package com.example.myapplication;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.*;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.face.core.EigenFaceDetector;
import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@androidx.camera.core.ExperimentalGetImage
public class MainActivity extends AppCompatActivity {
    private static final int W = 112, H = 112;
    private static final int CAMERA_PERMISSION_REQUEST = 1001;
    private EigenFaceDetector detector;
    private boolean trained = false;
    private PreviewView previewView;
    private Bitmap lastFrame = null;
    private ExecutorService cameraExecutor;
    private CascadeClassifier faceCascade;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        Button btnTrain = findViewById(R.id.btnTrain);
        Button btnPredict = findViewById(R.id.btnPredict);

        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "OpenCV init failed", Toast.LENGTH_LONG).show();
            return;
        }

        faceCascade = loadCascade();
        if (faceCascade == null || faceCascade.empty()) {
            Toast.makeText(this, "Failed to load face cascade classifier", Toast.LENGTH_LONG).show();
            return;
        }

        detector = new EigenFaceDetector(W, H, 80);
        cameraExecutor = Executors.newSingleThreadExecutor();

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST);
        } else {
            startCamera();
        }

        btnTrain.setOnClickListener(v -> {
            try {
                trainFromAssets();
                trained = true;
                Toast.makeText(this, "Training done", Toast.LENGTH_SHORT).show();
            } catch (Exception e) {
                Toast.makeText(this, "Train failed: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            }
        });

        btnPredict.setOnClickListener(v -> {
            if (lastFrame != null) {
                Toast.makeText(this, "Prediction Ready", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private CascadeClassifier loadCascade() {
        try {
            InputStream is = getAssets().open("haarcascades/haarcascade_frontalface_default.xml");
            File cascadeDir = getDir("cascade", MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_default.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close(); os.close();
            return new CascadeClassifier(mCascadeFile.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private void trainFromAssets() throws Exception {
        List<Mat> images = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        loadFaces("faces/kibae", 0, images, labels);
        loadFaces("faces/zhuoer", 1, images, labels);
        loadFaces("faces/third_party", 2, images, labels);
        detector.train(images, labels.stream().mapToInt(i -> i).toArray());
    }

    private void loadFaces(String folder, int label, List<Mat> images, List<Integer> labels) throws Exception {
        String[] files = getAssets().list(folder);
        if (files == null) return;
        for (String f : files) {
            InputStream is = getAssets().open(folder + "/" + f);
            Bitmap bmp = android.graphics.BitmapFactory.decodeStream(is);
            Mat rgba = new Mat();
            Utils.bitmapToMat(bmp, rgba);
            Mat gray = new Mat();
            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY);
            Imgproc.resize(gray, gray, new Size(W, H));
            gray.convertTo(gray, CvType.CV_8U);
            images.add(gray);
            labels.add(label);
            is.close();
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                analysis.setAnalyzer(cameraExecutor, image -> {
                    ImageProxy.PlaneProxy[] planes = image.getPlanes();
                    if (planes == null || planes.length == 0) {
                        image.close();
                        return;
                    }

                    ByteBuffer yBuffer = planes[0].getBuffer();
                    byte[] yData = new byte[yBuffer.remaining()];
                    yBuffer.get(yData);

                    int width = image.getWidth();
                    int height = image.getHeight();
                    Mat gray = new Mat(height, width, CvType.CV_8UC1);
                    gray.put(0, 0, yData);

                    MatOfRect faces = new MatOfRect();
                    faceCascade.detectMultiScale(gray, faces);
                    Mat rgba = new Mat();
                    Imgproc.cvtColor(gray, rgba, Imgproc.COLOR_GRAY2RGBA);

                    if (faces.toArray().length > 0) {
                        Imgproc.putText(rgba, "Face Detected", new Point(20, 40), Imgproc.FONT_HERSHEY_SIMPLEX, 1.2, new Scalar(0, 255, 0), 2);
                    } else {
                        Imgproc.putText(rgba, "No Face", new Point(20, 40), Imgproc.FONT_HERSHEY_SIMPLEX, 1.2, new Scalar(0, 0, 255), 2);
                    }

                    Bitmap bmp = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(rgba, bmp);
                    runOnUiThread(() -> previewView.setForeground(new android.graphics.drawable.BitmapDrawable(getResources(), bmp)));
                    image.close();
                });

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis);
            } catch (Exception e) {
                Log.e("CameraX", "Camera binding failed", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_LONG).show();
            }
        }
    }
}