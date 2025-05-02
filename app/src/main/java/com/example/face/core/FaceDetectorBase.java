// File: java/com/example/face/core/FaceDetectorBase.java
package com.example.face.core;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

public abstract class FaceDetectorBase {
    protected final int imgWidth;
    protected final int imgHeight;

    protected FaceDetectorBase(int imgWidth, int imgHeight) {
        this.imgWidth  = imgWidth;
        this.imgHeight = imgHeight;
    }

    public abstract void train(java.util.List<Mat> imagesGray, int[] labels);
    public abstract int predict(Mat gray);

    protected static Mat toRowVector(Mat gray) {
        if (gray.channels() != 1)
            throw new IllegalArgumentException("Expect grayscale image");
        Mat cont = gray.isContinuous() ? gray : gray.clone();
        Mat row = cont.reshape(1, 1);
        row.convertTo(row, CvType.CV_64F);
        return row;
    }
}