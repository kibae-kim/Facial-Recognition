// ===============================
// Package: com.example.face.core
// ===============================
package com.example.face.core;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.Utils;
import java.util.*;

/***********************************
 * FaceDetectorBase (public)
 ***********************************/
public abstract class FaceDetectorBase {
    protected final int imgWidth;
    protected final int imgHeight;

    protected FaceDetectorBase(int imgWidth, int imgHeight) {
        this.imgWidth  = imgWidth;
        this.imgHeight = imgHeight;
    }

    public abstract void train(List<Mat> imagesGray, int[] labels);
    public abstract int predict(Mat gray);

    /* --- shared util --- */
    protected static Mat toRowVector(Mat gray) {
        if (gray.channels() != 1)
            throw new IllegalArgumentException("expect 1‑channel image");
        Mat cont = gray.isContinuous() ? gray : gray.clone();
        Mat row  = cont.reshape(1, 1);
        row.convertTo(row, CvType.CV_64F);
        return row;
    }
}

