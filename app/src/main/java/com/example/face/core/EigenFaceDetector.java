// File: java/com/example/face/core/EigenFaceDetector.java
package com.example.face.core;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import java.util.*;

public class EigenFaceDetector extends FaceDetectorBase {
    private final int k;
    private Mat mean, eigVec, projections;
    private int[] labelsRef;

    public EigenFaceDetector(int w, int h, int k) {
        super(w, h);
        this.k = k;
    }

    @Override
    public void train(List<Mat> imgs, int[] labels) {
        if (imgs.size() != labels.length) throw new IllegalArgumentException();
        List<Mat> rows = new ArrayList<>(imgs.size());
        for (Mat g : imgs) rows.add(toRowVector(g));
        Mat X = new Mat(); Core.vconcat(rows, X);
        mean = new Mat(); eigVec = new Mat();
        Core.PCACompute(X, mean, eigVec, k);
        projections = new Mat();
        Core.PCAProject(X, mean, eigVec, projections);
        labelsRef = labels.clone();
    }

    @Override
    public int predict(Mat gray) {
        if (projections == null) throw new IllegalStateException("train first");
        Mat x = toRowVector(gray);
        Mat y = new Mat();
        Core.PCAProject(x, mean, eigVec, y);
        double best = Double.MAX_VALUE; int lab = -1;
        for (int i = 0; i < projections.rows(); i++) {
            double d = Core.norm(projections.row(i), y, Core.NORM_L2);
            if (d < best) {
                best = d; lab = labelsRef[i];
            }
        }
        return lab;
    }
}
