// ===============================
// Package: com.example.face.core
// ===============================
package com.example.face.core;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.Utils;
import java.util.*;


/***********************************
 * FisherFaceDetector (public)
 ***********************************/
public class FisherFaceDetector extends FaceDetectorBase {
    private final int pcaDim;
    private Mat mean,pcaVec,ldaVec,projections; int[] labelsRef;

    public FisherFaceDetector(int w,int h,int pcaDim){ super(w,h); this.pcaDim=pcaDim; }

    @Override public void train(List<Mat> imgs,int[] labels){
        int N=imgs.size(); if(N!=labels.length) throw new IllegalArgumentException();
        List<Mat> rows=new ArrayList<>(N); for(Mat g:imgs) rows.add(toRowVector(g));
        Mat X=new Mat(); Core.vconcat(rows,X);
        mean=new Mat(); pcaVec=new Mat();
        int redDim=Math.min(pcaDim,N-unique(labels));
        Core.PCACompute(X,mean,pcaVec,redDim);
        Mat Xp=new Mat(); Core.PCAProject(X,mean,pcaVec,Xp);
        int C=unique(labels);
        Mat Sw=Mat.zeros(redDim,redDim,CvType.CV_64F);
        Mat Sb=Mat.zeros(redDim,redDim,CvType.CV_64F);
        Map<Integer,List<Integer>> cls=new HashMap<>();
        for(int i=0;i<labels.length;i++) cls.computeIfAbsent(labels[i],k->new ArrayList<>()).add(i);
        Mat overall=new Mat(); Core.reduce(Xp,overall,0,Core.REDUCE_AVG);
        for(Map.Entry<Integer,List<Integer>> e:cls.entrySet()){
            List<Integer> idxs=e.getValue();
            Mat Xi=rowsOf(Xp,idxs);
            Mat mi=new Mat(); Core.reduce(Xi,mi,0,Core.REDUCE_AVG);
            Mat diff=new Mat(); Core.subtract(mi,overall,diff);
            Mat outer=new Mat(); Core.gemm(diff.t(),diff,1.0,new Mat(),0.0,outer);
            Core.scaleAdd(outer,idxs.size(),Sb,Sb);
            for(int idx:idxs){
                Mat xi=Xp.row(idx); Mat d=new Mat(); Core.subtract(xi,mi,d);
                Mat o=new Mat(); Core.gemm(d.t(),d,1.0,new Mat(),0.0,o);
                Core.add(Sw,o,Sw);
            }
        }
        Mat SwInv=new Mat(); Core.invert(Sw,SwInv,Core.DECOMP_SVD);
        Mat A=new Mat(); Core.gemm(SwInv,Sb,1.0,new Mat(),0.0,A);
        Mat eigVals=new Mat(); Mat eigVecs=new Mat(); Core.eigen(A,eigVals,eigVecs);
        ldaVec=eigVecs.rowRange(0,C-1).clone();
        projections=new Mat(); Core.gemm(Xp,ldaVec.t(),1.0,new Mat(),0.0,projections);
        labelsRef=labels.clone();
    }

    @Override public int predict(Mat gray){
        if(projections==null) throw new IllegalStateException();
        Mat x=toRowVector(gray);
        Mat xp=new Mat(); Core.PCAProject(x,mean,pcaVec,xp);
        Mat y=new Mat(); Core.gemm(xp,ldaVec.t(),1.0,new Mat(),0.0,y);
        double best=Double.MAX_VALUE; int lab=-1;
        for(int i=0;i<projections.rows();i++){
            double d=Core.norm(projections.row(i),y,Core.NORM_L2);
            if(d<best){best=d; lab=labelsRef[i];}
        }
        return lab;
    }

    private static int unique(int[] a){ return (int)java.util.Arrays.stream(a).distinct().count(); }
    private static Mat rowsOf(Mat src,List<Integer> idxs){
        Mat dst=new Mat(idxs.size(),src.cols(),src.type());
        for(int r=0;r<idxs.size();r++) src.row(idxs.get(r)).copyTo(dst.row(r));
        return dst; }
}
