package com.example.ncnn_android_nanocls;  //
import android.content.res.AssetManager;//
import android.graphics.Bitmap;

public class MobileNet {
    //与 JNI 中所对应的 java 方法
    public native boolean Init(AssetManager mgr);    //模型初始化
    public native String ImagePredict(Bitmap bitmap, boolean use_gpu); //模型推理

    static {
        //加载库
        System.loadLibrary("nanocls");
    }
}
