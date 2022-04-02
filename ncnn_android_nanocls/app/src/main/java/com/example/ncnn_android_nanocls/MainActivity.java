package com.example.ncnn_android_nanocls;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.FileNotFoundException;

public class MainActivity extends AppCompatActivity {

    private static final int SELECT_IMAGE = 1;
    //文字展示控件
    private TextView infoResult;
    //图片展示控件
    private ImageView imageView;
    //用来保存选中的图片数据
    private Bitmap selectedImage = null;
    //创建图片识别对象, MobileNet 是在 com.example.android_mobilenet 中定义的类
    private MobileNet mobileNet = new MobileNet();

    //第一次创建activity的时候调用这个方法
    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 初始化图像识别模型
        boolean ret_init = mobileNet.Init(getAssets());
        if(!ret_init){
            Log.e("MainActivity","mobilenet Init failed");
        }

        // 展示结果的 TextView
        infoResult = (TextView)findViewById(R.id.infoResult);
        // 显示图片的 ImageView
        imageView = (ImageView)findViewById(R.id.imageView);

        // 选图按钮
        Button buttonImage = (Button)findViewById(R.id.buttonImage);
        buttonImage.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                //调用图库获取本地所有图片
                Intent i = new Intent(Intent.ACTION_PICK);//??
                i.setType("image/*");//??
                startActivityForResult(i,SELECT_IMAGE);//??
            }
        });

        //点击CPU识别按钮触发的事件
        Button buttonDetectCPU = (Button)findViewById(R.id.buttonDetectCPU);
        buttonDetectCPU.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                if(selectedImage == null){
                    return;
                }
                //调用CPU来实现图片分类,获取识别的结果
                String result = mobileNet.ImagePredict(selectedImage,false);

                if(result == null){
                    infoResult.setText("detect failed");
                }else{
                    infoResult.setText(result);
                }
            }
        });

        //点击GPU识别按钮触发的事件
        Button buttonDetectGPU = (Button)findViewById(R.id.buttonDetectGPU);
        buttonDetectGPU.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                if(selectedImage == null){
                    return;
                }
                // 调用GPU来识别图片
                String result = mobileNet.ImagePredict(selectedImage,true);

                if(result == null){
                    infoResult.setText("detect failed");
                }else{
                    infoResult.setText(result);
                }
            }
        });
    }

    @Override //将图片添加到 ImageView 控件中显示出来
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && null != data) {
            //获取到选中的图片数据
            Uri selectedImageUri = data.getData();
            try
            {
                if (requestCode == SELECT_IMAGE) {
                    //解码图片数据,将图片数据转换为Bitmap格式
                    Bitmap bitmap = decodeUri(selectedImageUri);

                    Bitmap rgba = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                    selectedImage = Bitmap.createBitmap(rgba);//???
                    rgba.recycle();//???
                    //将图片显示在图片控件上
                    imageView.setImageBitmap(bitmap);
                }
            }
            catch (FileNotFoundException e)
            {
                Log.e("MainActivity", "FileNotFoundException");
                return;
            }
        }
    }
    //@Override 下面函数为何没有 Override
    private Bitmap decodeUri(Uri selectedImage) throws FileNotFoundException
    {
        // 解码图片,获取图片的size
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inJustDecodeBounds = true;
        BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage),
                null, o);

        // 将图片短边缩放到指定尺寸以下
        final int REQUIRED_SIZE = 400;

        // 每次对图片的size进行,50%的缩放,直到短边长度小于指定尺寸
        int width_tmp = o.outWidth, height_tmp = o.outHeight;
        int scale = 1;
        while (true) {
            if (width_tmp / 2 < REQUIRED_SIZE
                    || height_tmp / 2 < REQUIRED_SIZE) {
                break;
            }
            width_tmp /= 2;
            height_tmp /= 2;
            scale *= 2;
        }

        // 下采样图片,缩放图片
        BitmapFactory.Options o2 = new BitmapFactory.Options();
        o2.inSampleSize = scale;
        return BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage),
                null, o2);
    }

}
