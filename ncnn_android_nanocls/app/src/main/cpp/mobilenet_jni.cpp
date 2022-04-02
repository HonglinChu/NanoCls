#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>

//#include "gpu.h"
#include "net.h"
#include "benchmark.h"
#include "mobilenet1.0.id.h"

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

// 用来存放类别名称
static std::vector<std::string> mobilenet_words;

// 创建一个网络对象
static ncnn::Net mobilenet;

// 将类别字符串转换为vector,便于获取类别名称
static std::vector<std::string> split_string(const std::string& str, const std::string& delimiter)
{
    std::vector<std::string> strings;

    std::string::size_type pos = 0; //为了避免溢出，保存一个 stirng 对象 size 的最安全的方法就是使用标准库类型 string::size_type。
    std::string::size_type prev = 0;
    // 从下标prev(0)开始从str，字符串中寻找delimiter的位置,并返回位置
    // npos可以表示string的结束位子，是string::type_size 类型的，也就是find（）返回的类型。
    // find函数在找不到指定值得情况下会返回 string::npos
    while ((pos = str.find(delimiter, prev)) != std::string::npos)
    {
        // 裁剪字符串(去除换行符),将类别的名称存到 strings 中
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    // 2. 解释：str.substr(pos,n) 返回一个string，包含s中从pos开始的n个字符的拷贝（的默认值是0，n的默认值是s.size() - pos，即不加参数会默认拷贝整个s）
    strings.push_back(str.substr(prev));

    return strings;
}

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
   //在Android studio的Logcat输出日志信息
   __android_log_print(ANDROID_LOG_DEBUG, "MobileNet", "JNI_OnLoad");
   //创建一个GPU实例,以便于后面使用GPU进行推理
   //ncnn::create_gpu_instance();
   return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
   __android_log_print(ANDROID_LOG_DEBUG, "MobileNet", "JNI_OnUnload");
   //删除GPU实例,释放占用的内存
   //ncnn::destroy_gpu_instance();
}

JNIEXPORT jboolean JNICALL Java_com_example_ncnn_1android_1nanocls_MobileNet_Init(JNIEnv* env, jobject thiz,
        jobject assetManager)
{
    //网络相关设置
    ncnn::Option opt;
    opt.lightmode = true; //轻量级模式在网络推理中会不断地进行垃圾回收，顾名思义就是释放垃圾占用的空间，防止内存泄漏。对内存堆中已经死亡或者长时间没有使用的对象进行清理和回收
    opt.num_threads = 4;  //线程数量
    opt.blob_allocator = &g_blob_pool_allocator; // 内存分配器，对基本数据结构blob的内存分配  
    opt.workspace_allocator = &g_workspace_pool_allocator; //对计算空间worksapce的内存分配 

    // use vulkan compute
    //    if (ncnn::get_gpu_count() != 0)
    //        opt.use_vulkan_compute = true;

    //获取java中assert对象,用来访问assert目录下的资源文件 在JAVA中通过Context.getAssets()轻松获得AssetManager
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    //设置网络参数
    mobilenet.opt = opt;

    // init param
    {
        // 加载网络结构  nanocls_shufflenetv2_garbage_sim-opt.param
        // int ret = mobilenet.load_param_bin(mgr, "mobilenet1.0.param.bin");

        // 垃圾分类
        // int ret = mobilenet.load_param(mgr, "nanocls_shufflenetv2_garbage_sim-opt.param");
        int ret = mobilenet.load_param(mgr, "nanocls_mobilenetv2_garbage_sim-opt.param");

        // 判断网络结构是否加载成功,返回0表示加载成功
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "MobileNet", "load_param_bin failed");
            return JNI_FALSE;
        }
    }
    // init bin
    {
        //垃圾分类
        //int ret = mobilenet.load_model(mgr, "nanocls_shufflenetv2_garbage_sim-opt.bin");
        int ret = mobilenet.load_model(mgr, "nanocls_mobilenetv2_garbage_sim-opt.bin");

        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "MobileNet", "load_model failed");
            return JNI_FALSE;
        }
    }

    // init words
    {
        // 打开assert目录下的文件， 管理器， 文件路径，MODE
        AAsset* assetfile = AAssetManager_open(mgr, "synset_words.txt", AASSET_MODE_BUFFER);
        if (!assetfile)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "MobileNet", "open synset_words.txt failed");
            return JNI_FALSE;
        }
        // 获取文件的长度
        int len = AAsset_getLength(assetfile);

        //用来存放类别名称
        std::string words_buffer;
        words_buffer.resize(len);

        // 读取文件的内容  AAsset_getBuffer()
        int ret = AAsset_read(assetfile, (void*)words_buffer.data(), len);

        //关闭文件流
        AAsset_close(assetfile);

        if (ret != len)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "MobileNet", "read synset_words.txt failed");
            return JNI_FALSE;
        }
        //分割字符串,通过每行的换行符"\n"来分割不同类别的名称
        mobilenet_words = split_string(words_buffer, "\n");
    }

    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL Java_com_example_ncnn_1android_1nanocls_MobileNet_ImagePredict(JNIEnv* env, jobject thiz, jobject bitmap, jboolean use_gpu)
{
    //是否使用GPU,并且判断设备是否有GPU
//    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0)
//    {
//        return env->NewStringUTF("no vulkan capable gpu");
//    }
    //用来记录模型开始推理的时间
    double start_time = ncnn::get_current_time();
    //记录Android中Bitmap对象的信息
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    int width = info.width;
    int height = info.height;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;
    //缩放后图片的宽和高
    int wsize = 256;
    int hsize = 256;
    //等比例缩放图片,将图片的短边缩放到256,计算图片缩放后的大小
    if(height > width){
        hsize = int(height * wsize / width);
    }else{
        wsize = int(width * hsize / height);
    }
    //将bitmap的图片转为Mat,通道顺序为RGB
    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env,bitmap,
            ncnn::Mat::PIXEL_RGB,wsize,hsize);
    //图片中心裁剪
    ncnn::Mat copy_center_in;
    //图片裁剪后的大小
    //int cropSize = 224;

    // 垃圾分类
    int cropSize = 128;

    //计算裁剪图片的坐标位置
    int x = (wsize - cropSize) / 2;
    int y = (hsize - cropSize) / 2;
    //中心裁剪图片,将输入图片的size固定为224×224
    ncnn::copy_cut_border(in,copy_center_in,y,y,x,x);
    __android_log_print(ANDROID_LOG_DEBUG, "MobileNet",
            "%d,%d",copy_center_in.w,copy_center_in.h);

    // mobilenet 网络前向推理
    std::vector<float> cls_scores;
    {
        //对输入的图片像素的均值和方差进行归一化
        const float mean_vals[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
        const float norm_vals[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f };
        copy_center_in.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = mobilenet.create_extractor();
        //设置是否使用GPU
        //ex.set_vulkan_compute(use_gpu);

        //设置网络的输入 垃圾分类
        ex.input("input", copy_center_in);

        //获取网络的输出
        ncnn::Mat out;
        ex.extract("results", out);

        //创建一个Softmax层，将输出转换为概率
        {
            ncnn::Layer* softmax = ncnn::create_layer("Softmax");

            ncnn::ParamDict pd;
            softmax->load_param(pd);//定义的常量层的参数

            softmax->forward_inplace(out, mobilenet.opt);

            delete softmax;
        }

        out = out.reshape(out.w * out.h * out.c);

        cls_scores.resize(out.w);
        //从ncnn的Mat中获取输出结果
        for (int j = 0; j < out.w; j++)
        {
            cls_scores[j] = out[j];
        }
    }
    //计算概率最大的类别
    int top_class = 0;
    float max_score = 0.f;
    for (size_t i=0; i<cls_scores.size(); i++)
    {
        float s = cls_scores[i];
        if (s > max_score)
        {
            top_class = i;
            max_score = s;
        }
    }
    //获取类别对应的名称
    const std::string& word = mobilenet_words[top_class];
    char tmp[32];

    //将类别输出概率的float转为string
    sprintf(tmp, "%.2f", max_score);

    // 垃圾分类
    std::string result_str = std::string(word.c_str() + 2) + " = " + tmp;

    //计算前向推理耗时
    double elasped = ncnn::get_current_time() - start_time;
    char time_tmp[32];
    sprintf(time_tmp,"time:%.2fms",elasped);
    result_str = result_str + "\n" + time_tmp;
    //将string转换为 jstring 类型,传给java
    jstring result = env->NewStringUTF(result_str.c_str());
    //从Android studio的控制台输出日志信息便于调试
    __android_log_print(ANDROID_LOG_DEBUG, "MobileNet","%.2fms predict", elasped);

    return result;
}

}