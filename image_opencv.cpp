#include "image_opencv.h"

#ifdef OPENCV
#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <omp.h>
#include <fstream>
#include <algorithm>
#include <time.h>
#include <stdio.h>
#include <Windows.h>

#include <opencv2/core/version.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

// includes for OpenCV >= 3.x
#ifndef CV_VERSION_EPOCH
#include <opencv2/core/types.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#endif

// OpenCV includes for OpenCV 2.x
#ifdef CV_VERSION_EPOCH
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/types_c.h>
#include <opencv2/core/version.hpp>
#endif

//using namespace cv;

using std::cerr;
using std::endl;

#ifdef DEBUG
#define OCV_D "d"
#else
#define OCV_D
#endif//DEBUG


// OpenCV libraries
#ifndef CV_VERSION_EPOCH
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION) OCV_D
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
#endif    // USE_CMAKE_LIBS
#else   // CV_VERSION_EPOCH
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH)"" CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR) OCV_D
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#endif    // USE_CMAKE_LIBS
#endif    // CV_VERSION_EPOCH

#include "http_stream.h"

#ifndef CV_RGB
#define CV_RGB(r, g, b) cvScalar( (b), (g), (r), 0 )
#endif

#ifndef CV_FILLED
#define CV_FILLED cv::FILLED
#endif

#ifndef CV_AA
#define CV_AA cv::LINE_AA
#endif

extern "C" {

    struct mat_cv : cv::Mat { int a[0]; };
    struct cap_cv : cv::VideoCapture { int a[0]; };
    struct write_cv : cv::VideoWriter { int a[0]; };

    struct CvPBGMMGaussian
    {
        float sigma;
        float muR;
        float muG;
        float muB;
        float weight;
    };
    typedef struct CvPixelBackgroundGMM
    {
        /////////////////////////
        //very important parameters - things you will change
        ////////////////////////
        float fAlphaT;
        //alpha - speed of update - if the time interval you want to average over is T
        //set alpha=1/T. It is also usefull at start to make T slowly increase
        //from 1 until the desired T
        float fTb;
        //Tb - threshold on the squared Mahalan. dist. to decide if it is well described
        //by the background model or not. Related to Cthr from the paper.
        //This does not influence the update of the background. A typical value could be 4 sigma
        //and that is Tb=4*4=16;

        /////////////////////////
        //less important parameters - things you might change but be carefull
        ////////////////////////
        float fTg;
        //Tg - threshold on the squared Mahalan. dist. to decide 
        //when a sample is close to the existing components. If it is not close
        //to any a new component will be generated. I use 3 sigma => Tg=3*3=9.
        //Smaller Tg leads to more generated components and higher Tg might make
        //lead to small number of components but they can grow too large
        float fTB;//1-cf from the paper
        //TB - threshold when the component becomes significant enough to be included into
        //the background model. It is the TB=1-cf from the paper. So I use cf=0.1 => TB=0.
        //For alpha=0.001 it means that the mode should exist for approximately 105 frames before
        //it is considered foreground
        float fSigma;
        //initial standard deviation  for the newly generated components. 
        //It will will influence the speed of adaptation. A good guess should be made. 
        //A simple way is to estimate the typical standard deviation from the images.
        //I used here 10 as a reasonable value
        float fCT;//CT - complexity reduction prior
        //this is related to the number of samples needed to accept that a component
        //actually exists. We use CT=0.05 of all the samples. By setting CT=0 you get
        //the standard Stauffer&Grimson algorithm (maybe not exact but very similar)

        //even less important parameters
        int nM;//max number of modes - const - 4 is usually enough

        //shadow detection parameters
        int bShadowDetection;//do shadow detection
        float fTau;
        // Tau - shadow threshold. The shadow is detected if the pixel is darker
        //version of the background. Tau is a threshold on how much darker the shadow can be.
        //Tau= 0.5 means that if pixel is more than 2 times darker then it is not shadow
        //See: Prati,Mikic,Trivedi,Cucchiarra,"Detecting Moving Shadows...",IEEE PAMI,2003.

        //data
        int nNBands;//only RGB now ==3
        int nWidth;//image size
        int nHeight;
        int nSize;
        // dynamic array for the mixture of Gaussians
        CvPBGMMGaussian* rGMM;
        unsigned char* rnUsedModes;//number of Gaussian components per pixel
        int bRemoveForeground;
    };


// ====================================================================
// cv::Mat
// ====================================================================
    image mat_to_image(cv::Mat mat);
    cv::Mat image_to_mat(image img);
//    image ipl_to_image(mat_cv* src);
//    mat_cv *image_to_ipl(image img);
//    cv::Mat ipl_to_mat(IplImage *ipl);
//    IplImage *mat_to_ipl(cv::Mat mat);


mat_cv *load_image_mat_cv(const char *filename, int flag)
{
    try {
        cv::Mat *mat_ptr = new cv::Mat();
        cv::Mat &mat = *mat_ptr;
        mat = cv::imread(filename, flag);
        if (mat.empty())
        {
            delete mat_ptr;
            std::string shrinked_filename = filename;
            if (shrinked_filename.length() > 1024) {
                shrinked_filename += "name is too long: ";
                shrinked_filename.resize(1024);
            }
            cerr << "Cannot load image " << shrinked_filename << std::endl;
            std::ofstream bad_list("bad.list", std::ios::out | std::ios::app);
            bad_list << shrinked_filename << std::endl;
            //if (check_mistakes) getchar();
            return NULL;
        }
        if (mat.channels() == 3) cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        else if (mat.channels() == 4) cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGRA);

        return (mat_cv *)mat_ptr;
    }
    catch (...) {
        cerr << "OpenCV exception: load_image_mat_cv \n";
    }
    return NULL;
}
// ----------------------------------------

cv::Mat load_image_mat(char *filename, int channels)
{
    int flag = cv::IMREAD_UNCHANGED;
    if (channels == 0) flag = cv::IMREAD_COLOR;
    else if (channels == 1) flag = cv::IMREAD_GRAYSCALE;
    else if (channels == 3) flag = cv::IMREAD_COLOR;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }
    //flag |= IMREAD_IGNORE_ORIENTATION;    // un-comment it if you want

    cv::Mat *mat_ptr = (cv::Mat *)load_image_mat_cv(filename, flag);

    if (mat_ptr == NULL) {
        return cv::Mat();
    }
    cv::Mat mat = *mat_ptr;
    delete mat_ptr;

    return mat;
}
// ----------------------------------------

image load_image_cv(char *filename, int channels)
{
    cv::Mat mat = load_image_mat(filename, channels);

    if (mat.empty()) {
        return make_image(10, 10, channels);
    }
    return mat_to_image(mat);
}
// ----------------------------------------

image load_image_resize(char *filename, int w, int h, int c, image *im)
{
    image out;
    try {
        cv::Mat loaded_image = load_image_mat(filename, c);

        *im = mat_to_image(loaded_image);

        cv::Mat resized(h, w, CV_8UC3);
        cv::resize(loaded_image, resized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
        out = mat_to_image(resized);
    }
    catch (...) {
        cerr << " OpenCV exception: load_image_resize() can't load image %s " << filename << " \n";
        out = make_image(w, h, c);
        *im = make_image(w, h, c);
    }
    return out;
}
// ----------------------------------------

int get_width_mat(mat_cv *mat)
{
    if (mat == NULL) {
        cerr << " Pointer is NULL in get_width_mat() \n";
        return 0;
    }
    return ((cv::Mat *)mat)->cols;
}
// ----------------------------------------

int get_height_mat(mat_cv *mat)
{
    if (mat == NULL) {
        cerr << " Pointer is NULL in get_height_mat() \n";
        return 0;
    }
    return ((cv::Mat *)mat)->rows;
}
// ----------------------------------------

void release_mat(mat_cv **mat)
{
    try {
        cv::Mat **mat_ptr = (cv::Mat **)mat;
        if (*mat_ptr) delete *mat_ptr;
        *mat_ptr = NULL;
    }
    catch (...) {
        cerr << "OpenCV exception: release_mat \n";
    }
}

// ====================================================================
// IplImage
// ====================================================================
/*
int get_width_cv(mat_cv *ipl_src)
{
    IplImage *ipl = (IplImage *)ipl_src;
    return ipl->width;
}
// ----------------------------------------

int get_height_cv(mat_cv *ipl_src)
{
    IplImage *ipl = (IplImage *)ipl_src;
    return ipl->height;
}
// ----------------------------------------

void release_ipl(mat_cv **ipl)
{
    IplImage **ipl_img = (IplImage **)ipl;
    if (*ipl_img) cvReleaseImage(ipl_img);
    *ipl_img = NULL;
}
// ----------------------------------------

// ====================================================================
// image-to-ipl, ipl-to-image, image_to_mat, mat_to_image
// ====================================================================

mat_cv *image_to_ipl(image im)
{
    int x, y, c;
    IplImage *disp = cvCreateImage(cvSize(im.w, im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for (y = 0; y < im.h; ++y) {
        for (x = 0; x < im.w; ++x) {
            for (c = 0; c < im.c; ++c) {
                float val = im.data[c*im.h*im.w + y*im.w + x];
                disp->imageData[y*step + x*im.c + c] = (unsigned char)(val * 255);
            }
        }
    }
    return (mat_cv *)disp;
}
// ----------------------------------------

image ipl_to_image(mat_cv* src_ptr)
{
    IplImage* src = (IplImage*)src_ptr;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep;
    int i, j, k;

    for (i = 0; i < h; ++i) {
        for (k = 0; k < c; ++k) {
            for (j = 0; j < w; ++j) {
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k] / 255.;
            }
        }
    }
    return im;
}
// ----------------------------------------

cv::Mat ipl_to_mat(IplImage *ipl)
{
    Mat m = cvarrToMat(ipl, true);
    return m;
}
// ----------------------------------------

IplImage *mat_to_ipl(cv::Mat mat)
{
    IplImage *ipl = new IplImage;
    *ipl = mat;
    return ipl;
}
// ----------------------------------------
*/

cv::Mat image_to_mat(image img)
{
    int channels = img.c;
    int width = img.w;
    int height = img.h;
    cv::Mat mat = cv::Mat(height, width, CV_8UC(channels));
    int step = mat.step;

    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            for (int c = 0; c < img.c; ++c) {
                float val = img.data[c*img.h*img.w + y*img.w + x];
                mat.data[y*step + x*img.c + c] = (unsigned char)(val * 255);
            }
        }
    }
    return mat;
}
// ----------------------------------------

image mat_to_image(cv::Mat mat)
{
    int w = mat.cols;
    int h = mat.rows;
    int c = mat.channels();
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)mat.data;
    int step = mat.step;
    for (int y = 0; y < h; ++y) {
        for (int k = 0; k < c; ++k) {
            for (int x = 0; x < w; ++x) {
                //uint8_t val = mat.ptr<uint8_t>(y)[c * x + k];
                //uint8_t val = mat.at<Vec3b>(y, x).val[k];
                //im.data[k*w*h + y*w + x] = val / 255.0f;

                im.data[k*w*h + y*w + x] = data[y*step + x*c + k] / 255.0f;
            }
        }
    }
    return im;
}

image mat_to_image_cv(mat_cv *mat)
{
    return mat_to_image(*mat);
}

// ====================================================================
// Window
// ====================================================================
void create_window_cv(char const* window_name, int full_screen, int width, int height)
{
    try {
        int window_type = cv::WINDOW_NORMAL;
#ifdef CV_VERSION_EPOCH // OpenCV 2.x
        if (full_screen) window_type = CV_WINDOW_FULLSCREEN;
#else
        if (full_screen) window_type = cv::WINDOW_FULLSCREEN;
#endif
        cv::namedWindow(window_name, window_type);
        cv::moveWindow(window_name, 0, 0);
        cv::resizeWindow(window_name, width, height);
    }
    catch (...) {
        cerr << "OpenCV exception: create_window_cv \n";
    }
}
// ----------------------------------------

void destroy_all_windows_cv()
{
    try {
        cv::destroyAllWindows();
    }
    catch (...) {
        cerr << "OpenCV exception: destroy_all_windows_cv \n";
    }
}
// ----------------------------------------

int wait_key_cv(int delay)
{
    try {
        return cv::waitKey(delay);
    }
    catch (...) {
        cerr << "OpenCV exception: wait_key_cv \n";
    }
    return -1;
}
// ----------------------------------------

int wait_until_press_key_cv()
{
    return wait_key_cv(0);
}
// ----------------------------------------

void make_window(char *name, int w, int h, int fullscreen)
{
    try {
        cv::namedWindow(name, cv::WINDOW_NORMAL);
        if (fullscreen) {
#ifdef CV_VERSION_EPOCH // OpenCV 2.x
            cv::setWindowProperty(name, cv::WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
#else
            cv::setWindowProperty(name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
#endif
        }
        else {
            cv::resizeWindow(name, w, h);
            if (strcmp(name, "Demo") == 0) cv::moveWindow(name, 0, 0);
        }
    }
    catch (...) {
        cerr << "OpenCV exception: make_window \n";
    }
}
// ----------------------------------------

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}
// ----------------------------------------

void show_image_cv(image p, const char *name)
{
    try {
        image copy = copy_image(p);
        constrain_image(copy);

        cv::Mat mat = image_to_mat(copy);
        if (mat.channels() == 3) cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        else if (mat.channels() == 4) cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGR);
        cv::namedWindow(name, cv::WINDOW_NORMAL);
        cv::imshow(name, mat);
        free_image(copy);
    }
    catch (...) {
        cerr << "OpenCV exception: show_image_cv \n";
    }
}
// ----------------------------------------

/*
void show_image_cv_ipl(mat_cv *disp, const char *name)
{
    if (disp == NULL) return;
    char buff[256];
    sprintf(buff, "%s", name);
    cv::namedWindow(buff, WINDOW_NORMAL);
    cvShowImage(buff, disp);
}
// ----------------------------------------
*/

void show_image_mat(mat_cv *mat_ptr, const char *name)
{
    try {
        if (mat_ptr == NULL) return;
        cv::Mat &mat = *(cv::Mat *)mat_ptr;
        cv::namedWindow(name, cv::WINDOW_NORMAL);
        cv::imshow(name, mat);
    }
    catch (...) {
        cerr << "OpenCV exception: show_image_mat \n";
    }
}

// ====================================================================
// Video Writer
// ====================================================================
write_cv *create_video_writer(char *out_filename, char c1, char c2, char c3, char c4, int fps, int width, int height, int is_color)
{
    try {
    cv::VideoWriter * output_video_writer =
#ifdef CV_VERSION_EPOCH
        new cv::VideoWriter(out_filename, CV_FOURCC(c1, c2, c3, c4), fps, cv::Size(width, height), is_color);
#else
        new cv::VideoWriter(out_filename, cv::VideoWriter::fourcc(c1, c2, c3, c4), fps, cv::Size(width, height), is_color);
#endif

    return (write_cv *)output_video_writer;
    }
    catch (...) {
        cerr << "OpenCV exception: create_video_writer \n";
    }
    return NULL;
}

void write_frame_cv(write_cv *output_video_writer, mat_cv *mat)
{
    try {
        cv::VideoWriter *out = (cv::VideoWriter *)output_video_writer;
        out->write(*mat);
    }
    catch (...) {
        cerr << "OpenCV exception: write_frame_cv \n";
    }
}

void release_video_writer(write_cv **output_video_writer)
{
    try {
        if (output_video_writer) {
            std::cout << " closing...";
            cv::VideoWriter *out = *(cv::VideoWriter **)output_video_writer;
            out->release();
            delete out;
            output_video_writer = NULL;
            std::cout << " closed!";
        }
        else {
            cerr << "OpenCV exception: output_video_writer isn't created \n";
        }
    }
    catch (...) {
        cerr << "OpenCV exception: release_video_writer \n";
    }
}

/*
void *open_video_stream(const char *f, int c, int w, int h, int fps)
{
    VideoCapture *cap;
    if(f) cap = new VideoCapture(f);
    else cap = new VideoCapture(c);
    if(!cap->isOpened()) return 0;
    if(w) cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
    if(h) cap->set(CV_CAP_PROP_FRAME_HEIGHT, w);
    if(fps) cap->set(CV_CAP_PROP_FPS, w);
    return (void *) cap;
}


image get_image_from_stream(void *p)
{
    VideoCapture *cap = (VideoCapture *)p;
    Mat m;
    *cap >> m;
    if(m.empty()) return make_empty_image(0,0,0);
    return mat_to_image(m);
}

int show_image_cv(image im, const char* name, int ms)
{
    Mat m = image_to_mat(im);
    imshow(name, m);
    int c = waitKey(ms);
    if (c != -1) c = c%256;
    return c;
}
*/


// ====================================================================
// Video Capture
// ====================================================================

cap_cv* get_capture_video_stream(const char *path) {
    cv::VideoCapture* cap = NULL;
    try {
        cap = new cv::VideoCapture(path);
    }
    catch (...) {
        cerr << " OpenCV exception: video-stream " << path << " can't be opened! \n";
    }
    return (cap_cv*)cap;
}
// ----------------------------------------

cap_cv* get_capture_webcam(int index)
{
    cv::VideoCapture* cap = NULL;
    try {
        cap = new cv::VideoCapture(index);
        //cap->set(CV_CAP_PROP_FRAME_WIDTH, 1280);
        //cap->set(CV_CAP_PROP_FRAME_HEIGHT, 960);
    }
    catch (...) {
        cerr << " OpenCV exception: Web-camera " << index << " can't be opened! \n";
    }
    return (cap_cv*)cap;
}
// ----------------------------------------

void release_capture(cap_cv* cap)
{
    try {
        cv::VideoCapture *cpp_cap = (cv::VideoCapture *)cap;
        delete cpp_cap;
    }
    catch (...) {
        cerr << " OpenCV exception: cv::VideoCapture " << cap << " can't be released! \n";
    }
}
// ----------------------------------------

mat_cv* get_capture_frame_cv(cap_cv *cap) {
    cv::Mat *mat = new cv::Mat();
    try {
        if (cap) {
            cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
            if (cpp_cap.isOpened())
            {
                cpp_cap >> *mat;
            }
            else std::cout << " Video-stream stopped! \n";
        }
        else cerr << " cv::VideoCapture isn't created \n";
    }
    catch (...) {
        std::cout << " OpenCV exception: Video-stream stoped! \n";
    }
    return (mat_cv *)mat;
}
// ----------------------------------------

int get_stream_fps_cpp_cv(cap_cv *cap)
{
    int fps = 25;
    try {
        cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
#ifndef CV_VERSION_EPOCH    // OpenCV 3.x
        fps = cpp_cap.get(cv::CAP_PROP_FPS);
#else                        // OpenCV 2.x
        fps = cpp_cap.get(CV_CAP_PROP_FPS);
#endif
    }
    catch (...) {
        cerr << " Can't get FPS of source videofile. For output video FPS = 25 by default. \n";
    }
    return fps;
}
// ----------------------------------------

double get_capture_property_cv(cap_cv *cap, int property_id)
{
    try {
        cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
        return cpp_cap.get(property_id);
    }
    catch (...) {
        cerr << " OpenCV exception: Can't get property of source video-stream. \n";
    }
    return 0;
}
// ----------------------------------------

double get_capture_frame_count_cv(cap_cv *cap)
{
    try {
        cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
#ifndef CV_VERSION_EPOCH    // OpenCV 3.x
        return cpp_cap.get(cv::CAP_PROP_FRAME_COUNT);
#else                        // OpenCV 2.x
        return cpp_cap.get(CV_CAP_PROP_FRAME_COUNT);
#endif
    }
    catch (...) {
        cerr << " OpenCV exception: Can't get CAP_PROP_FRAME_COUNT of source videofile. \n";
    }
    return 0;
}
// ----------------------------------------

int set_capture_property_cv(cap_cv *cap, int property_id, double value)
{
    try {
        cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
        return cpp_cap.set(property_id, value);
    }
    catch (...) {
        cerr << " Can't set property of source video-stream. \n";
    }
    return false;
}
// ----------------------------------------

int set_capture_position_frame_cv(cap_cv *cap, int index)
{
    try {
        cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
#ifndef CV_VERSION_EPOCH    // OpenCV 3.x
        return cpp_cap.set(cv::CAP_PROP_POS_FRAMES, index);
#else                        // OpenCV 2.x
        return cpp_cap.set(CV_CAP_PROP_POS_FRAMES, index);
#endif
    }
    catch (...) {
        cerr << " Can't set CAP_PROP_POS_FRAMES of source videofile. \n";
    }
    return false;
}
// ----------------------------------------



// ====================================================================
// ... Video Capture
// ====================================================================

image get_image_from_stream_cpp(cap_cv *cap)
{
    cv::Mat *src = new cv::Mat();
    static int once = 1;
    if (once) {
        once = 0;
        do {
            src = get_capture_frame_cv(cap);
            if (!src) return make_empty_image(0, 0, 0);
        } while (src->cols < 1 || src->rows < 1 || src->channels() < 1);
        printf("Video stream: %d x %d \n", src->cols, src->rows);
    }
    else
        src = get_capture_frame_cv(cap);

    if (!src) return make_empty_image(0, 0, 0);
    image im = mat_to_image(*src);
    rgbgr_image(im);
    return im;
}
// ----------------------------------------

int wait_for_stream(cap_cv *cap, cv::Mat* src, int dont_close)
{
    if (!src) {
        if (dont_close) src = new cv::Mat(416, 416, CV_8UC(3)); // cvCreateImage(cvSize(416, 416), IPL_DEPTH_8U, 3);
        else return 0;
    }
    if (src->cols < 1 || src->rows < 1 || src->channels() < 1) {
        if (dont_close) {
            delete src;// cvReleaseImage(&src);
            int z = 0;
            for (z = 0; z < 20; ++z) {
                get_capture_frame_cv(cap);
                delete src;// cvReleaseImage(&src);
            }
            src = new cv::Mat(416, 416, CV_8UC(3)); // cvCreateImage(cvSize(416, 416), IPL_DEPTH_8U, 3);
        }
        else return 0;
    }
    return 1;
}
// ----------------------------------------

image get_image_from_stream_resize(cap_cv *cap, int w, int h, int c, mat_cv** in_img, int dont_close)
{
    c = c ? c : 3;
    cv::Mat *src = NULL;

    static int once = 1;
    if (once) {
        once = 0;
        do {
            src = get_capture_frame_cv(cap);
            if (!src) return make_empty_image(0, 0, 0);
        } while (src->cols < 1 || src->rows < 1 || src->channels() < 1);
        printf("Video stream: %d x %d \n", src->cols, src->rows);
    }
    else
        src = get_capture_frame_cv(cap);

    if (!wait_for_stream(cap, src, dont_close)) return make_empty_image(0, 0, 0);

    *(cv::Mat **)in_img = src;

    cv::Mat new_img = cv::Mat(h, w, CV_8UC(c));
    cv::resize(*src, new_img, new_img.size(), 0, 0, cv::INTER_LINEAR);
    if (c>1) cv::cvtColor(new_img, new_img, cv::COLOR_RGB2BGR);
    image im = mat_to_image(new_img);

    //show_image_cv(im, "im");
    //show_image_mat(*in_img, "in_img");
    return im;
}
// ----------------------------------------

image get_image_from_stream_letterbox(cap_cv *cap, int w, int h, int c, mat_cv** in_img, int dont_close)
{
    c = c ? c : 3;
    cv::Mat *src = NULL;
    static int once = 1;
    if (once) {
        once = 0;
        do {
            src = get_capture_frame_cv(cap);
            if (!src) return make_empty_image(0, 0, 0);
        } while (src->cols < 1 || src->rows < 1 || src->channels() < 1);
        printf("Video stream: %d x %d \n", src->cols, src->rows);
    }
    else
        src = get_capture_frame_cv(cap);

    if (!wait_for_stream(cap, src, dont_close)) return make_empty_image(0, 0, 0);   // passes (cv::Mat *)src while should be (cv::Mat **)src

    *in_img = (mat_cv *)new cv::Mat(src->rows, src->cols, CV_8UC(c));
    cv::resize(*src, **in_img, (*in_img)->size(), 0, 0, cv::INTER_LINEAR);

    if (c>1) cv::cvtColor(*src, *src, cv::COLOR_RGB2BGR);
    image tmp = mat_to_image(*src);
    image im = letterbox_image(tmp, w, h);
    free_image(tmp);
    release_mat((mat_cv **)&src);

    //show_image_cv(im, "im");
    //show_image_mat(*in_img, "in_img");
    return im;
}
// ----------------------------------------

// ====================================================================
// Image Saving
// ====================================================================
extern int stbi_write_png(char const *filename, int w, int h, int comp, const void  *data, int stride_in_bytes);
extern int stbi_write_jpg(char const *filename, int x, int y, int comp, const void  *data, int quality);

void save_mat_png(cv::Mat img_src, const char *name)
{
    cv::Mat img_rgb;
    if (img_src.channels() >= 3) cv::cvtColor(img_src, img_rgb, cv::COLOR_RGB2BGR);
    stbi_write_png(name, img_rgb.cols, img_rgb.rows, 3, (char *)img_rgb.data, 0);
}
// ----------------------------------------

void save_mat_jpg(cv::Mat img_src, const char *name)
{
    cv::Mat img_rgb;
    if (img_src.channels() >= 3) cv::cvtColor(img_src, img_rgb, cv::COLOR_RGB2BGR);
    stbi_write_jpg(name, img_rgb.cols, img_rgb.rows, 3, (char *)img_rgb.data, 80);
}
// ----------------------------------------


void save_cv_png(mat_cv *img_src, const char *name)
{
    cv::Mat* img = (cv::Mat* )img_src;
    save_mat_png(*img, name);
}
// ----------------------------------------

void save_cv_jpg(mat_cv *img_src, const char *name)
{
    cv::Mat* img = (cv::Mat*)img_src;
    save_mat_jpg(*img, name);
}
// ----------------------------------------


// ====================================================================
// Draw Detection
// ====================================================================
void draw_detections_cv_v3(mat_cv* mat, detection *dets, int num, float thresh, char **names, image **alphabet, int classes, int ext_output)
{
    try {
        cv::Mat *show_img = mat;
        int i, j;
        if (!show_img) return;
        static int frame_id = 0;
        frame_id++;

        for (i = 0; i < num; ++i) {
            char labelstr[4096] = { 0 };
            int class_id = -1;
            for (j = 0; j < classes; ++j) {
                int show = strncmp(names[j], "dont_show", 9);
                if (dets[i].prob[j] > thresh && show) {
                    if (class_id < 0) {
                        strcat(labelstr, names[j]);
                        class_id = j;
                        char buff[10];
                        sprintf(buff, " (%2.0f%%)", dets[i].prob[j] * 100);
                        strcat(labelstr, buff);
                    }
                    else {
                        strcat(labelstr, ", ");
                        strcat(labelstr, names[j]);
                    }
                    printf("%s: %.0f%% ", names[j], dets[i].prob[j] * 100);
                }
            }
            if (class_id >= 0 ) {
                int width = std::max(1.0f, show_img->rows * .002f);

                //if(0){
                //width = pow(prob, 1./2.)*10+1;
                //alphabet = 0;
                //}

                //printf("%d %s: %.0f%%\n", i, names[class_id], prob*100);
                int offset = class_id * 123457 % classes;
                float red = get_color(2, offset, classes);
                float green = get_color(1, offset, classes);
                float blue = get_color(0, offset, classes);
                float rgb[3];

                //width = prob*20+2;

                rgb[0] = red;
                rgb[1] = green;
                rgb[2] = blue;
                box b = dets[i].bbox;
                if (std::isnan(b.w) || std::isinf(b.w)) b.w = 0.5;
                if (std::isnan(b.h) || std::isinf(b.h)) b.h = 0.5;
                if (std::isnan(b.x) || std::isinf(b.x)) b.x = 0.5;
                if (std::isnan(b.y) || std::isinf(b.y)) b.y = 0.5;
                b.w = (b.w < 1) ? b.w : 1;
                b.h = (b.h < 1) ? b.h : 1;
                b.x = (b.x < 1) ? b.x : 1;
                b.y = (b.y < 1) ? b.y : 1;
                //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

                int left = (b.x - b.w / 2.)*show_img->cols;
                int right = (b.x + b.w / 2.)*show_img->cols;
                int top = (b.y - b.h / 2.)*show_img->rows;
                int bot = (b.y + b.h / 2.)*show_img->rows;

                if (left < 0) left = 0;
                if (right > show_img->cols - 1) right = show_img->cols - 1;
                if (top < 0) top = 0;
                if (bot > show_img->rows - 1) bot = show_img->rows - 1;

                //int b_x_center = (left + right) / 2;
                //int b_y_center = (top + bot) / 2;
                //int b_width = right - left;
                //int b_height = bot - top;
                //sprintf(labelstr, "%d x %d - w: %d, h: %d", b_x_center, b_y_center, b_width, b_height);

                float const font_size = show_img->rows / 1000.F;
                cv::Size const text_size = cv::getTextSize(labelstr, cv::FONT_HERSHEY_COMPLEX_SMALL, font_size, 1, 0);
                cv::Point pt1, pt2, pt_text, pt_text_bg1, pt_text_bg2;
                pt1.x = left;
                pt1.y = top;
                pt2.x = right;
                pt2.y = bot;

                pt_text.x = left;
                pt_text.y = top - 4;// 12;
                pt_text_bg1.x = left;
                pt_text_bg1.y = top - (3 + 18 * font_size);
                pt_text_bg2.x = right;
                if ((right - left) < text_size.width) pt_text_bg2.x = left + text_size.width;
                pt_text_bg2.y = top;
                cv::Scalar color;
                color.val[0] = red * 256;
                color.val[1] = green * 256;
                color.val[2] = blue * 256;

                // you should create directory: result_img
                //static int copied_frame_id = -1;
                //static IplImage* copy_img = NULL;
                //if (copied_frame_id != frame_id) {
                //    copied_frame_id = frame_id;
                //    if(copy_img == NULL) copy_img = cvCreateImage(cvSize(show_img->width, show_img->height), show_img->depth, show_img->nChannels);
                //    cvCopy(show_img, copy_img, 0);
                //}
                //static int img_id = 0;
                //img_id++;
                //char image_name[1024];
                //sprintf(image_name, "result_img/img_%d_%d_%d_%s.jpg", frame_id, img_id, class_id, names[class_id]);
                //CvRect rect = cvRect(pt1.x, pt1.y, pt2.x - pt1.x, pt2.y - pt1.y);
                //cvSetImageROI(copy_img, rect);
                //cvSaveImage(image_name, copy_img, 0);
                //cvResetImageROI(copy_img);

                cv::rectangle(*show_img, pt1, pt2, color, width, 8, 0);
                if (ext_output)
                    printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
                    (float)left, (float)top, b.w*show_img->cols, b.h*show_img->rows);
                else
                    printf("\n");

                cv::rectangle(*show_img, pt_text_bg1, pt_text_bg2, color, width, 8, 0);
                cv::rectangle(*show_img, pt_text_bg1, pt_text_bg2, color, CV_FILLED, 8, 0);    // filled
                cv::Scalar black_color = CV_RGB(0, 0, 0);
                cv::putText(*show_img, labelstr, pt_text, cv::FONT_HERSHEY_COMPLEX_SMALL, font_size, black_color, 2 * font_size, CV_AA);
                // cv::FONT_HERSHEY_COMPLEX_SMALL, cv::FONT_HERSHEY_SIMPLEX
            }
        }
        if (ext_output) {
            fflush(stdout);
        }
    }
    catch (...) {
        cerr << "OpenCV exception: draw_detections_cv_v3() \n";
    }
}
// ----------------------------------------

void pixel_counter(mat_cv* mat, detection *dets, int num, float thresh, char **names, int classes, int count)
{
    try {
        cv::Mat *show_img = mat;
        int i, j;
        if (!show_img) return;
        static int frame_id = 0;
        frame_id++;
        int class_id;

        std::vector<int> x1;
        std::vector<int> x2;
        std::vector<int> y1;
        std::vector<int> y2;
        std::vector<int> center_x;
        std::vector<int> center_y;

        for (i = 0; i < num; ++i)
        {
            char labelstr[4096] = { 0 };
            class_id = -1;
            for (j = 0; j < classes; ++j) {
                int show = strncmp(names[j], "dont_show", 9);
                if (dets[i].prob[j] > thresh && show) {
                    if (class_id < 0) {
                        strcat(labelstr, names[j]);
                        class_id = j;
                        char buff[10];
                        sprintf(buff, " (%2.0f%%)", dets[i].prob[j] * 100);
                        strcat(labelstr, buff);
                    }
                    else {
                        strcat(labelstr, ", ");
                        strcat(labelstr, names[j]);
                    }
                    //printf("%s: %.0f%% ", names[j], dets[i].prob[j] * 100);
                }
            }
            if (class_id >= 0)
            {
                int width = std::max(1.0f, show_img->rows * .002f);


                int offset = class_id * 123457 % classes;
                float red = get_color(2, offset, classes);
                float green = get_color(1, offset, classes);
                float blue = get_color(0, offset, classes);
                float rgb[3];

                //width = prob*20+2;

                rgb[0] = red;
                rgb[1] = green;
                rgb[2] = blue;
                box b = dets[i].bbox;
                if (std::isnan(b.w) || std::isinf(b.w)) b.w = 0.5;
                if (std::isnan(b.h) || std::isinf(b.h)) b.h = 0.5;
                if (std::isnan(b.x) || std::isinf(b.x)) b.x = 0.5;
                if (std::isnan(b.y) || std::isinf(b.y)) b.y = 0.5;
                b.w = (b.w < 1) ? b.w : 1;
                b.h = (b.h < 1) ? b.h : 1;
                b.x = (b.x < 1) ? b.x : 1;
                b.y = (b.y < 1) ? b.y : 1;
                //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

                int left = (b.x - b.w / 2.)*show_img->cols;
                int right = (b.x + b.w / 2.)*show_img->cols;
                int top = (b.y - b.h / 2.)*show_img->rows;
                int bot = (b.y + b.h / 2.)*show_img->rows;

                if (left < 0) left = 0;
                if (right > show_img->cols - 1) right = show_img->cols - 1;
                if (top < 0) top = 0;
                if (bot > show_img->rows - 1) bot = show_img->rows - 1;


                x1.push_back(left);
                y1.push_back(top);

                x2.push_back(right);
                y2.push_back(bot);

                center_x.push_back( (right + left) / 2);
                center_y.push_back((bot + top) / 2);

                // cv::FONT_HERSHEY_COMPLEX_SMALL, cv::FONT_HERSHEY_SIMPLEX
            } //  if (class_id >= 0) 
        } // for (i = 0; i < num; ++i)


        //printf("\n\nNumber of Boxes : %d\n\n", x1.size());

        cv::Mat src = *mat;

        src = src.clone();

        bool is_counted = false;

        LARGE_INTEGER Frequency;
        LARGE_INTEGER BeginTime;
        LARGE_INTEGER Endtime;
        double duringtime = 0;

        QueryPerformanceFrequency(&Frequency);
        QueryPerformanceCounter(&BeginTime);

        for (i = 0; i < x1.size(); i++)
        {
            bool overlap = false;

            for (j = 0; j < x1.size(); j++)
            {
                if (i == j)
                {
                    continue;
                }
                double length = pow(center_x[i] - center_x[j], 2) + pow(center_y[i] - center_y[j], 2);
                length = sqrt(length);

                if (length < 100)
                {
                    overlap = true;
                    break;
                }
            } // for (j = 0; j < x1.size(); j++)


            //double length = pow(src_center.x - center.x, 2) + pow(src_center.x - center.x, 2);

            //length = sqrt(length);

            float box_ratio = 3.0;

            float x_length = x2[i] - x1[i];
            float y_length = y2[i] - y1[i];

            bool cond1 = x_length / y_length >= box_ratio;
            bool cond2 = y_length / x_length >= box_ratio;

            bool cond = cond1 || cond2;

            if (overlap == false && cond == true)
            {

                printf("isolate\n");

                // printf("count : %d\n", count);

                int offset = class_id * 123457 % classes;
                float red = get_color(2, offset, classes);
                float green = get_color(1, offset, classes);
                float blue = get_color(0, offset, classes);

                float rgb[3];
                rgb[0] = red;
                rgb[1] = green;
                rgb[2] = blue;

                cv::Point point_pt1;
                cv::Point point_pt2;

                point_pt1.x = x1[i];
                point_pt1.y = y1[i];

                point_pt2.x = x2[i];
                point_pt2.y = y2[i];

                cv::Scalar color;
                color.val[0] = red * 256;
                color.val[1] = green * 256;
                color.val[2] = blue * 256;

                cv::rectangle(*show_img, point_pt1, point_pt2, color, 2.2, 8, 0);

                float size_rate = 0.15;

                point_pt1.x = x1[i] * (1.0 - size_rate);
                point_pt1.y = y1[i] * (1.0 - size_rate);

                point_pt2.x = x2[i] * (1.0 + size_rate);
                point_pt2.y = y2[i] * (1.0 + size_rate);

                if (point_pt2.x > show_img->cols - 1)
                {
                    point_pt2.x = show_img->cols - 1;
                }
                if (point_pt2.y > show_img->rows - 1)
                {
                    point_pt2.y = show_img->rows - 1;
                }


                //  printf("create binary frame\n");

                 // printf("pt1 : %d , %d\n", point_pt1.x, point_pt1.y);
                 // printf("pt2 : %d , %d\n", point_pt2.x, point_pt2.y);

                cv::Mat frame = src.clone();
                cv::rectangle(frame, point_pt1, point_pt2, cv::Scalar(0, 0, 0), 2.2, 8, 0);

                int _width = point_pt2.x - point_pt1.x;
                int _height = point_pt2.y - point_pt1.y;

                frame = cv::Mat(frame, cv::Rect(point_pt1, point_pt2));

                if (frame.channels() == 3)
                {
                    cv::cvtColor(frame, frame, cv::COLOR_RGB2GRAY);

                } // if (frame.channels() == 3)

                cv::threshold(frame, frame, 0, 255, CV_THRESH_OTSU);

                cv::Mat frame_c = frame.clone();



                cv::Mat img_labels, stats, centroids;

                int numOfLables = cv::connectedComponentsWithStats(frame, img_labels, stats, centroids, 8, CV_32S);

                int max_area = 0;
                int max_area_index = 1;

                for (int _i = 1; _i < numOfLables; _i++)
                {
                    int area = stats.at<int>(_i, cv::CC_STAT_AREA); // 레이블된 객체의 픽셀 수

                    if (area > max_area)
                    {
                        max_area = area;
                        max_area_index = _i;
                    } // if (area > max_area)

                } //for (int _i = 1; _i < numOfLables; _i++)

                point_pt1.x = x1[i];
                point_pt1.y = y1[i];

                point_pt2.x = x2[i];
                point_pt2.y = y2[i];

                cv::putText(*show_img, std::to_string(max_area), cv::Point(center_x[i], center_y[i]), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, color, 2);

                //if (frame_c.channels() == 1)
                //{
                //    cv::cvtColor(frame_c, frame_c, cv::COLOR_GRAY2BGR);
                //}

                //int l_left = stats.at<int>(max_area_index, cv::CC_STAT_LEFT); // 라벨링 영역의 가장 왼쪽좌표
                //int l_top = stats.at<int>(max_area_index, cv::CC_STAT_TOP); // 라벨링 영역의 가장 위쪽좌표
                //int l_width = stats.at<int>(max_area_index, cv::CC_STAT_WIDTH); // 라벨링 영역을 둘러싸는 박스의 너비
                //int l_height = stats.at<int>(max_area_index, cv::CC_STAT_HEIGHT); // 라벨링 영역을 둘러싸는 박스의 높이

                ////float box_size_rate = (l_width * l_height) / (x_length*y_length);
                //float box_size_rate = max_area / (x_length*y_length);

                is_counted = true;

                //cv::rectangle(frame_c, point_pt1, point_pt2, color, 2.2, 8, 0);

                //cv::rectangle(frame_c, cv::Point(l_left, l_top), cv::Point(l_left + l_width, l_top + l_height), cv::Scalar(0, 0, 255), 2.2, 8, 0);
                //cv::putText(frame_c, std::to_string(box_size_rate), cv::Point(20,20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, color, 2);

               // printf("save binary frame\n");
                //char fileName[100];
                //sprintf(fileName, "results/img_%d_binary_%d.jpg", count, i);
                //cv::imwrite(fileName, frame_c);

            } // if (overlap == false && cond == true && length < 500)
            else if(overlap == true)
            {
                cv::Point point_pt1, point_pt2;
                point_pt1.x = x1[i];
                point_pt1.y = y1[i];

                point_pt2.x = x2[i];
                point_pt2.y = y2[i];

            cv::rectangle(*show_img, point_pt1, point_pt2, cv::Scalar(0, 0, 255), 2.2, 8, 0);
            }
            else
            {
                cv::Point point_pt1, point_pt2;
                point_pt1.x = x1[i];
                point_pt1.y = y1[i];

                point_pt2.x = x2[i];
                point_pt2.y = y2[i];
                cv::rectangle(*show_img, point_pt1, point_pt2, cv::Scalar(255, 0, 0), 2.2, 8, 0);
            }

        } // for (i = 0; i < x1.size(); i++)

        QueryPerformanceCounter(&Endtime);
        duringtime = (double)(Endtime.QuadPart - BeginTime.QuadPart) / Frequency.QuadPart * 1000;

        printf("During time : %lf msec\n", duringtime);

        if (is_counted == true)
        {
            char fileName[100];
            sprintf(fileName, "results/img_%d.jpg", count);

            cv::Mat save_img = *mat;
            save_img = save_img.clone();

            cv::imwrite(fileName, save_img);


            // printf("\nsave img\n");
        }


    } // try
    catch (...) {
        cerr << "OpenCV exception: draw_detections_cv_v3() \n";
        printf("");
    }
}

int Count_Non_Zero(mat_cv* src)
{
    cv::Mat img = *src;

    if (img.empty())
    {
        return -1;

    }
    else
    {
        if (img.channels() == 3)
        {
            cv::cvtColor(img, img, CV_RGB2GRAY);
        }

        int thresh = cv::countNonZero(img);
        //img.release();
        return thresh;
    }
} // int Count_Non_Zero(mat_cv* src)

mat_cv * copyImg(mat_cv* src, mat_cv* det) {
    try {
        det = (mat_cv*)new cv::Mat(get_width_mat(src), get_height_mat(src), CV_8UC1);
        src->copyTo(*det);

        //cv::Mat &mat = *(cv::Mat *)src;
        //src->copyTo(*det);


        return det;
    }
    catch (...) {

    }
}

mat_cv* image_to_mat_cv(image img)
{
    cv::Mat *mat_ptr = new cv::Mat();
    cv::Mat &mat = *mat_ptr;
    *mat_ptr = image_to_mat(img);
    return (mat_cv *)mat_ptr;
}

mat_cv* diff_frame(mat_cv* src1, mat_cv* src2) {

    mat_cv* det1 = (mat_cv*)new cv::Mat(416, 416, CV_8UC1);
    mat_cv* det2 = (mat_cv*)new cv::Mat(416, 416, CV_8UC1);
    mat_cv* det3 = (mat_cv*)new cv::Mat(416, 416, CV_8UC1);

    src1->copyTo(*det1);
    src2->copyTo(*det2);
    src1->copyTo(*det3);

    cv::absdiff(*det1, *det2, *det3);

    //free(det1);
    //free(det2);

    release_mat(&det1);
    release_mat(&det2);

    return det3;
}

mat_cv* openning(mat_cv* src1)
{
    mat_cv* det1 = (mat_cv*)new cv::Mat(get_width_mat(src1), get_height_mat(src1), CV_8UC1);
    //mat_cv* det2 = (mat_cv*)new cv::Mat(get_width_mat(src1), get_height_mat(src1), CV_8UC1); // ???
    //src1->copyTo(*det1);
    //src1->copyTo(sample);

    //cv::Mat sample;
    //src1->copyTo(sample);
    cv::cvtColor(*src1, *det1, CV_RGB2GRAY);
    cv::cvtColor(*src1, *det1, CV_RGB2GRAY);

    cv::Mat kernel(7, 7, CV_8U, cv::Scalar(1));

    cv::morphologyEx(*src1, *det1, cv::MORPH_OPEN, kernel);

    kernel.release();


    release_mat(&src1);

    return det1;
}

mat_cv* closing(mat_cv* src1) {
    mat_cv* det1 = (mat_cv*)new cv::Mat(get_width_mat(src1), get_height_mat(src1), CV_8UC1);
    mat_cv* det2 = (mat_cv*)new cv::Mat(get_width_mat(src1), get_height_mat(src1), CV_8UC1);
    //src1->copyTo(*det1);
    //src1->copyTo(*det2);

    cv::cvtColor(*src1, *det1, CV_RGB2GRAY);
    cv::cvtColor(*src1, *det1, CV_RGB2GRAY);

    cv::Mat kernel(7, 7, CV_8U, cv::Scalar(1));

    cv::morphologyEx(*src1, *det1, cv::MORPH_CLOSE, kernel);



    return det1;
}

mat_cv* threshold_otsu(mat_cv* src1, int type) {
    mat_cv* src = (mat_cv*)new cv::Mat(get_width_mat(src1), get_height_mat(src1), CV_8UC1);
    mat_cv* det = (mat_cv*)new cv::Mat(get_width_mat(src1), get_height_mat(src1), CV_8UC1);

    cv::cvtColor(*src1, *src, CV_RGB2GRAY);
    cv::cvtColor(*src1, *det, CV_RGB2GRAY);

    double threshold = 100.0;
    if (type == 1) cv::threshold(*src, *det, threshold, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    else if (type == 2) cv::threshold(*src, *det, threshold, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
    else cv::threshold(*src, *det, threshold, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    int white_pxs = 0;

    for (int y = 0; y < src1->rows; y++) {
        for (int x = 0; x < src1->cols; x++) {
            if (det->data[y * det->rows + x] == 255)
                white_pxs++;
        }
    }
    cv::cvtColor(*det, *det, CV_GRAY2RGB);

    //free(src);
    release_mat(&src);

    return det;
}

mat_cv* HistEqual(mat_cv* src1)
{
    mat_cv* src = (mat_cv*)new cv::Mat(get_width_mat(src1), get_height_mat(src1), CV_8UC1);
    mat_cv* det = (mat_cv*)new cv::Mat(get_width_mat(src1), get_height_mat(src1), CV_8UC1);
    cv::cvtColor(*src1, *src, CV_RGB2GRAY);
    //cv::cvtColor(*src1, *det, CV_RGB2GRAY);

    cv::equalizeHist(*src, *det);
    cv::cvtColor(*det, *det, CV_GRAY2RGB);

    //free(src);
    release_mat(&src);

    return det;
}

mat_cv* img_resize(mat_cv* src1) {
    float x = 416.0 / get_width_mat(src1);
    float y = 416.0 / get_height_mat(src1);
    mat_cv* det1 = (mat_cv*)new cv::Mat(416, 416, CV_8UC3);

    cv::resize(*src1, *det1, cv::Size(), x, y);
    return det1;

}

mat_cv* AND_image(mat_cv* src1, mat_cv* src2) {
    mat_cv* SRC1 = (mat_cv*)new cv::Mat(get_width_mat(src1), get_height_mat(src1), CV_8UC1);
    mat_cv* SRC2 = (mat_cv*)new cv::Mat(get_width_mat(src2), get_height_mat(src2), CV_8UC1);
    mat_cv* det = (mat_cv*)new cv::Mat(get_width_mat(src1), get_height_mat(src1), CV_8UC1);

    cv::cvtColor(*src1, *SRC1, CV_RGB2GRAY);
    cv::cvtColor(*src2, *SRC2, CV_RGB2GRAY);
    cv::cvtColor(*src1, *det, CV_RGB2GRAY);

    cv::bitwise_and(*SRC1, *SRC2, *det);

    cv::cvtColor(*det, *det, CV_GRAY2RGB);

    //free(SRC1);
    //free(SRC2);

    release_mat(&SRC1);
    release_mat(&SRC2);

    return det;
}

mat_cv* Merge_image(mat_cv* src1, mat_cv* src2) {
    mat_cv* SRC1 = (mat_cv*)new cv::Mat(get_width_mat(src1), get_height_mat(src1), CV_8UC1);
    mat_cv* SRC2 = (mat_cv*)new cv::Mat(get_width_mat(src2), get_height_mat(src2), CV_8UC1);
    mat_cv* det = (mat_cv*)new cv::Mat(get_width_mat(src1), get_height_mat(src1), CV_8UC1);

    cv::cvtColor(*src1, *SRC1, CV_RGB2GRAY);
    cv::cvtColor(*src2, *SRC2, CV_RGB2GRAY);
    cv::cvtColor(*src1, *det, CV_RGB2GRAY);

    cv::bitwise_or(*SRC1, *SRC2, *det);

    cv::cvtColor(*det, *det, CV_GRAY2RGB);

    //free(SRC1);
    //free(SRC2);

    release_mat(&SRC1);
    release_mat(&SRC2);

    return det;

}

mat_cv* edge(mat_cv * src1, int type) { //1. canny, 2. sobel

    mat_cv* src = (mat_cv*)new cv::Mat(get_width_mat(src1), get_height_mat(src1), CV_8UC1);
    mat_cv* det1 = (mat_cv*)new cv::Mat(get_width_mat(src1), get_height_mat(src1), CV_8UC1);
    mat_cv* det2 = (mat_cv*)new cv::Mat(get_width_mat(src1), get_height_mat(src1), CV_8UC1); // return

    mat_cv* sobelX = (mat_cv*)new cv::Mat(get_width_mat(src1), get_height_mat(src1), CV_8UC1);
    mat_cv* sobelY = (mat_cv*)new cv::Mat(get_width_mat(src1), get_height_mat(src1), CV_8UC1);

    cv::Mat sobel;

    cv::cvtColor(*src1, *src, CV_RGB2GRAY);

    if (type == 1)
    {
        double a = 0.0;
        a = cv::threshold(*src, *src, a, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
        cv::Canny(*src, *det1, a*0.5, a);
    } // if (type == 1)
    else if (type == 2)
    {
        cv::Sobel(*src, *sobelX, CV_8U, 1, 0);
        cv::Sobel(*src, *sobelY, CV_8U, 0, 1);
        sobel = cv::abs(*sobelX) + cv::abs(*sobelY);
        sobel.copyTo(*det1);
    } // else if (type == 2)

    cv::threshold(*det1, *det2, 0, 255, CV_THRESH_BINARY_INV);

    cv::Mat kernel(5, 5, CV_8U, cv::Scalar(1));

    cv::morphologyEx(*det2, *det1, cv::MORPH_OPEN, kernel);

    //show_image_mat(det1, "edge");

    cv::cvtColor(*det1, *det2, CV_GRAY2RGB);

    kernel.release();

    sobel.release();

    //free(src);
    //free(sobelX);
    //free(sobelY);

    release_mat(&src);

    release_mat(&det1);
    release_mat(&sobelX);
    release_mat(&sobelY);

    //release_mat(&src1);

    return det2;

}

CvPixelBackgroundGMM* cvCreatePixelBackgroundGMM(int width, int height)
{
    CvPixelBackgroundGMM* pGMM = new(CvPixelBackgroundGMM);
    int size = width * height;
    pGMM->nWidth = width;
    pGMM->nHeight = height;
    pGMM->nSize = size;

    pGMM->nNBands = 3;//always 3 - not implemented for other values!

    //set parameters
    // K - max number of Gaussians per pixel
    pGMM->nM = 10;
    // Tb - the threshold - n var
    pGMM->fTb = 7 * 7;	// very important parameter, background threshold (mahalanobis distance)
    // Tbf - the threshold
    pGMM->fTB = 0.9f;//1-cf from the paper
    // Tgenerate - the threshold
    pGMM->fTg = 3.0f*3.0f;//update the mode or generate new
    pGMM->fSigma = 10.0f;//sigma for the new mode
    // alpha - the learning factor
    pGMM->fAlphaT = 0.003f;	// very important parameter
    // complexity reduction prior constant
    //pGMM->fCT=0.05f;
    pGMM->fCT = 0;

    //shadow
    // Shadow detection
    pGMM->bShadowDetection = 1;//turn on
    pGMM->fTau = 0.01f;// Tau - shadow threshold


    //GMM for each pixel
    pGMM->rGMM = (CvPBGMMGaussian*)malloc(size * pGMM->nM * sizeof(CvPBGMMGaussian));

    //used modes per pixel
    pGMM->rnUsedModes = (unsigned char*)malloc(size);
    memset(pGMM->rnUsedModes, 0, size);//no modes used
    pGMM->bRemoveForeground = 0;
    return pGMM;
}

void cvReleasePixelBackgroundGMM(CvPixelBackgroundGMM** ppGMM)
{
    delete (*ppGMM)->rGMM;
    delete (*ppGMM)->rnUsedModes;
    delete (*ppGMM);
    (*ppGMM) = 0;
}


//this might be usefull
void cvSetPixelBackgroundGMM(CvPixelBackgroundGMM* pGMM, unsigned char* data)
{
    int size = pGMM->nSize;
    unsigned char* pDataCurrent = data;

    for (int i = 0; i < size; i++)
    {
        // retrieve the colors
        float R = *pDataCurrent++;
        float G = *pDataCurrent++;
        float B = *pDataCurrent++;

        pGMM->rGMM[i].weight = 1.0;
        pGMM->rGMM[i].muR = R;
        pGMM->rGMM[i].muG = G;
        pGMM->rGMM[i].muB = B;
        pGMM->rGMM[i].sigma = pGMM->fSigma;
    }

    memset(pGMM->rnUsedModes, 1, size);//1 mode used

}

int _cvRemoveShadowGMM(long posPixel,
    float red, float green, float blue,
    unsigned char nModes,
    CvPBGMMGaussian* m_aGaussians,
    int m_nM,
    float m_fTb,
    float m_fTB,
    float m_fTg,
    float m_fTau)
{
    //calculate distances to the modes (+ sort???)
    //here we need to go in descending order!!!
//	long posPixel = pixel * m_nM;
    long pos;
    float tWeight = 0;
    float numerator, denominator;
    // check all the distributions, marked as background:
    for (int iModes = 0; iModes < nModes; iModes++)
    {
        pos = posPixel + iModes;
        float var = m_aGaussians[pos].sigma;
        float muR = m_aGaussians[pos].muR;
        float muG = m_aGaussians[pos].muG;
        float muB = m_aGaussians[pos].muB;
        float weight = m_aGaussians[pos].weight;
        tWeight += weight;

        numerator = red * muR + green * muG + blue * muB;
        denominator = muR * muR + muG * muG + muB * muB;
        // no division by zero allowed
        if (denominator == 0)
        {
            break;
        }
        float a = numerator / denominator;

        // if tau < a < 1 then also check the color distortion
        if ((a <= 1) && (a >= m_fTau))//m_nBeta=1
        {
            float dR = a * muR - red;
            float dG = a * muG - green;
            float dB = a * muB - blue;

            //square distance -slower and less accurate
            //float maxDistance = cvSqrt(m_fTb*var);
            //if ((fabs(dR) <= maxDistance) && (fabs(dG) <= maxDistance) && (fabs(dB) <= maxDistance))
            //circle
            float dist = (dR*dR + dG * dG + dB * dB);
            if (dist < m_fTb*var*a*a)
            {
                return 2;
            }
        }
        if (tWeight > m_fTB)
        {
            break;
        }
    }

    return 0;
}


void _cvReplacePixelBackgroundGMM(long pos,
    unsigned char* pData,
    CvPBGMMGaussian* m_aGaussians)
{
    pData[0] = (unsigned char)m_aGaussians[pos].muR;
    pData[1] = (unsigned char)m_aGaussians[pos].muG;
    pData[2] = (unsigned char)m_aGaussians[pos].muB;
}


int _cvUpdatePixelBackgroundGMM(long posPixel,
    float red, float green, float blue,
    unsigned char pModesUsed,
    CvPBGMMGaussian* m_aGaussians,
    int m_nM,
    float m_fAlphaT,
    float m_fTb,
    float m_fTB,
    float m_fTg,
    float m_fSigma,
    float m_fPrune,
    int* bDecision)
{
    //calculate distances to the modes (+ sort???)
    //here we need to go in descending order!!!

    //long posPixel = pixel * m_nM;
    long pos;

    //	long pos=posPixel-1;//because of ++ at the end

    bool bFitsPDF = 0;
    bool bBackground = 0;

    float m_fOneMinAlpha = 1 - m_fAlphaT;

    //bool bPrune=0;
    int nModes = pModesUsed;
    float totalWeight = 0.0f;

    //////
    //go through all modes
    for (int iModes = 0; iModes < nModes; iModes++)
    {
        pos = posPixel + iModes;
        float weight = m_aGaussians[pos].weight;

        ////
        //fit not found yet
        if (!bFitsPDF)
        {
            //check if it belongs to some of the modes
            //calculate distance
            float var = m_aGaussians[pos].sigma;
            float muR = m_aGaussians[pos].muR;
            float muG = m_aGaussians[pos].muG;
            float muB = m_aGaussians[pos].muB;

            float dR = muR - red;
            float dG = muG - green;
            float dB = muB - blue;

            ///////
            //check if it fits the current mode (Factor * sigma)

            //square distance -slower and less accurate
            //float maxDistance = cvSqrt(m_fTg*var);
            //if ((fabs(dR) <= maxDistance) && (fabs(dG) <= maxDistance) && (fabs(dB) <= maxDistance))
            //circle
            float dist = (dR*dR + dG * dG + dB * dB);
            //background? - m_fTb
            if ((totalWeight < m_fTB) && (dist < m_fTb*var))
                bBackground = 1;
            //check fit
            if (dist < m_fTg*var)
            {
                /////
                //belongs to the mode
                bFitsPDF = 1;

                //update distribution
                float k = m_fAlphaT / weight;
                weight = m_fOneMinAlpha * weight + m_fPrune;
                weight += m_fAlphaT;
                m_aGaussians[pos].muR = muR - k * (dR);
                m_aGaussians[pos].muG = muG - k * (dG);
                m_aGaussians[pos].muB = muB - k * (dB);

                float sigmanew = var + k * (dist - var);
                m_aGaussians[pos].sigma = sigmanew < 4 ? 4 : sigmanew>5 * m_fSigma ? 5 * m_fSigma : sigmanew;

                for (int iLocal = iModes; iLocal > 0; iLocal--)
                {
                    long posLocal = posPixel + iLocal;
                    if (weight < (m_aGaussians[posLocal - 1].weight))
                    {
                        break;
                    }
                    else
                    {
                        //swap
                        CvPBGMMGaussian temp = m_aGaussians[posLocal];
                        m_aGaussians[posLocal] = m_aGaussians[posLocal - 1];
                        m_aGaussians[posLocal - 1] = temp;
                    }
                } // for (int iLocal = iModes; iLocal > 0; iLocal--)

            }

            else
            {
                weight = m_fOneMinAlpha * weight + m_fPrune;
                if (weight < -m_fPrune)
                {
                    weight = 0.0;
                    nModes--;
                } // if (weight < -m_fPrune)
            }
        } // if (!bFitsPDF)
        else
        {
            weight = m_fOneMinAlpha * weight + m_fPrune;
            //check prune
            if (weight < -m_fPrune)
            {
                weight = 0.0;
                nModes--;
            } // if (weight < -m_fPrune)
        } // else - if (!bFitsPDF)
        totalWeight += weight;
        m_aGaussians[pos].weight = weight;
    } //  for (int iModes = 0; iModes < nModes; iModes++)

    //go through all modes
    //renormalize weights
    for (int iLocal = 0; iLocal < nModes; iLocal++)
    {
        m_aGaussians[posPixel + iLocal].weight = m_aGaussians[posPixel + iLocal].weight / totalWeight;
    } // for (int iLocal = 0; iLocal < nModes; iLocal++)

    //make new mode if needed and exit
    if (!bFitsPDF)
    {
        if (nModes == m_nM)
        {
            //replace the weakest
        } // if (nModes == m_nM)
        else
        {
            //add a new one
            nModes++;
        } // else - if (nModes == m_nM)
        pos = posPixel + nModes - 1;

        if (nModes == 1)
            m_aGaussians[pos].weight = 1;
        else
            m_aGaussians[pos].weight = m_fAlphaT;

        //renormalize weights
        int iLocal;
        for (iLocal = 0; iLocal < nModes - 1; iLocal++)
        {
            m_aGaussians[posPixel + iLocal].weight *= m_fOneMinAlpha;
        } // for (iLocal = 0; iLocal < nModes - 1; iLocal++)

        m_aGaussians[pos].muR = red;
        m_aGaussians[pos].muG = green;
        m_aGaussians[pos].muB = blue;
        m_aGaussians[pos].sigma = m_fSigma;

        //sort
        //find the new place for it
        for (iLocal = nModes - 1; iLocal > 0; iLocal--)
        {
            long posLocal = posPixel + iLocal;
            if (m_fAlphaT < (m_aGaussians[posLocal - 1].weight))
            {
                break;
            } // if (m_fAlphaT < (m_aGaussians[posLocal - 1].weight))
            else
            {
                //swap
                CvPBGMMGaussian temp = m_aGaussians[posLocal];
                m_aGaussians[posLocal] = m_aGaussians[posLocal - 1];
                m_aGaussians[posLocal - 1] = temp;
            } // else - if (m_fAlphaT < (m_aGaussians[posLocal - 1].weight))
        } // for (iLocal = nModes - 1; iLocal > 0; iLocal--)
    } // if (!bFitsPDF)

    //set the number of modes
    pModesUsed = nModes;

    *bDecision = bBackground;

    return pModesUsed;
}

void cvUpdatePixelBackgroundGMM(CvPixelBackgroundGMM* pGMM, unsigned char* data, unsigned char* output)
{
    int size = pGMM->nSize;
    unsigned char* pDataCurrent = data;
    unsigned char* pUsedModes = pGMM->rnUsedModes;
    unsigned char* pDataOutput = output;

    //some constants
    int m_nM = pGMM->nM;
    float m_fAlphaT = pGMM->fAlphaT;
    float m_fTb = pGMM->fTb;//Tb - threshold on the Mahalan. dist.
    float m_fTB = pGMM->fTB;//1-TF from the paper
    float m_fTg = pGMM->fTg;//Tg - when to generate a new component
    float m_fSigma = pGMM->fSigma;//initial sigma
    float m_fCT = pGMM->fCT;//CT - complexity reduction prior 
    float m_fPrune = -m_fAlphaT * m_fCT;
    float m_fTau = pGMM->fTau;
    CvPBGMMGaussian* m_aGaussians = pGMM->rGMM;
    long posPixel;
    int m_bShadowDetection = pGMM->bShadowDetection;

    //go through the image

    int result = 0;

    // TODO 
#pragma omp parallel for private( posPixel, result )
    for (int i = 0; i < size; i++)
    {
        // retrieve the colors
        float red = pDataCurrent[i * 3];
        float green = pDataCurrent[i * 3 + 1];
        float blue = pDataCurrent[i * 3 + 2];


        //update model+ background subtract
        posPixel = i * m_nM;

        pUsedModes[i] = _cvUpdatePixelBackgroundGMM(
            posPixel, red, green, blue, pUsedModes[i], m_aGaussians,
            m_nM, m_fAlphaT, m_fTb, m_fTB, m_fTg, m_fSigma, m_fPrune, &result);

        int nMLocal = pUsedModes[i];

        if (m_bShadowDetection)
            if (!result)
            {
                result = _cvRemoveShadowGMM(posPixel, red, green, blue, nMLocal, m_aGaussians,
                    m_nM,
                    m_fTb,
                    m_fTB,
                    m_fTg,
                    m_fTau);
            }


        switch (result)
        {
        case 0:
            //foreground
            pDataOutput[i] = 255;
            if (pGMM->bRemoveForeground)
            {
                _cvReplacePixelBackgroundGMM(posPixel, pDataCurrent - 3, m_aGaussians);
            } // if (pGMM->bRemoveForeground)
            break;
        case 1:
            //background
            pDataOutput[i] = 0;
            break;
        case 2:
            //shadow
            pDataOutput[i] = 0;
            if (pGMM->bRemoveForeground)
            {
                _cvReplacePixelBackgroundGMM(posPixel, pDataCurrent - 3, m_aGaussians);
            } // if (pGMM->bRemoveForeground)

            break;
        } // switch (result)
    } //  for (int i = 0; i < size; i++)
} // void cvUpdatePixelBackgroundGMM(CvPixelBackgroundGMM* pGMM, unsigned char* data, unsigned char* output)

CvPixelBackgroundGMM* GMM_creat(mat_cv* src) {
    CvPixelBackgroundGMM* pGMM = 0;
    pGMM = cvCreatePixelBackgroundGMM(416, 416);
    return pGMM;

}
mat_cv* GMM_update(CvPixelBackgroundGMM* pGMM, mat_cv* src) {

    //mat_cv* src1 = (mat_cv*)new cv::Mat(416, 416, CV_8UC3);
    mat_cv* det1 = (mat_cv*)new cv::Mat(416, 416, CV_8UC1);
    mat_cv* det2 = (mat_cv*)new cv::Mat(416, 416, CV_8UC1);
    //src->copyTo(*src1);
    //src->copyTo(*det1);
    //cv::cvtColor(*src, *src1, CV_RGB2HSV);
    cv::cvtColor(*src, *det1, CV_RGB2GRAY);
    cv::cvtColor(*src, *det2, CV_RGB2GRAY);

    unsigned char* data = src->data;
    unsigned char* result_data = (unsigned char*)malloc(src->cols*src->rows);

    cvUpdatePixelBackgroundGMM(pGMM, data, result_data);
    det1->data = result_data;
    /*
    show_image_mat(src, "src");
    printf("src쇼끝\n");
    show_image_mat(det1, "det");
    printf("det쇼끝\n");
    */

    cv::medianBlur(*det1, *det2, 5);

    cv::cvtColor(*det2, *det1, CV_GRAY2RGB);

    free(result_data);
    //free(data);
    //release_mat(&result_data);
    //free(det2);
    release_mat(&det2);

    return det1;
}

// ====================================================================
// Draw Loss & Accuracy chart
// ====================================================================
mat_cv* draw_train_chart(float max_img_loss, int max_batches, int number_of_lines, int img_size, int dont_show)
{
    int img_offset = 60;
    int draw_size = img_size - img_offset;
    cv::Mat *img_ptr = new cv::Mat(img_size, img_size, CV_8UC3, CV_RGB(255, 255, 255));
    cv::Mat &img = *img_ptr;
    cv::Point pt1, pt2, pt_text;

    try {
        char char_buff[100];
        int i;
        // vertical lines
        pt1.x = img_offset; pt2.x = img_size, pt_text.x = 30;
        for (i = 1; i <= number_of_lines; ++i) {
            pt1.y = pt2.y = (float)i * draw_size / number_of_lines;
            cv::line(img, pt1, pt2, CV_RGB(224, 224, 224), 1, 8, 0);
            if (i % 10 == 0) {
                sprintf(char_buff, "%2.1f", max_img_loss*(number_of_lines - i) / number_of_lines);
                pt_text.y = pt1.y + 3;

                cv::putText(img, char_buff, pt_text, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 0), 1, CV_AA);
                cv::line(img, pt1, pt2, CV_RGB(128, 128, 128), 1, 8, 0);
            }
        }
        // horizontal lines
        pt1.y = draw_size; pt2.y = 0, pt_text.y = draw_size + 15;
        for (i = 0; i <= number_of_lines; ++i) {
            pt1.x = pt2.x = img_offset + (float)i * draw_size / number_of_lines;
            cv::line(img, pt1, pt2, CV_RGB(224, 224, 224), 1, 8, 0);
            if (i % 10 == 0) {
                sprintf(char_buff, "%d", max_batches * i / number_of_lines);
                pt_text.x = pt1.x - 20;
                cv::putText(img, char_buff, pt_text, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 0), 1, CV_AA);
                cv::line(img, pt1, pt2, CV_RGB(128, 128, 128), 1, 8, 0);
            }
        }

        cv::putText(img, "Loss", cv::Point(10, 55), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 255), 1, CV_AA);
        cv::putText(img, "Iteration number", cv::Point(draw_size / 2, img_size - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 0), 1, CV_AA);
        char max_batches_buff[100];
        sprintf(max_batches_buff, "in cfg max_batches=%d", max_batches);
        cv::putText(img, max_batches_buff, cv::Point(draw_size - 195, img_size - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 0), 1, CV_AA);
        cv::putText(img, "Press 's' to save : chart.png", cv::Point(5, img_size - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 0), 1, CV_AA);
        if (!dont_show) {
            printf(" If error occurs - run training with flag: -dont_show \n");
            cv::namedWindow("average loss", cv::WINDOW_NORMAL);
            cv::moveWindow("average loss", 0, 0);
            cv::resizeWindow("average loss", img_size, img_size);
            cv::imshow("average loss", img);
            cv::waitKey(20);
        }
    }
    catch (...) {
        cerr << "OpenCV exception: draw_train_chart() \n";
    }
    return (mat_cv*)img_ptr;
}
// ----------------------------------------

void draw_train_loss(mat_cv* img_src, int img_size, float avg_loss, float max_img_loss, int current_batch, int max_batches,
    float precision, int draw_precision, char *accuracy_name, int dont_show, int mjpeg_port)
{
    try {
        cv::Mat &img = *(cv::Mat*)img_src;
        int img_offset = 60;
        int draw_size = img_size - img_offset;
        char char_buff[100];
        cv::Point pt1, pt2;
        pt1.x = img_offset + draw_size * (float)current_batch / max_batches;
        pt1.y = draw_size * (1 - avg_loss / max_img_loss);
        if (pt1.y < 0) pt1.y = 1;
        cv::circle(img, pt1, 1, CV_RGB(0, 0, 255), CV_FILLED, 8, 0);

        // precision
        if (draw_precision) {
            static float old_precision = 0;
            static float max_precision = 0;
            static int iteration_old = 0;
            static int text_iteration_old = 0;
            if (iteration_old == 0)
                cv::putText(img, accuracy_name, cv::Point(10, 12), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 0, 0), 1, CV_AA);

            cv::line(img,
                cv::Point(img_offset + draw_size * (float)iteration_old / max_batches, draw_size * (1 - old_precision)),
                cv::Point(img_offset + draw_size * (float)current_batch / max_batches, draw_size * (1 - precision)),
                CV_RGB(255, 0, 0), 1, 8, 0);

            sprintf(char_buff, "%2.1f%% ", precision * 100);
            cv::putText(img, char_buff, cv::Point(10, 28), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 255, 255), 5, CV_AA);
            cv::putText(img, char_buff, cv::Point(10, 28), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(200, 0, 0), 1, CV_AA);

            if ((std::fabs(old_precision - precision) > 0.1)  || (max_precision < precision) || (current_batch - text_iteration_old) >= max_batches / 10) {
                text_iteration_old = current_batch;
                max_precision = std::max(max_precision, precision);
                sprintf(char_buff, "%2.0f%% ", precision * 100);
                cv::putText(img, char_buff, cv::Point(pt1.x - 30, draw_size * (1 - precision) + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 255, 255), 5, CV_AA);
                cv::putText(img, char_buff, cv::Point(pt1.x - 30, draw_size * (1 - precision) + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(200, 0, 0), 1, CV_AA);
            }
            old_precision = precision;
            iteration_old = current_batch;
        }

        sprintf(char_buff, "current avg loss = %2.4f    iteration = %d", avg_loss, current_batch);
        pt1.x = 15, pt1.y = draw_size + 18;
        pt2.x = pt1.x + 460, pt2.y = pt1.y + 20;
        cv::rectangle(img, pt1, pt2, CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
        pt1.y += 15;
        cv::putText(img, char_buff, pt1, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 100), 1, CV_AA);

        int k = 0;
        if (!dont_show) {
            cv::imshow("average loss", img);
            k = cv::waitKey(20);
        }
        static int old_batch = 0;
        if (k == 's' || current_batch == (max_batches - 1) || (current_batch / 100 > old_batch / 100)) {
            old_batch = current_batch;
            save_mat_png(img, "chart.png");
            cv::putText(img, "- Saved", cv::Point(260, img_size - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 0, 0), 1, CV_AA);
        }
        else
            cv::putText(img, "- Saved", cv::Point(260, img_size - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 255, 255), 1, CV_AA);

        if (mjpeg_port > 0) send_mjpeg((mat_cv *)&img, mjpeg_port, 500000, 100);
    }
    catch (...) {
        cerr << "OpenCV exception: draw_train_loss() \n";
    }
}
// ----------------------------------------


// ====================================================================
// Data augmentation
// ====================================================================
static box float_to_box_stride(float *f, int stride)
{
    box b = { 0 };
    b.x = f[0];
    b.y = f[1 * stride];
    b.w = f[2 * stride];
    b.h = f[3 * stride];
    return b;
}

image image_data_augmentation(mat_cv* mat, int w, int h,
    int pleft, int ptop, int swidth, int sheight, int flip,
    float dhue, float dsat, float dexp,
    int blur, int num_boxes, float *truth)
{
    image out;
    try {
        cv::Mat img = *(cv::Mat *)mat;

        // crop
        cv::Rect src_rect(pleft, ptop, swidth, sheight);
        cv::Rect img_rect(cv::Point2i(0, 0), img.size());
        cv::Rect new_src_rect = src_rect & img_rect;

        cv::Rect dst_rect(cv::Point2i(std::max<int>(0, -pleft), std::max<int>(0, -ptop)), new_src_rect.size());
        cv::Mat sized;

        if (src_rect.x == 0 && src_rect.y == 0 && src_rect.size() == img.size()) {
            cv::resize(img, sized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
        }
        else {
            cv::Mat cropped(src_rect.size(), img.type());
            //cropped.setTo(cv::Scalar::all(0));
            cropped.setTo(cv::mean(img));

            img(new_src_rect).copyTo(cropped(dst_rect));

            // resize
            cv::resize(cropped, sized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
        }

        // flip
        if (flip) {
            cv::Mat cropped;
            cv::flip(sized, cropped, 1);    // 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)
            sized = cropped.clone();
        }

        // HSV augmentation
        // cv::COLOR_BGR2HSV, cv::COLOR_RGB2HSV, cv::COLOR_HSV2BGR, cv::COLOR_HSV2RGB
        if (dsat != 1 || dexp != 1 || dhue != 0) {
            if (img.channels() >= 3)
            {
                cv::Mat hsv_src;
                cvtColor(sized, hsv_src, cv::COLOR_RGB2HSV);    // RGB to HSV

                std::vector<cv::Mat> hsv;
                cv::split(hsv_src, hsv);

                hsv[1] *= dsat;
                hsv[2] *= dexp;
                hsv[0] += 179 * dhue;

                cv::merge(hsv, hsv_src);

                cvtColor(hsv_src, sized, cv::COLOR_HSV2RGB);    // HSV to RGB (the same as previous)
            }
            else
            {
                sized *= dexp;
            }
        }

        //std::stringstream window_name;
        //window_name << "augmentation - " << ipl;
        //cv::imshow(window_name.str(), sized);
        //cv::waitKey(0);

        if (blur) {
            cv::Mat dst(sized.size(), sized.type());
            if(blur == 1) cv::GaussianBlur(sized, dst, cv::Size(31, 31), 0);
            else cv::GaussianBlur(sized, dst, cv::Size((blur / 2) * 2 + 1, (blur / 2) * 2 + 1), 0);
            cv::Rect img_rect(0, 0, sized.cols, sized.rows);
            //std::cout << " blur num_boxes = " << num_boxes << std::endl;

            if (blur == 1) {
                int t;
                for (t = 0; t < num_boxes; ++t) {
                    box b = float_to_box_stride(truth + t*(4 + 1), 1);
                    if (!b.x) break;
                    int left = (b.x - b.w / 2.)*sized.cols;
                    int width = b.w*sized.cols;
                    int top = (b.y - b.h / 2.)*sized.rows;
                    int height = b.h*sized.rows;
                    cv::Rect roi(left, top, width, height);
                    roi = roi & img_rect;

                    sized(roi).copyTo(dst(roi));
                }
            }
            dst.copyTo(sized);
        }

        // Mat -> image
        out = mat_to_image(sized);
    }
    catch (...) {
        cerr << "OpenCV can't augment image: " << w << " x " << h << " \n";
        out = mat_to_image(*(cv::Mat*)mat);
    }
    return out;
}

// blend two images with (alpha and beta)
void blend_images_cv(image new_img, float alpha, image old_img, float beta)
{
    cv::Mat new_mat(cv::Size(new_img.w, new_img.h), CV_32FC(new_img.c), new_img.data);// , size_t step = AUTO_STEP)
    cv::Mat old_mat(cv::Size(old_img.w, old_img.h), CV_32FC(old_img.c), old_img.data);
    cv::addWeighted(new_mat, alpha, old_mat, beta, 0.0, new_mat);
}

// ====================================================================
// Show Anchors
// ====================================================================
void show_acnhors(int number_of_boxes, int num_of_clusters, float *rel_width_height_array, model anchors_data, int width, int height)
{
    cv::Mat labels = cv::Mat(number_of_boxes, 1, CV_32SC1);
    cv::Mat points = cv::Mat(number_of_boxes, 2, CV_32FC1);
    cv::Mat centers = cv::Mat(num_of_clusters, 2, CV_32FC1);

    for (int i = 0; i < number_of_boxes; ++i) {
        points.at<float>(i, 0) = rel_width_height_array[i * 2];
        points.at<float>(i, 1) = rel_width_height_array[i * 2 + 1];
    }

    for (int i = 0; i < num_of_clusters; ++i) {
        centers.at<float>(i, 0) = anchors_data.centers.vals[i][0];
        centers.at<float>(i, 1) = anchors_data.centers.vals[i][1];
    }

    for (int i = 0; i < number_of_boxes; ++i) {
        labels.at<int>(i, 0) = anchors_data.assignments[i];
    }

    size_t img_size = 700;
    cv::Mat img = cv::Mat(img_size, img_size, CV_8UC3);

    for (int i = 0; i < number_of_boxes; ++i) {
        cv::Point pt;
        pt.x = points.at<float>(i, 0) * img_size / width;
        pt.y = points.at<float>(i, 1) * img_size / height;
        int cluster_idx = labels.at<int>(i, 0);
        int red_id = (cluster_idx * (uint64_t)123 + 55) % 255;
        int green_id = (cluster_idx * (uint64_t)321 + 33) % 255;
        int blue_id = (cluster_idx * (uint64_t)11 + 99) % 255;
        cv::circle(img, pt, 1, CV_RGB(red_id, green_id, blue_id), CV_FILLED, 8, 0);
        //if(pt.x > img_size || pt.y > img_size) printf("\n pt.x = %d, pt.y = %d \n", pt.x, pt.y);
    }

    for (int j = 0; j < num_of_clusters; ++j) {
        cv::Point pt1, pt2;
        pt1.x = pt1.y = 0;
        pt2.x = centers.at<float>(j, 0) * img_size / width;
        pt2.y = centers.at<float>(j, 1) * img_size / height;
        cv::rectangle(img, pt1, pt2, CV_RGB(255, 255, 255), 1, 8, 0);
    }
    save_mat_png(img, "cloud.png");
    cv::imshow("clusters", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

}   // extern "C"


#else  // OPENCV
int wait_key_cv(int delay) { return 0; }
int wait_until_press_key_cv() { return 0; }
void destroy_all_windows_cv() {}
#endif // OPENCV
