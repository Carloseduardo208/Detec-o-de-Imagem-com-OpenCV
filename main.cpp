
#include <opencv2/core.hpp> 
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp> 
#include <opencv2/highgui.hpp> 
#include<opencv2/video.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/objdetect.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace cv;
using namespace dnn;
using namespace std;


float confThreshold = 0.5; 
float nmsThreshold = 0.4;  
int inpWidth = 416;  
int inpHeight = 416; 
vector<string> classes2;


void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame) {

    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 255, 255), 1);

    
    string conf_label = format("%.2f", conf);
    string label = "";
    if (!classes2.empty()) {
        label = classes2[classId] + ":" + conf_label;
    }

    
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height), Point(left + labelSize.width, top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1, LINE_AA);
}


void postprocess(Mat& frame, const vector<Mat>& outs) {
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i) {

        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    

    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
            box.x + box.width, box.y + box.height, frame);
    }
}


vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {

        vector<int> outLayers = net.getUnconnectedOutLayers();

        vector<String> layersNames = net.getLayerNames();

        names.resize(outLayers.size());
        for (int i = 0; i < outLayers.size(); ++i) {
            names[i] = layersNames[outLayers[i] - 1];
            cout << " Nomes: " + layersNames[outLayers[i] - 1] << endl;
        }
    }
    return names;
}


int main(int argc, const  char** argv) {

    const string keys =

        //"{ @image |vtest.avi| path to image file }";
        "{ @image |rua-de-arapiraca.mp4| path to image file }";
    CommandLineParser parser(argc, argv, keys);


    string classesFile = "coco.names.txt";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line))
        classes2.push_back(line);

    String modelConfiguration = "yolov3.cfg.txt";
    String modelWeights = "yolov3.weights";

    
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    Mat input, blob, img;
    
    CascadeClassifier plateCascade;
    plateCascade.load("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_russian_plate_number.xml");
    if (plateCascade.empty()) { cout << "xml file not load" << endl; }
    vector<Rect> plates;
    string filename = parser.get<string>("@image");
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
    VideoCapture capture(filename);
    if (!capture.isOpened()) {

        cerr << "Não foi possível abrir o arquivo" << endl;
        return 0;
    }
    while (true) {

        capture.read(input);
        capture.read(img);

        blobFromImage(input, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

        net.setInput(blob);
        
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));
        
        postprocess(input, outs);

        plateCascade.detectMultiScale(img, plates, 1.1, 10);
        for (int i = 0; i < plates.size(); i++) {

            
            vector<double> layersTimes;
            double freq = getTickFrequency() / 1000;
            double t = net.getPerfProfile(layersTimes) / freq;
            printf("\n \n Tempo total: %.2f ms\n\n", t);

            
        }
        
        
        
        imshow("YOLO", input);
       

        waitKey(20);
    }
    return 0;
}