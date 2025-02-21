#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem> // C++17 库用于遍历文件夹
#include <iomanip>

using cv::Mat;
using std::cout;
using std::endl;
using std::string;
using std::vector;
namespace fs = std::filesystem;

static const vector<string> class_name = { "A", "B" };

void print_result(const Mat& result, float conf = 0.7, int len_data = 7)
{
    float* pdata = (float*)result.data;
    for (int i = 0; i < result.total() / len_data; i++)
    {
        if (pdata[4] > conf)
        {
            for (int j = 0; j < len_data; j++)
            {
                cout << pdata[j] << " ";
            }
            cout << endl;
        }
        pdata += len_data;
    }
    return;
}

vector<vector<float>> get_info(const Mat& result, float conf = 0.7, int len_data = 7)
{
    float* pdata = (float*)result.data;
    vector<vector<float>> info;
    for (int i = 0; i < result.total() / len_data; i++)
    {
        if (pdata[4] > conf)
        {
            vector<float> info_line;
            for (int j = 0; j < len_data; j++)
            {
                info_line.push_back(pdata[j]);
            }
            info.push_back(info_line);
        }
        pdata += len_data;
    }
    return info;
}

void info_simplify(vector<vector<float>>& info)
{
    for (auto i = 0; i < info.size(); i++)
    {
        info[i][5] = std::max_element(info[i].cbegin() + 5, info[i].cend()) - (info[i].cbegin() + 5);
        info[i].resize(6);
        float x = info[i][0];
        float y = info[i][1];
        float w = info[i][2];
        float h = info[i][3];
        info[i][0] = x - w / 2.0;
        info[i][1] = y - h / 2.0;
        info[i][2] = x + w / 2.0;
        info[i][3] = y + h / 2.0;
    }
}

vector<vector<vector<float>>> split_info(vector<vector<float>>& info)
{
    vector<vector<vector<float>>> info_split;
    vector<int> class_id;
    for (auto i = 0; i < info.size(); i++)
    {
        if (std::find(class_id.begin(), class_id.end(), (int)info[i][5]) == class_id.end())
        {
            class_id.push_back((int)info[i][5]);
            vector<vector<float>> info_;
            info_split.push_back(info_);
        }
        info_split[std::find(class_id.begin(), class_id.end(), (int)info[i][5]) - class_id.begin()].push_back(info[i]);
    }
    return info_split;
}

void nms(vector<vector<float>>& info, float iou = 0.4)
{
    int counter = 0;
    vector<vector<float>> return_info;
    while (counter < info.size())
    {
        return_info.clear();
        float x1 = 0;
        float x2 = 0;
        float y1 = 0;
        float y2 = 0;
        std::sort(info.begin(), info.end(), [](vector<float> p1, vector<float> p2)
            { return p1[4] > p2[4]; });
        for (auto i = 0; i < info.size(); i++)
        {
            if (i < counter)
            {
                return_info.push_back(info[i]);
                continue;
            }
            if (i == counter)
            {
                x1 = info[i][0];
                y1 = info[i][1];
                x2 = info[i][2];
                y2 = info[i][3];
                return_info.push_back(info[i]);
                continue;
            }
            if (info[i][0] > x2 or info[i][2] < x1 or info[i][1] > y2 or info[i][3] < y1)
            {
                return_info.push_back(info[i]);
            }
            else
            {
                float over_x1 = std::max(x1, info[i][0]);
                float over_y1 = std::max(y1, info[i][1]);
                float over_x2 = std::min(x2, info[i][2]);
                float over_y2 = std::min(y2, info[i][3]);
                float s_over = (over_x2 - over_x1) * (over_y2 - over_y1);
                float s_total = (x2 - x1) * (y2 - y1) + (info[i][0] - info[i][2]) * (info[i][1] - info[i][3]) - s_over;
                if (s_over / s_total < iou)
                {
                    return_info.push_back(info[i]);
                }
            }
        }
        info = return_info;
        counter += 1;
    }
}

void print_info(const vector<vector<float>>& info)
{
    for (auto i = 0; i < info.size(); i++)
    {
        for (auto j = 0; j < info[i].size(); j++)
        {
            cout << info[i][j] << " ";
        }
        cout << endl;
    }
}
void draw_box(Mat& img, const vector<vector<float>>& info)
{
    for (int i = 0; i < info.size(); i++)
    {
        cv::Scalar color = (info[i][5] == 0) ? cv::Scalar(0, 255, 255) : cv::Scalar(255, 0, 0); // A类黄色，B类蓝色
        cv::rectangle(img, cv::Point(info[i][0], info[i][1]), cv::Point(info[i][2], info[i][3]), color, 1.7);

        string label;
        label += class_name[info[i][5]];
        label += "  ";

        // 格式化置信度为小数点后两位
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << info[i][4];
        label += ss.str();

        cv::putText(img, label, cv::Point(info[i][0], info[i][1]), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 1.7);
    }
}

// 使用 std::filesystem 进行文件夹遍历
vector<string> get_files_in_folder(const string& folder_path)
{
    vector<string> files;
    for (const auto& entry : fs::directory_iterator(folder_path))
    {
        if (entry.is_regular_file())
        {
            files.push_back(entry.path().string());
        }
    }
    return files;
}

int main()
{
    cv::dnn::Net net = cv::dnn::readNetFromONNX("C:\\Users\\xianuo\\Desktop\\test4\\runs\\weights\\best.onnx");
    string folder_path = "E:\\yolov5-6.0\\data\\dataset\\test\\images";
    string output_folder_path = "C:\\Users\\xianuo\\Desktop\\test4\\test_result1";
    vector<string> image_files = get_files_in_folder(folder_path);

    for (const auto& image_path : image_files)
    {
        Mat img = cv::imread(image_path);
        if (img.empty())
        {
            cout << "Could not read the image: " << image_path << endl;
            continue;
        }
        cv::resize(img, img, cv::Size(640, 640));
        Mat blob = cv::dnn::blobFromImage(img, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true);
        net.setInput(blob);
        vector<Mat> netoutput;
        vector<string> out_name = { "output" };
        net.forward(netoutput, out_name);
        Mat result = netoutput[0];
        vector<vector<float>> info = get_info(result);
        info_simplify(info);
        vector<vector<vector<float>>> info_split = split_info(info);

        for (auto i = 0; i < info_split.size(); i++)
        {
            nms(info_split[i]);
            draw_box(img, info_split[i]);
        }
        string output_path = output_folder_path + "\\" + image_path.substr(image_path.find_last_of("\\") + 1);
        output_path = output_path.substr(0, image_path.find_last_of('.')) + "_test.jpg";
        cv::imwrite(output_path, img);
        cout << "Processed and saved: " << output_path << endl;
    }

    return 0;
}
