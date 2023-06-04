#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "StructurePropagation.h"
#include "OpenCvUtility.h"
#include <opencv2/imgproc/types_c.h>

#include "TextureCompletion.h"

using namespace std;
using namespace cv;

enum _line_or_curve
{
    LINE,
    CURVE
};

char *image_path;
char *save_path;

Mat image_src;       // source, not changed
Mat mask;            // 255 for mask, 0 for background
Mat image_with_mask; // masked part of image_src is painted to (255, 0, 0)

vector<vector<Point>> structure_line_set;

int brush_size = 30;
int block_size = 20;
int sample_step = 10;
int line_or_curve = 0;

static void drawMaskMouseCallback(int event, int x, int y, int flags, void *param)
{
    // brush and mask are painted in special color, only for UI
    static Mat image_with_mask_and_brush;

    // flags is always zero here so I have to maintain left button status manually
    static bool hold_left = false;
    if (event == EVENT_LBUTTONDOWN)
    {
        hold_left = true;
    }
    else if (event == EVENT_LBUTTONUP)
    {
        hold_left = false;
    }

    // for the mouse event sample rate is relatively low, draw line is better than draw point
    // otherwise it's easily to draw series of sparse points when moving mouse quickly
    static Point pt_curr = Point(-1, -1);
    static Point pt_prev = Point(-1, -1);

    pt_prev = pt_curr.x == -1 ? Point(x, y) : pt_curr;
    pt_curr = Point(x, y);

    // draw mask
    if (hold_left)
    {
        line(mask, pt_prev, pt_curr, Scalar(255), 2 * brush_size);
        line(image_with_mask, pt_prev, pt_curr, Scalar(255, 0, 0), 2 * brush_size);
    }

    // draw brush
    image_with_mask_and_brush = image_with_mask.clone();
    circle(image_with_mask_and_brush, pt_curr, brush_size, Scalar(85, 85, 255), -1);

    imshow("Draw Mask", image_with_mask_and_brush);
}

void drawMask()
{
    mask = Mat::zeros(image_src.size(), CV_8UC1);
    image_with_mask = image_src.clone();

    namedWindow("Draw Mask");
    createTrackbar("Brush Size", "Draw Mask", &brush_size, 50);
    imshow("Draw Mask", image_with_mask);
    setMouseCallback("Draw Mask", drawMaskMouseCallback);

    waitKey(0);

    destroyWindow("Draw Mask");
}

void drawStructureLine(const Mat &input, Mat &output, const vector<vector<Point>> &structure_line_set, const vector<Point> &curr_point_set)
{
    output = input.clone();
    for (vector<Point> point_set : structure_line_set)
    {
        for (int i = 0; i < (int)(point_set.size()) - 1; i++)
        {
            line(output, point_set[i], point_set[i + 1], Scalar(85, 85, 255), 2);
        }
    }

    for (int i = 0; i < (int)(curr_point_set.size()) - 1; i++)
    {
        line(output, curr_point_set[i], curr_point_set[i + 1], Scalar(85, 85, 255), 2);
    }
}

static void structurePropagationMouseCallBack(int event, int x, int y, int flags, void *param)
{
    // image_with_mask with structure line on it, only for UI
    static Mat image_with_structure_line;

    // flags is always zero here so I have to maintain left button status manually
    static bool hold_left = false;
    static vector<Point> curr_point_set;

    static Point pt_curr = Point(-1, -1);
    pt_curr = Point(x, y);

    // left button down, add new point
    if (event == EVENT_LBUTTONDOWN)
    {
        curr_point_set.emplace_back(x, y);
    }
    // right bottom down, abort drawing
    // if it's a curve and two or more points is drawn, add it to structure_line_set
    else if (event == EVENT_RBUTTONDOWN)
    {
        if (line_or_curve == CURVE && curr_point_set.size() > 1)
        {
            structure_line_set.emplace_back(vector<Point>(curr_point_set));
        }

        curr_point_set.clear();
    }

    // if it's a line and two points is drawn, add it to structure_line_set
    if (line_or_curve == LINE && curr_point_set.size() == 2)
    {
        structure_line_set.emplace_back(vector<Point>(curr_point_set));
        curr_point_set.clear();
    }

    // draw sturcture_line on image in special color, only for UI
    drawStructureLine(image_with_mask, image_with_structure_line, structure_line_set, curr_point_set);

    // draw a line from the latest drawn point (if exists) to current mouse position
    if (curr_point_set.size() > 0)
    {
        line(image_with_structure_line, curr_point_set.back(), pt_curr, Scalar(85, 85, 255), 2);
    }

    imshow("Structure Propagation", image_with_structure_line);
}

void getDensePointSet(const vector<vector<Point>> &structure_line_set, vector<vector<Point>> &out)
{
    for (vector<Point> point_set : structure_line_set)
    {
        for (int i = 0; i < (int)(point_set.size()) - 1; i++)
        {
            int abs_x = std::abs(point_set[i].x - point_set[i + 1].x);
            int abs_y = std::abs(point_set[i].y - point_set[i + 1].y);
            int sign_x = point_set[i].x < point_set[i + 1].x ? 1 : -1;
            int sign_y = point_set[i].y < point_set[i + 1].y ? 1 : -1;

            // in case a "line" begin and end at the same pixel
            if (point_set[i].x != point_set[i + 1].x && point_set[i].y != point_set[i + 1].y)
            {
                vector<Point> point_samples;
                // sample the line with a step of 1 pixel
                if (abs_y > abs_x)
                {
                    double slope_y = (double)(point_set[i + 1].x - point_set[i].x) / (point_set[i + 1].y - point_set[i].y);
                    for (int dy = 0; dy < abs_y; dy++)
                    {
                        int y = sign_y * dy + point_set[i].y;
                        int x = slope_y * sign_y * dy + point_set[i].x;
                        point_samples.emplace_back(x, y);
                    }
                }
                else
                {
                    double slope_x = (double)(point_set[i + 1].y - point_set[i].y) / (point_set[i + 1].x - point_set[i].x);
                    for (int dx = 0; dx < abs_x; dx++)
                    {
                        int x = sign_x * dx + point_set[i].x;
                        int y = slope_x * sign_x * dx + point_set[i].y;
                        point_samples.emplace_back(x, y);
                    }
                }
                out.emplace_back(vector<Point>(point_samples));
            }
        }
    }
}

void structurePropagation()
{

    namedWindow("Structure Propagation");
    createTrackbar("Block Size", "Structure Propagation", &block_size, 50);
    createTrackbar("Sample Step", "Structure Propagation", &sample_step, 20);
    createTrackbar("Line or Curve", "Structure Propagation", &line_or_curve, 1);
    imshow("Structure Propagation", image_with_mask);
    setMouseCallback("Structure Propagation", structurePropagationMouseCallBack);

    vector<vector<Point>> dense_point_set;
    char c = waitKey(0);
    switch (c)
    {
    case 's':
        getDensePointSet(structure_line_set, dense_point_set);
        // TODO: structure propagation
        break;

    case 't':
        // TODO: texture propagation
        break;
    }

    destroyWindow("Structure Propagation");
}

int main(int argc, char **argv)
{
    // read source
    if (argc != 3)
    {
        std::cout << "Usage: ./main image_path save_path" << endl;
        return 0;
    }
    image_path = argv[1];
    save_path = argv[2];
    image_src = imread(image_path);

    if (image_src.empty())
    {
        std::cout << "Image is empty!" << endl;
        return 0;
    }

    drawMask();
    structurePropagation();

    waitKey(0);
    return 0;
}