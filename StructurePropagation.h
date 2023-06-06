#ifndef STRUCTURE_PROPAGATION_H
#define STRUCTURE_PROPAGATION_H

#include "OpenCvUtility.h"
#include "PointManager.h"
#include <map>
#include <list>
#include <set>
#include <math.h>

using namespace std;
using namespace cv;

class StructurePropagation {
public:
    StructurePropagation() = default;

    ~StructurePropagation() = default;

    void SetParam(int block_size, int sample_step, double ks, double ki);

    void Run(const Mat &mask, const Mat &img_masked, Mat &mask_after_propagation, vector<vector<Point>> &plist, Mat &result);

private:
    int block_size;
    int sample_step;
    double ks;
    double ki;
    PointManager point_manager;

    int *DP(const vector<PointPos> &known_points, vector<PointPos> &unknown_points, const Mat &image_src_grey);

    double computeEs(const PointPos &i, const PointPos &xi);

    double computeEi(const Mat &image_src, const PointPos &i, const PointPos &xi);

    double
    computeE2(const Mat &image_src, const PointPos &i1, const PointPos &i2, const PointPos &xi1, const PointPos &xi2);

    int *BP(const vector<PointPos> &known_points, vector<PointPos> &unknown_points, const Mat &image_src);

    void computeMij(MyNode &n, const list<shared_ptr<Edge>>::iterator &edge_iter, const Mat &image_src,
                    const vector<PointPos> &known_points);

    void
    getResult(Mat mask, int *sample_indices, const vector<PointPos> &known_points, vector<PointPos> &unknown_points,
              Mat &result);
};

#endif /* STRUCTURE_PROPAGATION_H */