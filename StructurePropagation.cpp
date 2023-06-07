#include "StructurePropagation.h"
#include <algorithm>
#include <queue>

#include "Photometric.h"

// set parameters
void StructurePropagation::SetParam(int block_size, int sample_step, double ks, double ki) {
    this->block_size = block_size;
    this->sample_step = sample_step;
    this->ks = ks;
    this->ki = ki;
}

// run structure propagation
void
StructurePropagation::Run(const Mat &mask, const Mat &img_masked, Mat &mask_after_propagation, vector<vector<Point>> &structure_line_points,
                          Mat &result) {
    mask_after_propagation = Mat::zeros(mask.size(), CV_8UC1);

    Mat image_src_grey;
    cvtColor(img_masked, image_src_grey, COLOR_BGR2GRAY);

    set<shared_ptr<list<int>>> line_sets;
    point_manager.init(structure_line_points, mask, block_size, line_sets);
    // HACK:
    // point_manager.init(structure_line_points, image_src_grey, block_size, line_sets);

    int *sample_indices;
    vector<PointPos> unknown_points;
    vector<PointPos> known_points;
    set<shared_ptr<list<int>>>::iterator iter;
    for (iter = line_sets.begin(); iter != line_sets.end(); iter++) {
        point_manager.getKnownPoint(known_points, sample_step, **iter);
        if (!known_points.empty()) {
            // if there's only one structure line, use DP, otherwise use BP
            if ((*iter)->size() == 1) {
                point_manager.getUnknownPoint(unknown_points, **iter);
                sample_indices = DP(known_points, unknown_points, image_src_grey);
            } else {
                point_manager.constructBpMap(**iter);
                sample_indices = BP(known_points, unknown_points, image_src_grey);
            }

            // update mask (mark anchored patches as known)
            for (auto p: unknown_points) {
                Point tar = point_manager.getPoint(p);
                for (int j = -block_size / 2; j < block_size / 2; j++) {
                    for (int k = -block_size / 2; k < block_size / 2; k++) {
                        int y = j + tar.y;
                        int x = k + tar.x;
                        if (x >= 0 && y >= 0 && x < mask_after_propagation.cols && y < mask_after_propagation.rows) {
                            mask_after_propagation.at<uchar>(y, x) = 255;
                        }
                    }
                }
            }

            // Photometric Correction
            getResult(mask, sample_indices, known_points, unknown_points, result);
        }
    }
}

// called when only one structure line exists
int *StructurePropagation::DP(const vector<PointPos> &known_points, vector<PointPos> &unknown_points,
                              const Mat &image_src_grey) {

    auto *M = (double *) malloc(2 * known_points.size() * sizeof(double));
    auto *record = (int *) malloc(known_points.size() * unknown_points.size() * sizeof(int));

    // first anchor point
    for (int xi = 0; xi < known_points.size(); xi++) {
        M[xi] = ks * computeEs(unknown_points[0], known_points[xi]) +
                ki * computeEi(image_src_grey, unknown_points[0], known_points[xi]);
    }

    // for each anchor point, for each xi, compute M
    int curr_offset = 0;
    int prev_offset = 0;
    for (int i = 1; i < unknown_points.size(); i++) {
        curr_offset = (i % 2) * known_points.size();
        prev_offset = ((i + 1) % 2) * known_points.size();

        for (int xi = 0; xi < known_points.size(); xi++) {
            // compute E1
            double E1 = ks * computeEs(unknown_points[i], known_points[xi]) +
                        ki * computeEi(image_src_grey, unknown_points[i], known_points[xi]);

            // compute E2 and
            double min = INT_MAX;
            int min_ind = 0;
            for (int xj = 0; xj < known_points.size(); xj++) {
                double tmp = computeE2(image_src_grey, unknown_points[i], unknown_points[i - 1], known_points[xi],
                                       known_points[xj]) + M[prev_offset + xj];
                if (tmp < min) {
                    min = tmp;
                    min_ind = xj;
                }
            }

            // record xi and M
            record[known_points.size() * i + xi] = min_ind;
            M[curr_offset + xi] = E1 + min;
        }
    }

    int *sample_indices = (int *) malloc(unknown_points.size() * sizeof(int));
    double min = INT_MAX;
    for (int xi = 0; xi < known_points.size(); xi++) {
        if (M[curr_offset + xi] < min) {
            sample_indices[unknown_points.size() - 1] = xi;
            min = M[curr_offset + xi];
        }
    }

    // trace back
    for (int i = unknown_points.size() - 2; i >= 0; i--) {
        sample_indices[i] = record[known_points.size() * (i + 1) + sample_indices[i + 1]];
    }

    free(M);
    free(record);

    return sample_indices;
}

// used in DP
double StructurePropagation::computeEs(const PointPos &i, const PointPos &xi) {
    // get points of curve segment contained in patch
    list<Point *> begin1, begin2;
    list<int> length1, length2;
    point_manager.getPointsInPatch(i, begin1, length1);
    point_manager.getPointsInPatch(xi, begin2, length2);

    int len1 = 0;
    for (auto l: length1) {
        len1 += l;
    }

    int len2 = 0;
    for (auto l: length2) {
        len2 += l;
    }
    static vector<int> min_dist1(len1), min_dist2(len2);

    // initialize minimal distance
    for (int i = 0; i < len1; i++) {
        min_dist1[i] = INT_MAX;
    }
    for (int i = 0; i < len2; i++) {
        min_dist2[i] = INT_MAX;
    }

    Point pi = point_manager.getPoint(i);
    Point pxi = point_manager.getPoint(xi);
    int offset_x = pxi.x - pi.x;
    int offset_y = pxi.y - pi.y;
    list<int>::iterator lenItor1, lenItor2;
    list<Point *>::iterator pointItor1, pointItor2;

    // compute minimal distance
    for (lenItor1 = length1.begin(), pointItor1 = begin1.begin(); lenItor1 != length1.end(); lenItor1++, pointItor1++) {
        Point *points1 = *pointItor1;
        for (int i = 0; i < *lenItor1; i++) {
            for (lenItor2 = length2.begin(), pointItor2 = begin2.begin();
                 lenItor2 != length2.end(); lenItor2++, pointItor2++) {
                for (int j = 0; j < *lenItor2; j++) {
                    Point *points2 = *pointItor2;
                    int dx = points1[i].x - points2[j].x + offset_x;
                    int dy = points1[i].y - points2[j].y + offset_y;
                    int dist = dx * dx + dy * dy;
                    if (dist < min_dist1[i]) {
                        min_dist1[i] = dist;
                    }
                    if (dist < min_dist2[j]) {
                        min_dist2[j] = dist;
                    }
                }
            }
        }
    }

    int Es = 0;
    for (auto d: min_dist1) {
        Es += d;
    }
    for (auto d: min_dist2) {
        Es += d;
    }
    return (double) Es / min_dist1.size();
}

// used in DP
double StructurePropagation::computeEi(const Mat &image_src, const PointPos &i, const PointPos &xi) {
    // compute Ei for every boundary patch
    if (point_manager.nearBoundary(i)) {
        Point pi = point_manager.getPoint(i);
        Point pxi = point_manager.getPoint(xi);
        int offset1 = block_size / 2;
        int offset2 = block_size - offset1;

        int cnt = 0;
        int ssd = 0;
        for (int i = -offset1; i < offset2; i++) {
            const auto *ptri = image_src.ptr<uchar>(i + pi.y);
            const auto *ptrxi = image_src.ptr<uchar>(i + pxi.y);
            for (int j = -offset1; j < offset2; j++) {
                if (ptri[j + pi.x] != 0) {
                    int diff = ptri[j + pi.x] - ptrxi[j + pxi.x];
                    ssd += diff * diff;
                    cnt++;
                }
            }
        }
        return (double) ssd / cnt;
    } else {
        return 0;
    }
}

// used in DP
double
StructurePropagation::computeE2(const Mat &image_src, const PointPos &i1, const PointPos &i2, const PointPos &xi1,
                                const PointPos &xi2) {
    Point p1 = point_manager.getPoint(i1);
    Point p2 = point_manager.getPoint(i2);
    Point px1 = point_manager.getPoint(xi1);
    Point px2 = point_manager.getPoint(xi2);

    int left1, left2, right1, right2;
    int up1, up2, down1, down2;
    if (p1.x > p2.x) {
        left1 = 0;
        left2 = p1.x - p2.x;
        right1 = block_size - left2;
        right2 = block_size;
    } else {
        left2 = 0;
        left1 = p2.x - p1.x;
        right2 = block_size - left1;
        right1 = block_size;
    }

    if (p1.y > p2.y) {
        up1 = 0;
        up2 = p1.y - p2.y;
        down1 = block_size - up2;
        down2 = block_size;
    } else {
        up2 = 0;
        up1 = p2.y - p1.y;
        down2 = block_size - up1;
        down1 = block_size;
    }

    // compute E2 between every pair of neighboring patches
    if (right1 >= 0 && right2 >= 0 && down1 >= 0 && down2 >= 0) {
        int cols = right1 - left1;
        int rows = down1 - up1;

        double ssd = 0;
        for (int i = 0; i < rows; i++) {
            const auto *ptr1 = image_src.ptr<uchar>(i + up1 + px1.y - block_size / 2);
            const auto *ptr2 = image_src.ptr<uchar>(i + up2 + px2.y - block_size / 2);
            for (int j = 0; j < cols; j++) {
                double diff = ptr1[j + left1 + px1.x - block_size / 2] - ptr2[j + left2 + px2.x - block_size / 2];
                ssd += diff * diff;
            }
        }
        return ssd / (cols * rows);
    } else {
        return 0;
    }
}

// called when more than one structure lines exist
int *
StructurePropagation::BP(const vector<PointPos> &known_points, vector<PointPos> &unknown_points, const Mat &image_src) {
    // initialization
    int size = point_manager.getStackSize();
    unknown_points.clear();
    unknown_points.reserve(size);

    list<shared_ptr<MyNode>>::iterator iter;
    list<shared_ptr<MyNode>>::iterator end;
    point_manager.getStackIter(iter, end);

    // receive messages from neighbors
    for (; iter != end; iter++) {
        shared_ptr<MyNode> n = *iter;
        list<shared_ptr<Edge>> edges;
        n->getEdges(edges);
        // message only for next neighbor (the node that enqueued this node)
        computeMij(*n, n->getEdgeBegin(), image_src, known_points);
    }

    auto *sample_scores = (int *) malloc(size * sizeof(int));
    auto *bp_visited = (double *) malloc(known_points.size() * sizeof(double));

    list<shared_ptr<MyNode>>::reverse_iterator rev_itor;
    list<shared_ptr<MyNode>>::reverse_iterator rev_end;
    point_manager.getStackReverseIter(rev_itor, rev_end);

    // send updated messages back to neighbors
    for (int i = 0; rev_itor != rev_end; rev_itor++, i++) {
        shared_ptr<MyNode> n = *rev_itor;
        auto begin = n->getEdgeBegin();
        auto end = n->getEdgeEnd();
        auto edge_iter = begin;
        unknown_points.push_back(n->p);

        // messages for all neighbors
        for (edge_iter++; edge_iter != end; edge_iter++) {
            computeMij(*n, edge_iter, image_src, known_points);
        }

        // compute E1 for all possible xi
        int min_ind = 0;
        double min = INT64_MAX;
        for (int xi = 0; xi < known_points.size(); xi++) {
            bp_visited[xi] = ks * computeEs(n->p, known_points[xi]) + ki * computeEi(image_src, n->p, known_points[xi]);
        }

        // add up all messages sent to this node
        for (edge_iter = begin; edge_iter != end; edge_iter++) {
            double **toMptr = (*edge_iter)->getMbyTo(n->id);
            for (int i = 0; i < known_points.size(); i++) {
                bp_visited[i] += (*toMptr)[i];
            }
        }

        // find the optimal xi
        for (int i = 0; i < known_points.size(); i++) {
            if (bp_visited[i] < min) {
                min = bp_visited[i];
                min_ind = i;
            }
        }
        sample_scores[i] = min_ind;
    }

    // release resources
    point_manager.getStackIter(iter, end);
    for (; iter != end; iter++) {
        shared_ptr<MyNode> n = *iter;
        auto edge_iter = n->getEdgeBegin();
        auto end = n->getEdgeEnd();
        for (; edge_iter != end; edge_iter++) {
            double **M = (*edge_iter)->getMbyFrom(n->id);
            if (*M != nullptr) {
                free(*M);
            }
        }
    }
    free(bp_visited);

    return sample_scores;
}

// used in both DP and BP
void
StructurePropagation::computeMij(MyNode &n, const list<shared_ptr<Edge>>::iterator &edge_iter, const Mat &image_src,
                                 const vector<PointPos> &known_points) {
    double **Mptr = (*edge_iter)->getMbyFrom(n.id);
    auto end = n.getEdgeEnd();

    if (*Mptr == nullptr) {
        *Mptr = (double *) malloc(known_points.size() * sizeof(double));
        memset(*Mptr, 0, known_points.size() * sizeof(double));

        for (int i = 0; i < known_points.size(); i++) {
            double E1 = ks * computeEs(n.p, known_points[i]) + ki * computeEi(image_src, n.p, known_points[i]);

            // add up messages sent from   (k != j)
            double msg = 0;
            for (auto iter = n.getEdgeBegin(); iter != end; iter++) {
                if (iter != edge_iter) {
                    double **toMptr = (*iter)->getMbyTo(n.id);
                    if (*toMptr == nullptr) {
                        assert(0);
                    }
                    msg += (*toMptr)[i];
                }
            }

            PointPos tmpPos = point_manager.getPointPos((*edge_iter)->getAnother(n.id));
            for (int j = 0; j < known_points.size(); j++) {
                // update each item in Mij
                double E2 = computeE2(image_src, n.p, tmpPos, known_points[i], known_points[j]);
                if ((*Mptr)[j] == 0 || E1 + E2 + msg < (*Mptr)[j]) {
                    (*Mptr)[j] = E1 + E2 + msg;
                }
            }
        }
    }
}

// update mask and unknown points, call photometric correction and get final result
void StructurePropagation::getResult(Mat mask, int *sample_indices, const vector<PointPos> &known_points,
                                     vector<PointPos> &unknown_points, Mat &result) {
    // update mask
    vector<vector<int>> tmp_mask(mask.rows, vector<int>(mask.cols, 0));
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            tmp_mask[i][j] = (mask.at<uchar>(i, j) > 0);
            if (tmp_mask[i][j]) {
                mask.at<uchar>(i, j) = 255;
            }
        }
    }

    Photometric::initMask(result, mask);

    int offset1 = block_size / 2;
    int offset2 = block_size - offset1;

    // copy all sample patches to corresponding unknown patches
    for (int i = 0; i < unknown_points.size(); i++) {
        Point src = point_manager.getPoint(known_points[sample_indices[i]]);
        Point tar = point_manager.getPoint(unknown_points[i]);

        Mat patch = result(Rect(src.x - offset1, src.y - offset1, block_size, block_size)).clone();
        Photometric::correctE(patch, src.x - offset1, src.y - offset1);

        for (int m = -offset1; m < offset2; m++) {
            const Vec3b *srcPtr = result.ptr<Vec3b>(src.y + m);
            for (int n = -offset1; n < offset2; n++) {
                Vec3b tmp = result.at<Vec3b>(tar.y + m, tar.x + n);
                if (tmp_mask[tar.y + m][tar.x + n] == 0) {
                    result.at<Vec3b>(tar.y + m, tar.x + n) = srcPtr[src.x + n];
                    tmp_mask[tar.y + m][tar.x + n] = 1;
                } else {
                    result.at<Vec3b>(tar.y + m, tar.x + n) = AlphaBlending(srcPtr[src.x + n],
                                                                           result.at<Vec3b>(tar.y + m, tar.x + n), 0.5);
                }
            }
        }
    }
    free(sample_indices);
}