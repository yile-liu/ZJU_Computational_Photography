#include "PointManager.h"

#define getLineIndex(x) (x - 1) >> 24
#define getPointIndex(x) (x - 1) << 8 >> 8
#define visit(l, p) ((l << 24) | (p + 1))

ushort MyNode::total_num = 0;

bool operator<(const PointPos &p1, const PointPos &p2)
{
	return (p1.line_index == p2.line_index) ? p1.point_index < p2.point_index : p1.line_index < p2.line_index;
}

bool operator==(const Edge &e1, const Edge &e2)
{
	return (e1.ni == e2.ni) & (e1.nj == e2.nj);
}

/// set parameters for the PointManager and contert the structure_lines to line_set
void PointManager::init(const vector<vector<Point>> &structure_lines, const Mat1b &mask, int block_size, set<shared_ptr<list<int>>> &line_set)
{
	// clean old data
	this->structure_line_points = structure_lines;
	this->mask = mask;
	this->block_size = block_size;
	line_ends.clear();
	boundary_points.clear();
	intersect_map.clear();
	out_intersect_map.clear();

	Mat visited_map = Mat::zeros(mask.rows, mask.cols, CV_32SC1);
	bool was_in_mask = false;   // indicate whether the previous point is in the mask area
	Endpoints endpoints;

    // traverse structure lines
	for (int j = 0; j < structure_lines.size(); j++)
	{
        // traverse points on a line
		int i;
		for (i = 0; i < structure_lines[j].size(); i++)
		{
			int y = structure_lines[j][i].y;
			int x = structure_lines[j][i].x;
			// 1.1 corner case: out of image border
            if (y < 0 || y >= mask.rows || x < 0 || x >= mask.cols)
			{
				continue;
			}
            // 1.2 current point is outside the mask
			else if (mask.at<uchar>(y, x))
			{
				// previous point is inside the mask, then the current point is the end of a line
				if (was_in_mask)
				{
					if (nearBoundary(structure_lines[j][i], true))
					{
						boundary_points.insert(PointPos(line_ends.size(), i));
					}
					else
					{
						endpoints.end_index = i;
						line_ends.push_back(endpoints);
                        was_in_mask = false;
					}
					endpoints.end_index = i;
					line_ends.push_back(endpoints);
                    was_in_mask = false;
				}
			}
            // 1.3 current point is inside the mask
			else
			{
                // previous point is outside the mask, the current point is the start of a line
                if (!was_in_mask)
                {
                    endpoints.start_index = i;
                    endpoints.true_line_index = j;
                    was_in_mask = true;
                }

				// mark this point if it is near the boundary
				if (nearBoundary(structure_lines[j][i], false))
				{
					boundary_points.insert(PointPos(line_ends.size(), i));
				}
			}

            // 2.1 if the current point is visited, it is an intersection point
            int visit_record = visited_map.at<int>(y, x);
            int line_index = getLineIndex(visit_record);
            int point_index = getPointIndex(visit_record);
            if (visit_record != 0 && line_index != line_ends.size())
            {
                // record the position info of overlapping points in the intersect_map
                list<PointPos> *intersect_list = &intersect_map[getHashValue(x, y)];
                if (intersect_list->empty())
                {
                    intersect_list->push_back(PointPos(line_index, point_index));
                }
                intersect_list->push_back(PointPos(line_ends.size(), i));
            }
            // 2.2 if the current point is not visited, mark it
            else
            {
                visited_map.at<int>(y, x) = visit(line_ends.size(), i);
            }
		}

		// handle if line ends in mask area
		if (was_in_mask)
		{
            was_in_mask = false;
			endpoints.end_index = i;
			line_ends.push_back(endpoints);
		}
	}

    // generate line_set based on how structure lines intersect
	vector<shared_ptr<list<int>>> line_set_record(line_ends.size());
	map<int, list<PointPos>>::iterator map_iter;
	list<PointPos>::iterator list_iter;
	for (int i = 0; i < line_ends.size(); i++)
	{
		shared_ptr<list<int>> ptr = make_shared<list<int>>();
		ptr->push_back(i);
		line_set_record[i] = ptr;
	}
	for (map_iter = intersect_map.begin(); map_iter != intersect_map.end(); map_iter++)
	{
		shared_ptr<list<int>> ptr = nullptr;
		for (list_iter = map_iter->second.begin(); list_iter != map_iter->second.end(); list_iter++)
		{
			if (line_set_record[list_iter->line_index] != nullptr)
			{
				if (ptr == nullptr)
				{
					ptr = line_set_record[list_iter->line_index];
				}
				else // merge two lists
				{
					ptr->insert(ptr->end(), line_set_record[list_iter->line_index]->begin(), line_set_record[list_iter->line_index]->end());
				}
			}
		}
		list<int>::iterator iter;
		for (iter = ptr->begin(); iter != ptr->end(); iter++)
		{
			line_set_record[*iter] = ptr;
		}
	}

	for (int i = 0; i < line_set_record.size(); i++)
	{
		line_set.insert(line_set_record[i]);
	}
}

bool PointManager::nearBoundary(const Point &p, bool is_sample)
{
	int leftBound = MAX(p.x - block_size / 2, 0);
	int rightBound = MIN(p.x + block_size - block_size / 2, mask.cols);
	int upBound = MAX(p.y - block_size / 2, 0);
	int downBound = MIN(p.y + block_size - block_size / 2, mask.rows);
	const uchar *upPtr = mask.ptr<uchar>(upBound);
	const uchar *downPtr = mask.ptr<uchar>(downBound - 1);

	// check if the mask boundary crosses up and down boundary of the patch
	for (int i = leftBound; i < rightBound; i++)
	{
		if (!upPtr[i] == is_sample || !downPtr[i] == is_sample)
		{
			return true;
		}
	}

	// check if the mask boundary crosses left and right boundary of the patch
	for (int i = upBound + 1; i < downBound - 1; i++)
	{
		if (!mask.at<uchar>(i, leftBound) == is_sample || !mask.at<uchar>(i, rightBound - 1) == is_sample)
		{
			return true;
		}
	}

	return false;
}

// get hash value of a point's coordinate
inline int PointManager::getHashValue(int x, int y)
{
	return x + y * mask.cols;
}

// get Point Object by PointPos object
inline Point PointManager::getPoint(PointPos p)
{
	return structure_line_points[line_ends[p.line_index].true_line_index][p.point_index];
}

// check if the patch crosses the boundary
bool PointManager::nearBoundary(PointPos p)
{
	return boundary_points.count(p);
}

// get all points of the line segment contained in this patch
void PointManager::getPointsInPatch(const PointPos &p, list<Point *> &begin, list<int> &length)
{
	Point center = getPoint(p);
	int leftBound = MAX(center.x - block_size / 2, 0);
	int rightBound = MIN(center.x + block_size - block_size / 2, mask.cols);
	int upBound = MAX(center.y - block_size / 2, 0);
	int downBound = MIN(center.y + block_size - block_size / 2, mask.rows);
	int hashValue = getHashValue(center.x, center.y);
	list<PointPos> pointPositions;
	bool in_mask = true;
	// check if the anchor point is an intersection
	// if it is, points of several line segments will be returned
	if (intersect_map.count(hashValue))
	{
		pointPositions = intersect_map[hashValue];
	}
	else if (out_intersect_map.count(hashValue))
	{
		pointPositions = out_intersect_map[hashValue];
		in_mask = false;
	}
	else
	{
		pointPositions.push_back(p);
	}
	for (auto & pointPosition : pointPositions)
	{
		int true_line_index = (in_mask) ? line_ends[pointPosition.line_index].true_line_index : pointPosition.line_index;
		Point *points = &structure_line_points[true_line_index][0];
		int beginIndex = pointPosition.point_index;
		// find the start index of the line segment
		for (int i = pointPosition.point_index; i >= 0; i--)
		{
			if (points[i].x < leftBound || points[i].y < upBound || points[i].x >= rightBound || points[i].y >= downBound)
			{
				beginIndex = i + 1;
				break;
			}
		}
		begin.push_back(points + beginIndex);
		// get anchor points backward
		int i;
		for (i = pointPosition.point_index; i < structure_line_points[true_line_index].size(); i++)
		{
			if (points[i].x < leftBound || points[i].y < upBound || points[i].x >= rightBound || points[i].y >= downBound)
			{
				length.push_back(i - beginIndex);
				break;
			}
		}
		if (i == structure_line_points[true_line_index].size())
		{
			length.push_back(structure_line_points[true_line_index].size() - beginIndex);
		}
	}
}

// generate the order of BP procedure, result is pushed in propagation_stack
void PointManager::constructBpMap(list<int> &line)
{
	map<int, list<PointPos>>::iterator map_iter;
	list<shared_ptr<MyNode>> bfs_stack;
	vector<vector<ushort>> visited_points(structure_line_points.size());
	vector<list<shared_ptr<MyNode>>> node_list(4);
	set<int> intersection_set;

	nodes.clear();
	propagation_stack.clear();
	MyNode::total_num = 0;

	// initialize visited map
	visited_points.resize(structure_line_points.size());
	for (int i = 0; i < structure_line_points.size(); i++)
	{
		visited_points[i].resize(structure_line_points[i].size());
	}

	int total = 0;
	list<int>::iterator iter;
	for (iter = line.begin(); iter != line.end(); iter++)
	{
		total += line_ends[*iter].end_index - line_ends[*iter].start_index;
		intersection_set.insert(*iter);
	}

	// vacate enough space for the node table
	nodes.reserve(total / block_size + 1);

	// skip the entry numbered 0
	nodes.resize(1);

	// enqueue all the intersections
	for (map_iter = intersect_map.begin(); map_iter != intersect_map.end(); map_iter++)
	{
		if (intersection_set.count(map_iter->second.begin()->line_index) == 0)
		{
			continue;
		}
		// enqueue the intersection (choose one point's position to represent all)
		bfs_stack.push_back(make_shared<MyNode>(*(map_iter->second.begin())));
		// mark all intersecting points as visited
		auto list_iter = map_iter->second.begin();
		for (; list_iter != map_iter->second.end(); list_iter++)
		{
            visited_points[line_ends[list_iter->line_index].true_line_index][list_iter->point_index] = MyNode::total_num;
		}
	}

	// enqueue all the neighbor nodes of intersections
	for (map_iter = intersect_map.begin(); map_iter != intersect_map.end(); map_iter++)
	{
		if (intersection_set.count(map_iter->second.begin()->line_index) == 0)
		{
			continue;
		}
		shared_ptr<MyNode> n = *bfs_stack.begin();
		auto list_iter = map_iter->second.begin();
		int neighbor_cnt = 0;
		for (; list_iter != map_iter->second.end(); list_iter++)
		{
            neighbor_cnt += addNeighbor(*n, *list_iter, visited_points, bfs_stack);
		}
		// Enlarge node_list if necessary
		if (neighbor_cnt > node_list.size())
		{
			node_list.resize(map_iter->second.size() * 2);
		}
		node_list[neighbor_cnt - 1].push_front(n);
		nodes.push_back(node_list[neighbor_cnt - 1].begin());
		bfs_stack.pop_front();
	}

	// pseudo-recurse to generate all neighbor relation
	while (!bfs_stack.empty())
	{
		shared_ptr<MyNode> n = *bfs_stack.begin();
		int neighborNum = addNeighbor(*n, n->p, visited_points, bfs_stack);
		node_list[neighborNum - 1].push_front(n);
		nodes.push_back(node_list[neighborNum - 1].begin());
		bfs_stack.pop_front();
	}

	// generate the sequence of message sending
	while (!node_list[0].empty())
	{
		shared_ptr<MyNode> n = *node_list[0].begin();
		if (n->getEdgeNum() != 1)
		{
			assert(n->getEdgeNum() == 1);
		}
		auto edge_iter = n->getEdgeBegin();
		nodes[n->id] = propagation_stack.insert(propagation_stack.end(), n);
		node_list[0].pop_front();
		// degrade the adjacent node
		int id = (*edge_iter)->getAnother(n->id);
		n = *nodes[id];
		int edgeNum = n->getEdgeNum();
		n->eraseEdge(*edge_iter);
		node_list[edgeNum - 1].erase(nodes[id]);
		if (edgeNum > 1)
		{
			nodes[id] = node_list[edgeNum - 2].insert(node_list[edgeNum - 2].end(), n);
		}
		else
		{
			nodes[id] = propagation_stack.insert(propagation_stack.end(), n);
		}
	}
}

int PointManager::addNeighbor(MyNode &n, const PointPos &pos, vector<vector<ushort>> &visited_points, list<shared_ptr<MyNode>> &bfs_stack)
{
	Endpoints endpoints = line_ends[pos.line_index];
	int line_index = endpoints.true_line_index;
	int point_index = pos.point_index;
	int prev_point_ind = point_index - block_size / 2;
	int next_point_ind = point_index + block_size / 2;
	int neighbor_cnt = 0;

	// check the point before current anchor point
	if (prev_point_ind >= endpoints.start_index)
	{
		int i;
		// try choosing a existed anchor point as its neighbor
		for (i = prev_point_ind; i < point_index; i++)
		{
			if (visited_points[line_index][i])
			{
				if (nodes.size() > visited_points[line_index][i])
				{
					// add an edge between two points
					shared_ptr<Edge> tmpEdge = make_shared<Edge>(n.id, visited_points[line_index][i]);
					n.push_front(tmpEdge);
					(*nodes[visited_points[line_index][i]])->push_back(tmpEdge);
				}
				break;
			}
		}
		// no existed point can be chosen, construct a new anchor point and enqueue it
		if (i == point_index)
		{
			bfs_stack.push_back(make_shared<MyNode>(PointPos(pos.line_index, prev_point_ind)));
            visited_points[line_index][prev_point_ind] = MyNode::total_num;
		}
		neighbor_cnt++;
	}

	// check the point behind current anchor point
	if (next_point_ind < endpoints.end_index)
	{
		int i;
		// try choosing an existed anchor point as its neighbor
		for (i = next_point_ind; i > point_index; i--)
		{
			if (visited_points[line_index][i])
			{
				if (nodes.size() > visited_points[line_index][i])
				{
					// add an edge between two points
					shared_ptr<Edge> tmpEdge = make_shared<Edge>(n.id, visited_points[line_index][i]);
					n.push_front(tmpEdge);
					(*nodes[visited_points[line_index][i]])->push_back(tmpEdge);
				}
				break;
			}
		}
		// no existed point can be chosen, construct a new anchor point and enqueue it
		if (i == point_index)
		{
			bfs_stack.push_back(make_shared<MyNode>(PointPos(pos.line_index, next_point_ind)));
            visited_points[line_index][next_point_ind] = MyNode::total_num;
		}
		neighbor_cnt++;
	}
	return neighbor_cnt;
}

void PointManager::getStackIter(list<shared_ptr<MyNode>>::iterator &begin, list<shared_ptr<MyNode>>::iterator &end)
{
	begin = propagation_stack.begin();
	end = propagation_stack.end();
}

void PointManager::getStackReverseIter(list<shared_ptr<MyNode>>::reverse_iterator &begin, list<shared_ptr<MyNode>>::reverse_iterator &end)
{
	begin = list<shared_ptr<MyNode>>::reverse_iterator(propagation_stack.end())++;
	end = list<shared_ptr<MyNode>>::reverse_iterator(propagation_stack.begin());
}

// sample the line and collect known points, which are outside mask
void PointManager::getKnownPoint(vector<PointPos> &samples, int sample_step, list<int> &line)
{
	if (line_ends.empty())
	{
		return;
	}
	samples.clear();
	Endpoints endpoints = line_ends[*line.begin()];

	// reserve enough space for samples
	int total = 0;
	int curr_line_ind = -1;
	list<int>::iterator iter;
	for (iter = line.begin(); iter != line.end(); iter++)
	{
		if (line_ends[*iter].true_line_index != curr_line_ind)
		{
            curr_line_ind = line_ends[*iter].true_line_index;
			total += structure_line_points[curr_line_ind].size();
		}
		total -= (line_ends[*iter].end_index - line_ends[*iter].start_index);
	}
	samples.reserve(total / sample_step);

    // sample the line with sample_step
	iter = line.begin();
	for (int i = 0; i < structure_line_points.size(); i++)
	{
		// +block_size: ensure all samples have complete line segments
		if (endpoints.true_line_index != i)
		{
			continue;
		}
		int begin_index = block_size;
		int end_index;
		while (endpoints.true_line_index == i)
		{
            end_index = endpoints.start_index;
			for (int j = end_index - 1; j >= begin_index; j -= sample_step)
			{
				if (j == begin_index)
				{
					int c = 0;
					c++;
				}
				if (!nearBoundary(structure_line_points[i][j], true))
				{
					samples.emplace_back(*iter, j);
				}
			}
            begin_index = endpoints.end_index;
			iter++;
			if (iter == line.end())
			{
				break;
			}
			endpoints = line_ends[*iter];
		}
		// -block_size: ensure all samples have complete line segments
		end_index = structure_line_points[i].size() - block_size;
		for (int j = end_index - 1; j >= begin_index; j -= sample_step)
		{
			if (!nearBoundary(structure_line_points[i][j], true))
			{
				samples.emplace_back(*(list<int>::reverse_iterator(iter)++), j);
			}
		}
	}
	samples.shrink_to_fit();
}

// sample the line and collect unknown points, which are inside mask
void PointManager::getUnknownPoint(vector<PointPos> &anchors, list<int> &line)
{
	anchors.clear();
	Endpoints endpoints = line_ends[*line.begin()];

	// reserve enough space for anchors
	int total = 0;
	list<int>::iterator iter;
	for (iter = line.begin(); iter != line.end(); iter++)
	{
		total += (line_ends[*(iter)].end_index - line_ends[*(iter)].start_index);
	}
	anchors.reserve(total / (block_size / 2));

	iter = line.begin();
	for (int i = 0; i < structure_line_points.size(); i++)
	{
		while (endpoints.true_line_index == i)
		{
			for (int j = endpoints.start_index; j < endpoints.end_index; j += block_size / 2)
			{
				anchors.emplace_back(*iter, j);
			}
			++iter;
			if (iter == line.end())
			{
				break;
			}
			endpoints = line_ends[*iter];
		}
	}
	anchors.shrink_to_fit();
}