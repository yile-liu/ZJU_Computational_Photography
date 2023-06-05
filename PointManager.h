#ifndef POINT_MANAGER_H
#define POINT_MANAGER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <memory>

using namespace std;
using namespace cv;

class Endpoints
{
public:
	int true_line_index;
	int start_index;
	int end_index;
};

class PointPos
{
public:
	int line_index;
	int point_index;

	explicit PointPos(int lineIndex = -1, int pointIndex = -1) : line_index(lineIndex), point_index(pointIndex)
	{
	}
};

class Edge;

class MyNode
{
private:
	list<shared_ptr<Edge>> edges;
	int edge_num;

public:
	const PointPos p;
	const ushort id;
	static ushort total_num;

	explicit MyNode(PointPos p) : p(p), id(++total_num)
	{
		edge_num = 0;
	}
	void push_front(const shared_ptr<Edge> &e)
	{
		edges.push_front(e);
		edge_num++;
	}
	void push_back(const shared_ptr<Edge> &e)
	{
		edges.push_back(e);
		edge_num++;
	}
	void eraseEdge(shared_ptr<Edge> &e)
	{
		edges.remove(e);
		edges.push_back(e);
		edge_num--;
	}
	list<shared_ptr<Edge>>::iterator getEdgeBegin()
	{
		return edges.begin();
	}
	list<shared_ptr<Edge>>::iterator getEdgeEnd()
	{
		return edges.end();
	}
	void getEdges(list<shared_ptr<Edge>> &edges)
	{
		edges = this->edges;
	}
	int getEdgeNum() const
	{
		return edge_num;
	}
};

class Edge
{
private:
	double *Mij;
	double *Mji;

public:
	ushort ni{};
	ushort nj{};

	Edge(ushort ni, ushort nj) : ni(ni), nj(nj)
	{
		Mij = Mji = nullptr;
	}

	Edge()
	{
		Mij = Mji = nullptr;
	}

	inline ushort getAnother(ushort n) const
	{
		if (n == ni)
		{
			return nj;
		}
		else
		{
			return ni;
		}
	}

	inline double **getMbyFrom(ushort from)
	{
		if (from == ni)
		{
			return &Mij;
		}
		else
		{
			return &Mji;
		}
	}

	inline double **getMbyTo(ushort to)
	{
		if (to == ni)
		{
			return &Mji;
		}
		else
		{
			return &Mij;
		}
	}
};

class PointManager
{
public:
	PointManager() = default;
	void init(const vector<vector<Point>> &structure_lines, const Mat1b &mask, int block_size, set<shared_ptr<list<int>>> &line_sets);
	Point getPoint(PointPos p);
	bool nearBoundary(PointPos p);
	void getPointsInPatch(const PointPos &p, list<Point *> &begin, list<int> &length);
	void getKnownPoint(vector<PointPos> &samples, int sample_step, list<int> &line);
	void constructBpMap(list<int> &line);
	void getUnknownPoint(vector<PointPos> &anchors, list<int> &line);
	void getStackIter(list<shared_ptr<MyNode>>::iterator &begin, list<shared_ptr<MyNode>>::iterator &end);
	void getStackReverseIter(list<shared_ptr<MyNode>>::reverse_iterator &begin, list<shared_ptr<MyNode>>::reverse_iterator &end);
	int getStackSize()
	{
		return propagation_stack.size();
	}
	PointPos getPointPos(ushort id)
	{
		return (*nodes[id])->p;
	}


private:
	vector<vector<Point>> structure_line_points; // 记录用户绘制的点的信息
	Mat1b mask;
	int block_size{};
	vector<Endpoints> line_ends;			// 用于记录经过PointManager再次划分后的线段的首尾信息
	set<PointPos> boundary_points;			// 用于记录所在patch与边界重叠的锚点
	map<int, list<PointPos>> intersect_map; // 用于记录交点，键值为根据交点的真实坐标计算出的hash值
	map<int, list<PointPos>> out_intersect_map;
	vector<list<shared_ptr<MyNode>>::iterator> nodes; // 一张根据Node id查找node的表，记录着Node对象在双向链表中的迭代器
	list<shared_ptr<MyNode>> propagation_stack;		  // 记录BP算法中信息传递的顺序

	bool nearBoundary(const Point &p, bool is_sample);
	int getHashValue(int x, int y);
	int addNeighbor(MyNode &n, const PointPos &pos, vector<vector<ushort>> &visited_points, list<shared_ptr<MyNode>> &bfs_stack);
};

#endif /* POINT_MANAGER_H */