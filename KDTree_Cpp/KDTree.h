#include <vector>
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>
#include <iostream>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

#include<boost/python.hpp>
#include<boost/python/numpy.hpp>

#ifndef KDTREE_LIBRARY_H
#define KDTREE_LIBRARY_H

namespace ublas = boost::numeric::ublas;
namespace py = boost::python;
namespace np = boost::python::numpy;

typedef ublas::matrix<double> RWMatrix;
typedef std::shared_ptr<const ublas::matrix<double>> ROMatrixSPtr;
typedef std::unique_ptr<const ublas::matrix<double>> ROMatrixUPtr;

typedef ublas::vector<double> RWVector;
typedef std::shared_ptr<const ublas::vector<double>> ROVectorSPtr;
typedef std::unique_ptr<const ublas::vector<double>> ROVectorUPtr;

typedef std::unique_ptr<std::vector<double>> StdVectorUPtr;

class DataPoint{
public:
    DataPoint() {};
    DataPoint(StdVectorUPtr &input_seq, bool with_orig_idx);
    DataPoint(const DataPoint & dataPoint);

    double GetValue(unsigned int idx) const { return data(idx);};
    ROVectorSPtr GetDataVectorPtr() { return std::make_shared<const RWVector>(data); };
    int GetOrigIdx() const {return orig_idx;};
    bool HasOrigIdx() const { return orig_idx != -1; }
private:
    RWVector data;
    int orig_idx = -1;
};

class AssociatedDataPoint : public DataPoint{
public:
    AssociatedDataPoint(const DataPoint & dataPoint, double distance) :
            DataPoint(dataPoint){
        this->distance = distance;
    };

    // we only need < , but why not have all three
    friend bool operator<(const AssociatedDataPoint& lhs, const AssociatedDataPoint& rhs) {
        return lhs.distance < rhs.distance;
    }
    friend bool operator>(const AssociatedDataPoint& lhs, const AssociatedDataPoint& rhs) {
        return lhs.distance > rhs.distance;
    }
    friend bool operator==(const AssociatedDataPoint& lhs, const AssociatedDataPoint& rhs) {
        return lhs.distance == rhs.distance;
    }
    double GetDistance() const { return distance; };
private:
    double distance;
};

class KDTreeNode {
public:
    KDTreeNode(StdVectorUPtr &input_seq, bool with_orig_idx);

    void SetLeft(std::shared_ptr<KDTreeNode> input) { this->left =  input; };
    void SetRight(std::shared_ptr<KDTreeNode> input) { this->right =  input; };

    const std::shared_ptr<KDTreeNode> GetLeft() { return left; };
    const std::shared_ptr<KDTreeNode> GetRight() { return right; }

    bool IsLeaf() { return ((left == nullptr) && (right == nullptr));};
    double GetValue(unsigned int idx) { return data_ptr->GetValue(idx);};
    int GetOrigIdx() const  {return data_ptr->GetOrigIdx();};
    bool HasOrigIdx() const { return  data_ptr->HasOrigIdx();};
    std::unique_ptr<DataPoint>& GetDataPtr() {return data_ptr;};

    double GetKernelPtDist(ROVectorUPtr &target, ROMatrixUPtr &cov_inv_matrix){
        RWVector diff = *(data_ptr->GetDataVectorPtr()) - (*target);
        RWVector temp = ublas::prod(diff, (*cov_inv_matrix));
        return 1.0 - std::exp((-1) * ublas::inner_prod(temp, diff));

//        return 1.0 - std::exp((-1) * ublas::prod(*(data_ptr->GetDataVectorPtr()) - (*target), (*cov_inv_matrix))(0));
    }

private:
    std::unique_ptr<DataPoint> data_ptr = nullptr;
    std::shared_ptr<KDTreeNode> left = nullptr;
    std::shared_ptr<KDTreeNode> right = nullptr;
};

class KDTreeInputDataComp {
public:
    KDTreeInputDataComp (int dim_idx) { this->dim_idx = dim_idx; };
    bool operator () (StdVectorUPtr & vec_a, StdVectorUPtr & vec_b ) {
        return (*vec_a)[dim_idx] > (*vec_b)[dim_idx];
    };
private:
    int dim_idx;
};

class KDTree {
public:
    std::shared_ptr<KDTreeNode> GetRoot() { return root; };
    void BuildTree(std::vector<StdVectorUPtr> &data);
    void BuildTreeNp(np::ndarray &data);

    void Insert(StdVectorUPtr &new_data_seq, bool with_orig_idx);
    void InsertNp(np::ndarray &data);

    void kNNSearch(int depth, std::shared_ptr<KDTreeNode> current, int kdim, int num_near,
                   ROVectorUPtr &center,
                   ROMatrixUPtr &cov_inv_matrix,
                   std::priority_queue<AssociatedDataPoint> &max_heap);

    np::ndarray kNNNp(np::ndarray &np_center, int num_near, np::ndarray &np_cov_inv_matrix);

    int GetTreeSize() { return tree_size; };
private:
    std::shared_ptr<KDTreeNode> BuildTreeRec(int depth, std::vector<StdVectorUPtr> &data);
    std::shared_ptr<KDTreeNode> root = nullptr;
    int tree_size = 0;
};

double static kernel_space_dist(double kd_node_val, double target_val, double cov_inv) {
    double diff = kd_node_val - target_val;
    return 1.0 - exp((-1) * diff * cov_inv * diff);
}


#endif