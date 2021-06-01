// *******************************************************
// This is a devil's version of KDTree
// It has not been tested and will not be tested
// You have been WARNED
// *******************************************************

#include "KDTree.h"

// Constructor for DataPoint
DataPoint::DataPoint(StdVectorUPtr &input_seq, bool with_orig_idx){
    auto size = input_seq->size();
    if (with_orig_idx) {
        orig_idx = (int) input_seq->back();
        size--;
    } else {
        orig_idx = -1;
    }

    this->data = RWVector(size);
    for (auto idx = 0; idx < size; ++idx)
        data(idx) = (*input_seq)[idx];
}

DataPoint::DataPoint(const DataPoint & dataPoint){
    orig_idx = dataPoint.orig_idx;
    auto size = dataPoint.data.size();
    this->data = RWVector(size);
    for (auto idx = 0; idx < size; ++idx)
        data(idx) = dataPoint.data(idx);
}

KDTreeNode::KDTreeNode(StdVectorUPtr &input_seq, bool with_orig_idx){
    data_ptr = std::unique_ptr<DataPoint>(new DataPoint(input_seq,with_orig_idx));
}

std::shared_ptr<KDTreeNode> KDTree::BuildTreeRec(int depth, std::vector<StdVectorUPtr> &data) {

    if (data.empty())
        return nullptr;

    if (data[0] == nullptr)
        return nullptr;

    auto ki_dim = data[0]->size();

    std::shared_ptr<KDTreeNode> sub_root = nullptr;
    auto num_samples = data.size();

    // if leaf node
    if (num_samples == 1) {
        // since there is only one sample
        sub_root = std::make_shared<KDTreeNode>(KDTreeNode(data[0], true));
        tree_size = tree_size + 1;
        return sub_root;
    }

    std::sort(data.begin(), data.end(), KDTreeInputDataComp(depth % (ki_dim - 1)));

    // build sub tree
    auto med_idx = num_samples / 2;
    sub_root = std::make_shared<KDTreeNode>(KDTreeNode(data[med_idx], true));
    tree_size = tree_size + 1;

    // populate left and right sub tree
    std::vector<StdVectorUPtr> left_data;
    std::vector<StdVectorUPtr> right_data;

    // Not extract necessary, but pre allocate memory make it faster
    auto num_left_samples = num_samples / 2;
    left_data.resize(num_left_samples);
    auto num_right_samples = num_samples - num_samples / 2 - 1;
    right_data.resize(num_right_samples);

    for (auto idx = 0; idx < data.size(); ++idx) {
        // if idx == med_idx
        // skip current node
        if (idx == med_idx)
            continue;
        // if idx < med point, move pointer into lower half
        if (idx < med_idx)
            left_data[idx] = std::move(data[idx]);
            // otherwise move pointer into higher half
        else
            right_data[idx-med_idx-1] = std::move(data[idx]);
    }

    // recursive call: construct left and right child
    // if data_ptr size is 0, BuildTreeRec will return nullptr
    sub_root->SetLeft(BuildTreeRec(depth + 1, left_data));
    sub_root->SetRight(BuildTreeRec(depth + 1, right_data));

    return sub_root;
}

void KDTree::BuildTree(std::vector<StdVectorUPtr> &data) {
    root = BuildTreeRec(0, data);
}

void KDTree::BuildTreeNp(np::ndarray &np_data){
    auto num_dim = np_data.get_nd();
    if(num_dim < 2)
        // should throw exception here
        return;

    auto num_row = np_data.shape(0);
    auto num_col = np_data.shape(1);

    std::vector<StdVectorUPtr> data;

    for(int i = 0; i < num_row; ++i){
        std::vector<double>* row = new std::vector<double>();
        for(int j = 0; j < num_col; ++j)
            row->push_back(std::atof(py::extract<char const *>(py::str(np_data[i][j]))));
        data.push_back(std::unique_ptr<std::vector<double>>(row));
    }

    BuildTree(data);
}

void KDTree::Insert(StdVectorUPtr &new_data_seq, bool with_orig_idx) {
    int depth = 0;
    int check_dim;
    double check_num_current;
    double check_num_new;
    auto current_node = root;

    int k_dim = (int) new_data_seq->size() - 1;

    bool is_leaf = current_node->IsLeaf();
    while (!is_leaf) {
        check_dim = depth % k_dim;

        check_num_new = (*new_data_seq)[check_dim];
        check_num_current = current_node->GetValue(check_dim);

        if (check_num_new <= check_num_current && current_node->GetLeft() != nullptr) {
            current_node = current_node->GetLeft();
        } else {
            is_leaf = true;
        }

        if(check_num_new > check_num_current && current_node->GetRight() != nullptr) {
            current_node = current_node->GetRight();
        } else {
            is_leaf = true;
        }

        ++depth;
    }

    check_dim = depth % k_dim;
    check_num_new = (*new_data_seq)[check_dim];
    check_num_current = current_node->GetValue(check_dim);

    auto new_node = std::make_shared<KDTreeNode>(KDTreeNode(new_data_seq, with_orig_idx));
    if (check_num_new <= check_num_current)
        current_node->SetLeft(new_node);
    else
        current_node->SetRight(new_node);

    tree_size = tree_size + 1;
}

void KDTree::InsertNp(np::ndarray &np_data){
    auto num_dim = np_data.get_nd();
//    if(num_dim < 1)
//        // should throw exception here
//        return;
    auto num_col = np_data.shape(0);

    StdVectorUPtr data;
    std::vector<double>* row = new std::vector<double>();

    for (int i = 0; i < num_col; ++i)
        row->push_back(std::atof(py::extract<char const *>(py::str(np_data[i]))));

    data = std::unique_ptr<std::vector<double>>(row);

    Insert(data, true);
}

void KDTree::kNNSearch(int depth, std::shared_ptr<KDTreeNode> current, int kdim, int num_near,
                       ROVectorUPtr &center,
                       ROMatrixUPtr &cov_inv_matrix,
                       std::priority_queue<AssociatedDataPoint> &max_heap) {

    if (current == nullptr) {
        return;
    }

    if (max_heap.size() < num_near) {
        double distance = current->GetKernelPtDist(center, cov_inv_matrix);
        // should use ptr instead ,but I am lazy
        max_heap.push(AssociatedDataPoint((*current->GetDataPtr()), distance));

        int dim = depth % kdim;
        double t_value = (*center)(dim);
        double r_value = current->GetValue(dim);
        if (t_value < r_value) {
            kNNSearch(depth + 1, current->GetLeft(), kdim, num_near, center, cov_inv_matrix, max_heap);
            kNNSearch(depth + 1, current->GetRight(), kdim, num_near, center, cov_inv_matrix, max_heap);
        } else {
            kNNSearch(depth + 1, current->GetRight(), kdim, num_near, center, cov_inv_matrix, max_heap);
            kNNSearch(depth + 1, current->GetLeft(), kdim, num_near, center, cov_inv_matrix, max_heap);
        }
    } else {
        double mh_max_dist = max_heap.top().GetDistance();
        int current_dim = depth % kdim;
        if (kernel_space_dist(current->GetValue(current_dim),
                              (*center)(current_dim), (*cov_inv_matrix)(current_dim, current_dim)) > mh_max_dist) {
            return;
        }

        double dist_to_center = current->GetKernelPtDist(center, cov_inv_matrix);

        if (dist_to_center < mh_max_dist) {
            max_heap.pop();
            max_heap.push(AssociatedDataPoint((*current->GetDataPtr()), dist_to_center));
        }

        int d = depth % kdim;
        double t_value = (*center)(d);
        double r_value = current->GetValue(d);
        if (t_value < r_value) {
            kNNSearch(depth + 1, current->GetLeft(), kdim, num_near, center, cov_inv_matrix, max_heap);
            kNNSearch(depth + 1, current->GetRight(), kdim, num_near, center, cov_inv_matrix, max_heap);
        } else {
            kNNSearch(depth + 1, current->GetRight(), kdim, num_near, center, cov_inv_matrix, max_heap);
            kNNSearch(depth + 1, current->GetLeft(), kdim, num_near, center, cov_inv_matrix, max_heap);
        }
    }
}

np::ndarray KDTree::kNNNp(np::ndarray &np_center, int num_near, np::ndarray &np_cov_inv_matrix){
    std::priority_queue<AssociatedDataPoint> max_heap;
    int kdim = np_center.shape(0);

    std::unique_ptr<RWVector>center_data_ptr(new RWVector(kdim));
    for (int i = 0; i < kdim; ++i)
        // NOTE, need fix this with data ptr and etc
        (*center_data_ptr)(i) = std::atof(py::extract<char const *>(py::str(np_center[i])));
    ROVectorUPtr center = ROVectorUPtr(std::move(center_data_ptr));

    std::unique_ptr<RWMatrix>cov_inv_matrix_ptr(new RWMatrix(kdim,kdim));
    for(int i = 0; i < kdim; ++i) {
        for(int j = 0; j < kdim; ++j) {
            (*cov_inv_matrix_ptr)(i, j) = std::atof(py::extract<char const *>(py::str(np_cov_inv_matrix[i][j])));
        }
    }
    ROMatrixUPtr cov_inv_matrix = ROMatrixUPtr(std::move(cov_inv_matrix_ptr));


    //Go with KNN Search
    kNNSearch(0, this->root, kdim, num_near, center, cov_inv_matrix, max_heap);

    // export data into 2d vector
    // in reverse order
    std::vector<std::vector<double>> data_vector;
    while(!max_heap.empty()){
        data_vector.push_back(std::vector<double>());
        for(int dim = 0 ; dim < kdim; ++dim){
            data_vector.back().push_back(max_heap.top().GetValue(dim));
        }

        // appending distance
        data_vector.back().push_back(max_heap.top().GetDistance());

        // appending orig idx
        if(max_heap.top().HasOrigIdx()) {
            data_vector.back().push_back(max_heap.top().GetOrigIdx());
        }

        max_heap.pop();
    }

    //Determine size of ndarray
    // ori array + distance
    int num_value_per_row = kdim + 1;
    if(max_heap.top().HasOrigIdx())
        num_value_per_row =  num_value_per_row + 1;

    //Make a 2d array with zeros
    np::ndarray knn_res = np::zeros(py::make_tuple(num_near,num_value_per_row),
            np::dtype::get_builtin<double>());

    // raw data ptr to the nd array, reinterpret_cast to double
    auto* knn_res_ptr = reinterpret_cast<double*>(knn_res.get_data());
    // populate return array in reverse
    // we want closed point on top
    for(auto record = data_vector.rbegin(); record != data_vector.rend(); ++record)
        for(auto & value : *record)
            (*knn_res_ptr ++) = value;

    return knn_res;
}


#include<boost/python.hpp>
#include<boost/python/numpy.hpp>

using namespace boost::python;
namespace np = boost::python::numpy;

BOOST_PYTHON_MODULE(pieKDTree){
    Py_Initialize();
    np::initialize();
    class_<KDTree>("KDTree")
            .def("BuildTree", &KDTree::BuildTreeNp)
            .def("Insert", &KDTree::InsertNp)
            .def("kNN", &KDTree::kNNNp)
            .def("GetTreeSize", &KDTree::GetTreeSize);
}