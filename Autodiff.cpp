//
// Created by 余乐 on 25-2-28.
//

#include "Autodiff.h"
using namespace std;


void build_topo(autodiff *node, vector<autodiff *> &topo, vector<autodiff *> &visited) {
    if (find(visited.begin(), visited.end(), node) == visited.end()) {
        visited.push_back(node);
        for (autodiff *child: node->_prev) {
            build_topo(child, topo, visited);
        }
        topo.push_back(node);
    }
}

void autodiff::backward() {
    vector<autodiff *> topo;
    vector<autodiff *> visited;
    build_topo(this, topo, visited);

    // 初始化所有梯度为 0
    for (auto node: topo) {
        node->grad = MatrixXf::Zero(node->data.rows(), node->data.cols());
    }

    this->grad = MatrixXf::Ones(this->data.rows(), this->data.cols());

    // 反向传播
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->_backward) {
            (*it)->_backward();
        }
    }
}