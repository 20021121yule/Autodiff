//
// Created by 余乐 on 25-2-28.
//

#ifndef AUTODIFF_H
#define AUTODIFF_H


#include <vector>
#include <memory>
#include <functional>
#include "Eigen/Dense"

using namespace Eigen;

class autodiff {
public:
    MatrixXf data; // 存储矩阵数据
    //对data继承了所有Eigen对矩阵的操作类型！
    mutable MatrixXf grad; // 梯度矩阵（与data同维度）可变参数
    std::vector<autodiff *> _prev; // 前驱节点
    std::function<void()> _backward; // 反向传播函数
    bool isCalculateGrad = false;

    // 禁用拷贝和移动
    autodiff(const autodiff &) = delete;

    autodiff &operator=(const autodiff &) = delete;

    // 构造函数：直接接受矩阵输入
    explicit autodiff(const MatrixXf &data): data(data), grad(MatrixXf::Zero(data.rows(), data.cols())) {
    }

    explicit autodiff(const MatrixXf &data, const bool &ComputeGradType)
        : data(data), grad(MatrixXf::Zero(data.rows(), data.cols())) {
        if (ComputeGradType) {
            isCalculateGrad = true;
        }
    }

    autodiff(const MatrixXf &data, std::vector<autodiff *> prev)
        : data(data), grad(MatrixXf::Zero(data.rows(), data.cols())), _prev(std::move(prev)) {
    }

    // 显式构造autodiff运算对象
    static std::shared_ptr<autodiff> create(const MatrixXf &data, const bool &requires_grad = false) {
        auto out = std::make_shared<autodiff>(data, requires_grad);
        return out;
    }

    // 运算符重载（需处理矩阵维度兼容性）, 此时 + 是矩阵加法
    friend std::shared_ptr<autodiff> operator+(const std::shared_ptr<autodiff> &a, const std::shared_ptr<autodiff> &b) {
        // 判断是否可以逐元素相加或广播
        const int a_rows = a->data.rows(), a_cols = a->data.cols();
        const int b_rows = b->data.rows(), b_cols = b->data.cols();

        // 检查广播是否合法
        const bool rows_ok = (a_rows == b_rows) || (a_rows == 1) || (b_rows == 1);
        const bool cols_ok = (a_cols == b_cols) || (a_cols == 1) || (b_cols == 1);
        if (!rows_ok || !cols_ok) {
            throw std::invalid_argument("Matrix dimension mismatch for broadcast");
        }

        MatrixXf result;

        // 逐元素加法
        if (a_rows == b_rows && a_cols == b_cols) {
            result = a->data + b->data;
        } else {
            // 计算广播后的维度
            const int out_rows = std::max(a_rows, b_rows);
            const int out_cols = std::max(a_cols, b_cols);

            // 扩展a和b到目标维度
            MatrixXf a_broadcast = a->data.replicate(out_rows / a_rows, out_cols / a_cols);
            MatrixXf b_broadcast = b->data.replicate(out_rows / b_rows, out_cols / b_cols);

            result = a_broadcast + b_broadcast;
        }

        auto out = std::make_shared<autodiff>(result, std::vector<autodiff *>{a.get(), b.get()});

        out->_backward = [a, b, out]() {
            // 处理梯度广播：将梯度累加到原始维度
            MatrixXf a_grad = out->grad;
            if (a->data.rows() == 1 && out->grad.rows() > 1) {
                a_grad = out->grad.colwise().sum();
            }
            if (a->data.cols() == 1 && out->grad.cols() > 1) {
                a_grad = a_grad.rowwise().sum();
            }
            a->grad += a_grad;

            MatrixXf b_grad = out->grad;
            if (b->data.rows() == 1 && out->grad.rows() > 1) {
                b_grad = out->grad.colwise().sum();
            }
            if (b->data.cols() == 1 && out->grad.cols() > 1) {
                b_grad = b_grad.rowwise().sum();
            }
            b->grad += b_grad;
        };

        return out;
    }

    // 张量 + 标量，我们认为加一个scalar相当于也是对a的一个操作，那么我们不需要返回scalar的梯度，并且不需要储存scalar的前向节点
    friend std::shared_ptr<autodiff> operator+(const std::shared_ptr<autodiff> &a, float scalar) {
        // 结果节点的输出
        auto out = std::make_shared<autodiff>((a->data.array() + scalar).eval(), std::vector<autodiff *>{a.get()});

        // 梯度计算
        out->_backward = [a, out]() {
            a->grad += out->grad; // 矩阵加法梯度直接传递，scalar相当于是一个常量矩阵，元素都是一样的
        };

        return out;
    }

    // 标量 + 张量
    friend std::shared_ptr<autodiff> operator+(float scalar, const std::shared_ptr<autodiff> &a) {
        // 结果节点的输出
        auto out = std::make_shared<autodiff>((a->data.array() + scalar).eval(), std::vector<autodiff *>{a.get()});

        // 梯度计算
        out->_backward = [a, out]() {
            a->grad += out->grad; // 矩阵加法梯度直接传递，scalar相当于是一个常量矩阵，元素都是一样的
        };

        return out;
    }


    // 运算符重载（需处理矩阵维度兼容性）
    friend std::shared_ptr<autodiff> operator-(const std::shared_ptr<autodiff> &a, const std::shared_ptr<autodiff> &b) {
        // 判断是否可以逐元素相减或广播
        const int a_rows = a->data.rows(), a_cols = a->data.cols();
        const int b_rows = b->data.rows(), b_cols = b->data.cols();

        // 检查广播是否合法
        const bool rows_ok = (a_rows == b_rows) || (a_rows == 1) || (b_rows == 1);
        const bool cols_ok = (a_cols == b_cols) || (a_cols == 1) || (b_cols == 1);
        if (!rows_ok || !cols_ok) {
            throw std::invalid_argument("Matrix dimension mismatch for broadcast");
        }

        MatrixXf result;

        // 逐元素加法
        if (a_rows == b_rows && a_cols == b_cols) {
            result = a->data - b->data;
        } else {
            // 计算广播后的维度
            const int out_rows = std::max(a_rows, b_rows);
            const int out_cols = std::max(a_cols, b_cols);

            // 扩展a和b到目标维度
            MatrixXf a_broadcast = a->data.replicate(out_rows / a_rows, out_cols / a_cols);
            MatrixXf b_broadcast = b->data.replicate(out_rows / b_rows, out_cols / b_cols);

            result = a_broadcast - b_broadcast;
        }

        auto out = std::make_shared<autodiff>(result, std::vector<autodiff *>{a.get(), b.get()});

        out->_backward = [a, b, out]() {
            // 处理梯度广播：将梯度累加到原始维度
            MatrixXf a_grad = out->grad;
            if (a->data.rows() == 1 && out->grad.rows() > 1) {
                a_grad = out->grad.colwise().sum();
            }
            if (a->data.cols() == 1 && out->grad.cols() > 1) {
                a_grad = a_grad.rowwise().sum();
            }
            a->grad += a_grad;

            MatrixXf b_grad = out->grad;
            if (b->data.rows() == 1 && out->grad.rows() > 1) {
                b_grad = out->grad.colwise().sum();
            }
            if (b->data.cols() == 1 && out->grad.cols() > 1) {
                b_grad = b_grad.rowwise().sum();
            }
            b->grad -= b_grad;
        };

        return out;
    }

    // 张量 - 标量
    friend std::shared_ptr<autodiff> operator-(const std::shared_ptr<autodiff> &a, float scalar) {
        // 结果节点的输出
        auto out = std::make_shared<autodiff>((a->data.array() - scalar).eval(), std::vector<autodiff *>{a.get()});

        // 梯度计算
        out->_backward = [a, out]() {
            a->grad += out->grad; // 矩阵加法梯度直接传递，scalar相当于是一个常量矩阵，元素都是一样的
        };

        return out;
    }

    // 标量 - 矩阵
    friend std::shared_ptr<autodiff> operator-(float scalar, const std::shared_ptr<autodiff> &a) {
        // 结果节点的输出
        auto out = std::make_shared<autodiff>((scalar - a->data.array()).eval(), std::vector<autodiff *>{a.get()});

        // 梯度计算
        out->_backward = [a, out]() {
            a->grad -= out->grad; // 矩阵加法梯度直接传递，scalar相当于是一个常量矩阵，元素都是一样的
        };

        return out;
    }

    friend std::shared_ptr<autodiff> operator*(const std::shared_ptr<autodiff> &a, const std::shared_ptr<autodiff> &b) {
        // 检查矩阵乘法维度兼容性
        if (a->data.cols() != b->data.rows()) {
            throw std::invalid_argument("Matrix multiplication dimension mismatch");
        }

        auto out = std::make_shared<autodiff>(a->data * b->data, std::vector<autodiff *>{a.get(), b.get()});

        out->_backward = [a, b, out]() {
            // 矩阵乘法梯度计算
            a->grad += out->grad * b->data.transpose(); // dL/da = dL/dout * b^T
            b->grad += a->data.transpose() * out->grad; // dL/db = a^T * dL/dout
        };

        return out;
    }

    // 逐元素乘积（Hadamard Product）
    friend std::shared_ptr<autodiff> mul(const std::shared_ptr<autodiff> &a, const std::shared_ptr<autodiff> &b) {
        // 检查维度是否完全一致
        if (a->data.rows() != b->data.rows() || a->data.cols() != b->data.cols()) {
            throw std::invalid_argument("Matrix dimension mismatch in element-wise multiplication");
        }

        // 创建输出节点
        auto out = std::make_shared<autodiff>(a->data.cwiseProduct(b->data).eval(),
                                              std::vector<autodiff *>{a.get(), b.get()});

        // 反向传播函数
        out->_backward = [a, b, out]() {
            // 梯度计算：dL/da = dL/dout ⊙ b, dL/db = dL/dout ⊙ a
            a->grad += out->grad.cwiseProduct(b->data); // 逐元素乘积
            b->grad += out->grad.cwiseProduct(a->data);
        };

        return out;
    }

    // 逐元素指数操作（注意：矩阵指数函数需特殊实现）
    std::shared_ptr<autodiff> exp(double power) const {
        auto out = std::make_shared<autodiff>(
            data.array().pow(power).matrix().eval(), // 逐元素幂运算
            std::vector<autodiff *>{const_cast<autodiff *>(this)}
        );

        out->_backward = [this, power, out]() {
            // 逐元素梯度计算
            this->grad += (power * data.array().pow(power - 1)).matrix().cwiseProduct(out->grad);
        };

        return out;
    }

    // 取平方+均值操作
    std::shared_ptr<autodiff> squared_mean() {
        // 前向计算：均值 = 所有元素平方和 / 元素总数
        float mean_value = ((this->data).array().pow(2)).mean();

        // 将标量转换为 MatrixXf（1x1 矩阵）
        MatrixXf mean_matrix(1, 1);
        mean_matrix(0, 0) = mean_value;

        auto out = std::make_shared<autodiff>(mean_matrix, std::vector<autodiff *>{const_cast<autodiff *>(this)});

        out->_backward = [this, out]() {
            const float grad_scale = 2.0f / static_cast<float>(this->data.size());
            this->grad.array() += grad_scale * out->grad(0, 0) * this->data.array();
        };

        return out;
    }

    // 误差函数
    static std::shared_ptr<autodiff> rmse(const std::shared_ptr<autodiff> &pred,
                                          const std::shared_ptr<autodiff> &true_vals) {
        // 计算误差平方均值
        auto error = pred - true_vals;
        auto mse = error->squared_mean();

        // 添加开平方操作
        auto rmse_val = std::make_shared<autodiff>(
            mse->data.array().sqrt().matrix().eval(), // sqrt(mean)
            std::vector<autodiff *>{mse.get()}
        );

        // RMSE梯度计算
        rmse_val->_backward = [mse, rmse_val]() {
            const float grad_scale = 0.5f / (rmse_val->data(0, 0) * mse->data.size());
            mse->grad.array() += grad_scale * rmse_val->grad(0, 0);
            mse->backward(); // 链式传播到error
        };

        return rmse_val;
    }

    void backward();
};


#endif //AUTODIFF_H
