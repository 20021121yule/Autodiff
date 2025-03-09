#include "Autodiff.h"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace Eigen;

int main() {
    // 生成线性回归数据 y = 3x + 2 + noise
    MatrixXf X(5, 1);
    X << 1.0, 2.0, 3.0, 4.0, 5.0;
    MatrixXf y(5, 1);
    y << 5.0, 8.0, 11.0, 14.0, 17.0;
    y.array() += MatrixXf::Random(5,1).array() * 0.1; // 添加噪声

    // 初始化模型参数（需要梯度）
    auto w = autodiff::create(MatrixXf::Random(1,1), true); // 权重
    auto b = autodiff::create(MatrixXf::Random(1,1), true); // 偏置

    // 训练参数
    const int epochs = 1000;
    const float lr = 0.01;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // 前向传播
        auto X_tensor = autodiff::create(X);
        auto pred = (X_tensor * w) + b;  // 广播加法
        auto loss = autodiff::rmse(pred, autodiff::create(y));

        // 反向传播
        loss->backward();

        // 参数更新
        w->data -= lr * w->grad;
        b->data -= lr * b->grad;

        // 打印训练信息
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch
                      << " | Loss: " << loss->data.mean()
                      << " | w: " << w->data.mean()
                      << " | b: " << b->data.mean()
                      << " | w_grad: " << w->grad.mean()
                      << " | b_grad: " << b->grad.mean()
                      << std::endl;
        }

        // 梯度清零
        w->grad.setZero();
        b->grad.setZero();
    }

    // 最终结果
    std::cout << "\nFinal parameters:\n";
    std::cout << "w = " << w->data(0,0) << std::endl;
    std::cout << "b = " << b->data(0,0) << std::endl;

    return 0;
}