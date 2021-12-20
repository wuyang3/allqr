#pragma once

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <math.h>
#include <vector>

constexpr size_t kStateDim = 4;
constexpr size_t kControlDim = 2; // control dim, delta control dim.
constexpr size_t kAugStateDim = 6; // state + control as the optimization state.

// determined at compile time.
using State = Eigen::Matrix<double, kStateDim, 1>;
using Control = Eigen::Matrix<double, kControlDim, 1>;

// determined at declaration time.
template <int n>
using VectorNd = Eigen::Matrix<double, n, 1>;

template <int n, int m>
using MatrixNMd = Eigen::Matrix<double, n, m>;

// vehicle model states: [global_x, global_y, heading, speed, front_wheel_angle (control), acceleration (control)]
// vehicle model controls: [delta_front_wheel_angle, delta_acceleration]

class VehicleModel
{
public:
    // set vehicle model parameters
    void InitializeVehicleModel(const double wheelbase, const double dt);

    // discrete-time forward dynamics x_{t+1} = f(x_t, u_t)
    State ForwardDynamics(const State &state, const Control &control);

    // discrete-time inverse dynamics x_t = \bar{f}(x_{t+1}, u_t)
    State InverseDynamics(const State &state, const Control &control);

    // discrete-time forward augmented dynamics x_{t+1} = f(x_t, u_t)
    Vector ForwardAugmentedDynamics(const Vector &state, const Control &control);

    // discrete-time inverse augmented dynamics x_t = \bar{f}(x_{t+1}, u_t)
    Vector InverseAugmentedDynamics(const Vector &state, const Control &control);

    std::pair<Matrix, Matrix> ComputeLinearModel(const State &state,
                                                 const Control &control);

    // augment control as state, and use delta_control as system input
    std::pair<Matrix, Matrix> ComputeAugmentedLinearModel(const Vector &state,
                                                          const Control &delta_control);

    double GetSamplingTime();

    const size_t state_dim_ = 4;
    const size_t control_dim_ = 2;

private:
    // continuous-time dynamics \dot{x} = f(x, u)
    State ComputeStateDerivatives(const State &state, const Control &control);

    Vector ComputeAugmentedStateDerivatives(const Vector &state, const Control &control);

    double dt_ = 0.0;
    double wheelbase_ = 0.0;

    // vehicle model state
    const int X_POS = 0; // rear axle center global x position
    const int Y_POS = 1; // rear axle center global y position
    const int HEADING = 2;
    const int VELOCITY = 3;
    const int FRONTWHEELANGLE = 4;
    const int ACCELERATION = 5;

    // control
    const int DELTA_FRONTWHEELANGLE = 0;
    const int DELTA_ACCELERATION = 1;
};
