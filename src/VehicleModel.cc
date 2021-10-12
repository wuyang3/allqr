#include "VehicleModel.h"

void VehicleModel::InitializeVehicleModel(const double wheelbase,
                                          const double dt)
{
    wheelbase_ = wheelbase;
    dt_ = dt;
}

double VehicleModel::GetSamplingTime()
{
    return dt_;
}

State VehicleModel::ComputeStateDerivatives(const State &state,
                                            const Control &control)
{
    const double x_dot = state(VELOCITY) * std::cos(state(HEADING));
    const double y_dot = state(VELOCITY) * std::sin(state(HEADING));
    const double heading_dot = state(VELOCITY) / wheelbase_ *
                               std::tan(control(FRONTWHEELANGLE));
    const double v_dot = control(ACCELERATION);
    return State(x_dot, y_dot, heading_dot, v_dot);
}

Vector VehicleModel::ComputeAugmentedStateDerivatives(const Vector &state, const Control &delta_control)
{
    const double x_dot = state(VELOCITY) * std::cos(state(HEADING));
    const double y_dot = state(VELOCITY) * std::sin(state(HEADING));
    const double heading_dot = state(VELOCITY) / wheelbase_ *
                               std::tan(state(FRONTWHEELANGLE));
    const double v_dot = state(ACCELERATION);
    const double front_wheel_angle_dot = delta_control(DELTA_FRONTWHEELANGLE);
    const double acceleration_dot = delta_control(DELTA_ACCELERATION);
    Vector state_derivatives;
    state_derivatives.resize(state_dim_ + control_dim_);
    state_derivatives << x_dot, y_dot, heading_dot, v_dot, front_wheel_angle_dot, acceleration_dot;
    return state_derivatives;
}

std::pair<Matrix, Matrix> VehicleModel::ComputeLinearModel(const State &state,
                                                           const Control &control)
{
    Matrix A = Matrix::Zero(state_dim_, state_dim_);
    Matrix B = Matrix::Zero(state_dim_, control_dim_);

    A(0, 2) = -state(VELOCITY) * std::sin(state(HEADING));
    A(0, 3) = std::cos(state(HEADING));
    A(1, 2) = state(VELOCITY) * std::cos(state(HEADING));
    A(1, 3) = std::sin(state(HEADING));
    A(2, 3) = std::tan(control(FRONTWHEELANGLE)) / wheelbase_;

    Matrix matrix_i = Matrix::Identity(4, 4);
    Matrix matrix_tmp_inverse = (matrix_i - dt_ * 0.5 * A).inverse();
    Matrix A_discrete = matrix_tmp_inverse * (matrix_i + dt_ * 0.5 * A);

    B(2, 0) = state(VELOCITY) / wheelbase_ / std::cos(control(FRONTWHEELANGLE)) / std::cos(control(FRONTWHEELANGLE));
    B(3, 1) = 1.0;

    std::pair<Matrix, Matrix> result;
    result.first = A_discrete;
    result.second = matrix_tmp_inverse * B * dt_;
    return result;
}

std::pair<Matrix, Matrix> VehicleModel::ComputeAugmentedLinearModel(const Vector &state,
                                                                    const Control &delta_control)
{
    Matrix A = Matrix::Zero(state_dim_ + control_dim_, state_dim_ + control_dim_);
    Matrix B = Matrix::Zero(state_dim_ + control_dim_, control_dim_);

    A(0, 2) = -state(VELOCITY) * std::sin(state(HEADING));
    A(0, 3) = std::cos(state(HEADING));

    A(1, 2) = state(VELOCITY) * std::cos(state(HEADING));
    A(1, 3) = std::sin(state(HEADING));

    A(2, 3) = std::tan(state(FRONTWHEELANGLE)) / wheelbase_;
    A(2, 4) = state(VELOCITY) / wheelbase_ / std::cos(state(FRONTWHEELANGLE)) / std::cos(state(FRONTWHEELANGLE));

    A(3, 5) = 1.0;

    Matrix matrix_i = Matrix::Identity(state_dim_ + control_dim_, state_dim_ + control_dim_);
    Matrix matrix_tmp_inverse = (matrix_i - dt_ * 0.5 * A).inverse();
    Matrix A_discrete = matrix_tmp_inverse * (matrix_i + dt_ * 0.5 * A);

    B(4, 0) = 1.0;
    B(5, 1) = 1.0;

    std::pair<Matrix, Matrix> result;
    result.first = A_discrete;
    result.second = matrix_tmp_inverse * B * dt_;
    return result;
}

State VehicleModel::ForwardDynamics(const State &state, const Control &control)
{
    State k1 = ComputeStateDerivatives(state, control);
    State k2 = ComputeStateDerivatives(state + 0.5 * dt_ * k1, control);
    State k3 = ComputeStateDerivatives(state + 0.5 * dt_ * k2, control);
    State k4 = ComputeStateDerivatives(state + dt_ * k3, control);
    return state + (dt_ / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

State VehicleModel::InverseDynamics(const State &state, const Control &control)
{
    State k1 = ComputeStateDerivatives(state, control);
    State k2 = ComputeStateDerivatives(state - 0.5 * dt_ * k1, control);
    State k3 = ComputeStateDerivatives(state - 0.5 * dt_ * k2, control);
    State k4 = ComputeStateDerivatives(state - dt_ * k3, control);
    return state - (dt_ / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

Vector VehicleModel::ForwardAugmentedDynamics(const Vector &state, const Control &control)
{
    Vector k1 = ComputeAugmentedStateDerivatives(state, control);
    Vector k2 = ComputeAugmentedStateDerivatives(state + 0.5 * dt_ * k1, control);
    Vector k3 = ComputeAugmentedStateDerivatives(state + 0.5 * dt_ * k2, control);
    Vector k4 = ComputeAugmentedStateDerivatives(state + dt_ * k3, control);
    return state + (dt_ / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

Vector VehicleModel::InverseAugmentedDynamics(const Vector &state, const Control &control)
{
    Vector k1 = ComputeAugmentedStateDerivatives(state, control);
    Vector k2 = ComputeAugmentedStateDerivatives(state - 0.5 * dt_ * k1, control);
    Vector k3 = ComputeAugmentedStateDerivatives(state - 0.5 * dt_ * k2, control);
    Vector k4 = ComputeAugmentedStateDerivatives(state - dt_ * k3, control);
    return state - (dt_ / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}
