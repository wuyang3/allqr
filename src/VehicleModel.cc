#include "VehicleModel.h"

void VehicleModel::InitializeVehicleModel(const double wheelbase,
                                          const double dt) {
  wheelbase_ = wheelbase;
  dt_ = dt;
}

double VehicleModel::GetSamplingTime() { return dt_; }

State VehicleModel::ComputeStateDerivatives(const State &state,
                                            const Control &control) {
  const double x_dot = state(VELOCITY) * std::cos(state(HEADING));
  const double y_dot = state(VELOCITY) * std::sin(state(HEADING));
  const double heading_dot =
      state(VELOCITY) / wheelbase_ * std::tan(control(FRONTWHEELANGLE));
  const double v_dot = control(ACCELERATION);
  return State(x_dot, y_dot, heading_dot, v_dot);
}

AugState
VehicleModel::ComputeAugmentedStateDerivatives(const AugState &state,
                                               const Control &delta_control) {
  const double x_dot = state(VELOCITY) * std::cos(state(HEADING));
  const double y_dot = state(VELOCITY) * std::sin(state(HEADING));
  const double heading_dot =
      state(VELOCITY) / wheelbase_ * std::tan(state(FRONTWHEELANGLE));
  const double v_dot = state(ACCELERATION);
  const double front_wheel_angle_dot = delta_control(DELTA_FRONTWHEELANGLE);
  const double acceleration_dot = delta_control(DELTA_ACCELERATION);
  AugState state_derivatives(x_dot, y_dot, heading_dot, v_dot,
                             front_wheel_angle_dot, acceleration_dot);
  return state_derivatives;
}

void VehicleModel::ComputeLinearModel(
    const State &state, const Control &control, const bool use_inverse_form,
    Eigen::Matrix<double, kStateDim, kStateDim> *Ad,
    Eigen::Matrix<double, kStateDim, kControlDim> *Bd) {
  Eigen::Matrix<double, kStateDim, kStateDim> A =
      Eigen::Matrix<double, kStateDim, kStateDim>::Zero();
  Eigen::Matrix<double, kStateDim, kControlDim> B =
      Eigen::Matrix<double, kStateDim, kControlDim>::Zero();

  A(0, 2) = -state(VELOCITY) * std::sin(state(HEADING));
  A(0, 3) = std::cos(state(HEADING));
  A(1, 2) = state(VELOCITY) * std::cos(state(HEADING));
  A(1, 3) = std::sin(state(HEADING));
  A(2, 3) = std::tan(control(FRONTWHEELANGLE - kStateDim)) / wheelbase_;

  B(2, 0) = state(VELOCITY) / wheelbase_ /
            std::cos(control(FRONTWHEELANGLE - kStateDim)) /
            std::cos(control(FRONTWHEELANGLE - kStateDim));
  B(3, 1) = 1.0;

  Eigen::Matrix<double, kStateDim, kStateDim> matrix_i =
      Eigen::Matrix<double, kStateDim, kStateDim>::Identity();
  Eigen::Matrix<double, kStateDim, kStateDim> matrix_tmp_inverse =
      (matrix_i - dt_ * 0.5 * A).inverse();
  Eigen::Matrix<double, kStateDim, kStateDim> matrix_tmp =
      (matrix_i + dt_ * 0.5 * A);

  if (!use_inverse_form) {
    (*Ad).noalias() = matrix_tmp_inverse * matrix_tmp;
    (*Bd).noalias() = matrix_tmp_inverse * B * dt_;
  } else {
    (*Ad).noalias() = (matrix_tmp_inverse * matrix_tmp).inverse();
    (*Bd).noalias() = -(*Ad) * matrix_tmp_inverse * B * dt_;
  }
}

void VehicleModel::ComputeAugmentedLinearModel(
    const AugState &state, const Control &delta_control,
    const bool use_inverse_form,
    Eigen::Matrix<double, kAugStateDim, kAugStateDim> *Ad,
    Eigen::Matrix<double, kAugStateDim, kControlDim> *Bd) {
  Eigen::Matrix<double, kAugStateDim, kAugStateDim> A =
      Eigen::Matrix<double, kAugStateDim, kAugStateDim>::Zero();
  Eigen::Matrix<double, kAugStateDim, kControlDim> B =
      Eigen::Matrix<double, kAugStateDim, kControlDim>::Zero();

  A(0, 2) = -state(VELOCITY) * std::sin(state(HEADING));
  A(0, 3) = std::cos(state(HEADING));

  A(1, 2) = state(VELOCITY) * std::cos(state(HEADING));
  A(1, 3) = std::sin(state(HEADING));

  A(2, 3) = std::tan(state(FRONTWHEELANGLE)) / wheelbase_;
  A(2, 4) = state(VELOCITY) / wheelbase_ / std::cos(state(FRONTWHEELANGLE)) /
            std::cos(state(FRONTWHEELANGLE));

  A(3, 5) = 1.0;

  B(4, 0) = 1.0;
  B(5, 1) = 1.0;

  Eigen::Matrix<double, kAugStateDim, kAugStateDim> matrix_i =
      Eigen::Matrix<double, kAugStateDim, kAugStateDim>::Identity();
  Eigen::Matrix<double, kAugStateDim, kAugStateDim> matrix_tmp_inverse =
      (matrix_i - dt_ * 0.5 * A).inverse();
  Eigen::Matrix<double, kAugStateDim, kAugStateDim> matrix_tmp =
      (matrix_i + dt_ * 0.5 * A);

  if (!use_inverse_form) {
    (*Ad).noalias() = matrix_tmp_inverse * matrix_tmp;
    (*Bd).noalias() = matrix_tmp_inverse * B * dt_;
  } else {
    (*Ad).noalias() = (matrix_tmp_inverse * matrix_tmp).inverse();
    (*Bd).noalias() = -(*Ad) * matrix_tmp_inverse * B * dt_;
  }
}

State VehicleModel::ForwardDynamics(const State &state,
                                    const Control &control) {
  State k1 = ComputeStateDerivatives(state, control);
  State k2 = ComputeStateDerivatives(state + 0.5 * dt_ * k1, control);
  State k3 = ComputeStateDerivatives(state + 0.5 * dt_ * k2, control);
  State k4 = ComputeStateDerivatives(state + dt_ * k3, control);
  return state + (dt_ / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

State VehicleModel::InverseDynamics(const State &state,
                                    const Control &control) {
  State k1 = ComputeStateDerivatives(state, control);
  State k2 = ComputeStateDerivatives(state - 0.5 * dt_ * k1, control);
  State k3 = ComputeStateDerivatives(state - 0.5 * dt_ * k2, control);
  State k4 = ComputeStateDerivatives(state - dt_ * k3, control);
  return state - (dt_ / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

AugState VehicleModel::ForwardAugmentedDynamics(const AugState &state,
                                                const Control &control) {
  AugState k1 = ComputeAugmentedStateDerivatives(state, control);
  AugState k2 =
      ComputeAugmentedStateDerivatives(state + 0.5 * dt_ * k1, control);
  AugState k3 =
      ComputeAugmentedStateDerivatives(state + 0.5 * dt_ * k2, control);
  AugState k4 = ComputeAugmentedStateDerivatives(state + dt_ * k3, control);
  return state + (dt_ / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

AugState VehicleModel::InverseAugmentedDynamics(const AugState &state,
                                                const Control &control) {
  AugState k1 = ComputeAugmentedStateDerivatives(state, control);
  AugState k2 =
      ComputeAugmentedStateDerivatives(state - 0.5 * dt_ * k1, control);
  AugState k3 =
      ComputeAugmentedStateDerivatives(state - 0.5 * dt_ * k2, control);
  AugState k4 = ComputeAugmentedStateDerivatives(state - dt_ * k3, control);
  return state - (dt_ / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}
