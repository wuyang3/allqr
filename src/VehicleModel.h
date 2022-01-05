#pragma once

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <math.h>
#include <vector>

constexpr size_t kStateDim = 4;    // plain state dim.
constexpr size_t kControlDim = 2;  // control dim, delta control dim.
constexpr size_t kAugStateDim = 6; // state + control as the optimization state.

using State = Eigen::Matrix<double, kStateDim, 1>;
using Control = Eigen::Matrix<double, kControlDim, 1>;
using AugState = Eigen::Matrix<double, kAugStateDim, 1>;

// determined at declaration time.
template <int n> using VectorNd = Eigen::Matrix<double, n, 1>;

template <int n, int m> using MatrixNMd = Eigen::Matrix<double, n, m>;

// vehicle model states: [global_x, global_y, heading, speed, front_wheel_angle
// (control), acceleration (control)] vehicle model controls:
// [delta_front_wheel_angle, delta_acceleration]

class VehicleModel {
public:
  // set vehicle model parameters
  void InitializeVehicleModel(const double wheelbase, const double dt);

  // discrete-time forward dynamics x_{t+1} = f(x_t, u_t)
  State ForwardDynamics(const State &state, const Control &control);

  // discrete-time inverse dynamics x_t = \bar{f}(x_{t+1}, u_t)
  State InverseDynamics(const State &state, const Control &control);

  // discrete-time forward augmented dynamics x_{t+1} = f(x_t, u_t)
  AugState ForwardAugmentedDynamics(const AugState &state,
                                    const Control &control);

  // discrete-time inverse augmented dynamics x_t = \bar{f}(x_{t+1}, u_t)
  AugState InverseAugmentedDynamics(const AugState &state,
                                    const Control &control);

  // plain state model
  void ComputeLinearModel(const State &state, const Control &control,
                          const bool use_inverse_form,
                          Eigen::Matrix<double, kStateDim, kStateDim> *Ad,
                          Eigen::Matrix<double, kStateDim, kControlDim> *Bd);

  // augment control as state, and use delta_control as system input
  void ComputeAugmentedLinearModel(
      const AugState &state, const Control &delta_control,
      const bool use_inverse_form,
      Eigen::Matrix<double, kAugStateDim, kAugStateDim> *Ad,
      Eigen::Matrix<double, kAugStateDim, kControlDim> *Bd);

  double GetSamplingTime();

  const size_t state_dim_ = kStateDim;
  const size_t control_dim_ = kControlDim;

private:
  // continuous-time dynamics \dot{x} = f(x, u)
  State ComputeStateDerivatives(const State &state, const Control &control);
  AugState ComputeAugmentedStateDerivatives(const AugState &state,
                                            const Control &delta_control);

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
