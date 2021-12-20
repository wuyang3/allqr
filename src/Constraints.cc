
#include "Constraints.h"
#include "VehicleModel.h"

void ConstraintsEvaluation::EvaluateConstraints(
  const State& x, const Control& u, VectorNd<kConstraintDim>* g) {
    (*g)(0) = x(4) - u_max_(0);
    (*g)(1) = -x(4) + u_min_(0);
    (*g)(2) = x(5) - u_max_(1);
    (*g)(3) = -x(5) + u_min_(1);
    (*g)(4) = u(0) - delta_u_max_(0);
    (*g)(5) = -u(0) + delta_u_min_(0);
    (*g)(6) = u(1) - delta_u_max_(1);
    (*g)(7) = -u(1) + delta_u_min_(1);
    // (*g)(8) =
    //  std::pow(x(3), 2) * std::tan(x(4)) / wheelbase_ - lat_acc_bound_;
    // (*g)(9) =
    // -std::pow(x(3), 2) * std:: tan(x(4)) / wheelbase_ + lat_acc_bound_;
}

void ConstraintsEvaluation::HessianConstraints(
  const State& x, const Control& u,
  const VectorNd<kConstraintDim>& g,
  const VectorNd<kConstraintDim>& lambda,
  const VectorNd<kConstraintDim>& penalty_in_cost,
  Eigen::Matrix<double, kStateDim, kStateDim>* Qt,
  Eigen::Matrix<double, kControlDim, kControlDim>* Rt) {
  (void) x;
  (void) u;
  (void) g;
  (void) lambda;
  (*Qt)(4, 4) += (penalty_in_cost(0) + penalty_in_cost(1));
  (*Qt)(5, 5) += (penalty_in_cost(2) + penalty_in_cost(3));

  (*Rt)(0, 0) += (penalty_in_cost(4) + penalty_in_cost(5));
  (*Rt)(1, 1) += (penalty_in_cost(6) + penalty_in_cost(7));
}

void ConstraintsEvaluation::GradientConstraints(
  const State& x, const Control& u,
  const VectorNd<kConstraintDim>& g,
  const VectorNd<kConstraintDim>& lambda,
  const VectorNd<kConstraintDim>& penalty_in_cost,
  Eigen::Matrix<double, kStateDim, 1>* qt,
  Eigen::Matrix<double, kControlDim, 1>* rt) {
  (*qt)(4) += (lambda(0) - lambda(1)
               + g(0) * penalty_in_cost(0) - g(1) * penalty_in_cost(1)
               - x(4) * (penalty_in_cost(0) + penalty_in_cost(1)));
  (*qt)(5) += (lambda(2) - lambda(3) - x(5) * (penalty_in_cost(2) + penalty_in_cost(3))
               + g(2) * penalty_in_cost(2) - g(3) * penalty_in_cost(3));

  (*rt)(0) += (lambda(4) - lambda(5) - u(0) * (penalty_in_cost(4) + penalty_in_cost(5))
               + g(4) * penalty_in_cost(4) - g(5) * penalty_in_cost(5));
  (*rt)(1) += (lambda(6) - lambda(7)
               - u(1) * (penalty_in_cost(6) + penalty_in_cost(7))
               + g(6) * penalty_in_cost(6) - g(7) * penalty_in_cost(7));
}

void ConstraintsEvaluation::EvaluateFinalConstraints(
  const State& x, const Control& u, VectorNd<kFinalConstraintDim>* g) {
  (void) u;
  (*g)(0) = x(4) - u_max_(0);
  (*g)(1) = -x(4) + u_min_(0);
  (*g)(2) = x(5) - u_max_(1);
  (*g)(3) = -x(5) + u_min_(1);
}

void ConstraintsEvaluation::HessianFinalConstraints(
  const State& x, const Control& u,
  const VectorNd<kFinalConstraintDim>& g,
  const VectorNd<kFinalConstraintDim>& lambda,
  const VectorNd<kFinalConstraintDim>& penalty_in_cost,
  Eigen::Matrix<double, kStateDim, kStateDim>* Qt,
  Eigen::Matrix<double, kControlDim, kControlDim>* Rt) {
  (void) x;
  (void) u;
  (void) g;
  (void) lambda;
  (void) Rt;
  (*Qt)(4, 4) += (penalty_in_cost(0) + penalty_in_cost(1));
  (*Qt)(5, 5) += (penalty_in_cost(2) + penalty_in_cost(3));
}

void ConstraintsEvaluation::GradientFinalConstraints(
  const State& x, const Control& u,
  const VectorNd<kFinalConstraintDim>& g,
  const VectorNd<kFinalConstraintDim>& lambda,
  const VectorNd<kFinalConstraintDim>& penalty_in_cost,
  Eigen::Matrix<double, kStateDim, 1>* qt,
  Eigen::Matrix<double, kControlDim, 1>* rt) {
  (void) u;
  (void) rt;
  (*qt)(4) += (lambda(0) - lambda(1)
               + g(0) * penalty_in_cost(0) - g(1) * penalty_in_cost(1)
               - x(4) * (penalty_in_cost(0) + penalty_in_cost(1)));
  (*qt)(5) += (lambda(2) - lambda(3) - x(5) * (penalty_in_cost(2) + penalty_in_cost(3))
               + g(2) * penalty_in_cost(2) - g(3) * penalty_in_cost(3));
}
