#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>
#include <functional>
#include "VehicleModel.h"

constexpr size_t kConstraintDim = 8;
constexpr size_t kFinalConstraintDim = 4;
constexpr size_t kInitEqualConstraintDim = 4;
constexpr double kZeroTolerance = 1e-6;

// pure static method class.
class EqualityAugLagrangian {
 public:
  EqualityAugLagrangian() = delete;
  static void UpdateDual(
    const Eigen::Ref<const Eigen::VectorXd>& g,
    const Eigen::Ref<const Eigen::VectorXd>& miu,
    Eigen::Ref<Eigen::VectorXd> lambda) {
    lambda = lambda + miu.cwiseProduct(g);
  }
  static void UpdatePenalty(
    const double phi,
    const Eigen::Ref<const Eigen::VectorXd>& g,
    Eigen::Ref<Eigen::VectorXd> miu) {
    (void) g;
    miu = phi * miu;
  }
  static void EvaluatePenaltyParam(
    const Eigen::Ref<const Eigen::VectorXd>& g,
    const Eigen::Ref<const Eigen::VectorXd>& miu,
    const Eigen::Ref<const Eigen::VectorXd>& lambda,
    Eigen::Ref<Eigen::VectorXd> miuInCost) {
    (void) g;
    (void) lambda;
    miuInCost = miu;
  }
  static double ComputeConstraintCost(
    const Eigen::Ref<const Eigen::VectorXd>& g,
    const Eigen::Ref<const Eigen::VectorXd>& miu,
    const Eigen::Ref<const Eigen::VectorXd>& lambda) {
    return g.transpose() * (lambda + miu.cwiseProduct(g));
  }
};

class InEqualityAugLagrangian {
 public:
  InEqualityAugLagrangian() = delete;
  static void UpdateDual(
    const Eigen::Ref<const Eigen::VectorXd>& g,
    const Eigen::Ref<const Eigen::VectorXd>& miu,
    Eigen::Ref<Eigen::VectorXd> lambda) {
    lambda = lambda + miu.cwiseProduct(g);
    lambda = lambda.cwiseMax(0.0);
  }
  static void UpdatePenalty(
    const double phi,
    const Eigen::Ref<const Eigen::VectorXd>& g,
    Eigen::Ref<Eigen::VectorXd> miu) {
    (void) g;
    miu = phi * miu;
  }
  static void EvaluatePenaltyParam(
    const Eigen::Ref<const Eigen::VectorXd>& g,
    const Eigen::Ref<const Eigen::VectorXd>& miu,
    const Eigen::Ref<const Eigen::VectorXd>& lambda,
    Eigen::Ref<Eigen::VectorXd> miuInCost) {
    miuInCost = (lambda.array().abs() < kZeroTolerance && g.array() < 0.0)
                .select(0.0, miu);
  }
  static double ComputeConstraintCost(
    const Eigen::Ref<const Eigen::VectorXd>& g,
    const Eigen::Ref<const Eigen::VectorXd>& miu,
    const Eigen::Ref<const Eigen::VectorXd>& lambda) {
    return g.transpose() * (lambda + miu.cwiseProduct(g));
  }
};

class InEqualityLogBarrier {
 public:
  InEqualityLogBarrier() = delete;
  // dual and penalty method left empty.
  static void UpdateDual(
    const Eigen::Ref<const Eigen::VectorXd>& g,
    const Eigen::Ref<const Eigen::VectorXd>& t,
    Eigen::Ref<Eigen::VectorXd> lambda) {
    (void) g;
    (void) t;
    (void) lambda;
  }
  static void UpdatePenalty(
    const double phi,
    const Eigen::Ref<const Eigen::VectorXd>& g,
    Eigen::Ref<Eigen::VectorXd> t) {
    (void) g;
    t = phi * t;
  }
  static void EvaluatePenaltyParam(
    const Eigen::Ref<const Eigen::VectorXd>& g,
    const Eigen::Ref<const Eigen::VectorXd>& t,
    const Eigen::Ref<const Eigen::VectorXd>& lambda,
    Eigen::Ref<Eigen::VectorXd> tInCost) {
    (void) g;
    (void) t;
    (void) lambda;
    (void) tInCost;
  }
  // TODO(wuyang): simplify the log barrier cost calculation.
  static double ComputeConstraintCost(
    const Eigen::Ref<const Eigen::VectorXd>& g,
    const Eigen::Ref<const Eigen::VectorXd>& t,
    const Eigen::Ref<const Eigen::VectorXd>& lambda) {
    return g.transpose() * (lambda - t.cwiseInverse().cwiseProduct((-g).array().log().matrix()));
  }
};

// Class type for the constraint vector object that is imposed on
// on the trajectory, e.g., constraint vector at the initial state
// and final state.
template <int p, class ConType>
class Constraints {
 public:
  // User defined functions for constraint value, gradient, hessian
  // evaluations. Initialized upon object construction.
  // Instead of using ref object, passing specific size matrix by pointer.
  using EvalFunc =
    std::function<void(const State& x, const Control& u, VectorNd<p>* g)>;
  using HessianFunc =
    std::function<void(const State& x, const Control& u,
                       const VectorNd<p>& g,
                       const VectorNd<p>& lambda,
                       const VectorNd<p>& penalty_in_cost,
                       Eigen::Matrix<double, kStateDim, kStateDim>* Qt,
                       Eigen::Matrix<double, kControlDim, kControlDim>* Rt)>;
  using GradientFunc =
    std::function<void(const State& x, const Control& u,
                       const VectorNd<p>& g,
                       const VectorNd<p>& lambda,
                       const VectorNd<p>& penalty_in_cost,
                       Eigen::Matrix<double, kStateDim, 1>* qt,
                       Eigen::Matrix<double, kControlDim, 1>* rt)>;
  Constraints(VectorNd<p> penaltyInit, double phi,
              EvalFunc evalFunc,
              HessianFunc hessianFunc,
              GradientFunc gradientFunc) {
    g_.setZero();
    lambda_.setZero();
    penalty_ = penaltyInit;
    penalty_in_cost_ = penaltyInit;
    phi_ = phi;
    EvalFunc_ = evalFunc;
    HessianFunc_ = hessianFunc;
    GradientFunc_ = gradientFunc;
  }

  // Evaluate the actual penalty parameter when compute the constraint cost
  // or when quadratize the constraint cost.
  // Whenever given new x, u pair, the constraint vector g is evaluated.
  double ComputeConstraintCost(
    const State& x, const Control& u) {
    EvalFunc_(x, u, g_);
    ConType::EvaluatePenaltyParam(g_, penalty_, lambda_, penalty_in_cost_);
    return ConType::ComputeConstraintCost(g_, penalty_in_cost_, lambda_);
  }
  void QuadratizeConstraint(
    const State& x, const Control& u,
    Eigen::Matrix<double, kStateDim, kStateDim>* Qt,
    Eigen::Matrix<double, kControlDim, kControlDim>* Rt,
    Eigen::Matrix<double, kStateDim, 1>* qt,
    Eigen::Matrix<double, kControlDim, 1>* rt) {
    EvalFunc_(x, u, g_);
    ConType::EvaluatePenaltyParam(g_, penalty_, lambda_, penalty_in_cost_);
    HessianFunc_(x, u, g_, lambda_, penalty_in_cost_, Qt, Rt);
    GradientFunc_(x, u, g_, lambda_, penalty_in_cost_, qt, rt);
  }

  VectorNd<p> GetConstraintVector() {
    return g_;
  }
  void UpdateDual() {
    ConType::UpdateDual(g_, penalty_, lambda_);
  }
  void UpdatePenalty() {
    ConType::UpdatePenalty(phi_, g_, penalty_);
  }

 private:
  EvalFunc EvalFunc_;
  HessianFunc HessianFunc_;
  GradientFunc GradientFunc_;
  VectorNd<p> g_;
  VectorNd<p> lambda_;
  VectorNd<p> penalty_;
  VectorNd<p> penalty_in_cost_;
  double phi_;
};

// utility class for actual constraints evaluation and quadratization.
// this class object will be used to intialize the actual constraint objects.
class ConstraintsEvaluation {
 public:
    ConstraintsEvaluation(
      Control u_min, Control u_max,
      Control delta_u_min, Control delta_u_max,
      double lat_acc_bound = 0.3,
      double lat_dist_bound = 0.5)
      : u_min_(u_min), u_max_(u_max),
        delta_u_min_(delta_u_min), delta_u_max_(delta_u_max),
        lat_acc_bound_(lat_acc_bound), lat_dist_bound_(lat_dist_bound) {}

    void EvaluateConstraints(
      const State& x, const Control& u, VectorNd<kConstraintDim>* g);
    void HessianConstraints(
      const State& x, const Control& u,
      const VectorNd<kConstraintDim>& g,
      const VectorNd<kConstraintDim>& lambda,
      const VectorNd<kConstraintDim>& penalty_in_cost,
      Eigen::Matrix<double, kStateDim, kStateDim>* Qt,
      Eigen::Matrix<double, kControlDim, kControlDim>* Rt);
    void GradientConstraints(
      const State& x, const Control& u,
      const VectorNd<kConstraintDim>& g,
      const VectorNd<kConstraintDim>& lambda,
      const VectorNd<kConstraintDim>& penalty_in_cost,
      Eigen::Matrix<double, kStateDim, 1>* qt,
      Eigen::Matrix<double, kControlDim, 1>* rt);

    void EvaluateFinalConstraints(
      const State& x, const Control& u, VectorNd<kFinalConstraintDim>* g);
    void HessianFinalConstraints(
      const State& x, const Control& u,
      const VectorNd<kFinalConstraintDim>& g,
      const VectorNd<kFinalConstraintDim>& lambda,
      const VectorNd<kFinalConstraintDim>& penalty_in_cost,
      Eigen::Matrix<double, kStateDim, kStateDim>* Qt,
      Eigen::Matrix<double, kControlDim, kControlDim>* Rt);
    void GradientFinalConstraints(
      const State& x, const Control& u,
      const VectorNd<kFinalConstraintDim>& g,
      const VectorNd<kFinalConstraintDim>& lambda,
      const VectorNd<kFinalConstraintDim>& penalty_in_cost,
      Eigen::Matrix<double, kStateDim, 1>* qt,
      Eigen::Matrix<double, kControlDim, 1>* rt);

    void EvaluateInitEqualConstraints(
      const State& x, const Control& u, VectorNd<kInitEqualConstraintDim>* g);
    void HessianInitConstraints(
      const State& x, const Control& u,
      const VectorNd<kInitEqualConstraintDim>& g,
      const VectorNd<kInitEqualConstraintDim>& lambda,
      const VectorNd<kInitEqualConstraintDim>& penalty_in_cost,
      Eigen::Matrix<double, kStateDim, kStateDim>* Qt,
      Eigen::Matrix<double, kControlDim, kControlDim>* Rt);
    void GradientInitConstraints(
      const State& x, const Control& u,
      const VectorNd<kInitEqualConstraintDim>& g,
      const VectorNd<kInitEqualConstraintDim>& lambda,
      const VectorNd<kInitEqualConstraintDim>& penalty_in_cost,
      Eigen::Matrix<double, kStateDim, 1>* qt,
      Eigen::Matrix<double, kControlDim, 1>* rt);
 private:
    Control u_min_ = Control::Zero();
    Control u_max_ = Control::Zero();
    Control delta_u_min_ = Control::Zero();
    Control delta_u_max_ = Control::Zero();
    double lat_acc_bound_;
    double lat_dist_bound_;
};