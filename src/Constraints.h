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
    static double ComputeConstraintCost(
        const Eigen::Ref<const Eigen::VectorXd>& g,
        const Eigen::Ref<const Eigen::VectorXd>& t,
        const Eigen::Ref<const Eigen::VectorXd>& lambda) {
        return g.transpose() * (lambda - t.cwiseInverse().cwiseProduct((-g).array().log().matrix()));
    }
};

// size and type of a constraint vector.
// input n is the length of the constraint vector.
template <int p, class ConType>
class Constraints {
public:
    using EvalFunc = std::function<void(const State& x, const Control& u, VectorNd<p>& g)>;
    using QuadratizeFunc =
        std::function<void(const State& x, const Control& u,
                           const VectorNd<p>& lambda, const VectorNd<p>& penalty,
                           Matrix& Pt, Matrix& Qt, Matrix& Rt, Vector& qt, Vector& rt)>;

    Constraints(VectorNd<p> penaltyInit, double phi,
                EvalFunc evalFunc, QuadratizeFunc quadratizeFunc) {
        g_.setZero();
        lambda_.setZero();
        penalty_ = penaltyInit;
        penalty_in_cost_ = penaltyInit;
        phi_ = phi;
        EvalCon_ = evalFunc;
        QuadratizeCon_ = quadratizeFunc;
    }

    void UpdateDual() {
        ConType::UpdateDual(g_, penalty_, lambda_);
    }

    void UpdatePenalty() {
        ConType::UpdatePenalty(phi_, g_, penalty_);
    }

    void EvaluatePenaltyParam() {
        ConType::EvaluatePenaltyParam(g_, penalty_, lambda_, penalty_in_cost_);
    }

    double ComputeConstraintCost() {
        return ConType::ComputeConstraintCost(g_, penalty_in_cost_, lambda_);
    }

private:
    EvalFunc EvalCon_;
    QuadratizeFunc QuadratizeCon_;
    VectorNd<p> g_;
    VectorNd<p> lambda_;
    VectorNd<p> penalty_;
    VectorNd<p> penalty_in_cost_;
    double phi_;
};

// utility class for actual constraints evaluation and quadratization.
class ConstraintsEvaluation {
public:
    void EvaluateConstraints(
        const VectorNd<kStatesDim>& x, const Control& u, VectorNd<kConstraintDim>& g);
    void QuadratizeConstraints(
        const VectorNd<kStatesDim>& x, const Control& u,
        const VectorNd<kConstraintDim>& lambda,
        const VectorNd<kConstraintDim>& penalty,
        MatrixNMd<kControlDim, kStatesDim>& Pt,
        MatrixNMd<kStatesDim, kStatesDim>& Qt,
        MatrixNMd<kControlDim, kControlDim>& Rt,
        VectorNd<kStatesDim>& qt,
        VectorNd<kControlDim>& rt);

    void EvaluateFinalConstraints(
        const VectorNd<kStatesDim>& x, const Control& u, VectorNd<kFinalConstraintDim>& g);
    void QuadratizeFinalConstraints(
        const VectorNd<kStatesDim>& x, const Control& u,
        const VectorNd<kFinalConstraintDim>& lambda,
        const VectorNd<kFinalConstraintDim>& penalty,
        MatrixNMd<kControlDim, kStatesDim>& Pt,
        MatrixNMd<kStatesDim, kStatesDim>& Qt,
        MatrixNMd<kControlDim, kControlDim>& Rt,
        VectorNd<kStatesDim>& qt,
        VectorNd<kControlDim>& rt);

    void EvaluateInitEqualConstraints(
        const VectorNd<kStatesDim>& x, const Control& u, VectorNd<kInitEqualConstraintDim>& g);
    void QuadratizeInitEqualConstraints(
        const VectorNd<kStatesDim>& x, const Control& u,
        const VectorNd<kInitEqualConstraintDim>& lambda,
        const VectorNd<kInitEqualConstraintDim>& penalty,
        MatrixNMd<kControlDim, kStatesDim>& Pt,
        MatrixNMd<kStatesDim, kStatesDim>& Qt,
        MatrixNMd<kControlDim, kControlDim>& Rt,
        VectorNd<kStatesDim>& qt,
        VectorNd<kControlDim>& rt);
private:
    Control u_min_ = Control::Zero();
    Control u_max_ = Control::Zero();
    Control delta_u_min_ = Control::Zero();
    Control delta_u_max_ = Control::Zero();
    double lat_acc_bound_;
    double lat_dist_bound_;
};