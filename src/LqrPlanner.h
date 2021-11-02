#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <random>
#include <sstream>
#include <stdio.h>
#include <string>
#include <time.h>
#include <vector>

#include "proto/conf.pb.h"
#include "VehicleModel.h"

#define INNER_UNCONVERGED        (0)
#define INNER_REACH_MAX_ITER     (1)
#define INNER_LOW_COST_REL_DELTA (2)
#define INNER_LOW_COST_ABS_DELTA (3)


#define OUTER_UNCONVERGED          (0)
#define OUTER_REACH_MAX_ITER        (1)
#define OUTER_LOW_CONSTR_VIOLATION  (2)

struct LineSegment
{
    Eigen::Vector3d coefficients;
    Eigen::Vector2d x_domain;
    Eigen::Vector2d y_domain;
    Eigen::Vector2d start_point;
    Eigen::Vector2d end_point;
    double length;
    double orientation;
    void PrintInfo()
    {
        // std::cout << "x: [" << x_domain(0) << ", " << x_domain(1) << "]" << std::endl;
        // std::cout << "y: [" << y_domain(0) << ", " << y_domain(1) << "]" << std::endl;
        std::cout << "orientation: " << orientation << " rad" << std::endl;
        std::cout << "length: " << length << " m" << std::endl;
        std::cout << "start point = (" << start_point(0) << ", " << start_point(1) << ")" << std::endl;
        std::cout << "end point = (" << end_point(0) << ", " << end_point(1) << ")" << std::endl;
        if (coefficients(1) < 0)
        {
            std::cout << "line seg equ: " << coefficients(0) << " * x - " << std::fabs(coefficients(1)) << " * y + " << coefficients(2) << " = 0.0" << std::endl;
        }
        else
        {
            std::cout << "line seg equ: " << coefficients(0) << " * x + " << coefficients(1) << " * y + " << coefficients(2) << " = 0.0" << std::endl;
        }
        std::cout << "----" << std::endl;
    }
};

class LqrPlanner
{
public:
    LqrPlanner(const conf::LqrPlannerConf &conf);

    double ControlCost(const Control &u);

    void QuadratizeControlCost(const Control &u, Matrix &Rt, Control &rt, Matrix &Qt, Vector &qt);

    void QuadratizeLatAccCost(const Vector &x, const size_t &t, Matrix &Qt, Vector &qt);

    void QuadratizeLatErrorCost(const Vector &x, const size_t &t, Matrix &Qt, Vector &qt);

    void QuadratizeCost(const Vector &x, const Control &u, const size_t &t, Matrix &Pt, Matrix &Qt, Matrix &Rt, Vector &qt, Control &rt);

    void QuadratizeCost2(const Vector &x, const Control &u, const size_t &t, Matrix &Pt, Matrix &Qt, Matrix &Rt, Vector &qt, Control &rt);

    void QuadratizeFinalCost(const Vector &x, Matrix &QFinal, Vector &qFinal);

    void QuadratizeConstraints(
        const Vector& x, const Control& u,
        const Constraint& lambda, const Constraint& miu,
        Matrix& Pt, Matrix& Qt, Matrix& Rt, Vector& qt, Control& rt);
    void QuadratizeFinalConstraints(
        const Vector& x,
        const FinalConstraint& lambda, const FinalConstraint& miu,
        Matrix& Qt, Vector& qt);
    void EvaluateConstraints(
        const Vector& x, const Control& u, Constraint& g);
    void EvaluateFinalConstraints(const Vector& x, FinalConstraint& g);

    void EvaluateMiuInPenalty(
        const Eigen::Ref<Eigen::VectorXd>& lambda, const Eigen::Ref<Eigen::VectorXd>& miu,
        const Eigen::Ref<Eigen::VectorXd>& g, Eigen::Ref<Eigen::VectorXd>& miu_I);

    void UpdateLambda(
        const Eigen::Ref<Eigen::VectorXd>& miu,
        const Eigen::Ref<Eigen::VectorXd>& g,
        Eigen::Ref<Eigen::VectorXd>& lambda);

    void UpdateMiu(
        const Eigen::Ref<Eigen::VectorXd>& g,
        Eigen::Ref<Eigen::VectorXd>& miu);

    double ComputeLateralError(const double x, const double y, const size_t t);

    void ComputeOptimalControlExtendedLQR(std::vector<size_t>& iters, std::vector<int>& flags);

    int ComputeInnerExtendedLQR(
        const std::vector<Eigen::Ref<Eigen::VectorXd>>& Lambda,
        const std::vector<Eigen::Ref<Eigen::VectorXd>>& Miu,
        std::vector<Eigen::Ref<Eigen::VectorXd>>& G,
        std::vector<Matrix>& L, std::vector<Control>& l,
        double* minInnerCost, size_t* innerIter);

    int ComputerOuterAugLagrangian(
        const std::vector<Eigen::Ref<Eigen::VectorXd>>& G,
        std::vector<Eigen::Ref<Eigen::VectorXd>>& Lambda,
        std::vector<Eigen::Ref<Eigen::VectorXd>>& Miu,
        size_t* outerIter);

    void PropogateOptimalTrajectory(
        const std::vector<Matrix>& L, const std::vector<Control>& l,
        const int innerConvergeFlag);

    size_t ComputeOptimalControlIterativeLQR();

    double ComputeConstraintCost(
        const Eigen::Ref<Eigen::VectorXd>& lambda,
        const Eigen::Ref<Eigen::VectorXd>& miu_I,
        const Eigen::Ref<Eigen::VectorXd>& g);
    double ComputeIntermediateCost(const Vector &x, const Control &u, const size_t &t);
    double ComputeFinalCost(const Vector &x);
    std::vector<double> ComputeLateralAccelerations();

    Eigen::Vector2d FindCloestPointToLineSeg(const double x, const double y, const LineSegment line_seg);

    void GenerateTrackingReference();

    void Spin();

    std::vector<Control> GetOptimalControl();
    std::vector<Vector> GetSolutionStates();
    std::vector<Vector> GetReferenceTrajectory();
    std::vector<double> GetOptimizationParameters();
    std::pair<std::vector<Vector>, std::vector<Vector>> GetReferenceTrajectoryLeftRightBounds();
    std::vector<double> GetLateralError();

    std::vector<Eigen::Vector2d> ConvergenceFlag;

private:
    VehicleModel vehicle_model_;
    double wheelbase_;
    Vector initial_state_;
    Vector final_state_;
    Control initial_control_ = Control::Zero();
    const Control nominal_control_ = Control::Zero();
    size_t horizon_ = 0;
    size_t default_horizon_ = 0;
    size_t max_iter_inner_ = 0;
    size_t max_iter_outer_ = 0;
    size_t state_dim_ = 0;
    size_t control_dim_ = 0;
    size_t constraint_dim_ = 0;
    size_t final_constraint_dim_ = 0;
    Matrix Q_;
    Matrix Q_t_;
    Matrix R_;
    double init_miu_;
    double phi_;

    Control u_min_ = Control::Zero();
    Control u_max_ = Control::Zero();
    Control delta_u_min_ = Control::Zero();
    Control delta_u_max_ = Control::Zero();
    double lat_acc_bound_;
    double lat_dist_bound_;

    double dt_ = 0.0;

    double reference_speed_ = 0.0;
    double lane_width_ = 0.0;
    double total_length_ = 0.0;

    std::vector<Control> optimal_control_;
    std::vector<Vector> solution_states_;

    conf::LqrPlannerConf conf_;

    double convergence_relative_tolerance_ = 0.0;  // relative changes.
    double convergence_abs_tolerance_ = 0.0;  // absolute changes.
    double constraint_tolerannce_ = 0.0;
    double zero_tolerance_ = 0.0;

    std::vector<LineSegment> ref_lines_;
    std::vector<Vector> ref_traj_;
    std::vector<Vector> ref_traj_full_;
    std::vector<size_t> ref_traj_points_line_seg_id_;
    std::vector<Vector> ref_traj_left_bound_;
    std::vector<Vector> ref_traj_right_bound_;
    std::vector<double> lateral_error_;

    bool ready_to_solve_ = false;
    size_t curr_line_seg_id_ = 0;

    double delta_f_penalty_factor_ = 0.0;
    double delta_f_scale_factor_ = 0.0;
    double acc_penalty_factor_ = 0.0;
    double acc_scale_factor_ = 0.0;
    double delta_control_penalty_factor_ = 0.0;
    double delta_control_scale_factor_ = 0.0;
    double lat_acc_penalty_factor_ = 0.0;
    double lat_acc_scale_factor_ = 0.0;
    double lat_dist_penalty_factor_ = 0.0;
    double lat_dist_scale_factor_ = 0.0;

    bool is_first_run_ = true;
};
