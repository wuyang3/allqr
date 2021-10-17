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
    void EvaluateConstraints(
        const Vector& x, const Control& u, Constraint& g);
    void EvaluateMiuInPenalty(const Constraint lambda, const Constraint miu,
                              const Constraint g, Constraint& miu_I);
    double ComputeConstraintCost(
        const Constraint& lambda, const Constraint& miu, const Constraint& g);
    void UpdateLambda(
        Constraint& lambda, const Constraint& miu, const Constraint& g);
    void UpdateMiu(Constraint& miu);

    double ComputeLateralError(const double x, const double y, const size_t t);

    size_t ComputeOptimalControlExtendedLQR();

    size_t ComputeOptimalControlIterativeLQR();

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

private:
    VehicleModel vehicle_model_;
    double wheelbase_;
    Vector initial_state_;
    Vector final_state_;
    Control initial_control_ = Control::Zero();
    const Control nominal_control_ = Control::Zero();
    size_t horizon_ = 0;
    size_t default_horizon_ = 0;
    size_t max_iter_ = 0;
    size_t state_dim_ = 0;
    size_t control_dim_ = 0;
    size_t constraint_dim_ = 0;
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
    double minCost_;

    conf::LqrPlannerConf conf_;

    double convergence_tolerance_ = 0.0;
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
