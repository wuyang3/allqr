#include "LqrPlanner.h"

LqrPlanner::LqrPlanner(const conf::LqrPlannerConf& conf)
{
    dt_ = conf.dt();
    wheelbase_ = conf.wheelbase();
    vehicle_model_.InitializeVehicleModel(wheelbase_, dt_);
    lane_width_ = conf.lane_width();
    horizon_ = conf.horizon() / conf.subdivision();
    default_horizon_ = horizon_;
    max_iter_ = conf.max_iter();
    convergence_tolerance_ = conf.convergence_tolerance();
    zero_tolerance_ = conf.zero_tolerance();

    lat_acc_bound_ = conf.lateral_acceleration_bound();
    lat_dist_bound_ = conf.lateral_distance_bound();
    reference_speed_ = conf.reference_speed();
    delta_f_penalty_factor_ = conf.delta_f_penalty_factor();
    delta_f_scale_factor_ = conf.delta_f_scale_factor();
    acc_penalty_factor_ = conf.acc_penalty_factor();
    acc_scale_factor_ = conf.acc_scale_factor();
    delta_control_penalty_factor_ = conf.delta_control_penalty_factor();
    delta_control_scale_factor_ = conf.delta_control_scale_factor();
    lat_acc_penalty_factor_ = conf.lat_acc_penalty_factor();
    lat_acc_scale_factor_ = conf.lat_acc_scale_factor();
    lat_dist_penalty_factor_ = conf.lat_distance_penalty_factor();
    lat_dist_scale_factor_ = conf.lat_distance_scale_factor();

    init_miu_ = conf.init_miu();
    phi_ = conf.phi();

    state_dim_ = vehicle_model_.state_dim_ + vehicle_model_.control_dim_;
    control_dim_ = vehicle_model_.control_dim_;
    constraint_dim_ = 10;  // temporarily given.

    initial_state_.resize(state_dim_);
    final_state_.resize(state_dim_);

    Q_ = Matrix::Zero(state_dim_, state_dim_);
    Q_t_ = Matrix::Zero(state_dim_, state_dim_);
    R_ = Matrix::Zero(control_dim_, control_dim_);

    for (size_t i = 0; i < state_dim_; i++)
    {
        Q_(i, i) = conf.matrix_q().value(i);
        Q_t_(i, i) = conf.matrix_q_intermediate().value(i);
    }

    for (size_t i = 0; i < control_dim_; i++)
    {
        initial_control_(i) = conf.initial_control().value(i);

        u_min_(i) = conf.control_bounds().lower_bound(i);
        u_max_(i) = conf.control_bounds().upper_bound(i);

        delta_u_min_(i) = conf.delta_control_bounds().lower_bound(i);
        delta_u_max_(i) = conf.delta_control_bounds().upper_bound(i);

        R_(i, i) = conf.matrix_r().value(i);
    }

    // construct reference line segments
    std::vector<double> ref_lines_x;
    std::vector<double> ref_lines_y;

    for (size_t i = 0; i < conf.reference_lines_x().value_size(); i++)
    {
        ref_lines_x.push_back(conf.reference_lines_x().value(i));
        ref_lines_y.push_back(conf.reference_lines_y().value(i));
    }

    double prev_ref_orientation = 0.0;
    for (size_t i = 0; i < ref_lines_x.size() - 1; i++)
    {
        LineSegment line_seg;
        const double x1 = ref_lines_x[i];
        const double x2 = ref_lines_x[i + 1];
        const double y1 = ref_lines_y[i];
        const double y2 = ref_lines_y[i + 1];
        line_seg.x_domain = Eigen::Vector2d(x1, x2);
        line_seg.y_domain = Eigen::Vector2d(y1, y2);
        line_seg.start_point = Eigen::Vector2d(x1, y1);
        line_seg.end_point = Eigen::Vector2d(x2, y2);
        line_seg.length = std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));
        line_seg.orientation = std::atan2(y2 - y1, x2 - x1);

        if (line_seg.orientation - prev_ref_orientation > M_PI)
        {
            while (line_seg.orientation - prev_ref_orientation > M_PI)
            {
                line_seg.orientation -= 2 * M_PI;
            }
        }
        else if (line_seg.orientation - prev_ref_orientation < -M_PI)
        {
            while (line_seg.orientation - prev_ref_orientation < -M_PI)
            {
                line_seg.orientation += 2 * M_PI;
            }
        }

        Eigen::Vector3d coefficients;
        if (x1 == x2)
        {
            coefficients(0) = 1.0;
            coefficients(1) = 0.0;
            coefficients(2) = -x1;
        }
        else if (y1 == y2)
        {
            coefficients(0) = 0.0;
            coefficients(1) = 1.0;
            coefficients(2) = -y1;
        }
        else
        {
            Eigen::Matrix2d mat;
            Eigen::Vector2d vec;
            vec(0) = -1;
            vec(1) = -1;
            mat(0, 0) = x1;
            mat(0, 1) = y1;
            mat(1, 0) = x2;
            mat(1, 1) = y2;
            Eigen::Vector2d ab = mat.inverse() * vec;
            coefficients(0) = ab(0);
            coefficients(1) = ab(1);
            coefficients(2) = 1.0;
        }
        line_seg.coefficients = coefficients;
        ref_lines_.push_back(line_seg);
        total_length_ += line_seg.length;
        // line_seg.PrintInfo();

        prev_ref_orientation = line_seg.orientation;
    }

    const double initial_position_std = conf.initial_position_std();
    const double initial_heading_std = conf.initial_heading_std();
    const double initial_speed_std = conf.initial_speed_std();
    const double initial_x = ref_lines_[0].x_domain(0);
    const double initial_y = ref_lines_[0].y_domain(0);
    const double initial_heading = ref_lines_[0].orientation;

    std::default_random_engine generator;
    if (conf.enable_random_initialization())
    {
        generator.seed(time(NULL));
    }
    std::normal_distribution<double> position_x_distribution(initial_x, initial_position_std);
    std::normal_distribution<double> position_y_distribution(initial_y, initial_position_std);
    std::normal_distribution<double> heading_distribution(initial_heading, initial_heading_std);
    std::normal_distribution<double> speed_distribution(reference_speed_, initial_speed_std);

    initial_state_(0) = position_x_distribution(generator);
    initial_state_(1) = position_y_distribution(generator);
    initial_state_(2) = heading_distribution(generator);
    initial_state_(3) = std::max(0.0, speed_distribution(generator));
    initial_state_(4) = initial_control_(0);
    initial_state_(5) = initial_control_(1);
}

Eigen::Vector2d LqrPlanner::FindCloestPointToLineSeg(const double x, const double y, const LineSegment line_seg)
{
    const Eigen::Vector3d coefficients = line_seg.coefficients;
    const double a = coefficients(0);
    const double b = coefficients(1);
    const double c = coefficients(2);

    const Eigen::Vector2d diff = line_seg.end_point - line_seg.start_point;
    const double new_c = b * x - a * y;

    Eigen::Matrix2d A;
    A(0, 0) = a;
    A(0, 1) = b;
    A(1, 0) = -b;
    A(1, 1) = a;
    Eigen::Vector2d B;
    B(0) = -c;
    B(1) = -new_c;
    Eigen::Vector2d p = A.inverse() * B;

    double t;
    if (std::fabs(diff(0)) < 1e-5)
    {
        t = (p(1) - line_seg.start_point(1)) / diff(1);
    }
    else if (std::fabs(diff(1)) < 1e-5)
    {
        t = (p(0) - line_seg.start_point(0)) / diff(0);
    }
    else
    {
        t = (p(0) - line_seg.start_point(0)) / diff(0);
    }

    if (t < 0)
    {
        return line_seg.start_point;
    }
    else if (t > 1)
    {
        return line_seg.end_point;
    }
    else
    {
        return p;
    }
}

void LqrPlanner::GenerateTrackingReference()
{
    // construct reference trajectory from the reference line segments
    const double max_length = reference_speed_ * dt_ * horizon_;
    const double unit_length = dt_ * reference_speed_;

    const double init_x = initial_state_(0);
    const double init_y = initial_state_(1);

    if (curr_line_seg_id_ >= ref_lines_.size())
    {
        ready_to_solve_ = false;
        return;
    }

    Eigen::Vector2d closest_point = FindCloestPointToLineSeg(init_x, init_y, ref_lines_[curr_line_seg_id_]);

    double curr_line_seg_length = sqrt((closest_point - ref_lines_[curr_line_seg_id_].start_point).transpose() * (closest_point - ref_lines_[curr_line_seg_id_].start_point));
    double curr_x = closest_point(0); //ref_lines_[0].start_point(0);
    double curr_y = closest_point(1); //ref_lines_[0].start_point(1);
    Eigen::VectorXd ref_state;
    Eigen::VectorXd ref_state_left_bound;
    Eigen::VectorXd ref_state_right_bound;
    ref_state.resize(state_dim_);
    ref_state_left_bound.resize(state_dim_);
    ref_state_right_bound.resize(state_dim_);

    ref_traj_.clear();
    ref_traj_points_line_seg_id_.clear();

    ref_state << curr_x, curr_y, ref_lines_[curr_line_seg_id_].orientation, reference_speed_, 0.0, 0.0;
    ref_traj_.push_back(ref_state);
    ref_traj_full_.push_back(ref_state);
    ref_traj_points_line_seg_id_.push_back(curr_line_seg_id_);

    const double left_orientation = ref_lines_[curr_line_seg_id_].orientation + M_PI / 2;
    const double right_orientation = ref_lines_[curr_line_seg_id_].orientation - M_PI / 2;
    const double left_x = curr_x + std::cos(left_orientation) * lane_width_ / 2;
    const double left_y = curr_y + std::sin(left_orientation) * lane_width_ / 2;
    const double right_x = curr_x + std::cos(right_orientation) * lane_width_ / 2;
    const double right_y = curr_y + std::sin(right_orientation) * lane_width_ / 2;

    ref_state_left_bound << left_x, left_y, ref_lines_[curr_line_seg_id_].orientation, reference_speed_, 0.0, 0.0;
    ref_state_right_bound << right_x, right_y, ref_lines_[curr_line_seg_id_].orientation, reference_speed_, 0.0, 0.0;

    ref_traj_left_bound_.push_back(ref_state_left_bound);
    ref_traj_right_bound_.push_back(ref_state_right_bound);

    double new_x, new_y;
    for (size_t i = 0; i < horizon_; i++)
    {
        new_x = curr_x + std::cos(ref_lines_[curr_line_seg_id_].orientation) * unit_length;
        new_y = curr_y + std::sin(ref_lines_[curr_line_seg_id_].orientation) * unit_length;
        curr_line_seg_length += unit_length;
        if (curr_line_seg_length >= ref_lines_[curr_line_seg_id_].length)
        {
            double length_left = ref_lines_[curr_line_seg_id_].length +
                                 unit_length - curr_line_seg_length;
            curr_line_seg_length = unit_length - length_left;
            double end_x = ref_lines_[curr_line_seg_id_].end_point(0);
            double end_y = ref_lines_[curr_line_seg_id_].end_point(1);
            curr_line_seg_id_++;
            if (curr_line_seg_id_ >= ref_lines_.size())
            {
                break;
            }
            new_x = end_x + std::cos(ref_lines_[curr_line_seg_id_].orientation) * curr_line_seg_length;
            new_y = end_y + std::sin(ref_lines_[curr_line_seg_id_].orientation) * curr_line_seg_length;
        }
        ref_state << new_x, new_y, ref_lines_[curr_line_seg_id_].orientation, reference_speed_, 0.0, 0.0;
        if (new_y < -11.0)
        {
            std::cout << "ref_state = \n"
                      << ref_state << std::endl;
            std::cout << "curr_line_seg_id_ = " << curr_line_seg_id_ << std::endl;
        }
        ref_traj_.push_back(ref_state);
        ref_traj_full_.push_back(ref_state);
        ref_traj_points_line_seg_id_.push_back(curr_line_seg_id_);
        curr_x = new_x;
        curr_y = new_y;

        const double left_orientation = ref_lines_[curr_line_seg_id_].orientation + M_PI / 2;
        const double right_orientation = ref_lines_[curr_line_seg_id_].orientation - M_PI / 2;
        const double left_x = new_x + std::cos(left_orientation) * lane_width_ / 2;
        const double left_y = new_y + std::sin(left_orientation) * lane_width_ / 2;
        const double right_x = new_x + std::cos(right_orientation) * lane_width_ / 2;
        const double right_y = new_y + std::sin(right_orientation) * lane_width_ / 2;

        ref_state_left_bound << left_x, left_y, ref_lines_[curr_line_seg_id_].orientation, reference_speed_, 0.0, 0.0;
        ref_state_right_bound << right_x, right_y, ref_lines_[curr_line_seg_id_].orientation, reference_speed_, 0.0, 0.0;

        ref_traj_left_bound_.push_back(ref_state_left_bound);
        ref_traj_right_bound_.push_back(ref_state_right_bound);
    }

    if (total_length_ >= max_length)
    {
        ready_to_solve_ = true;
    }

    final_state_(0) = ref_traj_full_.back()(0);
    final_state_(1) = ref_traj_full_.back()(1);
    final_state_(2) = ref_lines_.back().orientation;
    final_state_(3) = reference_speed_;
    final_state_(4) = nominal_control_(0);
    final_state_(5) = nominal_control_(1);

    // std::cout
    //     << "initial state = \n"
    //     << initial_state_ << std::endl;
    // std::cout << "--------------" << std::endl;
    // std::cout << "final state = \n"
    //           << final_state_ << std::endl;
    // std::cout << "--------------" << std::endl;

    if (ref_traj_.size() < horizon_)
    {
        horizon_ = ref_traj_.size() - 1;
        std::cout << " * reduced horizon: " << horizon_ << std::endl;
    }
}

double LqrPlanner::ComputeLateralError(const double x, const double y, size_t t)
{
    int line_seg_id = ref_traj_points_line_seg_id_[t];
    const double a = ref_lines_[line_seg_id].coefficients(0);
    const double b = ref_lines_[line_seg_id].coefficients(1);
    const double c = ref_lines_[line_seg_id].coefficients(2);
    return (a * x + b * y + c) / std::sqrt(a * a + b * b);
}

double LqrPlanner::ControlCost(const Control& u)
{
    return (u - nominal_control_).transpose() * R_ * (u - nominal_control_);
}

void LqrPlanner::QuadratizeCost(const Vector& x, const Control& u, const size_t& t, Matrix& Pt, Matrix& Qt, Matrix& Rt, Vector& qt, Control& rt)
{
    if (t == 0)
    {
        Qt = Q_t_;
        qt = -Q_t_ * initial_state_;
    }
    else
    {
        // if (t == horizon_ - 1)
        {
            Vector ref_state = ref_traj_[t];

            Qt = Q_t_;
            qt = -Qt * ref_state;
        }
    }

    Rt = R_;
    rt = -R_ * nominal_control_;  // why nominal control here?

    Pt.setZero();
}

void LqrPlanner::QuadratizeConstraints(
    const Vector& x, const Control& u,
    const Constraint& lambda, const Constraint& miu,
    Matrix& Pt, Matrix& Qt, Matrix& Rt, Vector& qt, Control& rt)
{
    double k1 = 2 * x(3) * std::tan(x(4)) / wheelbase_;
    double k2 = std::pow(x(3), 2) / std::pow(std::cos(x(4)), 2) / wheelbase_;
    Constraint g = Constraint::Zero();
    Constraint miu_I = Constraint::Zero();
    EvaluateConstraints(x, u, g);
    EvaluateMiuInPenalty(lambda, miu, g, miu_I);

    // Update the cost hessian.
    Qt(3, 3) += (miu_I(8) + miu_I(9)) * std::pow(k1, 2);
    Qt(4, 4) += (miu_I(0) + miu_I(1)) + (miu_I(8) + miu_I(9)) * std::pow(k2, 2);
    Qt(4, 3) += (miu_I(8) + miu_I(9)) * k1 * k2;
    Qt(3, 4) += (miu_I(8) + miu_I(9)) * k1 * k2;
    Qt(5, 5) += (miu_I(2) + miu_I(3));

    Rt(0, 0) += (miu_I(4) + miu_I(5));
    Rt(1, 1) += (miu_I(6) + miu_I(7));

    // Update the cost gradient.
    qt(3) += (k1 * (lambda(8) - lambda(9) + g(8) * miu_I(8) - g(9) * miu_I(9))
             - (x(3) * std::pow(k1, 2) + x(4) * k1 * k2) * (miu_I(8) + miu_I(9)));
    qt(4) += (lambda(0) - lambda(1)
             + k2 * (lambda(8) - lambda(9) + g(8) * miu_I(8) - g(9) * miu_I(9))
             + g(0) * miu_I(0) - g(1) * miu_I(1)
             - x(4) * (miu_I(0) + miu_I(1))
             - (x(4) * std::pow(k2, 2) + x(3) * k1 * k2) * (miu_I(8) + miu_I(9)));
    qt(5) += (lambda(2) - lambda(3) - x(5) * (miu_I(2) + miu_I(3))+ g(2) * miu_I(2) - g(3) * miu_I(3));

    rt(0) += (lambda(4) - lambda(5) - u(0) * (miu_I(4) + miu_I(5)) + g(4) * miu_I(4) - g(5) * miu_I(5));
    rt(1) += (lambda(6) - lambda(7) - u(1) * (miu_I(6) + miu_I(7)) + g(6) * miu_I(6) - g(7) * miu_I(7));
}

void LqrPlanner::QuadratizeFinalCost(const Vector& x, Matrix& QFinal, Vector& qFinal)
{
    QFinal = Q_;
    qFinal = -Q_ * x; //final_state_;
}

void LqrPlanner::EvaluateConstraints(const Vector& x, const Control& u, Constraint& g)
{
    g(0) = x(4) - u_max_(0);
    g(1) = -x(4) + u_min_(0);
    g(2) = x(5) - u_max_(1);
    g(3) = -x(5) + u_min_(1);
    g(4) = u(0) - delta_u_max_(0);
    g(5) = -u(0) + delta_u_min_(0);
    g(6) = u(1) - delta_u_max_(1);
    g(7) = -u(1) + delta_u_min_(1);
    g(8) = std::pow(x(3), 2) * std::tan(x(4)) / wheelbase_ - lat_acc_bound_;
    g(9) = -std::pow(x(3), 2) * std:: tan(x(4)) / wheelbase_ + lat_acc_bound_;
}

void LqrPlanner::EvaluateMiuInPenalty(const Constraint lambda, const Constraint miu, const Constraint g, Constraint& miu_I)
{
    miu_I = (lambda.array().abs() < zero_tolerance_ && g.array() < 0.0).select(Constraint::Zero(), miu);
}

double LqrPlanner::ComputeIntermediateCost(const Vector& x, const Control& u, const size_t& t)
{
    // std::cout << "t = " << t << std::endl;
    double cost = 0;
    if (t == 0)
    {
        cost = (x - initial_state_).transpose() * Q_ * (x - initial_state_);
    }
    // else if (t == horizon_ - 1)
    else
    {
        Vector ref_state = ref_traj_[t];
        cost += (x - ref_state).transpose() * Q_t_ * (x - ref_state);
    }
    cost += ControlCost(u);

    return cost;
}

double LqrPlanner::ComputeConstraintCost(const Constraint& lambda, const Constraint& miu_I, const Constraint& g)
{
    return g.transpose() * (lambda + miu_I.cwiseProduct(g));
}

void LqrPlanner::UpdateLambda(Constraint& lambda, const Constraint& miu, const Constraint& g)
{
    lambda = lambda + miu.cwiseProduct(g);
    lambda = lambda.cwiseMax(0.0);
}

void LqrPlanner::UpdateMiu(Constraint& miu)
{
    miu = miu * phi_;
}

double LqrPlanner::ComputeFinalCost(const Vector& x)
{
    double cost = (x - final_state_).transpose() * Q_ * (x - final_state_);
    return cost;
}

std::vector<Vector> LqrPlanner::GetSolutionStates()
{
    return solution_states_;
}

std::vector<Control> LqrPlanner::GetOptimalControl()
{
    return optimal_control_;
}

std::vector<Vector> LqrPlanner::GetReferenceTrajectory()
{
    return ref_traj_full_;
}

std::vector<double> LqrPlanner::GetLateralError()
{
    return lateral_error_;
}

std::pair<std::vector<Vector>, std::vector<Vector>> LqrPlanner::GetReferenceTrajectoryLeftRightBounds()
{
    std::pair<std::vector<Vector>, std::vector<Vector>> result;
    result.first = ref_traj_left_bound_;
    result.second = ref_traj_right_bound_;
    return result;
}

std::vector<double> LqrPlanner::GetOptimizationParameters()
{
    std::vector<double> result;
    result.push_back(dt_);
    result.push_back(default_horizon_);
    result.push_back(u_min_(0));
    result.push_back(u_min_(1));
    result.push_back(u_max_(0));
    result.push_back(u_max_(1));
    result.push_back(lat_acc_bound_);
    result.push_back(lat_dist_bound_);
    return result;
}

std::vector<double> LqrPlanner::ComputeLateralAccelerations()
{
    std::vector<double> result;
    for (size_t i = 0; i < solution_states_.size(); i++)
    {
        const double v = solution_states_[i](3);
        const double delta_f = solution_states_[i](4);
        const double lat_acc = v * v / wheelbase_ * std::tan(delta_f);
        result.push_back(lat_acc);
    }
    return result;
}

void LqrPlanner::Spin()
{
    ready_to_solve_ = true;
    size_t count = 1;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "frame #   |   iter  |   time (s)" << std::endl;
    while (ready_to_solve_)
    {
        clock_t beginTime = clock();
        size_t iter = ComputeOptimalControlExtendedLQR();
        clock_t endTime = clock();
        if (iter > 0)
        {
            std::cout << "--------------------------------" << std::endl;
            std::cout << std::setw(5) << count;
            std::cout << "    |   " << iter;
            std::cout << "    |   " << std::setprecision(3) << (endTime - beginTime) / (double)CLOCKS_PER_SEC;
            std::cout << std::endl;
        }
        count++;
    }
}

size_t LqrPlanner::ComputeOptimalControlExtendedLQR()
{
    GenerateTrackingReference();

    if (ready_to_solve_ == false)
    {
        // std::cout << "problem is not solvable" << std::endl;
        return 0;
    }

    // Initialization of multipliers, penalities, constraints.
    std::vector<Constraint> Lambda;
    std::vector<Constraint> Miu;
    std::vector<Constraint> G;

    for (size_t i = 0; i < horizon_; i++)
    {
        Lambda.push_back(Constraint::Zero());
        Miu.push_back(init_miu_ * Constraint::Ones());
        G.push_back(Constraint::Zero());
    }

    // Initialization
    std::vector<Matrix> L;
    std::vector<Control> l;
    for (size_t i = 0; i < horizon_; i++)
    {
        L.push_back(Matrix::Zero(control_dim_, state_dim_));
        l.push_back(nominal_control_);
    }

    std::vector<Matrix> S;
    std::vector<Vector> s;
    std::vector<Matrix> SBar;
    std::vector<Vector> sBar;
    for (size_t i = 0; i < horizon_ + 1; i++)
    {
        S.push_back(Matrix::Zero(state_dim_, state_dim_));
        s.push_back(Vector::Zero(state_dim_));

        SBar.push_back(Matrix::Zero(state_dim_, state_dim_));
        sBar.push_back(Vector::Zero(state_dim_));
    }

    double oldCost = -log(0.0);
    Vector xHat = initial_state_;

    for (size_t iter = 1; iter <= max_iter_; iter++)
    {
        xHat = initial_state_;
        // forward pass
        for (size_t t = 0; t < horizon_; ++t)
        {
            const Control uHat = L[t] * xHat + l[t];
            const Vector xHatPrime = vehicle_model_.ForwardAugmentedDynamics(xHat, uHat);

            // const std::pair<Matrix, Matrix> result = vehicle_model_.ComputeJacobian(xHatPrime, uHat, BACKWARD);

            // Not linearize the explicit inverse dynamic equations but linearzied forward dynamics at [xHat(t+1), u(t)].
            // The state derivative is calculated around [xHat(t+1), u(t)]. The state matrix can be calculated by
            // integrating the derivative backwards by one step and get the same result.
            const std::pair<Matrix, Matrix> result = vehicle_model_.ComputeAugmentedLinearModel(xHatPrime, uHat);

            // backward direction
            const Matrix ABar = result.first.inverse();
            const Matrix BBar = -ABar * result.second;
            const Vector cBar = xHat - ABar * xHatPrime - BBar * uHat;

            Matrix P = Matrix::Zero(control_dim_, state_dim_);
            Matrix Q = Matrix::Zero(state_dim_, state_dim_);
            Matrix R = Matrix::Zero(control_dim_, control_dim_);

            Vector q(state_dim_);
            q.setZero();
            Control r = Control::Zero();
            QuadratizeCost(xHat, uHat, t, P, Q, R, q, r);
            QuadratizeConstraints(xHat, uHat, Lambda[t], Miu[t], P, Q, R, q, r);

            const Matrix SBarQ = SBar[t] + Q;
            const Vector sBarqSBarQcBar = sBar[t] + q + SBarQ * cBar;

            const Matrix CBar = BBar.transpose() * SBarQ * ABar + P * ABar;
            const Matrix DBar = ABar.transpose() * SBarQ * ABar;
            const Matrix EBar = BBar.transpose() * SBarQ * BBar + R + P * BBar + BBar.transpose() * P.transpose();
            const Vector dBar = ABar.transpose() * sBarqSBarQcBar;
            const Control eBar = BBar.transpose() * sBarqSBarQcBar + r + P * cBar;

            L[t] = -EBar.inverse() * CBar;
            l[t] = -EBar.inverse() * eBar;

            SBar[t + 1] = DBar + CBar.transpose() * L[t];
            sBar[t + 1] = dBar + CBar.transpose() * l[t];

            xHat = -(S[t + 1] + SBar[t + 1]).inverse() * (s[t + 1] + sBar[t + 1]);
        }

        // backward pass
        QuadratizeFinalCost(xHat, S[horizon_], s[horizon_]);  // do we need to add constraint cost here?
        xHat = -(S[horizon_] + SBar[horizon_]).inverse() * (s[horizon_] + sBar[horizon_]);

        for (size_t t = horizon_ - 1; t != -1; --t)
        {
            const Control uHat = L[t] * xHat + l[t];
            const Vector xHatPrime = vehicle_model_.InverseAugmentedDynamics(xHat, uHat);
            const std::pair<Matrix, Matrix> result = vehicle_model_.ComputeAugmentedLinearModel(xHatPrime, uHat); // forward direction

            const Matrix A = result.first;
            const Matrix B = result.second;
            const Vector c = xHat - A * xHatPrime - B * uHat;

            Matrix P = Matrix::Zero(control_dim_, state_dim_);
            Matrix Q = Matrix::Zero(state_dim_, state_dim_);
            Matrix R = Matrix::Zero(control_dim_, control_dim_);
            Vector q(state_dim_);
            q.setZero();
            Control r = Control::Zero();

            QuadratizeCost(xHatPrime, uHat, t, P, Q, R, q, r);
            QuadratizeConstraints(xHatPrime, uHat, Lambda[t], Miu[t], P, Q, R, q, r);

            const Matrix C = B.transpose() * S[t + 1] * A + P;
            const Matrix D = A.transpose() * S[t + 1] * A + Q;
            const Matrix E = B.transpose() * S[t + 1] * B + R;
            const Vector d = A.transpose() * (s[t + 1] + S[t + 1] * c) + q;
            const Control e = B.transpose() * (s[t + 1] + S[t + 1] * c) + r;

            L[t] = -E.inverse() * C;
            l[t] = -E.inverse() * e;

            S[t] = D + C.transpose() * L[t];
            s[t] = d + C.transpose() * l[t];

            xHat = -(S[t] + SBar[t]).inverse() * (s[t] + sBar[t]);
        }

        // compute cost
        double newCost = 0;
        Vector x = xHat;

        for (size_t t = 0; t < horizon_; t++)
        {
            Control u = L[t] * x + l[t];
            newCost += ComputeIntermediateCost(x, u, t);

            Constraint g = Constraint::Zero();
            Constraint miu_I = Constraint::Zero();
            EvaluateConstraints(x, u, g);
            EvaluateMiuInPenalty(Lambda[t], Miu[t], g, miu_I);
            newCost += ComputeConstraintCost(Lambda[t], miu_I, g);
            G[t] = g;

            x = vehicle_model_.ForwardAugmentedDynamics(x, u);
        }

        newCost += ComputeFinalCost(x);

        for (size_t t = 0; t < horizon_; t++)
        {
            UpdateLambda(Lambda[t], Miu[t], G[t]);
            UpdateMiu(Miu[t]);
        }

        // std::cout << "oldcost = " << oldCost << std::endl;
        // std::cout << "newcost = " << newCost << std::endl;
        // std::cout << "abs((oldCost - newCost) = " << std::fabs(oldCost - newCost) << std::endl;

        if (newCost < oldCost)
        {
            minCost_ = newCost;
        }

        if (std::fabs(oldCost - newCost) / newCost < convergence_tolerance_)
        {
            // std::cout << "minimum cost = " << minCost_ << std::endl;

            // execute control
            // solution_states_.clear();
            // optimal_control_.clear();
            Vector x = initial_state_;
            if (is_first_run_)
            {
                solution_states_.push_back(x);
                is_first_run_ = false;
            }
            double lat_error = ComputeLateralError(x(0), x(1), 0);

            lateral_error_.push_back(lat_error);

            for (size_t t = 0; t < horizon_; t++)
            {
                Control u = L[t] * x + l[t];
                optimal_control_.push_back(u);
                x = vehicle_model_.ForwardAugmentedDynamics(x, u);
                solution_states_.push_back(x);
                lat_error = ComputeLateralError(x(0), x(1), t + 1);

                lateral_error_.push_back(lat_error);
            }

            initial_state_ = solution_states_.back();
            initial_control_(0) = 0.0; //solution_states_.back()(4);
            initial_control_(1) = 0.0; //solution_states_.back()(5);
            // initial_control_ = optimal_control_.back();

            return iter;
        }

        oldCost = newCost;
    }

    // execute control
    // solution_states_.clear();
    // optimal_control_.clear();
    Vector x = initial_state_;
    solution_states_.push_back(x);

    for (size_t t = 0; t < horizon_; t++)
    {
        Control u = L[t] * x + l[t];
        optimal_control_.push_back(u);
        x = vehicle_model_.ForwardAugmentedDynamics(x, u);
        solution_states_.push_back(x);
    }

    return max_iter_;
}

// size_t LqrPlanner::ComputeOptimalControlIterativeLQR()
// {
//     if (ready_to_solve_ == false)
//     {
//         std::cout << "problem is not solvable" << std::endl;
//         return 0;
//     }

//     // Initialization
//     std::vector<Matrix> L;
//     std::vector<Control> l;
//     for (size_t i = 0; i < horizon_; i++)
//     {
//         L.push_back(Matrix::Zero(control_dim_, state_dim_));
//         l.push_back(nominal_control_);
//     }

//     std::vector<State> xHat(horizon_ + 1, State::Zero());
//     std::vector<State> xHatNew(horizon_ + 1, State::Zero());
//     std::vector<Control> uHat;
//     std::vector<Control> uHatNew;
//     uHat.resize(horizon_);
//     uHatNew.resize(horizon_);

//     for (size_t i = 0; i < horizon_; i++)
//     {
//         uHat[i] = nominal_control_;
//     }

//     double oldCost = 1e8;

//     for (size_t iter = 0; iter < max_iter_; iter++)
//     {
//         double newCost;
//         double alpha = 1.0;

//         do
//         {
//             newCost = 0;

//             // init trajectory
//             xHatNew[0] = initial_state_;
//             for (size_t t = 0; t < horizon_; ++t)
//             {
//                 uHatNew[t] = (1.0 - alpha) * uHat[t] + L[t] * (xHatNew[t] - (1.0 - alpha) * xHat[t]) + alpha * l[t];
//                 xHatNew[t + 1] = vehicle_model_.ForwardDynamics(xHatNew[t], uHatNew[t]);

//                 newCost += ComputeIntermediateCost(xHatNew[t], uHatNew[t], t);
//             }
//             newCost += ComputeFinalCost(xHatNew[horizon_]);

//             alpha *= 0.5;
//         } while (!(newCost < oldCost || abs((oldCost - newCost) / newCost) < convergence_tolerance_));

//         xHat = xHatNew;
//         uHat = uHatNew;

//         if (abs((oldCost - newCost) / newCost) < convergence_tolerance_)
//         {
//             best_optimal_control_ = uHat;
//             best_solution_states_ = xHat;
//             return iter;
//         }

//         oldCost = newCost;

//         Matrix S = Matrix::Zero(state_dim_, state_dim_);
//         State s = State::Zero();

//         QuadratizeFinalCost(xHat[horizon_], S, s, iter);

//         for (size_t t = horizon_ - 1; t != -1; --t)
//         {

//             const std::pair<Matrix, Matrix> result = vehicle_model_.ComputeLinearModel(xHat[t], uHat[t]);

//             const Matrix A = result.first;
//             const Matrix B = result.second;

//             const State c = xHat[t + 1] - A * xHat[t] - B * uHat[t];

//             Matrix P = Matrix::Zero(control_dim_, state_dim_);
//             Matrix Q = Matrix::Zero(state_dim_, state_dim_);
//             Matrix R = Matrix::Zero(control_dim_, control_dim_);

//             State q;
//             Control r;

//             QuadratizeCost(xHat[t], uHat[t], t, P, Q, R, q, r, iter);

//             const Matrix C = B.transpose() * S * A + P;
//             const Matrix D = A.transpose() * S * A + Q;
//             const Matrix E = B.transpose() * S * B + R;
//             const State d = A.transpose() * (s + S * c) + q;
//             const Control e = B.transpose() * (s + S * c) + r;

//             L[t] = -E.inverse() * C;
//             l[t] = -E.inverse() * e;

//             S = D + C.transpose() * L[t];
//             s = d + C.transpose() * l[t];
//         }
//     }

//     best_optimal_control_ = uHat;
//     best_solution_states_ = xHat;

//     return max_iter_;
// }
