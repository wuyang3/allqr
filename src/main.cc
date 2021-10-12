#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <vector>

#include "LqrPlanner.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "proto/conf.pb.h"

int main(int argc, char **argv)
{
    // load configuration file
    const std::string file_name = "../conf.txt";
    conf::LqrPlannerConf conf;

    using google::protobuf::TextFormat;
    using google::protobuf::io::FileInputStream;
    using google::protobuf::io::ZeroCopyInputStream;
    int file_descriptor = open(file_name.c_str(), O_RDONLY);
    if (file_descriptor < 0)
    {
        std::cout << "failed to open configuration file" << std::endl;
    }
    ZeroCopyInputStream *input = new FileInputStream(file_descriptor);
    TextFormat::Parse(input, &conf);
    delete input;
    close(file_descriptor);

    LqrPlanner planner = LqrPlanner(conf);
    int planner_type = conf.planner_type();
    // size_t iter = planner.ComputeOptimalControlExtendedLQR();
    size_t iter = 0;
    planner.Spin();
    // if (planner_type == 0)
    // {
    //     iter = planner.ComputeOptimalControlExtendedLQR();
    // }
    // else if (planner_type == 1)
    // {
    //     // iter = planner.ComputeOptimalControlIterativeLQR();
    // }

    // if (iter == 0)
    // {
    //     return 1;
    // }

    std::vector<Vector> solution_states = planner.GetSolutionStates();
    std::vector<Control> optimal_controls = planner.GetOptimalControl();
    std::vector<Vector> reference_traj = planner.GetReferenceTrajectory();
    std::pair<std::vector<Vector>, std::vector<Vector>> reference_traj_bounds = planner.GetReferenceTrajectoryLeftRightBounds();
    std::vector<Vector> ref_traj_left_bounds = reference_traj_bounds.first;
    std::vector<Vector> ref_traj_right_bounds = reference_traj_bounds.second;
    std::vector<double> lat_acc = planner.ComputeLateralAccelerations();
    std::vector<double> optimization_parameters = planner.GetOptimizationParameters();
    std::vector<double> lateral_error = planner.GetLateralError();

    std::ofstream states_file;
    states_file.open("states.txt");
    for (size_t i = 0; i < solution_states.size(); i++)
    {
        states_file << conf.dt() * i << ", " << solution_states[i](0) << ", " << solution_states[i](1) << ", " << solution_states[i](2) << ", " << solution_states[i](3) << ", " << lat_acc[i] << ", " << lateral_error[i] << std::endl;
    }
    states_file.close();

    std::ofstream controls_file;
    controls_file.open("controls.txt");
    for (size_t i = 0; i < optimal_controls.size(); i++)
    {
        controls_file << conf.dt() * i << ", " << solution_states[i](4) << ", " << solution_states[i](5) << std::endl;
    }
    controls_file.close();

    std::ofstream ref_traj_file;
    ref_traj_file.open("ref_traj.txt");
    for (size_t i = 0; i < reference_traj.size(); i++)
    {
        ref_traj_file << conf.dt() * i << ", " << reference_traj[i](0) << ", " << reference_traj[i](1) << ", " << reference_traj[i](2) << ", " << reference_traj[i](3) << std::endl;
    }
    ref_traj_file.close();

    std::ofstream ref_traj_left_bound_file;
    ref_traj_left_bound_file.open("ref_traj_left_bounds.txt");
    for (size_t i = 0; i < ref_traj_left_bounds.size(); i++)
    {
        ref_traj_left_bound_file << conf.dt() * i << ", " << ref_traj_left_bounds[i](0) << ", " << ref_traj_left_bounds[i](1) << std::endl;
    }
    ref_traj_left_bound_file.close();

    std::ofstream ref_traj_right_bound_file;
    ref_traj_right_bound_file.open("ref_traj_right_bounds.txt");
    for (size_t i = 0; i < ref_traj_right_bounds.size(); i++)
    {
        ref_traj_right_bound_file << conf.dt() * i << ", " << ref_traj_right_bounds[i](0) << ", " << ref_traj_right_bounds[i](1) << std::endl;
    }
    ref_traj_right_bound_file.close();

    std::ofstream optimization_parameters_file;
    optimization_parameters_file.open("optimization_parameters.txt");
    for (size_t i = 0; i < optimization_parameters.size(); i++)
    {
        optimization_parameters_file << optimization_parameters[i] << std::endl;
    }
    optimization_parameters_file.close();

    return 0;
}
