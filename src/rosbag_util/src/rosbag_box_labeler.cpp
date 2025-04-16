// rosbag_box_labeler.cpp
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

#include <fstream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

class RosbagBoxLabeler : public rclcpp::Node
{
public:
    RosbagBoxLabeler()
    : Node("rosbag_box_labeler")
    {
        // Declare and get frame_id parameter
        this->declare_parameter<int>("frame_id", 0);
        frame_id_ = this->get_parameter("frame_id").as_int();

        // Load config
        std::string pkg_share_dir = ament_index_cpp::get_package_share_directory("rosbag_util");
        YAML::Node config = YAML::LoadFile(pkg_share_dir + "/../../../../config/lshape_detect.yaml");
        std::string folder_name = config["parameters"]["FOLDER_NAME"].as<std::string>();
        std::string bag_name = config["parameters"]["BAG_NAME"].as<std::string>();

        output_path_ = pkg_share_dir + folder_name + bag_name;
        json_path_ = output_path_ + "/frame_" + std::to_string(frame_id_) + ".json";

        RCLCPP_INFO(this->get_logger(), "Labeling frame %d, output: %s", frame_id_, json_path_.c_str());

        // Subscribe to /initialpose
        sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/initialpose", 10,
            std::bind(&RosbagBoxLabeler::pose_callback, this, std::placeholders::_1));
    }

    ~RosbagBoxLabeler()
    {
        save_json();
    }

private:
    struct BoxLabel {
        double x, y, z;
        double yaw;
    };

    void pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
    {
        double x = msg->pose.pose.position.x;
        double y = msg->pose.pose.position.y;
        double z = 0.7;  // 고정 높이

        tf2::Quaternion q(
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z,
            msg->pose.pose.orientation.w);

        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

        BoxLabel box = {x, y, z, yaw};
        labels_.emplace_back(box);

        RCLCPP_INFO(this->get_logger(), "Saved box at (%.2f, %.2f) yaw=%.2f", x, y, yaw);
    }

    void save_json()
    {
        json output;
        output["frame_id"] = frame_id_;

        for (const auto& label : labels_) {
            json box;
            box["position"] = {{"x", label.x}, {"y", label.y}, {"z", label.z}};
            box["orientation"] = {{"yaw", label.yaw}};
            box["size"] = {{"length", 4.5}, {"width", 1.8}, {"height", 1.5}};
            output["boxes"].push_back(box);
        }

        std::ofstream out(json_path_);
        out << output.dump(4);
        RCLCPP_INFO(this->get_logger(), "Saved %ld box labels to %s", labels_.size(), json_path_.c_str());
    }

    int frame_id_;
    std::string output_path_;
    std::string json_path_;
    std::vector<BoxLabel> labels_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RosbagBoxLabeler>());
    rclcpp::shutdown();
    return 0;
}
