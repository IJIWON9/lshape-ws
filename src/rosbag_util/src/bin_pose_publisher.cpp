#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

#include <fstream>
#include <string>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace fs = std::filesystem;

class StaticFramePublisher : public rclcpp::Node
{
public:
    StaticFramePublisher()
        : Node("static_frame_publisher")
    {
        config_data = YAML::LoadFile(yaml_config_path);
        config_parameters();

        FOLDER_NAME += BAG_NAME;
        DATASET_PATH = pkg_share_dir + FOLDER_NAME;

        pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/os1/lidar", 10);
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/localization/ego_pose", 10);

        // Load data
        std::stringstream bin_path, pose_path;
        bin_path << DATASET_PATH << "/frame_" << FRAME_ID << ".bin";
        pose_path << DATASET_PATH << "/frame_" << FRAME_ID << ".pose";

        pc_msg_ = load_bin_to_pcl(bin_path.str());
        odom_msg_ = load_pose_to_odom(pose_path.str());

        if (!pc_msg_ || !odom_msg_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load frame %d", FRAME_ID);
            rclcpp::shutdown();
            return;
        }

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / RATE)),
            std::bind(&StaticFramePublisher::publish_data, this));
    }

private:
    // Configuration
    YAML::Node config_data;
    std::string pkg_share_dir = ament_index_cpp::get_package_share_directory("rosbag_util");
    std::string FOLDER_NAME = "/../../../../folder_path";
    std::string BAG_NAME = "/multiego_bag2";
    std::string DATASET_PATH;
    std::string config_root_dir = "/../../../../config/lshape_detect.yaml";
    const std::string yaml_config_path = pkg_share_dir + config_root_dir;
    int FRAME_ID = 0;
    double RATE = 10;

    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // Messages
    sensor_msgs::msg::PointCloud2::SharedPtr pc_msg_;
    nav_msgs::msg::Odometry::SharedPtr odom_msg_;

    void config_parameters()
    {
        FOLDER_NAME = config_data["parameters"]["FOLDER_NAME"].as<std::string>();
        BAG_NAME = config_data["parameters"]["BAG_NAME"].as<std::string>();
        FRAME_ID = config_data["parameters"]["FRAME_ID"].as<int>();
        RATE = config_data["parameters"]["RATE"].as<double>();
    }

    sensor_msgs::msg::PointCloud2::SharedPtr load_bin_to_pcl(const std::string & bin_file_path)
    {
        std::ifstream input(bin_file_path, std::ios::binary);
        if (!input) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open: %s", bin_file_path.c_str());
            return nullptr;
        }

        // 파일 크기 계산
        input.seekg(0, std::ios::end);
        std::streamsize size = input.tellg();
        input.seekg(0, std::ios::beg);

        if (size % (3 * sizeof(float)) != 0) {
            RCLCPP_ERROR(this->get_logger(), "Invalid file size: not divisible by 12");
            return nullptr;
        }

        size_t num_points = size / (3 * sizeof(float));
        std::vector<float> buffer(num_points * 3);
        input.read(reinterpret_cast<char*>(buffer.data()), size);
        input.close();

        // 포인트 클라우드 할당
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        cloud->points.resize(num_points);

        for (size_t i = 0; i < num_points; ++i) {
            cloud->points[i].x = buffer[i * 3 + 0];
            cloud->points[i].y = buffer[i * 3 + 1];
            cloud->points[i].z = buffer[i * 3 + 2];
        }

        cloud->width = num_points;
        cloud->height = 1;
        cloud->is_dense = true;

        auto pcd_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(*cloud, *pcd_msg);
        pcd_msg->header.frame_id = "os1_frame";
        return pcd_msg;
    }

    nav_msgs::msg::Odometry::SharedPtr load_pose_to_odom(const std::string & pose_file_path)
    {
        if (!fs::exists(pose_file_path)) {
            std::cerr << "[ERROR] File not found: " << pose_file_path << std::endl;
            return nullptr;
        }

        std::ifstream pose_in(pose_file_path);
        double x, y, z, qx, qy, qz, qw;
        pose_in >> x >> y >> z >> qx >> qy >> qz >> qw;

        auto odom_msg = std::make_shared<nav_msgs::msg::Odometry>();
        odom_msg->header.frame_id = "world_frame";
        odom_msg->pose.pose.position.x = x;
        odom_msg->pose.pose.position.y = y;
        odom_msg->pose.pose.position.z = z;
        odom_msg->pose.pose.orientation.x = qx;
        odom_msg->pose.pose.orientation.y = qy;
        odom_msg->pose.pose.orientation.z = qz;
        odom_msg->pose.pose.orientation.w = qw;

        return odom_msg;
    }

    void publish_data()
    {
        rclcpp::Time now = this->now();

        pc_msg_->header.stamp = now;
        odom_msg_->header.stamp = now;

        pc_pub_->publish(*pc_msg_);
        odom_pub_->publish(*odom_msg_);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StaticFramePublisher>());
    rclcpp::shutdown();
    return 0;
}
