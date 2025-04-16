#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

#include <fstream>
#include <string>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

class StaticFramePublisher : public rclcpp::Node
{
public:
    StaticFramePublisher(const rclcpp::NodeOptions & options)
        : Node("static_frame_publisher", options)
    {
        this->get_parameter_or("frame_id", FRAME_ID, 0);

        config_data = YAML::LoadFile(yaml_config_path);
        config_parameters();

        FOLDER_NAME += BAG_NAME;
        DATASET_PATH = pkg_share_dir + FOLDER_NAME;

        pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/os1/lidar", 10);
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/localization/ego_pose", 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/label_gt_markers", 10);

        std::stringstream bin_path, pose_path, json_path;
        bin_path << DATASET_PATH << "/frame_" << FRAME_ID << ".bin";
        pose_path << DATASET_PATH << "/frame_" << FRAME_ID << ".pose";
        json_path << DATASET_PATH << "/frame_" << FRAME_ID << ".json";

        pc_msg_ = load_bin_to_pcl(bin_path.str());
        odom_msg_ = load_pose_to_odom(pose_path.str());
        marker_array_msg_ = load_json_to_markers(json_path.str());

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
    YAML::Node config_data;
    std::string pkg_share_dir = ament_index_cpp::get_package_share_directory("rosbag_util");
    std::string FOLDER_NAME = "/../../../../bag2bin_data";
    std::string BAG_NAME = "/multiego_bag2";
    std::string DATASET_PATH;
    std::string config_root_dir = "/../../../../config/lshape_detect.yaml";
    const std::string yaml_config_path = pkg_share_dir + config_root_dir;
    int FRAME_ID = 0;
    double RATE = 10;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    sensor_msgs::msg::PointCloud2::SharedPtr pc_msg_;
    nav_msgs::msg::Odometry::SharedPtr odom_msg_;
    visualization_msgs::msg::MarkerArray::SharedPtr marker_array_msg_;

    void config_parameters()
    {
        FOLDER_NAME = config_data["parameters"]["FOLDER_NAME"].as<std::string>();
        BAG_NAME = config_data["parameters"]["BAG_NAME"].as<std::string>();
        RATE = config_data["parameters"]["RATE"].as<double>();
    }

    sensor_msgs::msg::PointCloud2::SharedPtr load_bin_to_pcl(const std::string& path)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input) return nullptr;
        input.seekg(0, std::ios::end);
        size_t size = input.tellg(); input.seekg(0);
        size_t num_points = size / (3 * sizeof(float));
        std::vector<float> buffer(num_points * 3);
        input.read(reinterpret_cast<char*>(buffer.data()), size);

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

        auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(*cloud, *msg);
        msg->header.frame_id = "os1_frame";
        return msg;
    }

    nav_msgs::msg::Odometry::SharedPtr load_pose_to_odom(const std::string& path)
    {
        std::ifstream in(path);
        if (!in) return nullptr;
        double x, y, z, qx, qy, qz, qw;
        in >> x >> y >> z >> qx >> qy >> qz >> qw;
        auto msg = std::make_shared<nav_msgs::msg::Odometry>();
        msg->header.frame_id = "world_frame";
        msg->pose.pose.position.x = x;
        msg->pose.pose.position.y = y;
        msg->pose.pose.position.z = z;
        msg->pose.pose.orientation.x = qx;
        msg->pose.pose.orientation.y = qy;
        msg->pose.pose.orientation.z = qz;
        msg->pose.pose.orientation.w = qw;
        return msg;
    }

    visualization_msgs::msg::MarkerArray::SharedPtr load_json_to_markers(const std::string& path)
    {
        if (!fs::exists(path)) {
            RCLCPP_WARN(this->get_logger(), "GT JSON file does not exist: %s", path.c_str());
            return nullptr;
        }

        std::ifstream in(path);
        json j;
        in >> j;

        if (!j.contains("boxes")) {
            RCLCPP_WARN(this->get_logger(), "GT JSON missing 'boxes' field.");
            return nullptr;
        }

        auto msg = std::make_shared<visualization_msgs::msg::MarkerArray>();

        int id = 0;
        for (const auto& box : j["boxes"]) {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = "os1_frame";
            m.ns = "gt_box";
            m.id = id++;
            m.type = m.CUBE;
            m.action = m.ADD;

            m.pose.position.x = box["position"]["x"];
            m.pose.position.y = box["position"]["y"];
            m.pose.position.z = box["position"].value("z", -0.7);

            double yaw = box["orientation"]["yaw"];
            m.pose.orientation.x = 0.0;
            m.pose.orientation.y = 0.0;
            m.pose.orientation.z = std::sin(yaw / 2.0);
            m.pose.orientation.w = std::cos(yaw / 2.0);

            m.scale.x = box["size"].value("length", 4.6);
            m.scale.y = box["size"].value("width", 1.8);
            m.scale.z = box["size"].value("height", 1.4);

            m.color.a = 0.4;
            m.color.r = 0.2;
            m.color.g = 0.8;
            m.color.b = 0.2;

            m.lifetime = rclcpp::Duration::from_seconds(0.2);

            msg->markers.push_back(m);
        }

        RCLCPP_INFO(this->get_logger(), "Loaded %ld GT bounding boxes", msg->markers.size());
        return msg;
    }

    void publish_data()
    {
        rclcpp::Time now = this->now();
        pc_msg_->header.stamp = now;
        odom_msg_->header.stamp = now;
        pc_pub_->publish(*pc_msg_);
        odom_pub_->publish(*odom_msg_);
        if (marker_array_msg_) {
            for (auto& marker : marker_array_msg_->markers) {
                marker.header.stamp = now;
            }
            marker_pub_->publish(*marker_array_msg_);
        }
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto options = rclcpp::NodeOptions().allow_undeclared_parameters(true).automatically_declare_parameters_from_overrides(true);
    rclcpp::spin(std::make_shared<StaticFramePublisher>(options));
    rclcpp::shutdown();
    return 0;
}
