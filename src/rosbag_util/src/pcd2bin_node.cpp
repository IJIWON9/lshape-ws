#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include "ament_index_cpp/get_package_share_directory.hpp"

namespace fs = std::filesystem;

class PointCloudToBinNode : public rclcpp::Node
{
public:
    PointCloudToBinNode()
    : Node("pointcloud_to_bin_node")
    {
        // Load config
        pkg_share_dir = ament_index_cpp::get_package_share_directory("rosbag_util");
        config_data = YAML::LoadFile(pkg_share_dir + "/../../../../config/lshape_detect.yaml");
        config_parameters();

        // Init output folder
        output_dir_ = pkg_share_dir + FOLDER_NAME + BAG_NAME;
        if (!fs::exists(output_dir_)) {
            fs::create_directories(output_dir_);
        }

        // Subscribe
        pc_subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/os1/lidar", 10, std::bind(&PointCloudToBinNode::pointcloud_callback, this, std::placeholders::_1));

        odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/localization/ego_pose", 10, std::bind(&PointCloudToBinNode::odometry_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Initialized. Saving to: %s", output_dir_.c_str());
    }

private:
    // Config
    YAML::Node config_data;
    std::string pkg_share_dir;
    std::string FOLDER_NAME;
    std::string BAG_NAME;
    std::string output_dir_;
    int frame_counter_ = 0;

    // Buffer
    std::map<rclcpp::Time, nav_msgs::msg::Odometry::SharedPtr> odom_buffer_;

    // Subscriptions
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_subscription_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;

    void config_parameters()
    {
        FOLDER_NAME = config_data["parameters"]["FOLDER_NAME"].as<std::string>();
        BAG_NAME = config_data["parameters"]["BAG_NAME"].as<std::string>();
    }

    void odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        odom_buffer_[msg->header.stamp] = msg;
        if (odom_buffer_.size() > 200) {
            odom_buffer_.erase(odom_buffer_.begin());
        }
    }

    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {

        RCLCPP_INFO_ONCE(this->get_logger(), "PointCloud2 fields:");
 

        // Convert to PointXYZ
        auto pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::fromROSMsg(*msg, *pcl_cloud);

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

        for (size_t i = 0; i < 10; ++i, ++iter_x, ++iter_y, ++iter_z) {
            RCLCPP_INFO(this->get_logger(), "[%ld] x=%.3f y=%.3f z=%.3f", i, *iter_x, *iter_y, *iter_z);
        }
        // const auto& pt = pcl_cloud->points[150];
        // RCLCPP_INFO(this->get_logger(), "Sample point: x=%.2f y=%.2f z=%.2f intensity=%.2f", pt.x, pt.y, pt.z, pt.intensity);

        auto closest_odom = find_closest_odom(msg->header.stamp);
        if (!closest_odom) {
            RCLCPP_WARN(this->get_logger(), "No odometry for frame %d", frame_counter_);
            return;
        }

        RCLCPP_INFO(this->get_logger(), "point_step: %d, row_step: %d, width: %d, height: %d",
        msg->point_step, msg->row_step, msg->width, msg->height);
 



        save_to_bin_file(pcl_cloud);
        save_odometry_to_txt(*closest_odom);
    }

    nav_msgs::msg::Odometry::SharedPtr find_closest_odom(const rclcpp::Time& pc_time)
    {
        nav_msgs::msg::Odometry::SharedPtr closest = nullptr;
        rclcpp::Duration min_diff = rclcpp::Duration::from_nanoseconds(1000000000);  // 1초
        for (const auto& [t, odom] : odom_buffer_) {
            auto diff = (t > pc_time) ? (t - pc_time) : (pc_time - t);
            if (diff < min_diff) {
                min_diff = diff;
                closest = odom;
            }
        }
        return closest;
    }

    void save_to_bin_file(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud)
    {
        std::stringstream ss;
        ss << output_dir_ << "/frame_" << frame_counter_ << ".bin";
        std::ofstream out(ss.str(), std::ios::binary);

        int valid_count = 0;
        for (const auto& pt : cloud->points) {
            if (pt.x == 0.0f && pt.y == 0.0f && pt.z == 0.0f)
                continue;  // 0점 제거

            out.write(reinterpret_cast<const char*>(&pt.x), sizeof(float));
            out.write(reinterpret_cast<const char*>(&pt.y), sizeof(float));
            out.write(reinterpret_cast<const char*>(&pt.z), sizeof(float));
            valid_count++;
        }

        out.close();
        RCLCPP_INFO(this->get_logger(), "Saved %d valid points to: %s", valid_count, ss.str().c_str());
    }

    void save_odometry_to_txt(const nav_msgs::msg::Odometry& odom)
    {
        std::stringstream ss;
        ss << output_dir_ << "/frame_" << frame_counter_++ << ".pose";
        std::ofstream out(ss.str());
        const auto& p = odom.pose.pose.position;
        const auto& q = odom.pose.pose.orientation;
        out << p.x << " " << p.y << " " << p.z << " ";
        out << q.x << " " << q.y << " " << q.z << " " << q.w << std::endl;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudToBinNode>());
    rclcpp::shutdown();
    return 0;
}
