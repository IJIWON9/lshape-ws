#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include <unordered_map>
#include <chrono>

#include <pcl_conversions/pcl_conversions.h>
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/io/pcd_io.h"
#include "std_msgs/msg/string.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

class PointCloudToBinNode : public rclcpp::Node
{
public:
    PointCloudToBinNode()
    : Node("pointcloud_to_bin_node")
    {
        pc_subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/os1/lidar", 10,
            std::bind(&PointCloudToBinNode::pointcloud_callback, this, std::placeholders::_1));

        odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/localization/ego_pose", 10,
            std::bind(&PointCloudToBinNode::odometry_callback, this, std::placeholders::_1));

        bag_name_ = "multiego_bag2";
        output_dir_ = "/tmp/" + bag_name_;
        if (!fs::exists(output_dir_)) {
            fs::create_directory(output_dir_);
        }
    }

private:
    // 메시지 버퍼
    std::map<rclcpp::Time, nav_msgs::msg::Odometry::SharedPtr> odom_buffer_;


    void odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        rclcpp::Time stamp = msg->header.stamp;
        odom_buffer_[stamp] = msg;

        // 오래된 데이터는 버림 (예: 100개 이상 유지 금지)
        if (odom_buffer_.size() > 200) {
            odom_buffer_.erase(odom_buffer_.begin());
        }
    }

    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
        pcl::fromROSMsg(*msg, pcl_cloud);

        // 가장 가까운 timestamp의 Odometry 찾기
        rclcpp::Time pc_stamp = msg->header.stamp;
        auto closest_odom = find_closest_odom(pc_stamp);
        if (!closest_odom) {
            RCLCPP_WARN(this->get_logger(), "No matching odometry found for frame %d", frame_counter_);
            return;
        }

        // 저장
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

    void save_to_bin_file(const pcl::PointCloud<pcl::PointXYZ>& pcl_cloud)
    {
        std::stringstream ss;
        ss << output_dir_ << "/frame_" << frame_counter_ << ".bin";

        std::ofstream outfile(ss.str(), std::ios::binary);
        for (const auto& point : pcl_cloud.points) {
            outfile.write(reinterpret_cast<const char*>(&point.x), sizeof(float));
            outfile.write(reinterpret_cast<const char*>(&point.y), sizeof(float));
            outfile.write(reinterpret_cast<const char*>(&point.z), sizeof(float));
        }
        outfile.close();
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
        out.close();
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_subscription_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;

    std::string bag_name_;
    std::string output_dir_;
    int frame_counter_ = 0;
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudToBinNode>());
    rclcpp::shutdown();
    return 0;
}
