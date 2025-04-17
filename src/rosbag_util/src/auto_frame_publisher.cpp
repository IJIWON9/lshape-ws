#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <nlohmann/json.hpp>
#include "ament_index_cpp/get_package_share_directory.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <regex>

namespace fs = std::filesystem;
using json = nlohmann::json;

class AllFramesPublisher : public rclcpp::Node
{
public:
  AllFramesPublisher() : Node("all_frames_publisher")
  {
    pub_pc_ = create_publisher<sensor_msgs::msg::PointCloud2>("/os1/lidar", 10);
    pub_odom_ = create_publisher<nav_msgs::msg::Odometry>("/localization/ego_pose", 10);
    pub_marker_ = create_publisher<visualization_msgs::msg::MarkerArray>("/label_gt_markers", 10);

    root_path_ = pkg_share_dir_ + "/../../../../bag2bin_data";
    walk_and_collect_frames();

    if (frames_.empty()) {
      RCLCPP_ERROR(get_logger(), "No labeled frames found.");
      rclcpp::shutdown();
    }

    current_index_ = 0;
    timer_ = this->create_wall_timer(std::chrono::milliseconds(500), std::bind(&AllFramesPublisher::publish_next, this));
  }

private:
  std::string pkg_share_dir_ = ament_index_cpp::get_package_share_directory("rosbag_util");
  std::string root_path_;

  struct FrameInfo {
    std::string folder;
    int frame_id;
    std::string bin_path, pose_path, json_path;
  };

  std::vector<FrameInfo> frames_;
  size_t current_index_ = 0;
  std::string current_folder_logged_ = "";

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_pc_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_marker_;
  rclcpp::TimerBase::SharedPtr timer_;

  void walk_and_collect_frames()
  {
    for (const auto& dir : fs::directory_iterator(root_path_)) {
      if (!fs::is_directory(dir)) continue;
      std::string folder_name = dir.path().filename();

      for (const auto& file : fs::directory_iterator(dir.path())) {
        std::string name = file.path().filename();
        std::smatch match;
        if (std::regex_match(name, match, std::regex("frame_(\\d+)\\.json"))) {
          int id = std::stoi(match[1]);
          FrameInfo info;
          info.folder = folder_name;
          info.frame_id = id;
          info.json_path = file.path();
          info.bin_path = dir.path().string() + "/frame_" + std::to_string(id) + ".bin";
          info.pose_path = dir.path().string() + "/frame_" + std::to_string(id) + ".pose";

          if (fs::exists(info.bin_path) && fs::exists(info.pose_path)) {
            frames_.push_back(info);
          }
        }
      }
    }

    std::sort(frames_.begin(), frames_.end(), [](const FrameInfo &a, const FrameInfo &b) {
      return a.folder == b.folder ? a.frame_id < b.frame_id : a.folder < b.folder;
    });
  }

  void publish_next()
  {
    if (current_index_ >= frames_.size()) {
      RCLCPP_INFO(get_logger(), "Finished publishing all labeled frames.");
      return;
    }

    const auto& frame = frames_[current_index_++];

    if (frame.folder != current_folder_logged_) {
      current_folder_logged_ = frame.folder;
      RCLCPP_INFO(get_logger(), "[Folder] %s", frame.folder.c_str());
    }

    RCLCPP_INFO(get_logger(), "[Frame] Publishing frame_%d", frame.frame_id);

    pub_pc_->publish(*load_bin_to_msg(frame.bin_path));
    pub_odom_->publish(*load_pose_to_msg(frame.pose_path));
    pub_marker_->publish(*load_marker_from_json(frame.json_path));
  }

  sensor_msgs::msg::PointCloud2::SharedPtr load_bin_to_msg(const std::string& path)
  {
    std::ifstream input(path, std::ios::binary);
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
    cloud->width = num_points; cloud->height = 1; cloud->is_dense = true;

    auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
    pcl::toROSMsg(*cloud, *msg);
    msg->header.frame_id = "os1_frame";
    msg->header.stamp = now();
    return msg;
  }

  nav_msgs::msg::Odometry::SharedPtr load_pose_to_msg(const std::string& path)
  {
    std::ifstream in(path);
    double x, y, z, qx, qy, qz, qw;
    in >> x >> y >> z >> qx >> qy >> qz >> qw;
    auto msg = std::make_shared<nav_msgs::msg::Odometry>();
    msg->header.frame_id = "world_frame";
    msg->header.stamp = now();
    msg->pose.pose.position.x = x;
    msg->pose.pose.position.y = y;
    msg->pose.pose.position.z = z;
    msg->pose.pose.orientation.x = qx;
    msg->pose.pose.orientation.y = qy;
    msg->pose.pose.orientation.z = qz;
    msg->pose.pose.orientation.w = qw;
    return msg;
  }

  visualization_msgs::msg::MarkerArray::SharedPtr load_marker_from_json(const std::string& path)
  {
    std::ifstream in(path);
    json j; in >> j;

    auto arr = std::make_shared<visualization_msgs::msg::MarkerArray>();
    int id = 0;
    for (const auto& box : j["boxes"]) {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = "os1_frame";
      m.header.stamp = now();
      m.ns = "gt";
      m.id = id++;
      m.type = m.CUBE;
      m.action = m.ADD;
      m.pose.position.x = box["position"]["x"];
      m.pose.position.y = box["position"]["y"];
      m.pose.position.z = box["position"].value("z", 0.7);
      double yaw = box["orientation"]["yaw"];
      m.pose.orientation.z = sin(yaw / 2.0);
      m.pose.orientation.w = cos(yaw / 2.0);
      m.scale.x = box["size"].value("length", 4.5);
      m.scale.y = box["size"].value("width", 1.8);
      m.scale.z = box["size"].value("height", 1.6);
      m.color.a = 0.5; m.color.r = 1.0; m.color.g = 0.5; m.color.b = 0.0;
      m.lifetime = rclcpp::Duration::from_seconds(0.5);
      arr->markers.push_back(m);
    }
    return arr;
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AllFramesPublisher>());
  rclcpp::shutdown();
  return 0;
}