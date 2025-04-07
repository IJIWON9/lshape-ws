#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <typeinfo>
#include <cmath>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <geometry_msgs/msg/quaternion_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <custom_msgs/msg/float64_multi_array_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/string.hpp>

#include "rclcpp/rclcpp.hpp"

#include "rule_based_detect/timer_utils.hpp"
#include "rule_based_detect/lshape_detect.hpp"


using std::cout;
using std::endl;
using namespace std::chrono_literals;

using PointXYZIRT = OusterPointXYZIRT;

class ContourClassifier : public rclcpp::Node
{
public:
  ContourClassifier()
      : Node("contour_classifier_node")
  {
    
    subbed_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("/contour_classifier/subbed_contour", 10);

    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(1));
    contour_sub = this->create_subscription<custom_msgs::msg::Contours>(
        "/lshape_detect/outputContours",
        qos_profile,
        std::bind(&ContourClassifier::contour_sub_callback, this, std::placeholders::_1));

  }
  ~ContourClassifier() {}


  rclcpp::Subscription<custom_msgs::msg::Contours>::SharedPtr contour_sub;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr subbed_pub;

  const std::string FRAME_ID_LIDAR = "os1_frame";

private:

  void contour_sub_callback(const custom_msgs::msg::Contours msg)
  {
    TimeChecker tc(false);
    tc.start("total");

    std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> contours;

    pcl::PointCloud<pcl::PointXYZI>::Ptr subbed(new pcl::PointCloud<pcl::PointXYZI>);
    
    for (auto contour : msg.contours){
      double i = 10.0;
      std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> contour_segments;
      for (auto seg : contour.contour_segment){
        pcl::PointCloud<pcl::PointXYZ>::Ptr contour_segment(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(seg, *contour_segment);
        for (auto pt : contour_segment->points){
          pcl::PointXYZI pp;
          pp.x = pt.x;
          pp.y = pt.y;
          pp.z = pt.z;
          pp.intensity = i;
          subbed->points.push_back(pp);
        }
        i += 50.0;
        contour_segments.push_back(contour_segment);

        contour_segment->clear();
      }
      contours.push_back(contour_segments);
    }

    sensor_msgs::msg::PointCloud2 subbed_msg;
    pcl::toROSMsg(*subbed, subbed_msg);
    subbed_msg.header.frame_id = FRAME_ID_LIDAR;
    subbed_msg.header.stamp = this->get_clock()->now();
    subbed_pub->publish(subbed_msg);

    
    
    subbed->clear();
    
    tc.finish("total");
    tc.print();

    
    
  }
  
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ContourClassifier>());
  rclcpp::shutdown();
  return 0;
}

