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
    
    objCloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("/contour_classifier/subbed_contour", 10);

    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(1));
    contour_sub = this->create_subscription<custom_msgs::msg::Contours>(
        "/lshape_detect/outputContours",
        qos_profile,
        std::bind(&ContourClassifier::contour_sub_callback, this, std::placeholders::_1));

  }
  ~ContourClassifier() {}


  rclcpp::Subscription<custom_msgs::msg::Contours>::SharedPtr contour_sub;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr objCloud_pub;

  

private:

  void contour_sub_callback(const custom_msgs::msg::Contours msg)
  {

    TimeChecker tc(false);
    tc.start("total");
    
    cout << "subbed!!" << endl;
    
    tc.finish("total");
    // tc.print();

    
    
  }
  
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ContourClassifier>());
  rclcpp::shutdown();
  return 0;
}

