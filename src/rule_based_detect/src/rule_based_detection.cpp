#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <typeinfo>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <geometry_msgs/msg/quaternion_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <custom_msgs/msg/float64_multi_array_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cmath>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/pca.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/CXX11/Tensor>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "rule_based_detect/timer_utils.hpp"
#include "rule_based_detect/readcsv_utils.hpp"
#include <SimpleDBSCAN/dbscan.h>



using std::cout;
using std::endl;
using namespace std::chrono_literals;

// struct OusterPointXYZIRT           //os2
// {
//   float x;
//   float y;
//   float z;
//   float intensity;
//   float t;
//   float ring;
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW
// } EIGEN_ALIGN16;
// POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
//                                   (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, t, t)(float, ring, ring))

struct OusterPointXYZIRT        //os1
{
  PCL_ADD_POINT4D;
  float intensity;
  uint32_t t;
  uint16_t reflectivity;
  uint16_t ambient;
  uint32_t range;
  uint8_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(uint32_t, t, t)(uint16_t, reflectivity, reflectivity)(uint16_t, ambient, ambient)(uint32_t, range, range)(uint8_t, ring, ring))

using PointXYZIRT = OusterPointXYZIRT;

class RBdetect : public rclcpp::Node
{
public:
  RBdetect()
      : Node("rb_detect_node"), count_(0)
  {
    // for log //
    // mapdata_l_x = readColumnData(MAP_LOGGED_PATH, csv_index_x);
    // mapdata_l_y = readColumnData(MAP_LOGGED_PATH, csv_index_y);
    // mapdata_l_z = readColumnData(MAP_LOGGED_PATH, csv_index_z);
    // for log //
    
    mapdata_l_x = readColumnData(MAP_L_FILE_PATH, csv_index_x);
    mapdata_l_y = readColumnData(MAP_L_FILE_PATH, csv_index_y);
    mapdata_l_z = readColumnData(MAP_L_FILE_PATH, csv_index_z);
    mapdata_r_x = readColumnData(MAP_R_FILE_PATH, csv_index_x);
    mapdata_r_y = readColumnData(MAP_R_FILE_PATH, csv_index_y);
    mapdata_r_z = readColumnData(MAP_R_FILE_PATH, csv_index_z);

    mapdata_l_x_1m = readColumnData(MAP_L_FILE_PATH_1m, csv_index_x);
    mapdata_l_y_1m = readColumnData(MAP_L_FILE_PATH_1m, csv_index_y);
    mapdata_l_z_1m = readColumnData(MAP_L_FILE_PATH_1m, csv_index_z);
    mapdata_r_x_1m = readColumnData(MAP_R_FILE_PATH_1m, csv_index_x);
    mapdata_r_y_1m = readColumnData(MAP_R_FILE_PATH_1m, csv_index_y);
    mapdata_r_z_1m = readColumnData(MAP_R_FILE_PATH_1m, csv_index_z);

    mapdata_al_x = readColumnData(MAP_AL_FILE_PATH, csv_index_x);
    mapdata_al_y = readColumnData(MAP_AL_FILE_PATH, csv_index_y);
    mapdata_al_z = readColumnData(MAP_AL_FILE_PATH, csv_index_z);
    mapdata_ar_x = readColumnData(MAP_AR_FILE_PATH, csv_index_x);
    mapdata_ar_y = readColumnData(MAP_AR_FILE_PATH, csv_index_y);
    mapdata_ar_z = readColumnData(MAP_AR_FILE_PATH, csv_index_z);

    mapdata_al_x_1m = readColumnData(MAP_AL_FILE_PATH_1m, csv_index_x);
    mapdata_al_y_1m = readColumnData(MAP_AL_FILE_PATH_1m, csv_index_y);
    mapdata_al_z_1m = readColumnData(MAP_AL_FILE_PATH_1m, csv_index_z);
    mapdata_ar_x_1m = readColumnData(MAP_AR_FILE_PATH_1m, csv_index_x);
    mapdata_ar_y_1m = readColumnData(MAP_AR_FILE_PATH_1m, csv_index_y);
    mapdata_ar_z_1m = readColumnData(MAP_AR_FILE_PATH_1m, csv_index_z);

    lane_1_data_x = readColumnData(LANE_1_FILE_PATH, csv_index_x);
    lane_1_data_y = readColumnData(LANE_1_FILE_PATH, csv_index_y);
    lane_1_data_z = readColumnData(LANE_1_FILE_PATH, csv_index_z);

    lane_2_data_x = readColumnData(LANE_2_FILE_PATH, csv_index_x);
    lane_2_data_y = readColumnData(LANE_2_FILE_PATH, csv_index_y);
    lane_2_data_z = readColumnData(LANE_2_FILE_PATH, csv_index_z);

    lane_3_data_x = readColumnData(LANE_3_FILE_PATH, csv_index_x);
    lane_3_data_y = readColumnData(LANE_3_FILE_PATH, csv_index_y);
    lane_3_data_z = readColumnData(LANE_3_FILE_PATH, csv_index_z);

    // subtractOffset(mapdata_r_x, offset_x);
    // for (int i = 0; i < mapdata_r_x.size(); i++)
    // {
    //   cout << mapdata_r_x[i] << endl;
    // }
    resultmarker_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("/rulebased/result_marker", 10);
    objCloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("/rulebased/objCloud", 10);
    rb_detections_pub = this->create_publisher<custom_msgs::msg::Float64MultiArrayStamped>("/rulebased/detections", 10);
    localmap_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("/rulebased/localmap", 10);
    boundaryCloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("/rulebased/boundaryCloud", 10);

    line_pub = this->create_publisher<visualization_msgs::msg::Marker>("/rulebased/line_list_marker", 10);


    // localmap_pub_r = this->create_publisher<sensor_msgs::msg::PointCloud2>("localmap_r", 10);
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(1));
    pcd_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/os1/lidar",
        qos_profile,
        std::bind(&RBdetect::pcd_sub_callback, this, std::placeholders::_1));

    odometry_sub = this->create_subscription<nav_msgs::msg::Odometry>(
        "/localization/ego_pose",
        qos_profile,
        std::bind(&RBdetect::odometry_sub_callback, this, std::placeholders::_1));


    lidarPose.setIdentity();
    lidarPose.translation() = Eigen::Vector3d(1.2, 0.0, 1.0);
    

    // rawCloud.reset(new pcl::PointCloud<PointXYZIRT>());
    rawCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    nongroundCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
    boundaryCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());

    global_linkCloud_1.reset(new pcl::PointCloud<pcl::PointXYZ>());
    global_linkCloud_2.reset(new pcl::PointCloud<pcl::PointXYZ>());
    global_linkCloud_3.reset(new pcl::PointCloud<pcl::PointXYZ>());
    rotated_linkCloud_1.reset(new pcl::PointCloud<pcl::PointXYZ>());
    rotated_linkCloud_2.reset(new pcl::PointCloud<pcl::PointXYZ>());
    rotated_linkCloud_3.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_linkCloud_1.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_linkCloud_2.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_linkCloud_3.reset(new pcl::PointCloud<pcl::PointXYZ>());

    // global_mapCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    // rot_global_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    // local_mapCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_border.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_border_1m.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_border_a.reset(new pcl::PointCloud<pcl::PointXYZ>());

    global_mapCloud_l.reset(new pcl::PointCloud<pcl::PointXYZ>());
    rot_global_cloud_l.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_mapCloud_l.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_border_l.reset(new pcl::PointCloud<pcl::PointXYZ>());
    global_mapCloud_r.reset(new pcl::PointCloud<pcl::PointXYZ>());
    rot_global_cloud_r.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_mapCloud_r.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_border_r.reset(new pcl::PointCloud<pcl::PointXYZ>());

    global_mapCloud_l_1m.reset(new pcl::PointCloud<pcl::PointXYZ>());
    rot_global_cloud_l_1m.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_mapCloud_l_1m.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_border_l_1m.reset(new pcl::PointCloud<pcl::PointXYZ>());
    global_mapCloud_r_1m.reset(new pcl::PointCloud<pcl::PointXYZ>());
    rot_global_cloud_r_1m.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_mapCloud_r_1m.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_border_r_1m.reset(new pcl::PointCloud<pcl::PointXYZ>());

    global_mapCloud_al.reset(new pcl::PointCloud<pcl::PointXYZ>());
    rot_global_cloud_al.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_mapCloud_al.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_border_al.reset(new pcl::PointCloud<pcl::PointXYZ>());
    global_mapCloud_ar.reset(new pcl::PointCloud<pcl::PointXYZ>());
    rot_global_cloud_ar.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_mapCloud_ar.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_border_ar.reset(new pcl::PointCloud<pcl::PointXYZ>());

    global_mapCloud_al_1m.reset(new pcl::PointCloud<pcl::PointXYZ>());
    rot_global_cloud_al_1m.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_mapCloud_al_1m.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_border_al_1m.reset(new pcl::PointCloud<pcl::PointXYZ>());
    global_mapCloud_ar_1m.reset(new pcl::PointCloud<pcl::PointXYZ>());
    rot_global_cloud_ar_1m.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_mapCloud_ar_1m.reset(new pcl::PointCloud<pcl::PointXYZ>());
    local_border_ar_1m.reset(new pcl::PointCloud<pcl::PointXYZ>());

    clusterCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    roi_vis.reset(new pcl::PointCloud<pcl::PointXYZ>());
    objCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    
    //// mapdata loading ///
    for (int i = 0; i < mapdata_l_x.size(); i++)
    {
      pcl::PointXYZ pp;
      pp.x = mapdata_l_x[i];
      pp.y = mapdata_l_y[i];
      pp.z = mapdata_l_z[i];
      global_mapCloud_l->points.push_back(pp);
    }
    
    for (int i = 0; i < mapdata_r_x.size(); i++)
    {
      pcl::PointXYZ pp;
      pp.x = mapdata_r_x[i];
      pp.y = mapdata_r_y[i];
      pp.z = mapdata_r_z[i];
      global_mapCloud_r->points.push_back(pp);
    }

    for (int i = 0; i < mapdata_l_x_1m.size(); i++)
    {
      pcl::PointXYZ pp;
      pp.x = mapdata_l_x_1m[i];
      pp.y = mapdata_l_y_1m[i];
      pp.z = mapdata_l_z_1m[i];
      global_mapCloud_l_1m->points.push_back(pp);
    }
    for (int i = 0; i < mapdata_r_x_1m.size(); i++)
    {
      pcl::PointXYZ pp;
      pp.x = mapdata_r_x_1m[i];
      pp.y = mapdata_r_y_1m[i];
      pp.z = mapdata_r_z_1m[i];
      global_mapCloud_r_1m->points.push_back(pp);
    }

    for (int i = 0; i < mapdata_al_x.size(); i++)
    {
      pcl::PointXYZ pp;
      pp.x = mapdata_al_x[i];
      pp.y = mapdata_al_y[i];
      pp.z = mapdata_al_z[i];
      global_mapCloud_al->points.push_back(pp);
    }
    for (int i = 0; i < mapdata_ar_x.size(); i++)
    {
      pcl::PointXYZ pp;
      pp.x = mapdata_ar_x[i];
      pp.y = mapdata_ar_y[i];
      pp.z = mapdata_ar_z[i];
      global_mapCloud_ar->points.push_back(pp);
    }

    for (int i = 0; i < mapdata_al_x_1m.size(); i++)
    {
      pcl::PointXYZ pp;
      pp.x = mapdata_al_x_1m[i];
      pp.y = mapdata_al_y_1m[i];
      pp.z = mapdata_al_z_1m[i];
      global_mapCloud_al_1m->points.push_back(pp);
    }
    for (int i = 0; i < mapdata_r_x_1m.size(); i++)
    {
      pcl::PointXYZ pp;
      pp.x = mapdata_ar_x_1m[i];
      pp.y = mapdata_ar_y_1m[i];
      pp.z = mapdata_ar_z_1m[i];
      global_mapCloud_ar_1m->points.push_back(pp);
    }
    //// mapdata loading ///

    //// linkdata loading ///

    for (int i = 0; i < lane_1_data_x.size(); i++)
    {
      pcl::PointXYZ pp;
      pp.x = lane_1_data_x[i];
      pp.y = lane_1_data_y[i];
      pp.z = lane_1_data_z[i];
      global_linkCloud_1->points.push_back(pp);
    }
    for (int i = 0; i < lane_2_data_x.size(); i++)
    {
      pcl::PointXYZ pp;
      pp.x = lane_2_data_x[i];
      pp.y = lane_2_data_y[i];
      pp.z = lane_2_data_z[i];
      global_linkCloud_2->points.push_back(pp);
    }
    for (int i = 0; i < lane_3_data_x.size(); i++)
    {
      pcl::PointXYZ pp;
      pp.x = lane_3_data_x[i];
      pp.y = lane_3_data_y[i];
      pp.z = lane_3_data_z[i];
      global_linkCloud_3->points.push_back(pp);
    }
    
    //// linkdata loading ///


  }
  ~RBdetect() {}

  struct vec3f
  {
    float data[3];
    float operator[](int idx) const { return data[idx]; }
  };

  struct vec4f
  {
    float data[4];
    float operator[](int idx) const { return data[idx]; }
  };

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_sub;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub;

  // pcl::PointCloud<PointXYZIRT>::Ptr rawCloud;
  pcl::PointCloud<pcl::PointXYZ>::Ptr rawCloud;          //for sim
  pcl::PointCloud<pcl::PointXYZI>::Ptr nongroundCloud;
  pcl::PointCloud<pcl::PointXYZ>::Ptr boundaryCloud;

  pcl::PointCloud<pcl::PointXYZ>::Ptr global_linkCloud_1;
  pcl::PointCloud<pcl::PointXYZ>::Ptr global_linkCloud_2;
  pcl::PointCloud<pcl::PointXYZ>::Ptr global_linkCloud_3;

  pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_linkCloud_1;
  pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_linkCloud_2;
  pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_linkCloud_3;

  pcl::PointCloud<pcl::PointXYZ>::Ptr local_linkCloud_1;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_linkCloud_2;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_linkCloud_3;

  // pcl::PointCloud<pcl::PointXYZ>::Ptr global_mapCloud;
  // pcl::PointCloud<pcl::PointXYZ>::Ptr rot_global_cloud;
  // pcl::PointCloud<pcl::PointXYZ>::Ptr local_mapCloud;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_border;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_border_1m;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_border_a;


  pcl::PointCloud<pcl::PointXYZ>::Ptr global_mapCloud_l;
  pcl::PointCloud<pcl::PointXYZ>::Ptr rot_global_cloud_l;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_mapCloud_l;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_border_l;
  pcl::PointCloud<pcl::PointXYZ>::Ptr global_mapCloud_r;
  pcl::PointCloud<pcl::PointXYZ>::Ptr rot_global_cloud_r;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_mapCloud_r;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_border_r;

  pcl::PointCloud<pcl::PointXYZ>::Ptr global_mapCloud_l_1m;
  pcl::PointCloud<pcl::PointXYZ>::Ptr rot_global_cloud_l_1m;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_mapCloud_l_1m;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_border_l_1m;
  pcl::PointCloud<pcl::PointXYZ>::Ptr global_mapCloud_r_1m;
  pcl::PointCloud<pcl::PointXYZ>::Ptr rot_global_cloud_r_1m;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_mapCloud_r_1m;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_border_r_1m;

  pcl::PointCloud<pcl::PointXYZ>::Ptr global_mapCloud_al;
  pcl::PointCloud<pcl::PointXYZ>::Ptr rot_global_cloud_al;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_mapCloud_al;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_border_al;
  pcl::PointCloud<pcl::PointXYZ>::Ptr global_mapCloud_ar;
  pcl::PointCloud<pcl::PointXYZ>::Ptr rot_global_cloud_ar;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_mapCloud_ar;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_border_ar;

  pcl::PointCloud<pcl::PointXYZ>::Ptr global_mapCloud_al_1m;
  pcl::PointCloud<pcl::PointXYZ>::Ptr rot_global_cloud_al_1m;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_mapCloud_al_1m;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_border_al_1m;
  pcl::PointCloud<pcl::PointXYZ>::Ptr global_mapCloud_ar_1m;
  pcl::PointCloud<pcl::PointXYZ>::Ptr rot_global_cloud_ar_1m;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_mapCloud_ar_1m;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_border_ar_1m;

  pcl::PointCloud<pcl::PointXYZ>::Ptr clusterCloud;
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr roi_vis;
  pcl::PointCloud<pcl::PointXYZ>::Ptr objCloud;


  double utm_offset_x = 442000.0;
  double utm_offset_y = 3942660.0;
  double utm_offset_z = 0.0;             // offset : 23m

  double yaw_offset = -0.0;  // deg

  std::vector<double> translation_lidar2ego{-1.2, 0.0, -1.0};

  std::vector<double> mapdata_l_x, mapdata_l_y, mapdata_l_z;
  std::vector<double> mapdata_r_x, mapdata_r_y, mapdata_r_z;
  std::vector<double> mapdata_l_x_1m, mapdata_l_y_1m, mapdata_l_z_1m;
  std::vector<double> mapdata_r_x_1m, mapdata_r_y_1m, mapdata_r_z_1m;
  std::vector<double> mapdata_al_x, mapdata_al_y, mapdata_al_z;
  std::vector<double> mapdata_ar_x, mapdata_ar_y, mapdata_ar_z;
  std::vector<double> mapdata_al_x_1m, mapdata_al_y_1m, mapdata_al_z_1m;
  std::vector<double> mapdata_ar_x_1m, mapdata_ar_y_1m, mapdata_ar_z_1m;

  std::vector<double> lane_1_data_x, lane_1_data_y, lane_1_data_z;
  std::vector<double> lane_2_data_x, lane_2_data_y, lane_2_data_z;
  std::vector<double> lane_3_data_x, lane_3_data_y, lane_3_data_z;

  double ego_x, ego_y, ego_z;
  double ego_quat_w, ego_quat_x, ego_quat_y, ego_quat_z;
  double ego_roll, ego_pitch, ego_yaw;
  

  std::string pkg_share_dir = ament_index_cpp::get_package_share_directory("rule_based_detect");

  std::string LOCAL_MAP_LOGGED_FILE_PATH = "/../../../../src/rule_based_detect/mapdata/log.csv";
  const std::string MAP_LOGGED_PATH = pkg_share_dir + LOCAL_MAP_LOGGED_FILE_PATH;
  
  std::string LOCAL_MAP_R_FILE_PATH = "/../../../../src/rule_based_detect/mapdata/map_v2_r_5m.csv";
  std::string LOCAL_MAP_L_FILE_PATH = "/../../../../src/rule_based_detect/mapdata/map_v2_l_5m.csv";
  std::string LOCAL_MAP_R_FILE_PATH_1m = "/../../../../src/rule_based_detect/mapdata/map_v2_r_1m.csv";
  std::string LOCAL_MAP_L_FILE_PATH_1m = "/../../../../src/rule_based_detect/mapdata/map_v2_l_1m.csv";
  const std::string MAP_R_FILE_PATH = pkg_share_dir + LOCAL_MAP_R_FILE_PATH;
  const std::string MAP_L_FILE_PATH = pkg_share_dir + LOCAL_MAP_L_FILE_PATH;
  const std::string MAP_R_FILE_PATH_1m = pkg_share_dir + LOCAL_MAP_R_FILE_PATH_1m;
  const std::string MAP_L_FILE_PATH_1m = pkg_share_dir + LOCAL_MAP_L_FILE_PATH_1m;

  std::string LOCAL_MAP_AR_FILE_PATH = "/../../../../src/rule_based_detect/mapdata/map_v2_ar_5m.csv";
  std::string LOCAL_MAP_AL_FILE_PATH = "/../../../../src/rule_based_detect/mapdata/map_v2_al_5m.csv";
  std::string LOCAL_MAP_AR_FILE_PATH_1m = "/../../../../src/rule_based_detect/mapdata/map_v2_ar_1m.csv";
  std::string LOCAL_MAP_AL_FILE_PATH_1m = "/../../../../src/rule_based_detect/mapdata/map_v2_al_1m.csv";
  const std::string MAP_AR_FILE_PATH = pkg_share_dir + LOCAL_MAP_AR_FILE_PATH;
  const std::string MAP_AL_FILE_PATH = pkg_share_dir + LOCAL_MAP_AL_FILE_PATH;
  const std::string MAP_AR_FILE_PATH_1m = pkg_share_dir + LOCAL_MAP_AR_FILE_PATH_1m;
  const std::string MAP_AL_FILE_PATH_1m = pkg_share_dir + LOCAL_MAP_AL_FILE_PATH_1m;

  std::string LOCAL_LANE_1_FILE_PATH = "/../../../../src/rule_based_detect/mapdata/lane_1.csv";
  std::string LOCAL_LANE_2_FILE_PATH = "/../../../../src/rule_based_detect/mapdata/lane_2.csv";
  std::string LOCAL_LANE_3_FILE_PATH = "/../../../../src/rule_based_detect/mapdata/lane_3.csv";
  const std::string LANE_1_FILE_PATH = pkg_share_dir + LOCAL_LANE_1_FILE_PATH;
  const std::string LANE_2_FILE_PATH = pkg_share_dir + LOCAL_LANE_2_FILE_PATH;
  const std::string LANE_3_FILE_PATH = pkg_share_dir + LOCAL_LANE_3_FILE_PATH;

  const int csv_index_x = 0;
  const int csv_index_y = 1;
  const int csv_index_z = 2;

  double THRESHOLD_dM = 0.5;
  double MAX_RANGE = 150.0;
  double MIN_RANGE = 3.0;
  double RANGE_RESOLUTION = 0.3;
  int N_HORIZONTAL = 1024;
  int N_FILTER_CHANNEL = 40;
  double THRESHOLD_Z = 0.5;
  double THRESHOLD_HEIGHT_GAP = 0.15;
  double H_RES = M_PI * 2 / N_HORIZONTAL;
  double THRESHOLD_DISTANCE2BORDER = 1.7;
  double THRESHOLD_DISTANCE2LINK = 1.2;

  int CONTOUR_N = 720;
  int CONTOUR_RES = M_PI * 2 / CONTOUR_N;
  double CONTOUR_Z_THRH = 0.3;

  Eigen::Isometry3d lidarPose, currPose;

   

  uint DBSCAN_PTS = 6;
  uint DBSCAN_PTS_DOUBLED = 10;
  const float DBSCAN_EPS = 1.3;
  const std::string frame_id_lidar = "os1_frame";

private:
  std::vector<double> get_unit_vector(double x, double y)
  {
    double magnitude = std::sqrt(x*x + y*y);
    std::vector<double> vec = {x / magnitude, y / magnitude};
    return vec;
  }

  double larger_angle_singlewise(double theta1, double theta2)    //radian  // object angle over 180 error
  {
    double diff = fmod(theta2 - theta1, 2 * M_PI);

    if (diff < 0) {
        diff += 2 * M_PI;
    }

    return (diff > M_PI) ? theta1 : theta2;
  }
  
  void translatePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr global_pcd, pcl::PointCloud<pcl::PointXYZ>::Ptr local_pcd, Eigen::Vector3f ego_vec) 
  {
    for (const auto &point : global_pcd->points)     
    {
        pcl::PointXYZ pp;
        pp.x = point.x - ego_vec.x();
        pp.y = point.y - ego_vec.y();
        pp.z = point.z - ego_vec.z();
        local_pcd->points.push_back(pp);
    }
  }
  void rotatePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr global_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr rot_global_cloud, Eigen::Matrix3f rotation_matrix) 
  {
    
    for (auto& point : global_cloud->points) 
    {
        Eigen::Vector3f point_vector(point.x, point.y, point.z);
        Eigen::Vector3f rotated_point = rotation_matrix * point_vector;
        pcl::PointXYZ pp;
        
        pp.x = rotated_point.x();
        pp.y = rotated_point.y();
        pp.z = rotated_point.z();
        rot_global_cloud->points.push_back(pp);

    }
    
  }
  double get_object_local_yaw(pcl::PointCloud<pcl::PointXYZ>::Ptr local_link, std::vector<double> object)
  {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(local_link);
    pcl::PointXYZ searchpoint(0.0, 0.0, 0.0);
    std::vector<int> pointIdxNKNSearch(1);
    std::vector<float> pointNKNSquaredDistance(1);
    kdtree.nearestKSearch(searchpoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance);

    auto &nearest_pt = local_link->points.at(pointIdxNKNSearch[0]);
    auto &nearest_pt_above = local_link->points.at((local_link->points.size() + pointIdxNKNSearch[0] - 75) %local_link->points.size());
    return std::atan2(nearest_pt_above.y - nearest_pt.y, nearest_pt_above.x - nearest_pt.x);
  }
  void slice_map_l(pcl::PointCloud<pcl::PointXYZ>::Ptr local_pcd, pcl::PointCloud<pcl::PointXYZ>::Ptr local_border, int interval, bool is_alpha)
  {
    int front_idx = 100 / interval;
    int rear_idx = 50 / interval;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(local_pcd);
    pcl::PointXYZ searchpoint(0.0, 0.0, 0.0);
    std::vector<int> pointIdxNKNSearch(1);
    std::vector<float> pointNKNSquaredDistance(1);
    kdtree.nearestKSearch(searchpoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
    if (!is_alpha)
    {
      for (int i = pointIdxNKNSearch[0] - front_idx; i < pointIdxNKNSearch[0] + rear_idx; i++)
    {
      if (i < 0)
        local_border->points.push_back(local_pcd->points[(i+local_pcd->points.size())%local_pcd->points.size()]);
      else
        local_border->points.push_back(local_pcd->points[i%local_pcd->points.size()]);
    }

    }

    else{
      for (int i = pointIdxNKNSearch[0] - front_idx; i < pointIdxNKNSearch[0] + rear_idx; i++)
    {
      if (i < 0 || i > local_pcd->points.size() - 1)
        continue;
      else
        local_border->points.push_back(local_pcd->points[i%local_pcd->points.size()]);
    }
    }
    
  }
  void slice_map_r(pcl::PointCloud<pcl::PointXYZ>::Ptr local_pcd, pcl::PointCloud<pcl::PointXYZ>::Ptr local_border, int interval, bool is_alpha)
  {
    int front_idx = 100 / interval;
    int rear_idx = 50 / interval;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(local_pcd);
    pcl::PointXYZ searchpoint(0.0, 0.0, 0.0);
    std::vector<int> pointIdxNKNSearch(1);
    std::vector<float> pointNKNSquaredDistance(1);
    kdtree.nearestKSearch(searchpoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
    if (!is_alpha)
    {
    for (int i = pointIdxNKNSearch[0] + rear_idx; i > pointIdxNKNSearch[0] - front_idx; i--)
    {
      if (i < 0)
        local_border->points.push_back(local_pcd->points[(i+local_pcd->points.size())%local_pcd->points.size()]);
      else
        local_border->points.push_back(local_pcd->points[i%local_pcd->points.size()]);
    }
    }
    else{
    for (int i = pointIdxNKNSearch[0] + rear_idx; i > pointIdxNKNSearch[0] - front_idx; i--)
    {
      if (i < 0 || i > local_pcd->points.size() - 1)
        continue;
      else
        local_border->points.push_back(local_pcd->points[i%local_pcd->points.size()]);
    }
    }
  }
  double get_nearest_border_distance(pcl::PointCloud<pcl::PointXYZ>::Ptr local_border, std::vector<double> object)
  {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(local_border);
    pcl::PointXYZ searchpoint(object[0], object[1], object[2]);
    std::vector<int> pointIdxNKNSearch(1);
    std::vector<float> pointNKNSquaredDistance(1);
    kdtree.nearestKSearch(searchpoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
    auto &nearest_pt_1 = local_border->points.at(pointIdxNKNSearch[0]);
    auto &nearest_pt_2 = local_border->points.at((pointIdxNKNSearch[0]+1)%local_border->points.size());

    double a = nearest_pt_2.y - nearest_pt_1.y;
    double b = nearest_pt_1.x - nearest_pt_2.x;
    double c = nearest_pt_2.x * nearest_pt_1.y - nearest_pt_1.x * nearest_pt_2.y;

    double nearest_border_distance = std::abs(a*object[0] + b*object[1] + c) / std::sqrt(a*a + b*b);

    return nearest_border_distance;
  }

  double get_nearest_border_distance_v2(pcl::PointCloud<pcl::PointXYZ>::Ptr local_border, std::vector<double> object, std::vector<double> ymin_pt, std::vector<double> ymax_pt)
  {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(local_border);
    pcl::PointXYZ searchpoint(object[0], object[1], object[2]);
    std::vector<int> pointIdxNKNSearch(1);
    std::vector<float> pointNKNSquaredDistance(1);
    kdtree.nearestKSearch(searchpoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance);
    auto &nearest_pt_1 = local_border->points.at(pointIdxNKNSearch[0]);
    auto &nearest_pt_2 = local_border->points.at((pointIdxNKNSearch[0]+1)%local_border->points.size());

    double a = nearest_pt_2.y - nearest_pt_1.y;
    double b = nearest_pt_1.x - nearest_pt_2.x;
    double c = nearest_pt_2.x * nearest_pt_1.y - nearest_pt_1.x * nearest_pt_2.y;

    double nearest_border_distance_ymin = std::abs(a*ymin_pt[0] + b*ymin_pt[1] + c) / std::sqrt(a*a + b*b);
    double nearest_border_distance_ymax = std::abs(a*ymax_pt[0] + b*ymax_pt[1] + c) / std::sqrt(a*a + b*b);

    double nearest_border_distance = (nearest_border_distance_ymin > nearest_border_distance_ymax) ? nearest_border_distance_ymin : nearest_border_distance_ymax;

    return nearest_border_distance;
  }


  bool is_inside_polygon(pcl::PointCloud<pcl::PointXYZ>::Ptr polygon, float x, float y)
  {
    bool is_inside = false;
    int j = polygon->points.size() - 1;

    for (int i = 0; i < polygon->points.size(); i++){
      float p1_x = polygon->points[i].x;
      float p1_y = polygon->points[i].y;
      float p2_x = polygon->points[j].x;
      float p2_y = polygon->points[j].y;
      if (((p1_y > y) != (p2_y > y)) && (x < (p2_x - p1_x) * (y - p1_y) / (p2_y - p1_y+0.001) + p1_x))
        is_inside = !is_inside;
      j = i;
    }

    return is_inside;   
  }

  void odometry_sub_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);

    Eigen::Isometry3d insPose;
    insPose.setIdentity();
    insPose.linear() = q.matrix();
    insPose.translation() = Eigen::Vector3d(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);// + Eigen::Vector3d(0.5, -0.3, 22.5);

    currPose.matrix() = insPose.matrix() * lidarPose.matrix();

    ego_x = currPose.translation()(0) + utm_offset_x;
    ego_y = currPose.translation()(1) + utm_offset_y;
    ego_z = currPose.translation()(2) + utm_offset_z;
    

    // ego_x = msg->pose.pose.position.x + utm_offset_x + translation_lidar2ego[0];
    // ego_y = msg->pose.pose.position.y + utm_offset_y + translation_lidar2ego[1];
    
    
    tf2::Quaternion ego_qaut(msg->pose.pose.orientation.x,msg->pose.pose.orientation.y,msg->pose.pose.orientation.z,msg->pose.pose.orientation.w);  
    tf2::Matrix3x3 ego_rotation(ego_qaut);
    // tf2::Matrix3x3 ego_rotation;
    // ego_rotation.setValue(eigen_matrix(0, 0), eigen_matrix(0, 1), eigen_matrix(0, 2),
    //                    eigen_matrix(1, 0), eigen_matrix(1, 1), eigen_matrix(1, 2),
    //                    eigen_matrix(2, 0), eigen_matrix(2, 1), eigen_matrix(2, 2));

    ego_rotation.getRPY(ego_roll, ego_pitch, ego_yaw);
  

  }
  double calculateLogCurvature(const std::vector<double>& p1, const std::vector<double>& p2, const std::vector<double>& p3) 
  {
 
    double A = 0.5 * std::abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]));
    double d12 = std::hypot(p2[0] - p1[0], p2[1] - p1[1]);
    double d23 = std::hypot(p3[0] - p2[0], p3[1] - p2[1]);
    double d31 = std::hypot(p3[0] - p1[0], p3[1] - p1[1]);

    double R = (d12 * d23 * d31) / (4.0 * A);
    // cout << "Radius : " << R << endl;

    double log_d12 = std::log(d12);
    double log_d23 = std::log(d23);
    double log_d31 = std::log(d31);

    double log_R = log_d12 + log_d23 + log_d31 - std::log(4.0 * A);

    double log_K = -log_R;

    return log_K;
  }
  void pcd_sub_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {

    TimeChecker tc(false);
    tc.start("total");
    tc.start("map_slice");

    

    

    Eigen::Matrix3f ego_rot_mat;
    ego_rot_mat = 
        Eigen::AngleAxisf(ego_yaw + yaw_offset*(M_PI / 180), Eigen::Vector3f::UnitZ()) *
        Eigen::AngleAxisf(ego_pitch, Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(ego_roll, Eigen::Vector3f::UnitX());

    Eigen::Matrix3f inv_ego_rot_mat= ego_rot_mat.inverse();

    Eigen::Vector3f ego_vector(ego_x, ego_y, ego_z);
    
    Eigen::Vector3f rotated_ego = inv_ego_rot_mat * ego_vector;
    


    rotatePointCloud(global_mapCloud_l, rot_global_cloud_l, inv_ego_rot_mat);
    translatePointCloud(rot_global_cloud_l, local_mapCloud_l, rotated_ego);
    slice_map_l(local_mapCloud_l, local_border_l, 5, false);

    rotatePointCloud(global_mapCloud_r, rot_global_cloud_r, inv_ego_rot_mat);
    translatePointCloud(rot_global_cloud_r, local_mapCloud_r, rotated_ego);
    slice_map_r(local_mapCloud_r, local_border_r, 5, false);

    rotatePointCloud(global_mapCloud_l_1m, rot_global_cloud_l_1m, inv_ego_rot_mat);
    translatePointCloud(rot_global_cloud_l_1m, local_mapCloud_l_1m, rotated_ego);
    slice_map_l(local_mapCloud_l_1m, local_border_l_1m, 1, false);

    rotatePointCloud(global_mapCloud_r_1m, rot_global_cloud_r_1m, inv_ego_rot_mat);
    translatePointCloud(rot_global_cloud_r_1m, local_mapCloud_r_1m, rotated_ego);
    slice_map_r(local_mapCloud_r_1m, local_border_r_1m, 1, false);

    rotatePointCloud(global_mapCloud_al, rot_global_cloud_al, inv_ego_rot_mat);
    translatePointCloud(rot_global_cloud_al, local_mapCloud_al, rotated_ego);  

    rotatePointCloud(global_mapCloud_ar, rot_global_cloud_ar, inv_ego_rot_mat);
    translatePointCloud(rot_global_cloud_ar, local_mapCloud_ar, rotated_ego);    

    rotatePointCloud(global_mapCloud_al_1m, rot_global_cloud_al_1m, inv_ego_rot_mat);
    translatePointCloud(rot_global_cloud_al_1m, local_mapCloud_al_1m, rotated_ego);    

    rotatePointCloud(global_mapCloud_ar_1m, rot_global_cloud_ar_1m, inv_ego_rot_mat);
    translatePointCloud(rot_global_cloud_ar_1m, local_mapCloud_ar_1m, rotated_ego); 

    rotatePointCloud(global_linkCloud_1, rotated_linkCloud_1, inv_ego_rot_mat);
    translatePointCloud(rotated_linkCloud_1, local_linkCloud_1, rotated_ego);

    rotatePointCloud(global_linkCloud_2, rotated_linkCloud_2, inv_ego_rot_mat);
    translatePointCloud(rotated_linkCloud_2, local_linkCloud_2, rotated_ego);

    rotatePointCloud(global_linkCloud_3, rotated_linkCloud_3, inv_ego_rot_mat);
    translatePointCloud(rotated_linkCloud_3, local_linkCloud_3, rotated_ego);             



    // std::vector<double> rotated_ego_vector{rotated_ego.x(), rotated_ego.y(), rotated_ego.z()};
    std::vector<double> ego_zero{0, 0, 0};
    double alpha_l_distance = get_nearest_border_distance(local_mapCloud_al, ego_zero);
    double alpha_r_distance = get_nearest_border_distance(local_mapCloud_ar, ego_zero);

    if (alpha_l_distance < 20.0 || alpha_r_distance < 20.0)
    {
      
      slice_map_l(local_mapCloud_al, local_border_al, 5, true);
      slice_map_r(local_mapCloud_ar, local_border_ar, 5, true);
      slice_map_l(local_mapCloud_al_1m, local_border_al_1m, 1, true);
      slice_map_r(local_mapCloud_ar_1m, local_border_ar_1m, 1, true);
    }

    // for RECTANGLE ////
    // double rec_half_w = 10.0;
    // double rec_forward = 25.0;
    // double rec_backward = 3.0;
    // double rec_weight = 3.0;
    // pcl::PointXYZ pp1;        
    // pp1.x = rec_forward * weight;
    // pp1.y = rec_half_w * weight;
    // pp1.z = 0.0;
    // local_border->points.push_back(pp1);
    // pp1.x = rec_forward * weight;
    // pp1.y = -rec_half_w * weight;
    // pp1.z = 0.0;
    // local_border->points.push_back(pp1);
    // pp1.x = -rec_backward * weight;
    // pp1.y = -rec_half_w * weight;
    // pp1.z = 0.0;
    // local_border->points.push_back(pp1);
    // pp1.x = -rec_backward * weight;
    // pp1.y = rec_half_w * weight;
    // pp1.z = 0.0;
    // local_border->points.push_back(pp1);
    // for RECTANGLE ////

    // // for SQUARE //
    // double square_size = 60;
    // pcl::PointXYZ pp1;        
    // pp1.x = square_size;
    // pp1.y = square_size;
    // pp1.z = 0.0;
    // local_border->points.push_back(pp1);
    // pp1.x = square_size;
    // pp1.y = -square_size;
    // pp1.z = 0.0;
    // local_border->points.push_back(pp1);
    // pp1.x = -square_size;
    // pp1.y = -square_size;
    // pp1.z = 0.0;
    // local_border->points.push_back(pp1);
    // pp1.x = -square_size;
    // pp1.y = square_size;
    // pp1.z = 0.0;
    // local_border->points.push_back(pp1);
    // // for SQUARE //


    // // for logged //
    // for (const auto &point : local_border_l->points){
    //   local_border->points.push_back(point);
    // }
    // // fof logged //

   
    for (const auto &point : local_border_l->points){
      local_border->points.push_back(point);
    }
    for (const auto &point : local_border_r->points)
    {
      local_border->points.push_back(point);
    }

    for (const auto &point : local_border_l_1m->points){
      local_border_1m->points.push_back(point);
    }
    for (const auto &point : local_border_r_1m->points)
    {
      local_border_1m->points.push_back(point);
    }

    for (const auto &point : local_border_al->points){
      local_border_a->points.push_back(point);
    }
    for (const auto &point : local_border_ar->points)
    {
      local_border_a->points.push_back(point);
    }
    for (const auto &point : local_border_al_1m->points){
      local_border_1m->points.push_back(point);
    }
    for (const auto &point : local_border_ar_1m->points)
    {
      local_border_1m->points.push_back(point);
    }




    /// visualize ROI ///// for occupancygrid // 

    for (const auto &point : local_border->points)
    {
      roi_vis->points.push_back(point);
    }
    for (const auto &point : local_border_a->points)
    {
      roi_vis->points.push_back(point);
    }
    
    for (const auto &point : roi_vis->points)
    {
      boundaryCloud->points.push_back(point);
    }
    sensor_msgs::msg::PointCloud2 boundary_cloud_msg;
    pcl::toROSMsg(*boundaryCloud, boundary_cloud_msg);
    boundary_cloud_msg.header.frame_id = frame_id_lidar;
    boundary_cloud_msg.header.stamp = this->get_clock()->now();
    boundaryCloud_pub->publish(boundary_cloud_msg);
    // for occupancygrid // 

    

    // for occupancygrid // 
    // pcl::KdTreeFLANN<pcl::PointXYZ> border_kdtree;
    // int radius = 40;
    // std::vector<int> border_idices;
    // std::vector<float> sqr_dists;
    // border_kdtree.setInputCloud(local_border_1m);

    // pcl::PointXYZ ego_point(0.0, 0.0, 0.0);
    // border_kdtree.radiusSearch(ego_point, radius, border_idices, sqr_dists);
    // for (const auto& idx: border_idices){
    //     boundaryCloud->points.push_back(local_border_1m->points[idx]);
    // }
    // for occupancygrid // 


    
    


    /// visualize ROI ///

    tc.finish("map_slice");

    
    tc.start("initialize matrices & tensor");
    rawCloud->clear();
    nongroundCloud->clear();
    objCloud->clear();
    pcl::fromROSMsg(*msg, *rawCloud);
    int N_RANGE = int((MAX_RANGE - MIN_RANGE) / RANGE_RESOLUTION) + 1;  
    
    Eigen::Tensor<int, 3> idxTen(N_RANGE, N_HORIZONTAL, 2);

    Eigen::MatrixXd zmaxMat(N_RANGE, N_HORIZONTAL);
    Eigen::MatrixXd zminMat(N_RANGE, N_HORIZONTAL);
    Eigen::MatrixXd xsumMat(N_RANGE, N_HORIZONTAL);
    Eigen::MatrixXd ysumMat(N_RANGE, N_HORIZONTAL);
    Eigen::MatrixXd zsumMat(N_RANGE, N_HORIZONTAL);
    Eigen::MatrixXi nMat(N_RANGE, N_HORIZONTAL);
    Eigen::MatrixXd rMat(N_RANGE, N_HORIZONTAL);
    Eigen::MatrixXi idxMat(N_RANGE, N_HORIZONTAL);

    

    // Initialize the matrices
    zmaxMat.setConstant(-DBL_MAX);
    zminMat.setConstant(DBL_MAX);
    idxTen.setConstant(-1);
    xsumMat.setZero();
    ysumMat.setZero();
    zsumMat.setZero();
    nMat.setZero();
    rMat.setConstant(DBL_MAX);
    idxMat.setConstant(-1);
    tc.finish("initialize matrices & tensor");

    

    
    tc.start("insidepol(map filter)");    
    // for (const auto &point : rawCloud->points)
    for (int i = 0; i < rawCloud->points.size(); i++)
    {
      auto &point = rawCloud->points.at(i);
      // if (point.ring < N_FILTER_CHANNEL)
      //   continue;
      if (point.z > THRESHOLD_Z)
        continue;
      double ang = std::atan2(point.y, point.x);
      double r = std::hypot(point.x, point.y);
      auto colIdx = int((ang + M_PI) / H_RES);
      if (r < MIN_RANGE || r > MAX_RANGE)
        continue;
      int rowIdx = int((r - MIN_RANGE) / RANGE_RESOLUTION);
      if (!is_inside_polygon(local_border, point.x, point.y) && !is_inside_polygon(local_border_a, point.x, point.y))
        continue;

      if (zmaxMat(rowIdx, colIdx) < point.z)
        zmaxMat(rowIdx, colIdx) = point.z;
      if (zminMat(rowIdx, colIdx) > point.z)
        zminMat(rowIdx, colIdx) = point.z;

      
      if (nMat(rowIdx, colIdx) < 2)
      {
        idxTen(rowIdx, colIdx, nMat(rowIdx, colIdx)) = i;
      }

      nMat(rowIdx, colIdx)++;
      xsumMat(rowIdx, colIdx) += point.x;
      ysumMat(rowIdx, colIdx) += point.y;
      zsumMat(rowIdx, colIdx) += point.z; 

    }
    tc.finish("insidepol(map filter)");

    
    


    tc.start("ground_removal");

    /////////////////// zgap ////////////////////////
    for (int i = 0; i < N_RANGE; i++)
    {
      for (int j = 0; j < N_HORIZONTAL; j++)
      {
        // when number of points in grid is 0 or 1
        if (nMat(i, j) < 2)
          continue;

        // when gap of height is too small
        double range_limit = i * RANGE_RESOLUTION + MIN_RANGE;
        double DYNAMIC_GAP = THRESHOLD_HEIGHT_GAP;
        if (range_limit < 7.5)
        {
          DYNAMIC_GAP = THRESHOLD_HEIGHT_GAP * 2;
        }
        if (zmaxMat(i, j) - zminMat(i, j) < DYNAMIC_GAP)
          continue;
        // if ((zmaxMat(i, j) - zminMat(i, j)) / nMat(i, j) > 0.5)
        //   continue;

        // pcl::PointXYZ pp;
        // pp.x = xsumMat(i, j) / nMat(i, j);
        // pp.y = ysumMat(i, j) / nMat(i, j);
        // pp.z = zsumMat(i, j) / nMat(i, j);
        // nongroundCloud->points.push_back(pp);


        for (int k = 0; k < 2; k++){
          if (idxTen(i, j, k) != -1)
          {
            auto &pt = rawCloud->points.at(idxTen(i, j, k));
            if (pt.z > THRESHOLD_Z)
                continue;
            pcl::PointXYZI pp;
            pp.x = pt.x;
            pp.y = pt.y;
            pp.z = pt.z;
            pp.intensity = (zmaxMat(i, j) - zminMat(i, j)) / nMat(i, j);

            nongroundCloud->points.push_back(pp);
          }
        }

      
      }
    }
    

    /////////////////// zgap ////////////////////////

    


    tc.finish("ground_removal");

    
    ////////////////dbscan//////////////
    tc.start("clustering");
    std::vector<vec3f> data;
    std::vector<vec4f> final_data;

    visualization_msgs::msg::MarkerArray result_markers;
    std::vector<double> result_vector;
    result_vector.push_back(0);
    custom_msgs::msg::Float64MultiArrayStamped rb_detect;
    // rb_detect.header.stamp.nanosec = this->now();
    rb_detect.header.frame_id = frame_id_lidar;

    for (auto pt : nongroundCloud->points)
    {
      data.push_back(vec3f{pt.x, pt.y, pt.z});
    }

    auto dbscan = DBSCAN<vec3f, float>();

    // if (data.size() > 30)
    // {
    //   DBSCAN_PTS = DBSCAN_PTS_DOUBLED;
    // }

    dbscan.Run(&data, 3,  DBSCAN_EPS, DBSCAN_PTS);
    auto clusters = dbscan.Clusters;
    auto noises = dbscan.Noise;
    float intensity = 0;

    clusterCloud->clear();

    std::vector<std::vector<double>> line_pts;
    std::vector<std::vector<double>> dbscan_obj_list;
    std::vector<int> n_points_of_contour_vec;

    for (auto cluster : clusters)
    {
      int n_points_of_contour = 0;     
      float sum_z = 0;
      float sum_x = 0;
      float sum_y = 0;

      std::vector<int> contour_angle_check(CONTOUR_N, 0);
      std::vector<double> contour_range_check(CONTOUR_N, 0);
      std::vector<int> contour_idx_check(CONTOUR_N, 0);

      double y_min = DBL_MAX;
      double y_max = -DBL_MAX;



      std::vector<double> ymin_pt;
      std::vector<double> ymax_pt;


              
      for (auto idx : cluster)    // CHECKPOINT
      {
        // max_angle_idx = ()
        
        sum_x += data[idx][0];
        sum_y += data[idx][1];
        sum_z += data[idx][2];
        pcl::PointXYZ pp;
        pp.x = data[idx][0];
        pp.y = data[idx][1];
        pp.z = data[idx][2];
        clusterCloud->points.push_back(pp);    

        if (data[idx][1] < y_min)
        {
          ymin_pt = {data[idx][0], data[idx][1], data[idx][2]};
          y_min = data[idx][1];
        }

        if (data[idx][1] > y_max)
        {
          ymax_pt = {data[idx][0], data[idx][1], data[idx][2]};
          y_max = data[idx][1];
        }


      }



      std::vector<double> object = {sum_x/cluster.size(), sum_y/cluster.size(), sum_z/cluster.size()};


      double nearest_distance2border = get_nearest_border_distance_v2(local_border_1m, object, ymin_pt, ymax_pt);

      if (nearest_distance2border < THRESHOLD_DISTANCE2BORDER)
        continue;

      int max_angle_idx = 0;
      int min_angle_idx = 0;

      for (auto idx : cluster)
      {
        double angle = std::atan2(data[idx][1], data[idx][0]);
        double range = std::hypot(data[idx][1], data[idx][0]);
        
        // cout << data[idx][2] << endl;
        if (std::abs(data[idx][2] - object[2]) > CONTOUR_Z_THRH){
          // cout <<"diff:" << std::abs(data[idx][2] - object[2])<< endl;
          continue;
        }
        int contour_idx = static_cast<int>(std::round((angle + M_PI) * (180 / M_PI) * (CONTOUR_N / 360)));
        if (contour_idx == CONTOUR_N){
          contour_idx -= 1;
        }
        if (contour_angle_check[contour_idx] == 1 && range > contour_range_check[contour_idx]){
          continue;
        }
        contour_angle_check[contour_idx] = 1;
        contour_range_check[contour_idx] = range;
        contour_idx_check[contour_idx] = idx;
        max_angle_idx = idx;
        min_angle_idx = idx;
  
      }
      
      for (int id = 0; id < contour_idx_check.size(); id++)
      {
        if (contour_angle_check[id] == 0)
          continue;
        // pcl::PointXYZ rawpoint;
        // rawpoint.x = data[contour_idx_check[id]][0];
        // rawpoint.y = data[contour_idx_check[id]][1];
        // rawpoint.z = data[contour_idx_check[id]][2];
        // objCloud->points.push_back(rawpoint);

        double theta = std::atan2(data[contour_idx_check[id]][1], data[contour_idx_check[id]][0]);

        double prev_max_angle = std::atan2(data[max_angle_idx][1], data[max_angle_idx][0]);
        double prev_min_angle = std::atan2(data[min_angle_idx][1], data[min_angle_idx][0]);

        max_angle_idx = (larger_angle_singlewise(theta, prev_max_angle) == theta) ? contour_idx_check[id] : max_angle_idx;
        min_angle_idx = (larger_angle_singlewise(theta, prev_min_angle) == prev_min_angle) ? contour_idx_check[id] : min_angle_idx;

      }
      
     
      std::vector<double> pt1 = {data[max_angle_idx][0], data[max_angle_idx][1], object[2]};
      std::vector<double> pt2 = {data[min_angle_idx][0], data[min_angle_idx][1], object[2]};
      std::vector<double> closest_pt = {data[min_angle_idx][0], data[min_angle_idx][1], object[2]};

      double length_obj = std::hypot(pt1[0] - pt2[0], pt1[1] - pt2[1]);
      
      if(length_obj > 7.0)
      {
        // cout << "length_obj : " << length_obj << endl;
        continue;
      }
        
      

      for (int id = 0; id < contour_idx_check.size(); id++)
      {
        if (contour_angle_check[id] == 0)
          continue;
        pcl::PointXYZ rawpoint;
        rawpoint.x = data[contour_idx_check[id]][0];
        rawpoint.y = data[contour_idx_check[id]][1];
        rawpoint.z = data[contour_idx_check[id]][2];
        objCloud->points.push_back(rawpoint);
      }


      // for (auto idx : cluster)
      for (int id = 0; id < contour_idx_check.size(); id++)
      {
        if (contour_angle_check[id] == 0)
          continue;

        double a = pt2[1] - pt1[1];
        double b = pt1[0] - pt2[0];
        double c = pt2[0] * pt1[1] - pt1[0] * pt2[1];

        std::vector<double> cur_pt = {data[contour_idx_check[id]][0], data[contour_idx_check[id]][1], object[2]};
        
        double baseline_func = a*cur_pt[0] + b*cur_pt[1] + c;
        // if (std::abs(baseline_func) > 3.0)
        //   continue;
        
        double prev_baseline_distance = std::abs(a*closest_pt[0] + b*closest_pt[1] + c) / std::sqrt(a*a + b*b);
        double baseline_distance = std::abs(baseline_func) / std::sqrt(a*a + b*b);
        closest_pt = (prev_baseline_distance < baseline_distance) ? cur_pt : closest_pt;

        n_points_of_contour += 1;
        
      }
      

      dbscan_obj_list.push_back(object);
      n_points_of_contour_vec.push_back(n_points_of_contour);

      line_pts.push_back(pt1);
      line_pts.push_back(closest_pt);
      line_pts.push_back(closest_pt);
      line_pts.push_back(pt2);


      
                       

    }
    sensor_msgs::msg::PointCloud2 localmap_cloud_msg;
    pcl::toROSMsg(*clusterCloud, localmap_cloud_msg);
    localmap_cloud_msg.header.frame_id = frame_id_lidar;
    localmap_cloud_msg.header.stamp = this->get_clock()->now();
    localmap_pub->publish(localmap_cloud_msg);
    
    tc.finish("clustering");
    
    tc.start("heading estimation");
    std::vector<double> curvature_log;

    for (int line_idx_1st = 0; line_idx_1st < n_points_of_contour_vec.size(); line_idx_1st++)
    {
      if (n_points_of_contour_vec[line_idx_1st] < 3)
      {
        // cout << "not enough points : " << n_points_of_contour_vec[line_idx_1st] <<endl;
        continue; 
      }
      double log_curvature = calculateLogCurvature(line_pts[line_idx_1st * 4], line_pts[line_idx_1st * 4 + 1], line_pts[line_idx_1st * 4 + 3]);
      // cout << "log curvatre of " << line_idx_1st << " contour :" << log_curvature << endl;
      // cout << "UC distance from ego : " << std::hypot(line_pts[line_idx_1st * 4 + 1][0], line_pts[line_idx_1st * 4 + 1][1]) << endl;
    }
    

    int count = 0;
    for (const auto& object : dbscan_obj_list)
      {
        result_vector[0] += 1;
        result_vector.push_back(object[0] + translation_lidar2ego[0]);
        result_vector.push_back(object[1] + translation_lidar2ego[1]);
        // result_vector.push_back(object[2]);

        double lane_1_dist = get_nearest_border_distance(local_linkCloud_1, object);
        double lane_2_dist = get_nearest_border_distance(local_linkCloud_2, object);
        double lane_3_dist = get_nearest_border_distance(local_linkCloud_3, object);
        
        double link_yaw;

        int nearest_lane = (lane_1_dist < lane_2_dist) ? ((lane_1_dist < lane_3_dist) ? 1 : 3) : ((lane_2_dist < lane_3_dist) ? 2 : 3);
        switch (nearest_lane){
          case 1:
            link_yaw = get_object_local_yaw(local_linkCloud_1, object);
            break;
          case 2:
            link_yaw = get_object_local_yaw(local_linkCloud_2, object);
            break;
          case 3:
            link_yaw = get_object_local_yaw(local_linkCloud_3, object);
            break;
          default:
            link_yaw = 0.0;
            break;
        }

        

        std::vector<double> link_heading = {std::cos(link_yaw), std::sin(link_yaw)};
       
        // std::vector<double> heading_1 = {line_pts[count * 4][0] - line_pts[count * 4 + 1][0], line_pts[count * 4][1] - line_pts[count * 4 + 1][1]};
        // std::vector<double> heading_2 = {line_pts[count * 4 + 3][0] - line_pts[count * 4 + 1][0], line_pts[count * 4 + 3][1] - line_pts[count * 4 + 1][1]};

        std::vector<double> heading_1 = get_unit_vector(line_pts[count * 4][0] - line_pts[count * 4 + 1][0], line_pts[count * 4][1] - line_pts[count * 4 + 1][1]);
        std::vector<double> heading_2 = get_unit_vector(line_pts[count * 4 + 3][0] - line_pts[count * 4 + 1][0], line_pts[count * 4 + 3][1] - line_pts[count * 4 + 1][1]);

        double dot_pd_heading12 = heading_1[0]*heading_2[0] + heading_1[1]*heading_2[1];
        // line_pts[count * 4 + 1] = (dot_pd_heading12 < -0.7) ? line_pts[count * 4 + 3] : line_pts[count * 4 + 1];
        // line_pts[count * 4 + 2] = (dot_pd_heading12 < -0.7) ? line_pts[count * 4 + 3] : line_pts[count * 4 + 2];

        heading_1 = get_unit_vector(line_pts[count * 4][0] - line_pts[count * 4 + 1][0], line_pts[count * 4][1] - line_pts[count * 4 + 1][1]);
        heading_2 = get_unit_vector(line_pts[count * 4 + 3][0] - line_pts[count * 4 + 1][0], line_pts[count * 4 + 3][1] - line_pts[count * 4 + 1][1]);


        std::vector<double> heading = (std::pow(link_heading[0] * heading_1[0], 2) + std::pow(link_heading[1] * heading_1[1], 2) > std::pow(link_heading[0] * heading_2[0], 2) + std::pow(link_heading[1] * heading_2[1], 2)) ? heading_1 : heading_2;

        double heading_yaw = std::atan2(heading[1], heading[0]);

        

        result_vector.push_back(link_yaw);
        tf2::Quaternion object_quat;
        object_quat.setRPY(0, 0, link_yaw);


        geometry_msgs::msg::Pose pose;
        pose.position.x = object[0];
        pose.position.y = object[1];
        pose.position.z = object[2];
        pose.orientation.x = object_quat.getX();
        pose.orientation.y = object_quat.getY();
        pose.orientation.z = object_quat.getZ();
        pose.orientation.w = object_quat.getW();
        
        visualization_msgs::msg::Marker marker;

        marker.header.frame_id = frame_id_lidar;
        marker.header.stamp = this->now();
        marker.id = intensity;
        marker.type = visualization_msgs::msg::Marker::ARROW;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose = pose;
        marker.scale.x = 5.0;
        marker.scale.y = 0.5;
        marker.scale.z = 0.5;
        marker.color.a = 1.0;
        marker.color.r = intensity;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        marker.lifetime = rclcpp::Duration(300000000*1);

        result_markers.markers.push_back(marker);
        
        intensity += 1;
        count += 1; 
      }
      visualization_msgs::msg::Marker line_marker;
      line_marker.header.frame_id = "os1_frame";
      line_marker.header.stamp = this->now();
      line_marker.ns = "line_list_namespace";
      line_marker.id = 0;
      line_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
      line_marker.action = visualization_msgs::msg::Marker::ADD;
      line_marker.lifetime = rclcpp::Duration(100000000 * 1000);
      line_marker.scale.x = 0.2; 
      line_marker.color.r = 1.0f;
      line_marker.color.g = 0.0f;
      line_marker.color.b = 0.0f;
      line_marker.color.a = 1.0f; 

      for (auto pt : line_pts)
      {
        geometry_msgs::msg::Point point_msg;
        point_msg.x = pt[0];
        point_msg.y = pt[1];
        point_msg.z = pt[2];
        line_marker.points.push_back(point_msg);
      }

      line_pub->publish(line_marker);

   


    tc.finish("heading estimation");
    // cout << " # of object : " << result_vector[0] << endl;

    // sensor_msgs::msg::PointCloud2 boundary_cloud_msg;
    // pcl::toROSMsg(*local_border, boundary_cloud_msg);
    // // pcl::toROSMsg(*local_border_1m, boundary_cloud_msg);
    // boundary_cloud_msg.header.frame_id = frame_id_lidar;
    // boundary_cloud_msg.header.stamp = this->get_clock()->now();
    // // boundaryCloud_pub->publish(boundary_cloud_msg);

    rb_detect.data = result_vector;

    resultmarker_pub->publish(result_markers);
    rb_detections_pub->publish(rb_detect);
    //////////////
    //////////////////dbscan//////////////


    
    
    tc.finish("total");
    // tc.print();

    rotated_linkCloud_1->clear();
    rotated_linkCloud_2->clear();
    rotated_linkCloud_3->clear();
    local_linkCloud_1->clear();
    local_linkCloud_2->clear();
    local_linkCloud_3->clear();

    rot_global_cloud_l->clear();
    local_mapCloud_l->clear();
    local_border_l->clear();
    rot_global_cloud_r->clear();
    local_mapCloud_r->clear();
    local_border_r->clear();
    
    rot_global_cloud_l_1m->clear();
    local_mapCloud_l_1m->clear();
    local_border_l_1m->clear();
    rot_global_cloud_r_1m->clear();
    local_mapCloud_r_1m->clear();
    local_border_r_1m->clear();

    rot_global_cloud_al->clear();
    local_mapCloud_al->clear();
    local_border_al->clear();
    rot_global_cloud_ar->clear();
    local_mapCloud_ar->clear();
    local_border_ar->clear();
    
    rot_global_cloud_al_1m->clear();
    local_mapCloud_al_1m->clear();
    local_border_al_1m->clear();
    rot_global_cloud_ar_1m->clear();
    local_mapCloud_ar_1m->clear();
    local_border_ar_1m->clear();

    local_border->clear();
    local_border_1m->clear();
    local_border_a->clear();

    roi_vis -> clear();
    boundaryCloud -> clear();
    

    
      
    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(*objCloud, cloud_msg);
    cloud_msg.header.frame_id = frame_id_lidar;
    cloud_msg.header.stamp = this->get_clock()->now();
    objCloud_pub->publish(cloud_msg);

    
  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<custom_msgs::msg::Float64MultiArrayStamped>::SharedPtr rb_detections_pub;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr resultmarker_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr localmap_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr objCloud_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr boundaryCloud_pub;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr line_pub;
  
  // rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr localmap_pub_r;
  size_t count_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RBdetect>());
  rclcpp::shutdown();
  return 0;
}

