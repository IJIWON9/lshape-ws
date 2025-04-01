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

struct vec3f
  {
    float data[3];
    float operator[](int idx) const { return data[idx]; }
  };

using PointXYZIRT = OusterPointXYZIRT;

class LShapeDetect : public rclcpp::Node
{
public:
  LShapeDetect()
      : Node("rb_detect_node"), mat_of_PC(N_RANGE, N_HORIZONTAL)
  {
    // load mapdata
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
    
    // publishers
    resultmarker_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("/rulebased/result_marker", 10);
    contour_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("/rulebased/contour", 10);
    rb_detections_pub = this->create_publisher<custom_msgs::msg::Float64MultiArrayStamped>("/rulebased/detections", 10);
    clustercloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("/rulebased/clusterCloud", 10);
    boundaryCloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("/rulebased/boundaryCloud", 10);
    nongroundCloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("/rulebased/nongroundCloud", 10);

    line_pub = this->create_publisher<visualization_msgs::msg::Marker>("/rulebased/line_list_marker", 10);


    // subscribers
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(1));
    pcd_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/os1/lidar",
        qos_profile,
        std::bind(&LShapeDetect::pcd_sub_callback, this, std::placeholders::_1));

    odometry_sub = this->create_subscription<nav_msgs::msg::Odometry>(
        "/localization/ego_pose",
        qos_profile,
        std::bind(&LShapeDetect::odometry_sub_callback, this, std::placeholders::_1));

    // LiDAR Translation
    lidarPose.setIdentity();
    lidarPose.translation() = Eigen::Vector3d(1.2, 0.0, 1.0);
    
    // initialize pointclouds
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
    contourCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());

    
    
    // load mapdata into map pointclouds
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

    // linkdata(mapdata)

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

  }
  ~LShapeDetect() {}

  double get_object_local_yaw(pcl::PointCloud<pcl::PointXYZ>::Ptr local_link, std::vector<double> object);

  double calculateLogCurvature(const std::vector<double>& p1, const std::vector<double>& p2, const std::vector<double>& p3);

  void pcd_sub_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> getClusters(std::vector<std::vector<uint>>& clusters, std::vector<vec3f>& nonground_data);
  
  std::vector<std::vector<double>> pullClusters(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusterCloud_vector);
  
  void pushClusters(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusterCloud_vector, std::vector<std::vector<double>> dist_angle_list);

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> getContour(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusterCloud_vector,  
                                                                          std::vector<std::vector<double>>& dbscan_obj_list, const int contour_n, 
                                                                          const double contour_z_thresh);

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> getContourV2(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusterCloud_vector, std::vector<std::vector<double>>& dbscan_obj_list, 
                                                                          const double contour_res, const double contour_z_thresh, std::vector<std::vector<double>>& dist_angle_list);

  pcl::PointCloud<pcl::PointXYZ>::Ptr removeOutlier(pcl::PointCloud<pcl::PointXYZ>::Ptr obj_contour, 
                                                    double max_distance, std::vector<int>& contour_pt_idx,
                                                    double min_angle, double max_angle, double contour_res);

  void interpolateContour(pcl::PointCloud<pcl::PointXYZ>::Ptr filtered, pcl::PointCloud<pcl::PointXYZ>::Ptr cluster, 
                            int contour_n, std::vector<int>& contour_pt_idx, double average_z);

  std::vector<int> sortByAngle(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud, double contour_res);

  
  

  struct vec4f
  {
    float data[4];
    float operator[](int idx) const { return data[idx]; }
  };

  struct PointMatrices
  {
    Eigen::Tensor<int, 3> idxTen;
    Eigen::MatrixXd zmaxMat;
    Eigen::MatrixXd zminMat;
    Eigen::MatrixXd xsumMat;
    Eigen::MatrixXd ysumMat;
    Eigen::MatrixXd zsumMat;
    Eigen::MatrixXi nMat;
    Eigen::MatrixXd rMat;
    Eigen::MatrixXi idxMat;

    PointMatrices(int num_r, int num_h)
      : idxTen(num_r, num_h, 3),  
        zmaxMat(num_r, num_h),
        zminMat(num_r, num_h),
        xsumMat(num_r, num_h),
        ysumMat(num_r, num_h),
        zsumMat(num_r, num_h),
        nMat(num_r, num_h),
        rMat(num_r, num_h),
        idxMat(num_r, num_h) {}

  };

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pcd_sub;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub;

  pcl::PointCloud<pcl::PointXYZ>::Ptr rawCloud;         
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
  pcl::PointCloud<pcl::PointXYZ>::Ptr contourCloud;



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
  double THRESHOLD_Z = 2.0;
  double THRESHOLD_HEIGHT_GAP = 0.3;
  double H_RES = M_PI * 2 / N_HORIZONTAL;
  double THRESHOLD_DISTANCE2BORDER = 1.7;
  double THRESHOLD_DISTANCE2LINK = 1.2;
  int N_RANGE = int((MAX_RANGE - MIN_RANGE) / RANGE_RESOLUTION) + 1;  

  int CONTOUR_N = 720;
  double CONTOUR_RES = 0.5;
  double CONTOUR_Z_THRH = 0.5;

  Eigen::Isometry3d lidarPose, currPose;

   

  uint DBSCAN_PTS = 4;
  const float DBSCAN_EPS = 1.5;
  const std::string frame_id_lidar = "os1_frame";

  PointMatrices mat_of_PC;

  

private:
  rclcpp::Publisher<custom_msgs::msg::Float64MultiArrayStamped>::SharedPtr rb_detections_pub;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr resultmarker_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr clustercloud_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr contour_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr boundaryCloud_pub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr nongroundCloud_pub;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr line_pub;


  double larger_angle_singlewise(double theta1, double theta2)   
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
  
  void slice_map_l(pcl::PointCloud<pcl::PointXYZ>::Ptr local_pcd, pcl::PointCloud<pcl::PointXYZ>::Ptr local_border, int interval, bool is_alpha)
  {
    int front_idx = 100 / interval;
    int rear_idx = 20 / interval;
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
  int rear_idx = 20 / interval;
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

  void generate_mapdata_pointcloud(void)
  {
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
  }

  void assemble_mapdata(int mapdata_type)
  {
    if (mapdata_type == 0){
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
    } else if (mapdata_type == 1){
      // for RECTANGLE //
      double rec_half_w = 30.0;
      double rec_forward = 100.0;
      double rec_backward = 1.0;
      double rec_weight = 1.0;
      pcl::PointXYZ pp1;        
      pp1.x = rec_forward * rec_weight;
      pp1.y = rec_half_w * rec_weight;
      pp1.z = 0.0;
      local_border->points.push_back(pp1);
      pp1.x = rec_forward * rec_weight;
      pp1.y = -rec_half_w * rec_weight;
      pp1.z = 0.0;
      local_border->points.push_back(pp1);
      pp1.x = -rec_backward * rec_weight;
      pp1.y = -rec_half_w * rec_weight;
      pp1.z = 0.0;
      local_border->points.push_back(pp1);
      pp1.x = -rec_backward * rec_weight;
      pp1.y = rec_half_w * rec_weight;
      pp1.z = 0.0;
      local_border->points.push_back(pp1);
    } else if (mapdata_type == 2){
      // for SQUARE //
      double square_size = 60;
      pcl::PointXYZ pp1;        
      pp1.x = square_size;
      pp1.y = square_size;
      pp1.z = 0.0;
      local_border->points.push_back(pp1);
      pp1.x = square_size;
      pp1.y = -square_size;
      pp1.z = 0.0;
      local_border->points.push_back(pp1);
      pp1.x = -square_size;
      pp1.y = -square_size;
      pp1.z = 0.0;
      local_border->points.push_back(pp1);
      pp1.x = -square_size;
      pp1.y = square_size;
      pp1.z = 0.0;
      local_border->points.push_back(pp1);
    } else if (mapdata_type == 3) {
      // for logged mapdata//
      for (const auto &point : local_border_l->points){
        local_border->points.push_back(point);
      }
    }   

    // visualize ROI //
    for (const auto &point : local_border->points)
    {
      boundaryCloud->points.push_back(point);
    }
    for (const auto &point : local_border_a->points)
    {
      boundaryCloud->points.push_back(point);
    }
    sensor_msgs::msg::PointCloud2 boundary_cloud_msg;
    pcl::toROSMsg(*boundaryCloud, boundary_cloud_msg);
    boundary_cloud_msg.header.frame_id = frame_id_lidar;
    boundary_cloud_msg.header.stamp = this->get_clock()->now();
    boundaryCloud_pub->publish(boundary_cloud_msg);
    // visualize ROI //
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
    
    tf2::Quaternion ego_qaut(msg->pose.pose.orientation.x,msg->pose.pose.orientation.y,msg->pose.pose.orientation.z,msg->pose.pose.orientation.w);  
    tf2::Matrix3x3 ego_rotation(ego_qaut);


    ego_rotation.getRPY(ego_roll, ego_pitch, ego_yaw);

    
  }

  void init_matrices(PointMatrices& mat_of_PC)
  {
    mat_of_PC.zmaxMat.setConstant(-DBL_MAX);
    mat_of_PC.zminMat.setConstant(DBL_MAX);
    mat_of_PC.idxTen.setConstant(-1);
    mat_of_PC.xsumMat.setZero();
    mat_of_PC.ysumMat.setZero();
    mat_of_PC.zsumMat.setZero();
    mat_of_PC.nMat.setZero();
    mat_of_PC.rMat.setConstant(DBL_MAX);
    mat_of_PC.idxMat.setConstant(-1);
    
  }

  void select_roi(pcl::PointCloud<pcl::PointXYZ>::Ptr rawCloud, PointMatrices& mat_of_PC)
  {
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

      if (mat_of_PC.zmaxMat(rowIdx, colIdx) < point.z)
        mat_of_PC.zmaxMat(rowIdx, colIdx) = point.z;
      if (mat_of_PC.zminMat(rowIdx, colIdx) > point.z)
        mat_of_PC.zminMat(rowIdx, colIdx) = point.z;

      
      if (mat_of_PC.nMat(rowIdx, colIdx) < 3)
      {
        mat_of_PC.idxTen(rowIdx, colIdx, mat_of_PC.nMat(rowIdx, colIdx)) = i;
      }

      mat_of_PC.nMat(rowIdx, colIdx)++;
      mat_of_PC.xsumMat(rowIdx, colIdx) += point.x;
      mat_of_PC.ysumMat(rowIdx, colIdx) += point.y;
      mat_of_PC.zsumMat(rowIdx, colIdx) += point.z; 

    }
  }
  void ground_removal(pcl::PointCloud<pcl::PointXYZ>::Ptr rawCloud, pcl::PointCloud<pcl::PointXYZI>::Ptr nongroundCloud, PointMatrices& mat_of_PC)
  {
    for (int i = 0; i < N_RANGE; i++)
    {
      for (int j = 0; j < N_HORIZONTAL; j++)
      {
        double range_limit = i * RANGE_RESOLUTION + MIN_RANGE;
        double DYNAMIC_GAP = THRESHOLD_HEIGHT_GAP;

        // when number of points in grid is 0 or 1
        if (mat_of_PC.nMat(i, j) < 2){
          if (range_limit > 60.0){
            for (int k = 0; k < 3; k++){
              if (mat_of_PC.idxTen(i, j, k) != -1)
              {
                auto &pt = rawCloud->points.at(mat_of_PC.idxTen(i, j, k));
                if (pt.z > THRESHOLD_Z)
                    continue;
                pcl::PointXYZI pp;
                pp.x = pt.x;
                pp.y = pt.y;
                pp.z = pt.z;
                pp.intensity = (mat_of_PC.zmaxMat(i, j) - mat_of_PC.zminMat(i, j)) / mat_of_PC.nMat(i, j);

                nongroundCloud->points.push_back(pp);
              }
            }
          }
          continue;
        }
          

        // when gap of height is too small
        if (range_limit < 7.5)
        {
          DYNAMIC_GAP = THRESHOLD_HEIGHT_GAP * 2;
        } else if (range_limit > 40.0){
          DYNAMIC_GAP = THRESHOLD_HEIGHT_GAP * 0.5;
        } else if (range_limit > 60.0){
          DYNAMIC_GAP = THRESHOLD_HEIGHT_GAP * 0.25;
        }
        if (mat_of_PC.zmaxMat(i, j) - mat_of_PC.zminMat(i, j) < DYNAMIC_GAP)
          continue;



        for (int k = 0; k < 3; k++){
          if (mat_of_PC.idxTen(i, j, k) != -1)
          {
            auto &pt = rawCloud->points.at(mat_of_PC.idxTen(i, j, k));
            if (pt.z > THRESHOLD_Z)
                continue;
            pcl::PointXYZI pp;
            pp.x = pt.x;
            pp.y = pt.y;
            pp.z = pt.z;
            pp.intensity = (mat_of_PC.zmaxMat(i, j) - mat_of_PC.zminMat(i, j)) / mat_of_PC.nMat(i, j);

            nongroundCloud->points.push_back(pp);
          }
        }
    
      }
    }
  }
  std::vector<std::vector<uint>> dbscan_clustering(std::vector<vec3f> &data, pcl::PointCloud<pcl::PointXYZ>::Ptr clusterCloud)
  {
    auto dbscan = DBSCAN<vec3f, float>();

    dbscan.Run(&data, 3,  DBSCAN_EPS, DBSCAN_PTS);
    auto clusters = dbscan.Clusters;
    auto noises = dbscan.Noise;
    clusterCloud->clear();

    
    for (auto cluster : clusters)   // for visualize
    {
      for (auto idx : cluster)    
      {
        pcl::PointXYZ pp;
        pp.x = data[idx][0];
        pp.y = data[idx][1];
        pp.z = data[idx][2];
        clusterCloud->points.push_back(pp);
      }
    }
    return clusters;
  }

  std::vector<std::vector<double>> getObjectList(std::vector<vec3f> &data, std::vector<std::vector<uint>> &clusters, int mapdata_type)
  {
    std::vector<std::vector<double>> dbscan_obj_list;
    std::vector<int> removed_indices;
    for (int c_idx = 0; c_idx < clusters.size(); c_idx++)
    {
      auto cluster = clusters.at(c_idx);
      float sum_x = 0;
      float sum_y = 0;
      float sum_z = 0;

      double y_min = DBL_MAX;
      double y_max = -DBL_MAX;
      std::vector<double> ymin_pt;
      std::vector<double> ymax_pt;
              
      for (auto idx : cluster)    
      {        
        sum_x += data[idx][0];
        sum_y += data[idx][1];
        sum_z += data[idx][2];

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
      
      if (mapdata_type == 0){
        double nearest_distance2border = get_nearest_border_distance_v2(local_border_1m, object, ymin_pt, ymax_pt);
        if (nearest_distance2border < THRESHOLD_DISTANCE2BORDER)
        {
          removed_indices.push_back(c_idx);
          continue;
        }
          
        
      }
      dbscan_obj_list.push_back(object);
    }

    for (int i = removed_indices.size() -1; i >= 0; i--){
      clusters.erase(clusters.begin() + removed_indices.at(i));
    }
    
    return dbscan_obj_list;
  }
};
