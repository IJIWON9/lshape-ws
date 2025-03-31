#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <tf2/LinearMath/Quaternion.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <typeinfo>
#include <cstdlib>
#include <functional>
#include <algorithm>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

using namespace std::chrono_literals;

class OGMapGenerator : public rclcpp::Node
{
public:

  OGMapGenerator(rclcpp::NodeOptions options)
    : rclcpp::Node("ogmap_generator", options)
  {
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(1));
    
    boundarySub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/rulebased/boundaryCloud",
      qos_profile,
      std::bind(&OGMapGenerator::pcdCallback, this, std::placeholders::_1)
    );
    objCloudSub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/rulebased/localmap",
      qos_profile,
      std::bind(&OGMapGenerator::objCloudCallback, this, std::placeholders::_1)
    );

    mapPub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/rulebased/og_map",
      10
    );

    // initialize
    pcd_data.reset(new pcl::PointCloud<pcl::PointXYZ>());
    objCloud.reset(new pcl::PointCloud<pcl::PointXYZ>());

    // timer_ = this->create_wall_timer(
    //   100ms, std::bind(&OGMapGenerator::timer_callback, this));

    ogmap_data = new int [map_shape_w * map_shape_h];
    
  }
  ~OGMapGenerator() {}

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr boundarySub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr objCloudSub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr mapPub_;
  int map_width = 30;
  int map_height = 30; 
  double map_cell_size = 0.2;
  int map_shape_w = map_width / map_cell_size;
  int map_shape_h = map_height / map_cell_size;
  int* ogmap_data;
  std::vector<int> ego_trans_to_map = {-map_width/10, -map_height/2, 10};   // meter
  std::vector<int> ego_vec_inmap = {-ego_trans_to_map[0]/map_cell_size, -ego_trans_to_map[1]/map_cell_size};    // map coordinate

  pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_data;
  pcl::PointCloud<pcl::PointXYZ>::Ptr objCloud;
  
  
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
  void objCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (objCloud->points.size() > 0)
    {
      objCloud->clear();
    }
    pcl::fromROSMsg(*msg, *objCloud);
  }
  // void timer_callback()
  // {
  //   init_ogmap();
  //   if (pcd_data->points.size() > 0)
  //   {
  //     for (int i = 0; i < map_shape_w; i++)
  //     {
  //       for (int j = 0; j < map_shape_h; j++)
  //       {
  //         float x_out_map = i*map_cell_size + ego_trans_to_map[0];
  //         float y_out_map = j*map_cell_size + ego_trans_to_map[1];
  //         if (!is_inside_polygon(pcd_data, x_out_map, y_out_map))
  //         {
  //           ogmap_data[j*map_shape_w + i] = -1;
  //         }
  //       }
  //     }
  //       for (const auto &point : pcd_data->points)
  //       {
  //           int x_in_map = (point.x - ego_trans_to_map[0])/map_cell_size ;
  //           int y_in_map = (point.y - ego_trans_to_map[1])/map_cell_size ;
  //           if (y_in_map < 0 || x_in_map < 0 ||
  //               y_in_map > map_shape_h - 1 ||
  //               x_in_map > map_shape_w - 1)
  //               continue;
  //           ogmap_data[y_in_map*map_shape_w + x_in_map] = 100;
  //       }
  //   }

  //   publishOccupancyGrid();
  //   pcd_data->clear();
  // }
  
 

  void pcdCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    pcl::fromROSMsg(*msg, *pcd_data);

    std::vector<int> map_x;
    std::vector<int> map_y;

    init_ogmap();
    if (pcd_data->points.size() > 0)
    {
      for (int i = 0; i < map_shape_w; i++)
      {
        for (int j = 0; j < map_shape_h; j++)
        {
          float x_out_map = i*map_cell_size + ego_trans_to_map[0];
          float y_out_map = j*map_cell_size + ego_trans_to_map[1];
          if (!is_inside_polygon(pcd_data, x_out_map, y_out_map))
          {
            ogmap_data[j*map_shape_w + i] = -1;
          }
        }
      }
        for (const auto &point : pcd_data->points)
        {
          int x_in_map = (point.x - ego_trans_to_map[0])/map_cell_size ;
          int y_in_map = (point.y - ego_trans_to_map[1])/map_cell_size ;
          map_x.push_back(x_in_map);
          map_y.push_back(y_in_map);
          if (y_in_map < 0 || x_in_map < 0 ||
              y_in_map > map_shape_h - 1 ||
              x_in_map > map_shape_w - 1)
              continue;
          ogmap_data[y_in_map*map_shape_w + x_in_map] = 100;
        }
        for (int k = 0; k < map_x.size()-1; k++)
        {
          int mid_x = (map_x[k] + map_x[k+1]) / 2;
          int mid_y = (map_y[k] + map_y[k+1]) / 2;
          if (mid_y < 0 || mid_x < 0 ||
              mid_y > map_shape_h - 1 ||
              mid_x > map_shape_w - 1)
              continue;
          ogmap_data[mid_y*map_shape_w + mid_x] = 100;
        }
    }

    for (const auto &point : objCloud->points)
    {
      double angle = std::atan2(point.y, point.x);
      double range = std::hypot(point.x, point.y);
      double prob = 90.0;
      double itr = 0.0;
      double sampling_step = 3.0;
      double base = 0.9;
      double itr_limit = 10;

      while(prob > 10)
      {
        itr += 1.0;
        if (itr > itr_limit)
        {
          break;
        }
        int uk_x_in_map = (point.x - ego_trans_to_map[0])/map_cell_size +  map_cell_size * itr * sampling_step * std::cos(angle);
        int uk_y_in_map = (point.y - ego_trans_to_map[1])/map_cell_size +  map_cell_size * itr * sampling_step * std::sin(angle);
        if (uk_y_in_map < 0 || uk_x_in_map < 0 ||
          uk_y_in_map > map_shape_h - 1 ||
          uk_x_in_map > map_shape_w - 1)
          continue;

        ogmap_data[uk_y_in_map*map_shape_w + uk_x_in_map] = prob;
        
        
        prob *= base;
        
      }
      
      int x_in_map = (point.x - ego_trans_to_map[0])/map_cell_size ;
      int y_in_map = (point.y - ego_trans_to_map[1])/map_cell_size ;
      if (y_in_map < 0 || x_in_map < 0 ||
          y_in_map > map_shape_h - 1 ||
          x_in_map > map_shape_w - 1)
          continue;
      ogmap_data[y_in_map*map_shape_w + x_in_map] = 100;
    }
    // // sample saving //
    // std::ofstream file("gridmap_sample.csv");
    // if (file.is_open()) {
    //     for (int i = 0; i < map_shape_h; ++i) {
    //         for (int j = 0; j < map_shape_h; ++j) {
    //             file << ogmap_data[i * map_shape_h + j];  
    //             if (j < map_shape_h - 1) {
    //                 file << ",";
    //             }
    //         }
    //         file << "\n";
    //     }
    // }
    //     file.close();


    publishOccupancyGrid();
    pcd_data->clear();
    
  }
  void init_ogmap()
  {
    for (int i = 0; i < map_shape_w * map_shape_h; i++)
    {
        ogmap_data[i] = 0;
    }
  }
  void publishOccupancyGrid()
  {
    nav_msgs::msg::OccupancyGrid ogmap;

    ogmap.header.stamp = rclcpp::Clock().now();
    ogmap.header.frame_id = std::string("os1_frame");

    ogmap.info.map_load_time = rclcpp::Clock().now();

    ogmap.info.resolution = map_cell_size;
    ogmap.info.width  = map_shape_w;
    ogmap.info.height = map_shape_h;
    
    ogmap.info.origin.position.x = (double)ego_trans_to_map[0];
    ogmap.info.origin.position.y = (double)ego_trans_to_map[1];
    ogmap.info.origin.position.z = (double)ego_trans_to_map[2];

    tf2::Quaternion orientation;
    // orientation.setRPY(0, 0, -90*M_PI/ 180);
    orientation.setRPY(0, 0, 0);

    ogmap.info.origin.orientation.x = orientation[0];
    ogmap.info.origin.orientation.y = orientation[1];
    ogmap.info.origin.orientation.z = orientation[2];
    ogmap.info.origin.orientation.w = orientation[3];

    std::vector<int8_t> map_data(ogmap_data, ogmap_data+map_shape_w*map_shape_h);
    ogmap.data = map_data;
    mapPub_->publish(ogmap);
  }

};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;

  auto node = std::make_shared<OGMapGenerator>(options);
  
  rclcpp::spin(node);
  rclcpp::shutdown();
  
  return 0;
}
