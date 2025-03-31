#include "rule_based_detect/lshape_detect.hpp"

double LShapeDetect::get_object_local_yaw(pcl::PointCloud<pcl::PointXYZ>::Ptr local_link, std::vector<double> object)
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

double LShapeDetect::calculateLogCurvature(const std::vector<double>& p1, const std::vector<double>& p2, const std::vector<double>& p3)
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
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> LShapeDetect::getClusters(std::vector<std::vector<uint>>& clusters, std::vector<vec3f>& nonground_data)
{
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusterCloud_vector;
  for (int c_idx = 0; c_idx < clusters.size(); c_idx++)
  {
    auto cluster = clusters.at(c_idx);
    pcl::PointCloud<pcl::PointXYZ>::Ptr obj_cluster(new pcl::PointCloud<pcl::PointXYZ>);
    for (auto idx : cluster)
    {
      pcl::PointXYZ cluster_point;        
      cluster_point.x = nonground_data[idx][0];
      cluster_point.y = nonground_data[idx][1];
      cluster_point.z = nonground_data[idx][2];
      obj_cluster->points.push_back(cluster_point);
    }
    clusterCloud_vector.push_back(obj_cluster);
  }
  return clusterCloud_vector;
}

void LShapeDetect::pushClusters(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusterCloud_vector, std::vector<std::vector<double>> dist_angle_list){
  for (int c_idx = 0; c_idx < clusterCloud_vector.size(); c_idx++){
    auto cluster = clusterCloud_vector.at(c_idx);
    double dist = dist_angle_list.at(c_idx)[0];
    double angle = dist_angle_list.at(c_idx)[1];
    for (auto& pt : cluster->points){
      pt.x += dist * std::cos(angle);
      pt.y += dist * std::sin(angle);
    }
  }
}

std::vector<std::vector<double>> LShapeDetect::pullClusters(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusterCloud_vector)
{
  std::vector<std::vector<double>> dist_angle_list;
  for (int c_idx = 0; c_idx < clusterCloud_vector.size(); c_idx++){
    auto cluster = clusterCloud_vector.at(c_idx);
    double minimum_range = 999;
    int minimum_idx;
    double reference_range = 10;
    for (int i = 0; i < cluster->points.size(); i++){
      auto pt = cluster->points.at(i);
      double range = std::hypot(pt.y, pt.x);
      if (range < minimum_range){
        minimum_range = range;
        minimum_idx = i;
      }
    }
    double angle = std::atan2(cluster->points.at(minimum_idx).y, cluster->points.at(minimum_idx).x);
    double distance = minimum_range - reference_range;
    std::vector<double> dist_angle{distance, angle};
    dist_angle_list.push_back(dist_angle);
    for (auto& pt : cluster->points){
      pt.x -= distance * std::cos(angle);
      pt.y -= distance * std::sin(angle);
    }
  }
  return dist_angle_list;
}
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> LShapeDetect::getContourV2(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusterCloud_vector, std::vector<std::vector<double>>& dbscan_obj_list, const double contour_res, const double contour_z_thresh)
{
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> contourCloud_vector;
  cout << "111111111111111" << endl;
  for (int c_idx = 0; c_idx < clusterCloud_vector.size(); c_idx++){
    auto cluster = clusterCloud_vector.at(c_idx);
    double max_angle = std::atan2(cluster->points.at(0).y, cluster->points.at(0).x);
    double min_angle = std::atan2(cluster->points.at(0).y, cluster->points.at(0).x);
    cout << "22222222222" << endl;
    for (int idx = 0; idx < cluster->points.size(); idx++){
      auto pt = cluster->points.at(idx);
      double angle = std::atan2(pt.y, pt.x);
      max_angle = (larger_angle_singlewise(angle, max_angle) == angle) ? angle : max_angle;
      min_angle = (larger_angle_singlewise(angle, min_angle) == angle) ? min_angle : angle;
    }
    cout << "333333333333" << endl;
    double min = std::round(min_angle * (180 / M_PI) / contour_res);
    double max = std::round(max_angle * (180 / M_PI) / contour_res);
    int contour_n = (max - min > 0) ? static_cast<int>(max - min + 1) : static_cast<int>(max - min + 720 + 1);
    cout << "contour_n : " << contour_n << endl;
    cout << "min_angle : " << min_angle << endl;
    cout << "max_angle : " << max_angle << endl;
    cout << "min : " << min << endl;
    cout << "max : " << max << endl;

    std::vector<int> contour_pt_idx(contour_n, -1);
    std::vector<int> contour_angle_check(contour_n, 0);
    std::vector<double> contour_range_check(contour_n, 0);

    for (int idx = 0; idx < cluster->points.size(); idx++){
      auto pt = cluster->points.at(idx);
      if (std::abs(pt.z - dbscan_obj_list[c_idx][2]) > contour_z_thresh)
          continue;
      double range = std::hypot(pt.y, pt.x);
      double angle = std::atan2(pt.y, pt.x) * (180 / M_PI);
      double min_angle_deg = (max_angle > min_angle) ? min_angle * (180 / M_PI) : max_angle * (180 / M_PI);
      
      // cout << "min_angle_deg : " << min_angle_deg << endl;
      // double rel_angle = (angle - min_angle_deg > 0) ? angle - min_angle_deg : angle - min_angle_deg + M_PI;
      double rel_angle = angle - min_angle_deg;
      // cout << "rel_angle : " << rel_angle << endl;
      int angle_idx = static_cast<int>(std::round(rel_angle / contour_res));
      if (angle_idx >= contour_n)
        continue;
      
      if (contour_angle_check[angle_idx] == 1 && range > contour_range_check[angle_idx])
        continue;
      // cout << "rel_angle : " << rel_angle << endl;
      cout << "angle_idx : " << angle_idx << endl;
      contour_angle_check[angle_idx] = 1;
      contour_range_check[angle_idx] = range;
      contour_pt_idx[angle_idx] = idx;
    }
    cout << "7777777" << endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr obj_contour(new pcl::PointCloud<pcl::PointXYZ>);
    for (int id = 0; id < contour_pt_idx.size(); id++)
    {
      if (contour_pt_idx[id] == -1)
        continue;
        
      pcl::PointXYZ contour_point;        
      contour_point.x = cluster->points.at(contour_pt_idx[id]).x;
      contour_point.y = cluster->points.at(contour_pt_idx[id]).y;
      contour_point.z = dbscan_obj_list[c_idx][2];
      obj_contour->points.push_back(contour_point);
    }
    cout << "888888" << endl;
    contourCloud_vector.push_back(obj_contour);
  }
  cout << "99999" << endl;
  return contourCloud_vector;
}

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> LShapeDetect::getContour(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusterCloud_vector, 
                                                                          std::vector<std::vector<double>>& dbscan_obj_list, const int contour_n, 
                                                                          const double contour_z_thresh)
{
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> contourCloud_vector;
  
  
  for (int c_idx = 0; c_idx < clusterCloud_vector.size(); c_idx++){
    auto cluster = clusterCloud_vector.at(c_idx);

    std::vector<int> contour_angle_check(contour_n, 0);
    std::vector<double> contour_range_check(contour_n, 0);
    std::vector<int> contour_pt_idx(contour_n, -1);

    pcl::PointCloud<pcl::PointXYZ>::Ptr obj_contour(new pcl::PointCloud<pcl::PointXYZ>);

    // for (auto pt : cluster->points){
    for (int idx = 0; idx < cluster->points.size(); idx++){
      auto pt = cluster->points.at(idx);
      double angle = std::atan2(pt.y, pt.x);
      double range = std::hypot(pt.y, pt.x);
      if (std::abs(pt.z - dbscan_obj_list[c_idx][2]) > contour_z_thresh)
          continue;
      
      int contour_idx = static_cast<int>(std::round((angle + M_PI) * (180 / M_PI) * (contour_n / 360)));
      if (contour_idx == contour_n){
        contour_idx -= 1;
      }
      if (contour_angle_check[contour_idx] == 1 && range > contour_range_check[contour_idx])
        continue;

      contour_angle_check[contour_idx] = 1;
      contour_range_check[contour_idx] = range;
      contour_pt_idx[contour_idx] = idx;
      
    }

    for (int id = 0; id < contour_pt_idx.size(); id++)
    {
      if (contour_pt_idx[id] == -1)
        continue;
        
      pcl::PointXYZ contour_point;        
      contour_point.x = cluster->points.at(contour_pt_idx[id]).x;
      contour_point.y = cluster->points.at(contour_pt_idx[id]).y;
      contour_point.z = dbscan_obj_list[c_idx][2];
      obj_contour->points.push_back(contour_point);

    }

    contourCloud_vector.push_back(obj_contour);
  }

  return contourCloud_vector;
}
void LShapeDetect::pcd_sub_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{

  TimeChecker tc(false);
  tc.start("total");
  int md_type = 0;
  generate_mapdata_pointcloud();    
  assemble_mapdata(md_type); 

  rawCloud->clear();
  nongroundCloud->clear();
  contourCloud->clear();
  pcl::fromROSMsg(*msg, *rawCloud);
  
  tc.start("ground_removal");
  init_matrices(mat_of_PC);
  select_roi(rawCloud, mat_of_PC);  
  ground_removal(rawCloud, nongroundCloud, mat_of_PC);
  sensor_msgs::msg::PointCloud2 nongroundCloud_msg;
  pcl::toROSMsg(*nongroundCloud, nongroundCloud_msg);
  nongroundCloud_msg.header.frame_id = frame_id_lidar;
  nongroundCloud_msg.header.stamp = this->get_clock()->now();
  nongroundCloud_pub->publish(nongroundCloud_msg);
  tc.finish("ground_removal");

  tc.start("getObjectList");

  std::vector<vec3f> nonground_data;
  for (auto& pt : nongroundCloud->points)
  {
    nonground_data.push_back(vec3f{pt.x, pt.y, pt.z});
  }
  auto clusters = dbscan_clustering(nonground_data, clusterCloud);

  sensor_msgs::msg::PointCloud2 cluster_cloud_msg;
  pcl::toROSMsg(*clusterCloud, cluster_cloud_msg);
  cluster_cloud_msg.header.frame_id = frame_id_lidar;
  cluster_cloud_msg.header.stamp = this->get_clock()->now();
  clustercloud_pub->publish(cluster_cloud_msg);

  auto dbscan_obj_list = getObjectList(nonground_data, clusters);
  tc.finish("getObjectList");
  
  auto clusterCloud_vector = getClusters(clusters, nonground_data);
  

  tc.start("getContour");
  auto dist_ang_list = pullClusters(clusterCloud_vector);
  // auto contourCloud_vector = getContour(clusterCloud_vector, dbscan_obj_list, CONTOUR_N, CONTOUR_Z_THRH);
  auto contourCloud_vector = getContourV2(clusterCloud_vector, dbscan_obj_list, CONTOUR_RES, CONTOUR_Z_THRH);
  cout << "10101010" << endl;
  pushClusters(contourCloud_vector, dist_ang_list);
  cout << "11.11.11.11" << endl;
  // visualization
  for (auto& contour : contourCloud_vector)
  {
    for (auto& pt : contour->points){
      contourCloud->points.push_back(pt);
    }
  }
  tc.start("getContour");
  cout << "121212121212" << endl;
  
  

  tc.finish("total");
  tc.print();

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

  boundaryCloud -> clear();


  
    
  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(*contourCloud, cloud_msg);
  cloud_msg.header.frame_id = frame_id_lidar;
  cloud_msg.header.stamp = this->get_clock()->now();
  contour_pub->publish(cloud_msg);

  for (auto cluster : clusterCloud_vector)
  {
    cluster->clear();
  }
  for (auto contour : contourCloud_vector)
  {
    contour->clear();
  }
  cout << "131313131313" << endl;
  
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LShapeDetect>());
  rclcpp::shutdown();
  return 0;
}

