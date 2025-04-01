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
pcl::PointCloud<pcl::PointXYZ>::Ptr LShapeDetect::removeOutlier(pcl::PointCloud<pcl::PointXYZ>::Ptr obj_contour, 
                                                                  double max_distance, std::vector<int>& contour_pt_idx, 
                                                                  double min_angle, double max_angle, double contour_res)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < obj_contour->points.size(); i++) {

    auto& curr = obj_contour->points[i];

    double angle = (max_angle - min_angle > 0) ? std::atan2(curr.y, curr.x) : ((std::atan2(curr.y, curr.x) < 0) ? std::atan2(curr.y, curr.x) + 2 * M_PI : std::atan2(curr.y, curr.x));
    double angle_deg = angle * (180 / M_PI);
    double min_angle_deg = min_angle * (180 / M_PI);
    double rel_angle = angle_deg - min_angle_deg;
    int angle_idx = static_cast<int>(std::round(rel_angle / contour_res));

    if (i == 0){
      auto& next = obj_contour->points[i + 1];
      float dist = std::hypot(next.x - curr.x, next.y - curr.y);
      if (dist < max_distance) {
        filtered->points.push_back(curr);
      } else {
        // cout << "deleted 1st " << dist << endl;
        contour_pt_idx[0] = -1;
      }
      continue;
    } 
    if (i == obj_contour->points.size() - 1){
      auto& prev = obj_contour->points[i - 1];
      float dist = std::hypot(curr.x - prev.x, curr.y - prev.y);
      if (dist < max_distance) {
        filtered->points.push_back(curr);
      } else {
        // cout << "deleted last " << dist << endl;
        contour_pt_idx[contour_pt_idx.size() - 1] = -1;
      }
      continue;
    }

    auto& prev = obj_contour->points[i - 1];
    auto& next = obj_contour->points[i + 1];

    float dist1 = std::hypot(curr.x - prev.x, curr.y - prev.y);
    float dist2 = std::hypot(next.x - curr.x, next.y - curr.y);

    if (dist1 < max_distance && dist2 < max_distance) {
      filtered->points.push_back(curr);
    } else {
      // cout << "deleted : " << dist1 << " " << dist2 << endl;
      contour_pt_idx[angle_idx] = -1;
    }
  }
  // obj_contour->points.assign(filtered.begin(), filtered.end());
  return filtered;
}

void LShapeDetect::interpolateContour(pcl::PointCloud<pcl::PointXYZ>::Ptr filtered, pcl::PointCloud<pcl::PointXYZ>::Ptr cluster, int contour_n, std::vector<int>& contour_pt_idx, double average_z)
{
  // for interpolation
  std::vector<int> prev_valid_idx(contour_n, -1);
  std::vector<int> next_valid_idx(contour_n, -1);
  int last_valid = -1;
  for (int i = 0; i < contour_n; ++i) {
    if (contour_pt_idx[i] != -1) last_valid = i;
    prev_valid_idx[i] = last_valid;
  }
  last_valid = -1;
  for (int i = contour_n - 1; i >= 0; --i) {
    if (contour_pt_idx[i] != -1) last_valid = i;
    next_valid_idx[i] = last_valid;
  }

  // interpolation
  for (int id = 0; id < contour_pt_idx.size(); id++)
  {
    if (contour_pt_idx[id] == -1) {
      if (id == 0 || id == contour_pt_idx.size() - 1) continue;

      int prev = prev_valid_idx[id];
      int next = next_valid_idx[id];

      if (prev != -1 && next != -1 && prev < next) {
        const auto& pt1 = cluster->points.at(contour_pt_idx[prev]);
        const auto& pt2 = cluster->points.at(contour_pt_idx[next]);

        float t = static_cast<float>(id - prev) / (next - prev);

        pcl::PointXYZ interp_point;
        interp_point.x = pt1.x + t * (pt2.x - pt1.x);
        interp_point.y = pt1.y + t * (pt2.y - pt1.y);
        interp_point.z = average_z;

        filtered->points.push_back(interp_point);
        contour_pt_idx[id] = 9999;
      }
    }
  }
}
std::vector<int> LShapeDetect::sortByAngle(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud, double contour_res)
{
  double max_angle = std::atan2(input_cloud->points.at(0).y, input_cloud->points.at(0).x);
  double min_angle = std::atan2(input_cloud->points.at(0).y, input_cloud->points.at(0).x);
  for (int idx = 0; idx < input_cloud->points.size(); idx++){
    auto pt = input_cloud->points.at(idx);
    double angle = std::atan2(pt.y, pt.x);
    max_angle = (larger_angle_singlewise(angle, max_angle) == angle) ? angle : max_angle;
    min_angle = (larger_angle_singlewise(angle, min_angle) == angle) ? min_angle : angle;
  }
  double min = std::round(min_angle * (180 / M_PI) / contour_res);
  double max = std::round(max_angle * (180 / M_PI) / contour_res);
  int contour_n = (max - min > 0) ? static_cast<int>(max - min + 1) : static_cast<int>(max - min + 720 + 1);

  std::vector<int> sorted_contour_idx(contour_n, -1);
  for (int idx = 0; idx < input_cloud->points.size(); idx++){
    auto pt = input_cloud->points.at(idx);
    double angle = (max_angle - min_angle > 0) ? std::atan2(pt.y, pt.x) : ((std::atan2(pt.y, pt.x) < 0) ? std::atan2(pt.y, pt.x) + 2 * M_PI : std::atan2(pt.y, pt.x));
    double angle_deg = angle * (180 / M_PI);
    double min_angle_deg = min_angle * (180 / M_PI);
    double rel_angle = angle_deg - min_angle_deg;
    int angle_idx = static_cast<int>(std::round(rel_angle / contour_res));
    if (angle_idx >= contour_n)
      continue;
    sorted_contour_idx[angle_idx] = idx;
  }
  

  return sorted_contour_idx;
}

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> LShapeDetect::getContourV2(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusterCloud_vector, std::vector<std::vector<double>>& dbscan_obj_list, 
                                                                            const double contour_res, const double contour_z_thresh, std::vector<std::vector<double>>& dist_angle_list)
{
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> contourCloud_vector;
  for (int c_idx = 0; c_idx < clusterCloud_vector.size(); c_idx++){
    auto cluster = clusterCloud_vector.at(c_idx);

    double max_angle = std::atan2(cluster->points.at(0).y, cluster->points.at(0).x);
    double min_angle = std::atan2(cluster->points.at(0).y, cluster->points.at(0).x);
    for (int idx = 0; idx < cluster->points.size(); idx++){
      auto pt = cluster->points.at(idx);
      double angle = std::atan2(pt.y, pt.x);
      max_angle = (larger_angle_singlewise(angle, max_angle) == angle) ? angle : max_angle;
      min_angle = (larger_angle_singlewise(angle, min_angle) == angle) ? min_angle : angle;
    }

    double min = std::round(min_angle * (180 / M_PI) / contour_res);
    double max = std::round(max_angle * (180 / M_PI) / contour_res);
    int contour_n = (max - min > 0) ? static_cast<int>(max - min + 1) : static_cast<int>(max - min + 720 + 2);

    std::vector<int> contour_pt_idx(contour_n, -1);
    std::vector<int> contour_angle_check(contour_n, 0);
    std::vector<double> contour_range_check(contour_n, 0);
    for (int idx = 0; idx < cluster->points.size(); idx++){
      auto pt = cluster->points.at(idx);

      if (std::abs(pt.z - dbscan_obj_list[c_idx][2]) > contour_z_thresh)
          continue;
      double range = std::hypot(pt.y, pt.x);
      double angle = (max_angle - min_angle > 0) ? std::atan2(pt.y, pt.x) : ((std::atan2(pt.y, pt.x) < 0) ? std::atan2(pt.y, pt.x) + 2 * M_PI : std::atan2(pt.y, pt.x));
      double angle_deg = angle * (180 / M_PI);
      double min_angle_deg = min_angle * (180 / M_PI);

      
      double rel_angle = angle_deg - min_angle_deg;
      int angle_idx = static_cast<int>(std::round(rel_angle / contour_res));
      if (angle_idx >= contour_n)
        continue;

      
      if (contour_angle_check[angle_idx] == 1 && range > contour_range_check[angle_idx])
        continue;
      contour_angle_check[angle_idx] = 1;
      contour_range_check[angle_idx] = range;
      contour_pt_idx[angle_idx] = idx;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr obj_contour(new pcl::PointCloud<pcl::PointXYZ>);
    for (int id = 0; id < contour_pt_idx.size(); id++)
    {
      if (contour_pt_idx[id] != -1){
      auto pt = cluster->points.at(contour_pt_idx[id]);   

      pcl::PointXYZ contour_point;        
      contour_point.x = pt.x;
      contour_point.y = pt.y;
      contour_point.z = dbscan_obj_list[c_idx][2];
      obj_contour->points.push_back(contour_point);

      } 
    }
    double max_dist = 1.0;
    auto filtered = removeOutlier(obj_contour, max_dist, contour_pt_idx, min_angle, max_angle, contour_res);
    interpolateContour(filtered, cluster, contour_n, contour_pt_idx, dbscan_obj_list[c_idx][2]);


    // smoothing
    // for (int i = 0; i < contour_pt_idx.size(); i++){
    //   if (contour_pt_idx[i] != -1){
    //     cout << i << " ";
    //   } else if (contour_pt_idx[i] == 9999){
    //     cout << contour_pt_idx[i] << " ";
    //   }
    // }
    // cout << endl;
    double area = computeClosedAreaFromPointCloud(filtered);
    cout << "area " << area << endl;

    contourCloud_vector.push_back(filtered);
  }
  // cout << "contours : " << contourCloud_vector.size() << endl;
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
  int md_type = 1;
  if (md_type == 0)
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
  auto dbscan_obj_list = getObjectList(nonground_data, clusters, md_type);
  tc.finish("getObjectList");
  auto clusterCloud_vector = getClusters(clusters, nonground_data);
  
  tc.start("getContour");
  auto dist_ang_list = pullClusters(clusterCloud_vector);
  // auto contourCloud_vector = getContour(clusterCloud_vector, dbscan_obj_list, CONTOUR_N, CONTOUR_Z_THRH);
  
  auto contourCloud_vector = getContourV2(clusterCloud_vector, dbscan_obj_list, CONTOUR_RES, CONTOUR_Z_THRH, dist_ang_list);
  // pushClusters(contourCloud_vector, dist_ang_list);
  
  

  // visualization
  for (auto& contour : contourCloud_vector)
  {
    for (auto& pt : contour->points){
      contourCloud->points.push_back(pt);
    }
  }
  tc.start("getContour");

  
  
  

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
  
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LShapeDetect>());
  rclcpp::shutdown();
  return 0;
}

