#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/empty.hpp"
#include <termios.h>
#include <unistd.h>
#include <iostream>

class FrameTriggerNode : public rclcpp::Node
{
public:
  FrameTriggerNode() : Node("frame_trigger_node")
  {
    pub_ = create_publisher<std_msgs::msg::Empty>("/next_frame_trigger", 10);
    RCLCPP_INFO(this->get_logger(), "Press SPACE to publish next frame...");
    enableKeyboardInput();
  }

private:
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr pub_;

  void enableKeyboardInput()
  {
    struct termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);           // save old settings
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);         // disable buffering
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);  // apply new settings

    while (rclcpp::ok()) {
      char c = getchar();
      if (c == ' ') {
        std_msgs::msg::Empty msg;
        pub_->publish(msg);
        RCLCPP_INFO(this->get_logger(), "Published next frame trigger.");
      } else if (c == 'q') {
        RCLCPP_INFO(this->get_logger(), "Quitting...");
        break;
      }
    }

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);  // restore settings
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FrameTriggerNode>());
  rclcpp::shutdown();
  return 0;
}
