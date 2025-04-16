from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    frame_id_arg = DeclareLaunchArgument(
        'frame_id',
        default_value='0',
        description='Frame index to label and publish'
    )

    frame_id = LaunchConfiguration('frame_id')

    return LaunchDescription([
        frame_id_arg,

        Node(
            package='rosbag_util',
            executable='bin_pose_publisher',
            name='bin_pose_publisher',
            parameters=[{'frame_id': frame_id}]
        ),

        Node(
            package='rosbag_util',
            executable='rosbag_box_labeler',
            name='rosbag_box_labeler',
            parameters=[{'frame_id': frame_id}]
        )
    ])
