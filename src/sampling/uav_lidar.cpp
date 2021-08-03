#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "gazebo_msgs/SetLinkState.h"
#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/Quaternion.h"
#include <ros/package.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream> 

#define NUM_SAMPLES       2500
#define NUM_SAMPLES_CYCLE 720
#define CENTRO_X 0
#define CENTRO_Y 0
#define CENTRO_Z 9

#define PI 3.14159265359

using namespace std;

  ros::ServiceClient client;
  gazebo_msgs::SetLinkState srv;
  //ros::Rate loop_rate(2);
//

sensor_msgs::LaserScan scan;
int new_reading= 0;


void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg){
	scan.header = msg->header;
	scan.angle_min = msg->angle_min;
	scan.angle_max = msg->angle_max;
	scan.angle_increment =msg->angle_increment;
	scan.time_increment = msg->time_increment;
	scan.scan_time = msg->scan_time;
	scan.range_min = msg->range_min;
	scan.range_max = msg->range_max;
	scan.ranges = msg->ranges;
	scan.intensities = msg->intensities;
	new_reading= 1;
}
sensor_msgs::LaserScan scanv;
int new_readingv= 0;


void scanVerticalCallback(const sensor_msgs::LaserScan::ConstPtr& msg){
	scanv.header = msg->header;
	scanv.angle_min = msg->angle_min;
	scanv.angle_max = msg->angle_max;
	scanv.angle_increment =msg->angle_increment;
	scanv.time_increment = msg->time_increment;
	scanv.scan_time = msg->scan_time;
	scanv.range_min = msg->range_min;
	scanv.range_max = msg->range_max;
	scanv.ranges = msg->ranges;
	scanv.intensities = msg->intensities;
	new_readingv= 1;
}


void changePosition(std::string link_name){
	tf2::Quaternion quat_tf;
	geometry_msgs::Quaternion q;
	float angle=(((float)rand()/(float)(RAND_MAX)) * PI * 2) - PI;// (((float)rand()/(float)(RAND_MAX)) * 0.035) - 0.0174533;
	quat_tf.setRPY(0,0,angle);
	  ros::Rate loop(4);

	q = tf2::toMsg(quat_tf);
	srv.request.link_state.link_name = link_name;
	srv.request.link_state.pose.position.x = (((float)rand()/(float)(RAND_MAX)) * 50) - 25 + sin(angle)*50 + CENTRO_X;
	srv.request.link_state.pose.position.y = (((float)rand()/(float)(RAND_MAX)) * 50) - 25 + cos(angle)*50 + CENTRO_X;	
	srv.request.link_state.pose.position.z = 0;
	srv.request.link_state.pose.orientation = q;

	srv.request.link_state.twist.linear.x = 0;
	srv.request.link_state.twist.linear.y = 0;
	srv.request.link_state.twist.linear.z = 0;
	srv.request.link_state.twist.angular.x = 0;
	srv.request.link_state.twist.angular.y = 0;
	srv.request.link_state.twist.angular.z = 0;

	srv.request.link_state.reference_frame = "world";
	srv.response.success=false;
	while (!client.call(srv))
	{
		loop.sleep();
    		ROS_WARN("Failed to call service change position");
		
	}
	loop.sleep();
  	ros::spinOnce(); 
	

}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "lidar_samples");
  if (argc != 3)
  {
    ROS_INFO("usage: lidar_samples_node File_name isTorre");
    return 1;
  }
  if(strcmp(argv[2], "true") && strcmp(argv[2], "false")){
    ROS_INFO("isTorre must be either true or false");
    return 1;
  }
  std::string path = ros::package::getPath("lidar_samples")+"/datasets/" + argv[1] + ".csv";
  std::string pathv = ros::package::getPath("lidar_samples")+"/datasets/" + argv[1] + "V.csv";
    	  ROS_INFO("Changing Houses\n");
  cout << "Path do arquivo: " << path << "\n";
  ofstream resultados(path);
  ofstream resultadosv(pathv);

  ros::NodeHandle n;
  ros::Subscriber sub = n.subscribe("scan", 1, scanCallback);
  ros::Subscriber sub1 = n.subscribe("scan_vertical", 1, scanVerticalCallback);
  client = n.serviceClient<gazebo_msgs::SetLinkState>("gazebo/set_link_state");
//  ros::ServiceClient client = n.serviceClient<gazebo_msgs::SetLinkState>("gazebo/set_link_state");
//  gazebo_msgs::SetLinkState srv;
  srand(time(NULL));
  float x,y,z;
  float angle_sensor;
  int isValid=0;
  int isTorre=0;
  ros::Rate loop_rate(2);

//  ros::Rate loop_rate(2);
  int i=0;
  float angle;
  tf2::Quaternion quat_tf;
  geometry_msgs::Quaternion q;
    	  ROS_INFO("Starting\n");

  while(i<NUM_SAMPLES){
    	  ROS_INFO("Changing uav\n");
	  angle= 6.29*((float)rand()/(float)(RAND_MAX));
	  x=(((float)rand()/(float)(RAND_MAX)) * 10) - 5 + sin(angle)*12 + CENTRO_X;
	  y=(((float)rand()/(float)(RAND_MAX)) * 10) - 5 + cos(angle)*12 + CENTRO_Y;
	  z=(((float)rand()/(float)(RAND_MAX)) * 12) - 6 + CENTRO_Z;
	  angle_sensor=(((float)rand()/(float)(RAND_MAX)) * PI * 2) - PI;// (((float)rand()/(float)(RAND_MAX)) * 0.035) - 0.0174533;
	  quat_tf.setRPY(0,0,angle_sensor);
	  q = tf2::toMsg(quat_tf);
	  srv.request.link_state.link_name = "lidar::base_footprint";
	  srv.request.link_state.pose.position.x = x;
	  srv.request.link_state.pose.position.y = y;
	  srv.request.link_state.pose.position.z = z;
	  srv.request.link_state.pose.orientation = q;

	  srv.request.link_state.twist.linear.x = 0;
	  srv.request.link_state.twist.linear.y = 0;
	  srv.request.link_state.twist.linear.z = 0;

	  srv.request.link_state.twist.angular.x = 0;
	  srv.request.link_state.twist.angular.y = 0;
	  srv.request.link_state.twist.angular.z = 0;

	  srv.request.link_state.reference_frame = "world";
	  srv.response.success=false;
	  while (!client.call(srv))
	  {
		loop_rate.sleep();
    		ROS_WARN("Failed to call service lidar_samples");
		
	  }

	  angle_sensor=atan2(y - CENTRO_Y, x - CENTRO_X) - PI + (((float)rand()/(float)(RAND_MAX)) * 0.07) - 0.035;//(((float)rand()/(float)(RAND_MAX)) * 0.035) - 0.0174533;
//    	  ROS_INFO("y= %f, x=%f, Angle=%f\n", y, x, angle_sensor);
	  quat_tf.setRPY(0,0,angle_sensor);
	  q = tf2::toMsg(quat_tf);

	  srv.request.link_state.link_name = "lidar_vertical::base_footprint";
	  srv.request.link_state.pose.position.x = x;
	  srv.request.link_state.pose.position.y = y;
	  srv.request.link_state.pose.position.z = z+0.1;
	  srv.request.link_state.pose.orientation = q;

	  srv.request.link_state.twist.linear.x = 0;
	  srv.request.link_state.twist.linear.y = 0;
	  srv.request.link_state.twist.linear.z = 0;

	  srv.request.link_state.twist.angular.x = 0;
	  srv.request.link_state.twist.angular.y = 0;
	  srv.request.link_state.twist.angular.z = 0;

	  srv.request.link_state.reference_frame = "world";
	  srv.response.success=false;
	  while (!client.call(srv))
	  {
		loop_rate.sleep();
    		ROS_WARN("Failed to call service lidar_samples");
		
	  }
	  loop_rate.sleep();
  	  ros::spinOnce(); 
    	  ROS_INFO("Changing Houses\n");
	  changePosition("House 1::link");
	  changePosition("House 2::link");
	  changePosition("House 3::link");
    	  ROS_INFO("Changing Trees\n");
	  changePosition("oak_tree2_clone::link");
	  changePosition("oak_tree2_clone_0::link");
	  changePosition("oak_tree2_clone_1::link");
	  changePosition("oak_tree2_clone_2::link");
	  changePosition("oak_tree2_clone_3::link");
	  changePosition("oak_tree2_clone_4::link");
	  changePosition("oak_tree2_clone_5::link");

  	  new_reading=false;
  	  new_readingv=false;
    	  ROS_INFO("Esperando msg H\n");
	  while(!new_reading){
  	  	ros::spinOnce();
	  } 
    	  ROS_INFO("Esperando msg V\n");

	  while(!new_readingv){
  	  	ros::spinOnce();
	  } 
          ROS_INFO("Msg received: %d", i);
          isValid=0;
	  for(int j=0; j < NUM_SAMPLES_CYCLE;j++){
	  	if(!isinf(scan.ranges[j])){
			isValid=1;
		}
	  }
	 for(int j=0; j < NUM_SAMPLES_CYCLE/4;j++){
	  	if(!isinf(scanv.ranges[j])){
			isValid=1;
		}
	  }


          if(isValid){
	          ROS_INFO("Amostras Validas");

		  i++;
		  resultados << CENTRO_X - x*sin(angle_sensor) << ", " << CENTRO_Y - y*cos(angle_sensor) << ", ";
		  for(int j=0; j < NUM_SAMPLES_CYCLE; j++){
			if(isinf(scan.ranges[j]))
				resultados << "100" << ", ";
			else
				resultados << scan.ranges[j] << ", ";
		  }
		  resultados << argv[2];
   		  resultados <<"\n";


		  resultadosv << CENTRO_X - x*sin(angle_sensor) << ", " << CENTRO_Y - y*cos(angle_sensor) << ", ";
		  for(int j=0; j < NUM_SAMPLES_CYCLE/4; j++){
			if(isinf(scan.ranges[j]))
				resultadosv << "100" << ", ";
			else
				resultadosv << scan.ranges[j] << ", ";
		  }
		  resultadosv << argv[2];
   		  resultadosv <<"\n";
	 }


	}
  resultados.close();
  resultadosv.close();
  return 0;
}
