#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/String.h"
#include "sensor_msgs/Image.h"
#include "geometry_msgs/Point.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <math.h>
#include <string.h>
#include <vector>
#include <string>

#include "sensor_msgs/LaserScan.h"
#include "gazebo_msgs/SetLinkState.h"
#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/Quaternion.h"
#include <ros/package.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <time.h>
#include <stdlib.h>
#include <fstream> 

#define NUM_SAMPLES       2500
#define NUM_SAMPLES_CYCLE 720
#define CENTRO_X 0
#define CENTRO_Y 0
#define CENTRO_Z 9

#define PI 3.14159265359



using namespace cv;
using namespace std;


image_transport::Publisher pubImage;
Mat image;
float altura=0;
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
}


void getImage(const sensor_msgs::Image::ConstPtr& msg){
	image =  cv_bridge::toCvShare(msg, "bgr8")->image;
	new_reading= 1;
}


void locate_base()
{
	//imread(imageName, IMREAD_COLOR ); // Read the file
	if( image.empty() )                      // Check for invalid input
	{
        	cout <<  "Could not open or find the image" << std::endl ;
        	return;
    	}

//	cv::resize(image, img2, Size(image.cols, image.rows),0,0,CV_INTER_LINEAR);

//    	namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
//    	imshow( "Display window", img2 );                // Show our image inside it.
//    	waitKey(0); // Wait for a keystroke in the window


}




int main(int argc, char** argv)
{
    ros::init(argc, argv, "sampleImage");
  if (argc != 3)
  {
    ROS_INFO("usage: lidar_samples_node File_name isTorre");
    return 1;
  }
  if(strcmp(argv[2], "true") && strcmp(argv[2], "false")){
    ROS_INFO("isTorre must be either true or false");
    return 1;
  }


    ros::NodeHandle n;

    image_transport::ImageTransport it(n);
//    pub = it.advertise("/image_raw", 1);



/*     ros::ServiceClient uavGoto = n.serviceClient<mrs_msgs::Vec4>("/uav1/control_manager/goto");
    ros::ServiceClient uavLand = n.serviceClient<std_srvs::Trigger>("/uav1/uav_manager/land");
	ros::ServiceClient uavTakeOff = n.serviceClient<std_srvs::Trigger>("/uav1/uav_manager/takeoff");
	ros::ServiceClient uavArming = n.serviceClient<mavros_msgs::CommandBool>("/uav1/mavros/cmd/arming");


    ros::Subscriber baro = n.subscribe("/uav1/odometry/altitude", 1, ler_altura);
*/

  ros::Subscriber sub = n.subscribe("/iris/camera/image_raw", 1, getImage);
  string path = ros::package::getPath("lidar_samples")+"/datasets/" + argv[1];
  cout << "Path do arquivo: " << path << "\n";
  ofstream resultados(path);
  string imgFile;
  ros::Subscriber subScan = n.subscribe("scan", 1, scanCallback);
  ros::ServiceClient client = n.serviceClient<gazebo_msgs::SetLinkState>("gazebo/set_link_state");
  gazebo_msgs::SetLinkState srv;
  srand(time(NULL));
  float x,y,z;
  float angle_sensor;
  int isValid=0;
  int isTorre=0;
  ros::Rate loop_rate(2);
  int i=0;
  float angle;
  tf2::Quaternion quat_tf;
  geometry_msgs::Quaternion q;




  while(i<NUM_SAMPLES){
	  angle= 6.29*((float)rand()/(float)(RAND_MAX));
	  x=(((float)rand()/(float)(RAND_MAX)) * 10) - 5 + sin(angle)*12 + CENTRO_X;
	  y=(((float)rand()/(float)(RAND_MAX)) * 10) - 5 + cos(angle)*12 + CENTRO_Y;
	  z=(((float)rand()/(float)(RAND_MAX)) * 12) - 6 + CENTRO_Z;
	  angle_sensor= atan2(y - CENTRO_Y, x - CENTRO_X) - PI +
			//(((float)rand()/(float)(RAND_MAX)) * PI * 2) - PI;
			 (((float)rand()/(float)(RAND_MAX)) * 0.07) - 0.035;
	  quat_tf.setRPY(0,0,angle_sensor);
	  q = tf2::toMsg(quat_tf);
	  srv.request.link_state.link_name = "cam_link";
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
	  loop_rate.sleep();
  	  ros::spinOnce(); 
  	  new_reading=false;
    	  ROS_INFO("Esperando msg\n");
	  while(!new_reading){
  	  	ros::spinOnce();

	  } 
          ROS_INFO("Msg received: %d", i);
          isValid=0;
	  for(int j=0; j < NUM_SAMPLES_CYCLE;j++){
	  	if(!isinf(scan.ranges[j])){
			isValid=1;
		}
	  }
          if(isValid){
	          ROS_INFO("Amostra Valida");
		  imgFile = to_string(CENTRO_X - x*sin(angle_sensor)) + ", " + to_string(CENTRO_Y - y*cos(angle_sensor));
		  imwrite(path + imgFile + ".jpg",image);
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
	 }
	}
  resultados.close();

    return 0;
}





