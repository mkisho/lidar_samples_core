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

enum state_machine{
GO_TO_POINT,
CHECK_FOR_TOWER,
ALIGN_WITH_TOWER,
APPROACH_TOWER
}




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

std_msgs::Header trajectoryHeader;

void set_next_point(Point3f goal, geometry_msgs::Point oldPos, mrs_msgs::TrajectoryReference *traj){
//	goal.x;
//	goal->request.goal[1]=y;
//	goal->request.goal[2]=z;
//	goal->request.goal[3]=w;
//	goal->response.success=false;

	if(trajectoryHeader.seq==0){
		trajectoryHeader.seq=0;
	}
	else{
		trajectoryHeader.seq+=1;
	}
	trajectoryHeader.stamp=ros::Time::now();
	trajectoryHeader.frame_id="";
	traj->header=trajectoryHeader;
	traj->use_heading=false;
	traj->fly_now=true;
	traj->loop=false;
	traj->dt=0.2;

	vector<mrs_msgs::Reference> points;
	mrs_msgs::Reference newPoint;
	double	distance=9999;
	newPoint.position.x=oldPos.x;
	newPoint.position.y=oldPos.y;
	newPoint.position.z=oldPos.z;
	
	while(distance>0.05){
		distance=sqrt(pow(goal.x-newPoint.position.x,2)+pow(goal.y-newPoint.position.y,2)+pow(goal.z-newPoint.position.z,2));
		newPoint.position.x=aproxima(goal.x, newPoint.position.x);
		newPoint.position.y=aproxima(goal.y, newPoint.position.y);
		newPoint.position.z=aproxima(goal.z, newPoint.position.z);
		newPoint.heading=0;
		points.insert(points.end(), newPoint);
	}
	traj->points=points;
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
    	ros::ServiceClient uavGoto = n.serviceClient<mrs_msgs::TrajectoryReferenceSrv>("/uav1/control_manager/trajectory_reference");
	mrs_msgs::TrajectoryReferenceSrv trajectorySrv;
	mrs_msgs::TrajectoryReference next_trajectory;
	ros::Subscriber sub = n.subscribe("/iris/camera/image_raw", 1, getImage);
	ros::Subscriber subScanH = n.subscribe("/uav1/rplidar/scan", 1, getImage);
	ros::Subscriber subScanV = n.subscribe("/uav1/rplidar_vertical/scan", 1, getImage);
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

	while(1){
		switch(state){
			case GO_TO_POINT:
				
			break;
			case CHECK_FOR_TOWER:
				
			break;
			case ALIGN_WITH_TOWER:
			break;
			case APPORACH_TOWER:
			break;

		}
	}
	return 0;
}





