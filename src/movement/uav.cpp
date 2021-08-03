#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/Bool.h"
#include "std_msgs/String.h"
#include "sensor_msgs/Image.h"
#include "geometry_msgs/Point.h"
#include "mrs_msgs/TrajectoryReferenceSrv.h"
#include "mrs_msgs/Reference.h"
#include "mrs_msgs/Vec4.h"
#include "nav_msgs/Odometry.h"

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

Point3f gpsOdom;

bool isVertTower = false;
bool isHorizTower= false;


typedef enum state_machine{
GO_TO_POINT,
ONGOING_TRAJECTORY,
CHECK_FOR_TOWER,
ALIGN_WITH_TOWER,
APPROACH_TOWER,
INSPECT

} State_machine;


void scanHoriz(const std_msgs::Bool::ConstPtr& msg){
	isHorizTower= msg->data;
}

void scanVert(const std_msgs::Bool::ConstPtr& msg){
	isVertTower= msg->data;
}

void odomDroneCallback(const nav_msgs::Odometry::ConstPtr& msg){
	gpsOdom.x= msg->pose.pose.position.x;
	gpsOdom.y= msg->pose.pose.position.y;
	gpsOdom.z= msg->pose.pose.position.z;
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


float  aproxima(float objetivo, float origem){
	float diff = objetivo-origem;
	if(diff==0){
		return objetivo;
	}
	else if (diff>0){
		if(diff>0.2){
			origem+=0.2;
			return origem;
		}
		return objetivo;
	}
	else{
		if(diff<-0.2){
			origem-=0.2;
			return origem;
		}
		return objetivo;
	}
}

void set_next_point_relat(mrs_msgs::Vec4 *point, float x, float y, float z, float w){
	point->request.goal[0]=x;
	point->request.goal[1]=y;
	point->request.goal[2]=z;
	point->request.goal[3]=w;
	point->response.success=false;
}

void set_next_point(geometry_msgs::Point goal, geometry_msgs::Point oldPos, mrs_msgs::TrajectoryReference *traj){
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
	if (argc != 4)
	{
		ROS_INFO("usage: rosrun lidar_samples_node X Y Z");
		return 1;
	}
	geometry_msgs::Point goal;
	char *ptr;
	goal.x=strtod(argv[1], &ptr);
	goal.y=strtod(argv[2], &ptr);
	goal.z=strtod(argv[3], &ptr);

	printf("goal= %f, %f, %f\n",goal.x,goal.y,goal.z);

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
    	ros::ServiceClient uavGotoRelat = n.serviceClient<mrs_msgs::Vec4>("/uav1/control_manager/goto_relative");
	mrs_msgs::TrajectoryReferenceSrv trajectorySrv;
	mrs_msgs::TrajectoryReference next_trajectory;
	ros::Subscriber sub = n.subscribe("/iris/camera/image_raw", 1, getImage);
	ros::Subscriber subScanH = n.subscribe("/horiz", 1, scanHoriz);
	ros::Subscriber subScanV = n.subscribe("/vert", 1, scanVert);
//	ros::Subscriber subScanH = n.subscribe("/uav1/rplidar/scan", 1, scanHoriz);
//	ros::Subscriber subScanV = n.subscribe("/uav1/rplidar_vertical/scan", 1, scanVert);
        ros::Subscriber subOdom = n.subscribe("/uav1/odometry/odom_gps", 1000, odomDroneCallback);
	string path = ros::package::getPath("lidar_samples")+"/datasets/" + argv[1];
	ofstream resultados(path);
	string imgFile;
	ros::Subscriber subScan = n.subscribe("scan", 1, scanCallback);
	ros::ServiceClient client = n.serviceClient<gazebo_msgs::SetLinkState>("gazebo/set_link_state");
	mrs_msgs::Vec4 girar;
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
	geometry_msgs::Point oldPos;
	geometry_msgs::Point point;
	cout << "waiting for gps \n";

	ros::topic::waitForMessage<nav_msgs::Odometry>("/uav1/odometry/odom_gps");
	oldPos.x=0;
	oldPos.y=0;
	oldPos.z=0;
	State_machine state=GO_TO_POINT;
	float distanceT;
	cout << "iniciando máquina de estados \n";

	while(1){
		switch(state){
			case GO_TO_POINT:
				cout << "setting point\n";
				set_next_point(point, oldPos, &next_trajectory);
				trajectorySrv.request.trajectory=next_trajectory;
				trajectorySrv.response.success=false;
				while(!trajectorySrv.response.success){
					uavGoto.call(trajectorySrv)/
					sleep(1);
				}
				state=ONGOING_TRAJECTORY;

			break;
			case ONGOING_TRAJECTORY:
				cout <<"ongoing route\n";
				distanceT= sqrt(pow(point.x-gpsOdom.x,2)+pow(point.y-gpsOdom.y,2)+pow(point.z-gpsOdom.z,2));
				if(distanceT < 0.5){
					cout <<"goal reached\n";
					state=INSPECT;
				}

			break;
			case INSPECT:
				isHorizTower=false;
				ros::topic::waitForMessage<std_msgs::Bool>("/horiz");
				
				if(isHorizTower){
					isVertTower=false;
					ros::topic::waitForMessage<std_msgs::Bool>("/vert");
					if(isVertTower){
						set_next_point_relat(&girar, 0.1, 0, 0, 0);
						while (girar.response.success == false)
						{
							uavGotoRelat.call(girar);
							sleep(1);
						}
					}
					else{
						
						set_next_point_relat(&girar, 0, 0, 0 , -0.10);
						while (girar.response.success == false)
						{
							uavGotoRelat.call(girar);
							sleep(1);
						}
					}
				}
				
			break;
			case ALIGN_WITH_TOWER:
			break;
			case APPROACH_TOWER:
			break;
		}
	}
	return 0;
}





