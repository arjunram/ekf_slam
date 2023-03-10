#include <basics.h>
#include <iostream>
#include "ros/ros.h"
#include <Eigen/Dense>
#include "apriltag_ros/common_functions.h"
#include "sensor_msgs/Imu.h"
#define _USE_MATH_DEFINES
using Eigen::MatrixXd;
using Eigen::VectorXd;


using namespace std;
using namespace ros;
using namespace Eigen;

double angleWrap(double input)
{
  //ROS_INFO_STREAM("AngleWrap: "<<input);
  if(input>M_PI)
    return (input - 2*M_PI);
  else if(input<-M_PI)
    return (input + 2*M_PI);
  else
    return input;
}


class EkfSlamNode
{
public:
  EkfSlamNode()
  {
    imu_sub_ = n_.subscribe("imu/data_raw", 1000, &EkfSlamNode::imuCallback, this);
    apriltag_sub_ = n_.subscribe("tag_detections", 10, &EkfSlamNode::apriltagCallback, this);
    pose_pub_ = n_.advertise<geometry_msgs::PoseWithCovarianceStamped>("robot_pose", 1000);
    initialize_();
  }

  void initialize_()
  {
    num_state_ = 5;
    num_features_ = 0;
    state_ = VectorXd::Zero(num_state_);
    
    global_state_ = VectorXd::Zero(num_state_);
    cov_ = MatrixXd::Zero(num_state_,num_state_);
    F = MatrixXd::Zero(num_state_,num_state_);
    B = MatrixXd::Zero(num_state_,3);
    last_imu_ = ros::Time::now();
    ROS_INFO_STREAM(last_imu_.toSec());
    imu_dt_ = 0;
    //state_.head(3) = Matrix<double,3,1>(-0.85, 0.2, 0);
    state_.head(3) = Matrix<double,3,1>(0, 0, 0);
    Q_ << 0.1, 0,      0,
          0,      0.1, 0,
          0,      0,      0.1;
    R_ << 0.05, 0,
          0,    0.1;
    bx_ = -0.1391;
    by_ = 0.2746;
    ROS_INFO_STREAM("Initialized");
  }

  void apriltagCallback(const apriltag_ros::AprilTagDetectionArray::ConstPtr& tag)
  {
    geometry_msgs::Pose tag_pose;
    try
    {
      for (int i = 0;i<tag->detections.size();i++)
      {
        tag_pose = tag->detections[i].pose.pose.pose;
        update_state_(tag->detections[i].id[0], tag_pose);
      }
      

    }
    catch (std::exception& e)
    {
      std::cerr << "Exception caught : " << e.what() << std::endl;
    }
  }

  void imuCallback(const sensor_msgs::Imu::ConstPtr& msg)
  {
    try
    {
      update_dt_(*msg);
      propagate_state_(*msg);
    }
    catch (std::exception& e)
    {
      std::cerr << "Exception caught : " << e.what() << std::endl;
    }
    
  }

  void publish_state();

private:
  ros::Subscriber apriltag_sub_;
  ros::Subscriber imu_sub_;
  ros::Publisher pose_pub_;
  ros::NodeHandle n_;
  ros::Time last_imu_;
  double imu_dt_;
  double bx_,by_;
  int num_state_;
  int num_features_;
  Eigen::VectorXd state_, global_state_;
  Eigen::MatrixXd cov_, F, B;
  Eigen::Matrix3d Q_;
  Eigen::Matrix2d R_;
  std::vector<int> features_seen_;
  void update_dt_(sensor_msgs::Imu);
  void propagate_state_(sensor_msgs::Imu);
  void update_state_(int, geometry_msgs::Pose);
  void expand_state_();
  
};

void EkfSlamNode::update_dt_(sensor_msgs::Imu msg)
{
  ros::Duration dt = msg.header.stamp - last_imu_;
  last_imu_ = msg.header.stamp;
  imu_dt_ = dt.toSec();
  if (imu_dt_>0.1 || imu_dt_<0) imu_dt_ = 0;
  
}

void EkfSlamNode::propagate_state_(sensor_msgs::Imu msg)
{
  //ROS_INFO_STREAM("Prop");
  double ax = msg.linear_acceleration.x - bx_;
  double ay = msg.linear_acceleration.y - by_; 
  double w = msg.angular_velocity.z;
  //imu_dt_ = 0.01;
  Eigen::Vector3d u(ax,ay,w);
  double theta = state_(2);
  //theta = 0;
  //ROS_INFO_STREAM("dt="<<imu_dt_<<" ax="<<ax<<" ay="<<ay<<" w="<<w<<" theta="<<theta);
  Eigen::Matrix2d ThetaLocalToGlobalDCM, ThetaGlobalToLocalDCM, wLocalToGlobalDCM, wGlobalToLocalDCM;
  ThetaLocalToGlobalDCM << cos(theta), -sin(theta), sin(theta), cos(theta);
  ThetaGlobalToLocalDCM = ThetaLocalToGlobalDCM.transpose();
  wLocalToGlobalDCM << cos(w*imu_dt_), -sin(w*imu_dt_), sin(w*imu_dt_), cos(w*imu_dt_);
  wGlobalToLocalDCM = wLocalToGlobalDCM.transpose();
  F.block(0,0,5,5) = MatrixXd::Identity(5,5);
  F.block(0,3,2,2) = imu_dt_*ThetaLocalToGlobalDCM;
  B.block(0,0,2,2) = 0.5*imu_dt_*imu_dt_*ThetaLocalToGlobalDCM;
  B(2,2) = imu_dt_;
  B.block(3,0,2,2) = imu_dt_*ThetaLocalToGlobalDCM;
  for(int i = 0;i<num_features_;i++)
  {
    F.block(5+i*2,3,2,2) = imu_dt_*wLocalToGlobalDCM;
    F.block(5+i*2,5+i*2,2,2) = wGlobalToLocalDCM;
    B.block(5+i*2,0,2,2) = 0.5*imu_dt_*imu_dt_*wLocalToGlobalDCM;
  }
  
  state_ = F*state_ + B*u;
  cov_ = F*cov_*F.transpose() + B*Q_*B.transpose();
  //ROS_INFO_STREAM(imu_dt_<<" "<<u.transpose()<<endl<<B*u<<endl<<state_);
  //ROS_INFO_STREAM("XY:"<<state_(1)<<" "<<state_(2)<<" "<<state_(3)<<" "<<state_(4)<<" "<<state_(5)<<endl);
}

void EkfSlamNode::expand_state_()
{
  VectorXd tempVec = VectorXd::Zero(num_state_+2);
  tempVec.head(num_state_) = state_;
  state_ = tempVec;
  tempVec.head(num_state_) = global_state_;
  global_state_ = tempVec;
  MatrixXd temp = MatrixXd::Zero(num_state_+2,num_state_+2);
  temp.block(0,0,num_state_,num_state_) = cov_;
  cov_ = temp;
  temp.block(0,0,num_state_,num_state_) = F;
  F = temp;
  num_state_+= 2;
  num_features_ += 1;
  B = MatrixXd::Zero(num_state_,3);
  //ROS_INFO_STREAM("Expanded "<<num_state_<<" "<<num_features_);
}

void EkfSlamNode::update_state_(int id, geometry_msgs::Pose tag_pose)
{
  auto iter = find(features_seen_.begin(),features_seen_.end(),id);
  double x = tag_pose.position.x;
  double z = tag_pose.position.z;
  double r = sqrt(x*x + z*z);
  double theta = atan2(x,z);
  if(iter==features_seen_.end())
  {
    ROS_INFO_STREAM("New feature"<<id);
    features_seen_.push_back(id);
    expand_state_();
    state_(3+num_features_*2) = r*sin(theta);
    state_(4+num_features_*2) = r*cos(theta);
    ROS_INFO_STREAM(x<<" "<<z<<" "<<r<<" "<<theta<<" state\n "<<state_);
    //ROS_INFO_STREAM("Added");
  }
  else
  {

    int index = iter - features_seen_.begin();
    double xFeature = state_(5+index*2);
    double yFeature = state_(6+index*2);
    double range = sqrt(xFeature*xFeature + yFeature*yFeature);
    double angle = atan2(xFeature,yFeature);
    //ROS_INFO_STREAM("index = "<<index<<" "<<xFeature<<" "<<yFeature<<" "<<range<<" "<<angle);
    MatrixXd jH = MatrixXd::Zero(2,num_state_);
    jH(0,5+index*2) = xFeature/range;
    jH(0,6+index*2) = yFeature/range;
    jH(1,5+index*2) = -yFeature/(range*range);
    jH(1,6+index*2) = xFeature/(range*range);
    Vector2d residual(r - range,angleWrap(angle-theta));
    MatrixXd S(jH*cov_*jH.transpose() + R_);
    MatrixXd K(cov_*jH.transpose()*S.inverse());
    VectorXd innovation(K*residual);
    //ROS_INFO_STREAM("Innov = "<<residual.transpose()<<" "<<innovation.transpose());
    state_ = state_ + innovation;
    state_(2) = angleWrap(state_(2));
    MatrixXd I_KH(MatrixXd::Identity(num_state_,num_state_) - K*jH);
    cov_ = I_KH*cov_*I_KH.transpose() + K*R_*K.transpose();
  }
}

void EkfSlamNode::publish_state()
{
  geometry_msgs::PoseWithCovarianceStamped pose;
  std_msgs::Header h;
  h.stamp = ros::Time::now();
  pose.header = h;
  for(int i = 0;i<3;i++)
  {

    for(int j = 0;j<3;j++)
    {
      pose.pose.covariance[i+3*j] = cov_(i,j);
      //cout<<cov_(i,j)<<" ";
    }
    //cout<<endl;
  }
  pose.pose.pose.position.x = state_(0);
  pose.pose.pose.position.y = state_(1);
  pose.pose.pose.position.z = state_(2);
  pose_pub_.publish(pose);
}

int main(int argc, char **argv)
{
  // Initialize ROS

  ros::init(argc, argv, "ekf_slam");
  EkfSlamNode obj;
  ROS_INFO("Running");
  //ros::spin();
  ros:Rate loop_rate(10);
  while(ros::ok())
  {
    obj.publish_state();
    ros::spinOnce();
    loop_rate.sleep();
  } 
  return 0;
}