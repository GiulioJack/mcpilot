# SPDX-License-Identifier: AGPL-3.0-or-later
#include "ros/callback_queue.h"
#include "ros/ros.h"
#include "ros/subscribe_options.h"
#include "std_msgs/Float32.h"
#include <functional>
#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <ignition/math.hh>
#include <math.h>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>
#include <ignition/math/Vector3.hh>
#include <cmath>
#include <thread>

namespace gazebo {
class FluidResitance : public ModelPlugin {
public:
  void Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf) {

    if (_sdf->HasElement("fluid_resitanceTopicName")) {
      this->fluid_resitanceTopicName =
          _sdf->Get<std::string>("fluid_resitanceTopicName");
    } else {
      ROS_WARN("No fluid_resitance Topic Given name given, setting default name %s",
               this->fluid_resitanceTopicName.c_str());
    }

    if (_sdf->HasElement("reynolds_multTopicName")) {
      this->reynolds_multTopicName =
          _sdf->Get<std::string>("reynolds_multTopicName");
    } else {
      ROS_WARN("No reynolds_mult Topic Given name given, setting default name %s",
               this->reynolds_multTopicName.c_str());
    }

    if (_sdf->HasElement("NameLinkToApplyResitance")) {
      this->NameLinkToApplyResitance =
          _sdf->Get<std::string>("NameLinkToApplyResitance");
    } else {
      ROS_WARN("No NameLinkToApplyResitance Given name given, setting default "
               "name %s",
               this->NameLinkToApplyResitance.c_str());
    }

    if (_sdf->HasElement("rate")) {
      this->rate = _sdf->Get<double>("rate");
    } else {
      ROS_WARN("No rate Given name given, setting default "
               "name %f",
               this->rate);
    }

    if (_sdf->HasElement("fluid_resitance_mult")) {
      this->fluid_resitance_mult =
          _sdf->Get<double>("fluid_resitance_mult");
    }

    if (_sdf->HasElement("reynolds_mult")) {
      this->reynolds_mult = _sdf->Get<double>("reynolds_mult");
    }

    // Store the pointer to the model
    this->model = _parent;
    this->world = this->model->GetWorld();
    this->link_to_apply_resitance =
        this->model->GetLink(this->NameLinkToApplyResitance);

    // Listen to the update event. This event is broadcast every
    // simulation iteration.
    this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&FluidResitance::OnUpdate, this));

    // Create a topic name
    // std::string fluid_resitance_mult_topicName = "/fluid_resitance_mult";

    // Initialize ros, if it has not already bee initialized.
    if (!ros::isInitialized()) {
      int argc = 0;
      char **argv = NULL;
      ros::init(argc, argv, "model_mas_controler_rosnode",
                ros::init_options::NoSigintHandler);
    }

    // Create our ROS node. This acts in a similar manner to
    // the Gazebo node
    this->rosNode.reset(new ros::NodeHandle("model_mas_controler_rosnode"));

#if (GAZEBO_MAJOR_VERSION >= 8)
    this->last_time = this->world->SimTime().Double();
#else
    this->last_time = this->world->GetSimTime().Double();
#endif

    // Freq
    ros::SubscribeOptions so = ros::SubscribeOptions::create<std_msgs::Float32>(
        this->fluid_resitanceTopicName, 1,
        boost::bind(&FluidResitance::OnRosMsg, this, _1), ros::VoidPtr(),
        &this->rosQueue);
    this->fluidResitanceSub = this->rosNode->subscribe(so);

    ros::SubscribeOptions so_reynolds = ros::SubscribeOptions::create<std_msgs::Float32>(
        this->reynolds_multTopicName, 1,
        boost::bind(&FluidResitance::OnRosMsgReynolds, this, _1), ros::VoidPtr(),
        &this->rosQueue);
    this->reynoldsMultSub = this->rosNode->subscribe(so_reynolds);


    // Spin up the queue helper thread.
    this->rosQueueThread =
        std::thread(std::bind(&FluidResitance::QueueThread, this));

    ROS_INFO("Loaded FluidResitance Plugin with parent...%s, With Fluid "
             "Resitance = %f "
             "Reynolds coeff = %f "
             "Rate = %.3f Hz "
             "Started ",
             this->model->GetName().c_str(), this->fluid_resitance_mult,
             this->reynolds_mult, this->rate);
  }

  // Called by the world update start event
public:
    float get_drag_coeff(float Re){
        float phi_1 = pow(24/Re, 10) + pow(21 * pow(Re, -0.67), 10) + pow(4 * pow(Re,-0.33), 10) + pow(0.4, 10);
        float phi_2 = 1 / (pow(0.148 * pow(Re, 0.11), -10) + pow(0.5, -10));
        float phi_3 = pow(1.57 * pow(10, 8) * pow(Re, -1.625), 10);
        float phi_4 = 1 / (pow(6 * pow(10, -17) * pow(Re, 2.63), -10) + pow(0.2, -10));

        return pow(1/(1/(phi_1+phi_2) + 1/phi_3) + phi_4, 0.1);
    }

  void OnUpdate() {
    // Use double-precision relative time and an accumulator. The previous
    // float absolute time silently reduced the force application rate as the
    // Gazebo clock grew.
#if (GAZEBO_MAJOR_VERSION >= 8)
    const double current_time = this->world->SimTime().Double();
#else
    const double current_time = this->world->GetSimTime().Double();
#endif
    const double dt = current_time - this->last_time;
    this->last_time = current_time;

    if (dt < 0.0) {
      // The world clock was reset.
      this->elapsed_time = 0.0;
      return;
    }

    if (this->rate <= 0.0) {
      this->ApplyResitance();
      return;
    }

    const double period = 1.0 / this->rate;
    this->elapsed_time += dt;
    if (this->elapsed_time + 1e-12 < period) {
      return;
    }

    // Preserve the remainder to avoid drift when the physics and drag
    // periods are not integer multiples of one another.
    const unsigned int applications =
        static_cast<unsigned int>(std::floor(this->elapsed_time / period));
    this->elapsed_time -= applications * period;
    for (unsigned int i = 0; i < applications; ++i) {
      this->ApplyResitance();
    }
  }

public:
  void SetResitance(const double &_force) {
    this->fluid_resitance_mult = _force;
    ROS_WARN("model_fluid_resitance_mult changed >> %f",
             this->fluid_resitance_mult);
  }
    void SetReynoldsMult(const double &_mult) {
    this->reynolds_mult = _mult;
    ROS_WARN("model_reynolds_mult changed >> %f",
             this->reynolds_mult);
  }

  void UpdateLinearVel() {
#if (GAZEBO_MAJOR_VERSION >= 8)
    this->now_lin_vel = this->model->RelativeLinearVel();
#else
    this->now_lin_vel = this->model->GetRelativeLinearVel();
#endif
  }

  void ApplyResitance() {

    this->UpdateLinearVel();

    float drag_coeff = 0.0;

#if (GAZEBO_MAJOR_VERSION >= 8)
    ignition::math::Vector3d force, torque;
//    ROS_INFO("%s linearSpeed = [%f,%f,%f] ", this->model->GetName().c_str(), this->now_lin_vel.X(), this->now_lin_vel.Y(), this->now_lin_vel.Z());

    float sq_vel_norm = pow(this->now_lin_vel.X(), 2) + pow(this->now_lin_vel.Y(), 2) + pow(this->now_lin_vel.Z(), 2);
    float vel_norm = sqrt(sq_vel_norm);


    float reynolds_num = this->reynolds_mult * vel_norm;
    // The drag coefficient
    float cd = this->get_drag_coeff(reynolds_num); // cd = phi(Re);

/*
    if(cd > 20)
        ROS_WARN("%s: CD=%f, Re=%f, 0.5 ro A = %f, D/v = %f", this->model->GetName().c_str(), cd, reynolds_num, this->fluid_resitance_mult, this->reynolds_mult);
    else
        ROS_INFO("%s: CD=%f, Re=%f, 0.5 ro A = %f, D/v = %f", this->model->GetName().c_str(), cd, reynolds_num, this->fluid_resitance_mult, this->reynolds_mult);
*/

    // Limit the coeficient to sensible values
    cd = std::min(cd, static_cast<float>(20.0));

    force.X(-1.0 * cd * this->fluid_resitance_mult * vel_norm * this->now_lin_vel.X());
    force.Y(-1.0 * cd * this->fluid_resitance_mult * vel_norm * this->now_lin_vel.Y());
    force.Z(-1.0 * cd * this->fluid_resitance_mult * vel_norm * this->now_lin_vel.Z());

#else
    math::Vector3 force, torque;

//    ROS_INFO("%s linearSpeed = [%f,%f,%f] ", this->model->GetName().c_str(), this->now_lin_vel.x, this->now_lin_vel.y, this->now_lin_vel.z);


    float sq_vel_norm = pow(this->now_lin_vel.x, 2) + pow(this->now_lin_vel.y, 2) + pow(this->now_lin_vel.z, 2);
    float vel_norm = sqrt(sq_vel_norm);

    // The drag coefficient
    float cd = this->get_drag_coeff(this->reynolds_mult * vel_norm); // cd = phi(Re);

    force.x = -1.0 * cd * this->fluid_resitance_mult * vel_norm * this->now_lin_vel.x;
    force.y = -1.0 * cd * this->fluid_resitance_mult * vel_norm * this->now_lin_vel.y;
    force.z = -1.0 * cd * this->fluid_resitance_mult * vel_norm * this->now_lin_vel.z;
#endif


    // Changing the mass
    this->link_to_apply_resitance->AddRelativeForce(force);
#if (GAZEBO_MAJOR_VERSION >= 8)
    this->link_to_apply_resitance->AddRelativeTorque(
        torque -
        this->link_to_apply_resitance->GetInertial()->CoG().Cross(force));
//        ROS_INFO("%s fluidResitance = [%f,%f,%f] ",this->model->GetName().c_str(), force.X(), force.Y(), force.Z());
#else
    this->link_to_apply_resitance->AddRelativeTorque(
        torque -
        this->link_to_apply_resitance->GetInertial()->GetCoG().Cross(force));
//        ROS_INFO("%s fluidResitance = [%f,%f,%f] ", this->model->GetName().c_str(), force.x, force.y, force.z);
#endif


  }

public:
  void OnRosMsg(const std_msgs::Float32ConstPtr &_msg) {
    this->SetResitance(_msg->data);
  }
  void OnRosMsgReynolds(const std_msgs::Float32ConstPtr &_msg) {
    this->SetReynoldsMult(_msg->data);
  }

  /// \brief ROS helper function that processes messages
private:
  void QueueThread() {
    static const double timeout = 0.01;
    while (this->rosNode->ok()) {
      this->rosQueue.callAvailable(ros::WallDuration(timeout));
    }
  }

  // Pointer to the model
private:
  physics::ModelPtr model;

  // Pointer to the update event connection
private:
  event::ConnectionPtr updateConnection;

  // Mas of model
  double fluid_resitance_mult = 0.0009388203894374999;
  double reynolds_mult = 2925.1700680272106; // air at 15°C

  /// \brief A node use for ROS transport
private:
  std::unique_ptr<ros::NodeHandle> rosNode;

  /// \brief A ROS subscriber
private:
  ros::Subscriber fluidResitanceSub;
  ros::Subscriber reynoldsMultSub;
  /// \brief A ROS callbackqueue that helps process messages
private:
  ros::CallbackQueue rosQueue;
  /// \brief A thread the keeps running the rosQueue
private:
  std::thread rosQueueThread;

  /// \brief A ROS subscriber
private:
  physics::LinkPtr link_to_apply_resitance;

private:
  std::string fluid_resitanceTopicName = "fluid_resitance";
  std::string reynolds_multTopicName = "reynolds_mult";

private:
  std::string NameLinkToApplyResitance = "base_link";

private:
#if (GAZEBO_MAJOR_VERSION >= 8)
  ignition::math::Vector3d now_lin_vel;
#else
  math::Vector3 now_lin_vel;
#endif


private:
  double rate = 1.0;

private:
  double last_time = 0.0;
  double elapsed_time = 0.0;

private:
  /// \brief The parent World
  physics::WorldPtr world;
};

// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(FluidResitance)
} // namespace gazebo
