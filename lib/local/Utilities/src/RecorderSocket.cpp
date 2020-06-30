#include "stdafx_ut.h"
#include "RecorderSocket.h"

using namespace Utilities;

bool RecorderSocket::Open(int port, bool is_sequence, bool output_2D_landmarks, bool output_3D_landmarks, bool output_model_params, bool output_pose, bool output_AUs, bool output_gaze,
	int num_face_landmarks, int num_model_modes, int num_eye_landmarks, const std::vector<std::string>& au_names_class, const std::vector<std::string>& au_names_reg)
{
	this->port = port;
	this->is_sequence = is_sequence;

	this->output_2D_landmarks = output_2D_landmarks;
	this->output_3D_landmarks = output_3D_landmarks;
	this->output_model_params = output_model_params;
	this->output_pose = output_pose;
	this->output_AUs = output_AUs;
	this->output_gaze = output_gaze;

	// Open connection
	zmq::context_t context(1);
	this->socket = new zmq::socket_t(context, ZMQ_PUB);
	this->socket->bind("tcp://*:" + std::to_string(this->port));
}

bool RecorderSocket::IsOpen()
{
	return this->socket != NULL;
}

void RecorderSocket::Close()
{
	if (this->IsOpen())
		this->socket->close();
}

void RecorderSocket::WriteMessage(int face_id, int frame_num, double time_stamp, bool landmark_detection_success, double landmark_confidence,
	const cv::Mat_<float>& landmarks_2D, const cv::Mat_<float>& landmarks_3D, const cv::Mat_<float>& pdm_model_params, const cv::Vec6f& rigid_shape_params, cv::Vec6f& pose_estimate,
	const cv::Point3f& gazeDirection0, const cv::Point3f& gazeDirection1, const cv::Vec2f& gaze_angle, const std::vector<cv::Point2f>& eye_landmarks2d, const std::vector<cv::Point3f>& eye_landmarks3d,
	const std::vector<std::pair<std::string, double> >& au_intensities, const std::vector<std::pair<std::string, double> >& au_occurences)
{
	if (!this->IsOpen())
	{
		std::cout << "The socket is not open, exiting" << std::endl;
		exit(1);
	}

	zmq::message_t message(20);
	std::stringstream ss;

	ss << std::fixed << std::noshowpoint;	
	ss << "Meta:";

	if (is_sequence)
	{
		ss << std::setprecision(3);
		ss << ",frame:" << frame_num << ",face_id:" << face_id << ",timestamp:" << time_stamp;
		ss << std::setprecision(2);
		ss << ",confidence:" << landmark_confidence;
		ss << std::setprecision(0);
		ss << ",success:" << landmark_detection_success;
	}
	else
	{
		ss << std::setprecision(3);
		ss << "face_id:" << face_id << ",confidence:" << landmark_confidence;
	}

	SendData(ss);

	if (output_gaze)
	{
		ss << "Gaze:";

		ss << std::setprecision(6);
		ss << ",gaze_0_x:" << gazeDirection0.x << ",gaze_0_y:" << gazeDirection0.y << ",gaze_0_z:" << gazeDirection0.z;
		ss << ",gaze_1_x:" << gazeDirection1.x << ",gaze_1_y:" << gazeDirection1.y << ",gaze_1_z:" << gazeDirection1.z;

		ss << std::setprecision(3);
		ss << ",gaze_angle_x:" << gaze_angle[0] << ",gaze_angle_y:" << gaze_angle[1];

		ss << std::setprecision(1);
		for (int i = 0; i < eye_landmarks2d.size(); i++)
		{
			auto eye_lmk = eye_landmarks2d[i];
			ss << ",eye_lmk_x_" << i << ":" << eye_lmk.x;
		}

		for (int i = 0; i < eye_landmarks2d.size(); i++)
		{
			auto eye_lmk = eye_landmarks2d[i];
			ss << ",eye_lmk_y_" << i << ":" << eye_lmk.y;
		}

		for (int i = 0; i < eye_landmarks3d.size(); i++)
		{
			auto eye_lmk = eye_landmarks3d[i];
			ss << ",eye_lmk_X_" << i << ":" << eye_lmk.x;
		}

		for (int i = 0; i < eye_landmarks3d.size(); i++)
		{
			auto eye_lmk = eye_landmarks3d[i];
			ss << ",eye_lmk_Y_" << i << ":" << eye_lmk.y;
		}

		for (int i = 0; i < eye_landmarks3d.size(); i++)
		{
			auto eye_lmk = eye_landmarks3d[i];
			ss << ",eye_lmk_Z_" << i << ":" << eye_lmk.z;
		}

		SendData(ss);
	}

	if (output_pose)
	{
		ss << "Pose:";
		ss << std::setprecision(1);
		ss << ",pose_Tx:" << pose_estimate[0] << ",pose_Ty:" << pose_estimate[1] << ",pose_Tz:" << pose_estimate[2];
		ss << std::setprecision(3);
		ss << ",pose_Rx:" << pose_estimate[3] << ",pose_Ry:" << pose_estimate[4] << ",pose_Rz:" << pose_estimate[5];
		SendData(ss);
	}

	if (output_2D_landmarks)
	{
		ss << "Landmarks2D:";
		ss << std::setprecision(1);
		int i = 0;
		int num_landmarks = landmarks_2D.size().area() / 2;
		for (auto lmk : landmarks_2D)
		{
			if (i < num_landmarks)
			{
				ss << ",x_" << i << ":" << lmk;
			}
			else
			{
				ss << ",y_" << i % num_landmarks << ":" << lmk;
			}
		}
		SendData(ss);
	}

	if (output_3D_landmarks)
	{
		ss << "Landmarks3D:";
		ss << std::setprecision(1);
		int i = 0;
		int num_landmarks = landmarks_3D.size().area() / 3;
		for (auto lmk : landmarks_3D)
		{
			if (i < num_landmarks)
			{
				ss << ",X_" << i << ":" << lmk;
			}
			else if (i < 2 * num_landmarks)
			{
				ss << ",Y_" << i % num_landmarks << ":" << lmk;
			}
			else
			{
				ss << ",Z_" << i % num_landmarks << ":" << lmk;
			}
		}
		SendData(ss);
	}

	if (output_model_params)
	{
		ss << "ModelParams:";
		ss << std::setprecision(3);
		ss << ",p_scale:" << rigid_shape_params[0] << ",p_rx:" << rigid_shape_params[1] << ",p_ry:" << rigid_shape_params[2];
		ss << ",p_rz:" << rigid_shape_params[3] << ",p_tx:" << rigid_shape_params[4] << ",p_ty:" << rigid_shape_params[5];

		int i = 0;
		for (auto model_param : pdm_model_params)
		{
			ss << ",p_" << i << ":" << model_param;
			i++;
		}
		SendData(ss);
	}

	if (output_AUs)
	{
		ss << "AUs:";
		ss << std::setprecision(2);
		std::sort(au_intensities.begin(), au_intensities.end());
		for (auto au : au_intensities)
		{
			ss << "," << au.first << ":" << au.second;
		}
		
		ss << std::setprecision(1);
		std::sort(au_occurences.begin(), au_occurences.end());
		for (auto au : au_occurences)
		{
			ss << "," << au.first << ":" << au.second;
		}
		SendData(ss);
	}
}

void RecorderSocket::SendData(std::stringstream &ss)
{
	std::string data = ss.str();
	socket->send(zmq::buffer(data), zmq::send_flags::none);
	ss.str("");
}