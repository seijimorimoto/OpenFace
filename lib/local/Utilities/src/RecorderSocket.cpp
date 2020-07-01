#include "stdafx_ut.h"
#include "RecorderSocket.h"

using namespace Utilities;

// Creates a RecorderSocket for pushing data to a socket identified by the given port.
RecorderSocket::RecorderSocket(int port) : RecorderResults()
{
	this->port = port;
}

// Initializes the object with the flags specifying which type of data is to be sent via the socket in each write.
// Only recordFlags parameter is used, the rest is only included to conform with RecorderResults abstract class.
void RecorderSocket::Init(const RecorderOpenFaceParameters& recordFlags, int num_face_landmarks, int num_model_modes, int num_eye_landmarks,
	const std::vector<std::string>& au_names_class, const std::vector<std::string>& au_names_reg)
{
	this->recordFlags = recordFlags;
}

// Opens a socket so that it can later send data to the port pointed by the socket.
bool RecorderSocket::Open()
{
	zmq::context_t context(1); // Use a single IO thread.
	this->socket = new zmq::socket_t(context, ZMQ_PUB /* Set this socket as a publisher. */);
	this->socket->bind("tcp://*:" + std::to_string(this->port));
}

// Checks whether the internal socket managed by this class has been successfully opened or not.
bool RecorderSocket::IsOpen()
{
	return this->socket != NULL;
}

// Closes the internal socket managed by this class.
void RecorderSocket::Close()
{
	if (this->IsOpen())
		this->socket->close();
}

// Prepares a message and sends it through the socket.
void RecorderSocket::Write(int face_id, int frame_num, double time_stamp, bool landmark_detection_success, double landmark_confidence,
	const cv::Mat_<float>& landmarks_2D, const cv::Mat_<float>& landmarks_3D, const cv::Mat_<float>& pdm_model_params, const cv::Vec6f& rigid_shape_params, cv::Vec6f& pose_estimate,
	const cv::Point3f& gazeDirection0, const cv::Point3f& gazeDirection1, const cv::Vec2f& gaze_angle, const std::vector<cv::Point2f>& eye_landmarks2d, const std::vector<cv::Point3f>& eye_landmarks3d,
	const std::vector<std::pair<std::string, double> >& au_intensities, const std::vector<std::pair<std::string, double> >& au_occurences)
{
	// Only proceeds if the socket is opened.
	if (!this->IsOpen())
	{
		std::cout << "The socket is not open, exiting" << std::endl;
		exit(1);
	}

	// String stream that will be used for creating the message that will be sent.
	std::stringstream ss;

	// Setting the format that will be used for floating point numbers in ss.
	ss << std::fixed << std::noshowpoint;	

	// Set the title of the first message that will be sent (metadata).
	// Set the content of the message based on whether the data computed is from a sequence or not.
	ss << "Meta:";
	if (recordFlags.isSequence())
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

	// Send the first message.
	SendData(ss);

	// Prepare and send a message containing the gaze information if it is supposed to be sent.
	if (recordFlags.outputGaze())
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

	// Prepare and send a message containing the pose information if it is supposed to be sent.
	if (recordFlags.outputPose())
	{
		ss << "Pose:";
		ss << std::setprecision(1);
		ss << ",pose_Tx:" << pose_estimate[0] << ",pose_Ty:" << pose_estimate[1] << ",pose_Tz:" << pose_estimate[2];
		ss << std::setprecision(3);
		ss << ",pose_Rx:" << pose_estimate[3] << ",pose_Ry:" << pose_estimate[4] << ",pose_Rz:" << pose_estimate[5];
		SendData(ss);
	}

	// Prepare and send a message containing the 2D landmarks information if it is supposed to be sent.
	if (recordFlags.output2DLandmarks())
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

	// Prepare and send a message containing the 3D landmarks information if it is supposed to be sent.
	if (recordFlags.output3DLandmarks())
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

	// Prepare and send a message containing the model parameters information if it is supposed to be sent.
	if (recordFlags.outputPDMParams())
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

	// Prepare and send a message containing the AUs information if it is supposed to be sent.
	if (recordFlags.outputAUs())
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

// Sends the data contained in a string stream through the socket. It also clears the string stream.
void RecorderSocket::SendData(std::stringstream &ss)
{
	std::string data = ss.str();
	socket->send(zmq::buffer(data), zmq::send_flags::none);
	ss.str("");
}