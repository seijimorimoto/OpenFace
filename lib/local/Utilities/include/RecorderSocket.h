#ifndef RECORDER_SOCKET_H
#define RECORDER_SOCKET_H

// Standard library includes
#include <string>
#include <vector>

// Third-party libraries includes
#include <opencv2/core/core.hpp>
#include <zmq.hpp>

namespace Utilities
{
	/**
	A class for sending the data computed by OpenFace through a socket.
	*/
	class RecorderSocket
	{
	public:

		// Opens a socket so that it can later send data to the port pointed by the socket.
		bool Open(int port, bool is_sequence, bool output_2D_landmarks, bool output_3D_landmarks, bool output_model_params, bool output_pose, bool output_AUs, bool output_gaze,
			int num_face_landmarks, int num_model_modes, int num_eye_landmarks, const std::vector<std::string>& au_names_class, const std::vector<std::string>& au_names_reg);
		
		// Checks whether the internal socket managed by this class has been successfully opened or not.
		bool IsOpen();

		// Closes the internal socket managed by this class.
		void Close();

		// Prepares a message and sends it through the socket.
		void WriteMessage(int face_id, int frame_num, double time_stamp, bool landmark_detection_success, double landmark_confidence,
			const cv::Mat_<float>& landmarks_2D, const cv::Mat_<float>& landmarks_3D, const cv::Mat_<float>& pdm_model_params, const cv::Vec6f& rigid_shape_params, cv::Vec6f& pose_estimate,
			const cv::Point3f& gazeDirection0, const cv::Point3f& gazeDirection1, const cv::Vec2f& gaze_angle, const std::vector<cv::Point2f>& eye_landmarks2d, const std::vector<cv::Point3f>& eye_landmarks3d,
			const std::vector<std::pair<std::string, double> >& au_intensities, const std::vector<std::pair<std::string, double> >& au_occurences);

	private:
		// Port for which the socket will be opened. It is here where the data will be sent to.
		int port;

		// Flag that indicates whether the data is being computed from a sequence or not.
		// If we are recording results from a sequence, each row refers to a frame. If we are recording an image, each row is a face.
		bool is_sequence;

		// Whether to include the 2D landmarks computed by OpenFace in the data to send.
		bool output_2D_landmarks;

		// Whether to include the 3D landmarks computed by OpenFace in the data to send.
		bool output_3D_landmarks;

		// Whether to include the model parameters of OpenFace in the data to send.
		bool output_model_params;

		// Whether to include the pose computed by OpenFace in the data to send.
		bool output_pose;

		// Whether to include the AUs results (intensity and occurence) of OpenFace in the data to send.
		bool output_AUs;

		// Whether to include the gaze computed by OpenFace in the data to send.
		bool output_gaze;

		// Socket that will be used to send the data to a given location.
		zmq::socket_t* socket;

		// Sends the data contained in a string stream through the socket. It also clears the string stream.
		void SendData(std::stringstream& ss);
	};
}

#endif // RECORDER_SOCKET_H