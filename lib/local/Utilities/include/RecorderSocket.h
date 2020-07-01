#ifndef RECORDER_SOCKET_H
#define RECORDER_SOCKET_H

// Standard library includes
#include <string>
#include <vector>

// Third-party libraries includes
#include <opencv2/core/core.hpp>
#include <zmq.hpp>

// Local files includes
#include <RecorderResults.h>

namespace Utilities
{
	/**
	A class for sending the data computed by OpenFace through a socket.
	*/
	class RecorderSocket : public RecorderResults
	{
	public:
		// Creates a RecorderSocket for pushing data to a socket identified by the given port.
		RecorderSocket(int port);

		// Initializes the object with the flags specifying which type of data is to be sent via the socket in each write.
		// Only recordFlags parameter is used, the rest is only included to conform with RecorderResults abstract class.
		void Init(const RecorderOpenFaceParameters& recordFlags, int num_face_landmarks, int num_model_modes, int num_eye_landmarks,
			const std::vector<std::string>& au_names_class, const std::vector<std::string>& au_names_reg);

		// Opens a socket so that it can later send data to the port pointed by the socket.
		bool Open();
		
		// Checks whether the internal socket managed by this class has been successfully opened or not.
		bool IsOpen();

		// Closes the internal socket managed by this class.
		void Close();

		// Prepares a message and sends it through the socket.
		void Write(int face_id, int frame_num, double time_stamp, bool landmark_detection_success, double landmark_confidence,
			const cv::Mat_<float>& landmarks_2D, const cv::Mat_<float>& landmarks_3D, const cv::Mat_<float>& pdm_model_params, const cv::Vec6f& rigid_shape_params, cv::Vec6f& pose_estimate,
			const cv::Point3f& gazeDirection0, const cv::Point3f& gazeDirection1, const cv::Vec2f& gaze_angle, const std::vector<cv::Point2f>& eye_landmarks2d, const std::vector<cv::Point3f>& eye_landmarks3d,
			const std::vector<std::pair<std::string, double> >& au_intensities, const std::vector<std::pair<std::string, double> >& au_occurences);

	private:
		// Port for which the socket will be opened. It is here where the data will be sent to.
		int port;

		// Socket that will be used to send the data to a given location.
		zmq::socket_t* socket;

		// Sends the data contained in a string stream through the socket. It also clears the string stream.
		void SendData(std::stringstream& ss);
	};
}

#endif // RECORDER_SOCKET_H