#include <RecorderOpenFaceParameters.h>

namespace Utilities
{
	/**
	Abstract class for representing objects that send/store results computed by OpenFace.
	*/
	class RecorderResults
	{
	public:
		// Default constructor
		RecorderResults() {}

		// Initializes the object.
		virtual void Init(const RecorderOpenFaceParameters& recordFlags, int num_face_landmarks, int num_model_modes, int num_eye_landmarks,
			const std::vector<std::string>& au_names_class, const std::vector<std::string>& au_names_reg) = 0;

		// Opens the internal stream or mechanism for sending/storing the data.
		virtual bool Open() = 0;

		// Checks whether the internal stream or mechanism has been opened.
		virtual bool IsOpen() = 0;

		// Closes the internal stream or mechanism for sending/storing the data.
		virtual void Close() = 0;

		// Stores/sends the data received as parameters through the internal stream or mechanism.
		virtual void Write(int face_id, int frame_num, double time_stamp, bool landmark_detection_success, double landmark_confidence,
			const cv::Mat_<float>& landmarks_2D, const cv::Mat_<float>& landmarks_3D, const cv::Mat_<float>& pdm_model_params,
			const cv::Vec6f& rigid_shape_params, cv::Vec6f& pose_estimate, const cv::Point3f& gazeDirection0, const cv::Point3f& gazeDirection1,
			const cv::Vec2f& gaze_angle, const std::vector<cv::Point2f>& eye_landmarks2d, const std::vector<cv::Point3f>& eye_landmarks3d,
			const std::vector<std::pair<std::string, double> >& au_intensities, const std::vector<std::pair<std::string, double> >& au_occurences) = 0;

	protected:
		// Object that contains information about which types of data should be stored/sent in each write.
		RecorderOpenFaceParameters recordFlags;
	};
}