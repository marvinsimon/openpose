// ------------------------- OpenPose C++ API Tutorial - Example 11 - Custom Output -------------------------
// Asynchronous mode: ideal for fast prototyping when performance is not an issue.
// In this function, the user can implement its own way to render/display/storage the results.

// Third-party dependencies
#include <opencv2/opencv.hpp>
// Command-line user interface
#define OPENPOSE_FLAGS_DISABLE_DISPLAY
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

constexpr long double PI = 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844;

// Custom OpenPose flags
// Display
DEFINE_bool(no_display,                 false,
    "Enable to disable the visual display.");

// This worker will just read and return all the jpg files in a directory
class UserOutputClass
{
private :
    struct coordinate {
        float x;
        float y;
    };
    std::vector<std::vector<coordinate>> keyPointsPerson;
public:
    bool display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
    {
        // User's displaying/saving/other processing here
            // datumPtr->cvOutputData: rendered frame with pose or heatmaps
            // datumPtr->poseKeypoints: Array<float> with the estimated pose
        
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            // Display image and sleeps at least 1 ms (it usually sleeps ~5-10 msec to display the image)
            const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData);
            if (!cvMat.empty())
                cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", cvMat);
            else
                op::opLog("Empty cv::Mat as output.", op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
        }
        else
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
        const auto key = (char)cv::waitKey(1);
        return (key == 27);
    }
    void printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
    {
        // Example: How to use the pose keypoints
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            keyPointsPerson.clear();
            //op::opLog("\nKeypoints:");
            // Accessing each element of the keypoints
            const auto& poseKeypoints = datumsPtr->at(0)->poseKeypoints;
            //op::opLog("Person pose keypoints:");
            for (auto person = 0 ; person < poseKeypoints.getSize(0) ; person++)
            {
                //op::opLog("Person " + std::to_string(person) + " (x, y, score):");
                std::vector<coordinate> personToAdd;
                keyPointsPerson.push_back(personToAdd);
                for (auto bodyPart = 0 ; bodyPart < poseKeypoints.getSize(1) ; bodyPart++)
                {

                    // Save coordinates of the body parts
                    coordinate bodyPartCoordinates{ poseKeypoints[{person, bodyPart, 0}], poseKeypoints[{person, bodyPart, 1}] };
                    keyPointsPerson.at(person).push_back(bodyPartCoordinates);

                    std::string valueToPrint;
                    for (auto xyscore = 0 ; xyscore < poseKeypoints.getSize(2) ; xyscore++)
                    {
                        valueToPrint += std::to_string(   poseKeypoints[{person, bodyPart, xyscore}]   ) + " ";
                    }
                    //op::opLog(valueToPrint, op::Priority::NoOutput);
                }
            }
            //op::opLog(" ");
            
        }
        else
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
    }

    void splitPersons(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
    {
        static double prevFrameTime = 0.0;
        if (datumsPtr == nullptr || datumsPtr->empty()) {
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
            return;
        }

        const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData);
        int width = cvMat.cols;
        int height = cvMat.rows;

        int boxWidth = width / 2;
        int boxHeight = height / 2;

        const cv::Rect boxes[4] = {
            cv::Rect(0, 0, boxWidth, boxHeight),
            cv::Rect(boxWidth, 0, boxWidth, boxHeight),
            cv::Rect(0, boxHeight, boxWidth, boxHeight),
            cv::Rect(boxWidth, boxHeight, boxWidth, boxHeight)
        };

        for (const auto& keypoints : keyPointsPerson) {
            const float x = keypoints[1].x;
            const float y = keypoints[1].y;

            int boxIndex = -1;

            for (int i = 0; i < 4; i++) {
                const cv::Rect& box = boxes[i];
                if (x >= box.x && x < box.x + box.width && y >= box.y && y < box.y + box.height) {
                    boxIndex = i + 1;
                    break;
                }
            }

            if (boxIndex != -1) {
                const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData);
                cv::Scalar color(0, 255, 0);
                std::string label = "Person in box " + std::to_string(boxIndex);
                cv::Point textPosition(static_cast<int>(x), static_cast<int>(y) - 10);
                cv::putText(cvMat, label, textPosition, cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);

                if (checkSitting(std::move(keypoints))) {
                    op::opLog("Person in box: " + std::to_string(boxIndex));
                }
            }



        }
    }

    bool checkSitting(const std::vector<coordinate>&& keypointsSinglePerson) {
        coordinate midHip = keypointsSinglePerson[8];
        const auto &rKnee = keypointsSinglePerson[10];
        const auto &lKnee = keypointsSinglePerson[13];
        const auto& neck = keypointsSinglePerson[1];
        coordinate knee = rKnee.x != 0 ? rKnee : lKnee;

        knee.x -= neck.x;
        knee.y -= neck.y;

        midHip.x -= neck.x;
        midHip.y -= neck.y;

        if (midHip.x == 0 || knee.x == 0) {
            return true; // Person sitting
        }

        const double lengthHip = sqrt(pow(midHip.x, 2) + pow(midHip.y, 2));
        const double lengthKnee = sqrt(pow(knee.x, 2) + pow(knee.y, 2));

        const double nSkalar = (midHip.x * knee.x) + (midHip.y * knee.y);

        const double kneeAngleInRad = acos(nSkalar / (lengthHip * lengthKnee));
        const auto angleInDegree = kneeAngleInRad * 360.0 / (2 * PI);
        if (angleInDegree > 40) {
            op::opLog("\n------------------------");
            op::opLog("Hip Points: " + std::to_string(midHip.x) + "/" + std::to_string(midHip.y));
            op::opLog("Knee Points: " + std::to_string(knee.x) + "/" + std::to_string(knee.y));
            op::opLog("Knee angle: " + std::to_string(angleInDegree), op::Priority::Max);
            op::opLog("------------------------");
        }
        return angleInDegree > 40;
    }
};

void configureWrapper(op::Wrapper& opWrapper)
{
    try
    {
        // Configuring OpenPose

        // logging_level
        op::checkBool(
            0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
            __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        // Applying user defined configuration - GFlags to program variables
        // producerType
        op::ProducerType producerType;
        op::String producerString;
        std::tie(producerType, producerString) = op::flagsToProducer(
            op::String(FLAGS_image_dir), op::String(FLAGS_video), op::String(FLAGS_ip_camera), FLAGS_camera,
            FLAGS_flir_camera, FLAGS_flir_camera_index);
        // cameraSize
        const auto cameraSize = op::flagsToPoint(op::String(FLAGS_camera_resolution), "-1x-1");
        // outputSize
        const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");
        // faceNetInputSize
        const auto faceNetInputSize = op::flagsToPoint(op::String(FLAGS_face_net_resolution), "368x368 (multiples of 16)");
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(op::String(FLAGS_hand_net_resolution), "368x368 (multiples of 16)");
        // poseMode
        const auto poseMode = op::flagsToPoseMode(FLAGS_body);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::opLog(
                "Flag `write_keypoint` is deprecated and will eventually be removed. Please, use `write_json`"
                " instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                      FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1 || FLAGS_flir_camera);
        // Face and hand detectors
        const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
        const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;

        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, FLAGS_net_resolution_dynamic, outputSize, keypointScaleMode, FLAGS_num_gpu,
            FLAGS_num_gpu_start, FLAGS_scale_number, (float)FLAGS_scale_gap,
            op::flagsToRenderMode(FLAGS_render_pose, multipleView), poseModel, !FLAGS_disable_blending,
            (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap, FLAGS_part_to_show, op::String(FLAGS_model_folder),
            heatMapTypes, heatMapScaleMode, FLAGS_part_candidates, (float)FLAGS_render_threshold,
            FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max, op::String(FLAGS_prototxt_path),
            op::String(FLAGS_caffemodel_path), (float)FLAGS_upsampling_ratio, enableGoogleLogging};
        opWrapper.configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{
            FLAGS_face, faceDetector, faceNetInputSize,
            op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
            (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
        opWrapper.configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{
            FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
            op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
            (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
        opWrapper.configure(wrapperStructHand);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{
            FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads};
        opWrapper.configure(wrapperStructExtra);
        // Producer (use default to disable any input)
        const op::WrapperStructInput wrapperStructInput{
            producerType, producerString, FLAGS_frame_first, FLAGS_frame_step, FLAGS_frame_last,
            FLAGS_process_real_time, FLAGS_frame_flip, FLAGS_frame_rotate, FLAGS_frames_repeat,
            cameraSize, op::String(FLAGS_camera_parameter_path), FLAGS_frame_undistort, FLAGS_3d_views};
        opWrapper.configure(wrapperStructInput);
        // Output (comment or use default argument to disable any output)
        const op::WrapperStructOutput wrapperStructOutput{
            FLAGS_cli_verbose, op::String(FLAGS_write_keypoint), op::stringToDataFormat(FLAGS_write_keypoint_format),
            op::String(FLAGS_write_json), op::String(FLAGS_write_coco_json), FLAGS_write_coco_json_variants,
            FLAGS_write_coco_json_variant, op::String(FLAGS_write_images), op::String(FLAGS_write_images_format),
            op::String(FLAGS_write_video), FLAGS_write_video_fps, FLAGS_write_video_with_audio,
            op::String(FLAGS_write_heatmaps), op::String(FLAGS_write_heatmaps_format), op::String(FLAGS_write_video_3d),
            op::String(FLAGS_write_video_adam), op::String(FLAGS_write_bvh), op::String(FLAGS_udp_host),
            op::String(FLAGS_udp_port)};
        opWrapper.configure(wrapperStructOutput);
        // No GUI. Equivalent to: opWrapper.configure(op::WrapperStructGui{});
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

int tutorialApiCpp()
{
    try
    {
        op::opLog("Starting OpenPose demo...", op::Priority::High);
        const auto opTimer = op::getTimerInit();

        // Configuring OpenPose
        op::opLog("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper{op::ThreadManagerMode::AsynchronousOut};
        configureWrapper(opWrapper);

        // Start, run, and stop processing - exec() blocks this thread until OpenPose wrapper has finished
        op::opLog("Starting thread(s)...", op::Priority::High);
        opWrapper.start();

        // User processing
        UserOutputClass userOutputClass;
        bool userWantsToExit = false;
        while (!userWantsToExit)
        {
            // Pop frame
            std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> datumProcessed;
            if (opWrapper.waitAndPop(datumProcessed))
            {
                userOutputClass.printKeypoints(datumProcessed);
                userOutputClass.splitPersons(datumProcessed);
                if (!FLAGS_no_display)
                    userWantsToExit = userOutputClass.display(datumProcessed);
            }
            // If OpenPose finished reading images
            else if (!opWrapper.isRunning())
                break;
            // Something else happened
            else
                op::opLog("Processed datum could not be emplaced.", op::Priority::High);
        }

        op::opLog("Stopping thread(s)", op::Priority::High);
        opWrapper.stop();

        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

        // Return
        return 0;
    }
    catch (const std::exception&)
    {
        return -1;
    }
}

int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running tutorialApiCpp
    return tutorialApiCpp();
}
