// ------------------------- OpenPose C++ API Tutorial - Example 16 - Custom Output -------------------------
// Synchronous mode: ideal for production integration. It provides the fastest results with respect to runtime
// performance.
// In this function, the user can implement its own way to render/display/storage the results.

// Third-party dependencies
#include <opencv2/opencv.hpp>
// Command-line user interface
#define OPENPOSE_FLAGS_DISABLE_DISPLAY
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>
#include <algorithm>


#include <WinSock2.h>
#include <Ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

constexpr long double PI = 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844;
constexpr int UDP_PORT = 3065;

// Custom OpenPose flags
// 
// Display
DEFINE_bool(no_display, false,
    "Enable to disable the visual display.");

// This worker will just read and return all the jpg files in a directory
class WUserOutput : public op::WorkerConsumer<std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>>
{
private:
    struct coordinate {
        float x;
        float y;
    };
    std::vector<std::vector<coordinate>> keyPointsPerson;
    std::array<int, 4> maxPersonsPerBox{ 0 };
    SOCKET sock;
    WSADATA wsaData;

public:
    void initializationOnThread() {
        
        // Initialize Winsock
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            std::cerr << "Failed to initialize Winsock." << std::endl;
            return;
        }

        // Create a UDP socket
        sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock == INVALID_SOCKET) {
            std::cerr << "Failed to create socket: " << WSAGetLastError() << std::endl;
            WSACleanup();
            return;
        }
        op::opLog("Socket initialized");
    }

    void printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
    {
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            keyPointsPerson.clear();
            //op::opLog("\nKeypoints:");
            // Accessing each element of the keypoints
            const auto& poseKeypoints = datumsPtr->at(0)->poseKeypoints;
            //op::opLog("Person pose keypoints:");
            for (auto person = 0; person < poseKeypoints.getSize(0); person++)
            {
                //op::opLog("Person " + std::to_string(person) + " (x, y, score):");
                std::vector<coordinate> personToAdd;
                keyPointsPerson.push_back(personToAdd);
                for (auto bodyPart = 0; bodyPart < poseKeypoints.getSize(1); bodyPart++)
                {

                    // Save coordinates of the body parts
                    coordinate bodyPartCoordinates{ poseKeypoints[{person, bodyPart, 0}], poseKeypoints[{person, bodyPart, 1}] };
                    keyPointsPerson.at(person).push_back(bodyPartCoordinates);

                    std::string valueToPrint;
                    for (auto xyscore = 0; xyscore < poseKeypoints.getSize(2); xyscore++)
                    {
                        valueToPrint += std::to_string(poseKeypoints[{person, bodyPart, xyscore}]) + " ";
                    }
                    //op::opLog(valueToPrint, op::Priority::NoOutput);
                }
            }
            //op::opLog(" ");

        }
        else
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
    }

    coordinate splitPersons(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
    {
        if (datumsPtr == nullptr || datumsPtr->empty()) {
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
            return coordinate{};
        }

        // Get the OpenCV Matrix for the frame
        const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData);
        int width = cvMat.cols;
        int height = cvMat.rows;

        int boxWidth = width / 2;
        int boxHeight = height / 2;

        //Calculate the boxes to be a quarter of the screens and stacked
        const std::vector<cv::Rect> boxes = {
            cv::Rect(0, boxHeight, boxWidth, boxHeight / 2),							//Left Upper
            cv::Rect(boxWidth, boxHeight, boxWidth, boxHeight / 2),						//Right Upper
            cv::Rect(0, (boxHeight + boxHeight / 2), boxWidth, boxHeight / 2),			//Left Down
            cv::Rect(boxWidth, (boxHeight + boxHeight / 2), boxWidth, boxHeight / 2),	//Right Down
        };
        for (int i = 0; i < boxes.size(); i++)
        {
            // Draw the rectangles on a board
            cv::Scalar color(0, 255, 0);
            cv::rectangle(cvMat, boxes[i], color, 4);
        }


        std::array<int, 4> personsInBox{ 0 };
        std::array<int, 4> personsInBoxStanding{ 0 };
        // Iterate over the persons keypoints
        for (const auto& keypointsOfPerson : keyPointsPerson) {
            // Take either the neck or hip for the association
            const float x = keypointsOfPerson[8].x != 0 ? keypointsOfPerson[8].x : keypointsOfPerson[1].x;
            const float y = keypointsOfPerson[8].y != 0 ? keypointsOfPerson[8].y : keypointsOfPerson[1].y;

            int boxIndex = -1;

            // Check if the keypoint is in one of the boxes
            for (int i = 0; i < 4; i++) {
                const cv::Rect& box = boxes[i];
                if (x >= box.x && x < box.x + box.width && y >= box.y && y < box.y + box.height) {
                    // Save the box and leave the loop early
                    boxIndex = i;
                    break;
                }
            }

            // Check if the keypoint was associated with one of the boxes
            if (boxIndex != -1) {

                cv::Scalar color(0, 255, 0);
                std::string label = "Person in box " + std::to_string(boxIndex + 1);
                cv::Point textPosition(static_cast<int>(x), static_cast<int>(y) - 10);

                // Count up the persons in the box
                personsInBox[boxIndex]++;

                // Check if a person is sitting
                if (checkSitting(keypointsOfPerson)) {
                    op::opLog(label);
                    label += " sitting";
                }
                else {
                    label += " standing";
                    personsInBoxStanding[boxIndex]++;
                }
                // Add a label to the person
                cv::putText(cvMat, label, textPosition, cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
            }
        }
        return calculateCoordinatesStanding(personsInBox, personsInBoxStanding);
    }

    coordinate calculateCoordinatesStanding(std::array<int, 4> personsInBox, std::array<int, 4> personsInBoxStanding) {

        // Check if a new max in a box was reached and save if it was
        for (int i = 0; i < personsInBox.size(); i++) {
            maxPersonsPerBox[i] = max(maxPersonsPerBox[i], personsInBox[i]);
        }


        // Calculate the coordinates after the provided formula
        std::array<coordinate, 4> averagePersonsStanding{};
        for (int i = 0; i < 4; i++) {
            float ratio = maxPersonsPerBox[i] != 0 ? static_cast<float>(personsInBoxStanding[i]) / static_cast<float>(maxPersonsPerBox[i]) : 0.0;

            averagePersonsStanding[i].x = (i % 2 == 0 ? -1.0 : 1.0) * ratio;
            averagePersonsStanding[i].y = (i < 2 ? 1.0 : -1.0) * ratio;
        }

        coordinate cord{};
        // Summarize the average persons per box and add them together
        cord.x = averagePersonsStanding[0].x + averagePersonsStanding[1].x + averagePersonsStanding[2].x + averagePersonsStanding[3].x;
        cord.y = averagePersonsStanding[0].y + averagePersonsStanding[1].y + averagePersonsStanding[2].y + averagePersonsStanding[3].y;

        op::opLog("x: " + std::to_string(cord.x) + ", y: " + std::to_string(cord.y));
        return cord;
    }

    bool checkSitting(const std::vector<coordinate>& keypointsSinglePerson) {

        // Get all relevant Keypoints
        coordinate midHip = keypointsSinglePerson[8];
        const auto& rKnee = keypointsSinglePerson[10];
        const auto& lKnee = keypointsSinglePerson[13];
        const auto& neck = keypointsSinglePerson[1];
        coordinate knee = rKnee.x != 0 ? rKnee : lKnee;

        // Check if any of these is 0,#
        // If so the person is probably sitting as the skeleton couldn´t be applied
        if (midHip.x == 0 || knee.x == 0 || neck.x == 0) {
            return true; // Person sitting
        }

        // Calculate the Vector Neck->Knee
        knee.x -= neck.x;
        knee.y -= neck.y;

        // Calculate the Vector Neck->Hip
        midHip.x -= neck.x;
        midHip.y -= neck.y;


        // Calculate magnitude/size/length
        const float lengthHip = sqrt(pow(midHip.x, 2) + pow(midHip.y, 2));
        const float lengthKnee = sqrt(pow(knee.x, 2) + pow(knee.y, 2));

        // Calculate skalar
        const float nSkalar = (midHip.x * knee.x) + (midHip.y * knee.y);

        // Calculate the angle in rad
        const float kneeAngleInRad = acos(nSkalar / (lengthHip * lengthKnee));

        // Convert into degree
        const float angleInDegree = kneeAngleInRad * 360.0 / (2 * PI);

        return angleInDegree >= 40;
    }

    bool sendUDPMessage(coordinate coordinates) {
        // Set up the server address and port
        sockaddr_in serverAddress{};
        serverAddress.sin_family = AF_INET;
        serverAddress.sin_port = htons(UDP_PORT);
        if (inet_pton(AF_INET, "127.0.0.1", &(serverAddress.sin_addr)) <= 0) {
            std::cerr << "Invalid address." << std::endl;
            closesocket(sock);
            WSACleanup();
            return false;
        }

        // Send data to the server
        char* message = reinterpret_cast<char*>(&coordinates);
        int messageLength = sizeof(coordinate);
        int bytesSent = sendto(sock, message, messageLength, 0, (sockaddr*)&serverAddress, sizeof(serverAddress));
        if (bytesSent == SOCKET_ERROR) {
            std::cerr << "Failed to send data: " << WSAGetLastError() << std::endl;
            closesocket(sock);
            WSACleanup();
            return false;
        }
        return true;
    }

    coordinate normalizeVector(coordinate coord) 
    {
        float minValue = min(coord.x, coord.y);
        float maxValue = max(coord.x, coord.y);

        // Calculate the range
        float range = maxValue - minValue;

        // Normalize each element in the coordinate struct
        float normalizedX = ((coord.x - minValue) / range) * 2 - 1;
        float normalizedY = ((coord.y - minValue) / range) * 2 - 1;

        // Clamp the values within the range of -1 and 1
        normalizedX = std::clamp(normalizedX, -1.0f, 1.0f);
        normalizedY = std::clamp(normalizedY, -1.0f, 1.0f);

        return { normalizedX, normalizedY };
    }

    void workConsumer(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
    {
        try
        {
            // User's displaying/saving/other processing here
                // datumPtr->cvOutputData: rendered frame with pose or heatmaps
                // datumPtr->poseKeypoints: Array<float> with the estimated pose
            if (datumsPtr != nullptr && !datumsPtr->empty())
            {
                printKeypoints(datumsPtr);

                auto result = splitPersons(datumsPtr);
                
                sendUDPMessage(normalizeVector(result));

                // Display results (if enabled)
                if (!FLAGS_no_display)
                {
                    // Display rendered output image
                    const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData);
                    if (!cvMat.empty())
                    {
                        cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - PoseSplitter", cvMat);
                        // Display image and sleeps at least 1 ms (it usually sleeps ~5-10 msec to display the image)
                        const char key = (char)cv::waitKey(1);
                        if (key == 27)
                            this->stop();
                    }
                    else
                        op::opLog("Empty cv::Mat as output.", op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
                }
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
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

        // Initializing the user custom classes
        // GUI (Display)
        auto wUserOutput = std::make_shared<WUserOutput>();
        // Add custom processing
        const auto workerOutputOnNewThread = true;
        opWrapper.setWorker(op::WorkerType::Output, wUserOutput, workerOutputOnNewThread);

        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, FLAGS_net_resolution_dynamic, outputSize, keypointScaleMode, FLAGS_num_gpu,
            FLAGS_num_gpu_start, FLAGS_scale_number, (float)FLAGS_scale_gap,
            op::flagsToRenderMode(FLAGS_render_pose, multipleView), poseModel, !FLAGS_disable_blending,
            (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap, FLAGS_part_to_show, op::String(FLAGS_model_folder),
            heatMapTypes, heatMapScaleMode, FLAGS_part_candidates, (float)FLAGS_render_threshold,
            FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max, op::String(FLAGS_prototxt_path),
            op::String(FLAGS_caffemodel_path), (float)FLAGS_upsampling_ratio, enableGoogleLogging };
        opWrapper.configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{
            FLAGS_face, faceDetector, faceNetInputSize,
            op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
            (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold };
        opWrapper.configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{
            FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
            op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
            (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold };
        opWrapper.configure(wrapperStructHand);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{
            FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads };
        opWrapper.configure(wrapperStructExtra);
        // Producer (use default to disable any input)
        const op::WrapperStructInput wrapperStructInput{
            producerType, producerString, FLAGS_frame_first, FLAGS_frame_step, FLAGS_frame_last,
            FLAGS_process_real_time, FLAGS_frame_flip, FLAGS_frame_rotate, FLAGS_frames_repeat,
            cameraSize, op::String(FLAGS_camera_parameter_path), FLAGS_frame_undistort, FLAGS_3d_views };
        opWrapper.configure(wrapperStructInput);
        // Output (comment or use default argument to disable any output)
        const op::WrapperStructOutput wrapperStructOutput{
            FLAGS_cli_verbose, op::String(FLAGS_write_keypoint), op::stringToDataFormat(FLAGS_write_keypoint_format),
            op::String(FLAGS_write_json), op::String(FLAGS_write_coco_json), FLAGS_write_coco_json_variants,
            FLAGS_write_coco_json_variant, op::String(FLAGS_write_images), op::String(FLAGS_write_images_format),
            op::String(FLAGS_write_video), FLAGS_write_video_fps, FLAGS_write_video_with_audio,
            op::String(FLAGS_write_heatmaps), op::String(FLAGS_write_heatmaps_format), op::String(FLAGS_write_video_3d),
            op::String(FLAGS_write_video_adam), op::String(FLAGS_write_bvh), op::String(FLAGS_udp_host),
            op::String(FLAGS_udp_port) };
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

        // OpenPose wrapper
        op::opLog("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper;
        configureWrapper(opWrapper);

                // Start, run, and stop processing - exec() blocks this thread until OpenPose wrapper has finished
        op::opLog("Starting thread(s)...", op::Priority::High);
        opWrapper.exec();

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

int main(int argc, char* argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running tutorialApiCpp
    return tutorialApiCpp();
}
