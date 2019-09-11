#include "System.h"

namespace meshflow {

System::System(const std::string& filename) : filename{filename} {}

void System::run() {
    cv::VideoCapture cap(filename);

    // Propagate motion vectors and generate vertex profiles
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    Stabilizer stab(filename, cv::Size(width, height));

    stab.readVideo(cap);

    // Stabilize the  vertex profiles
    stab.stabilize(false);

    // Get updated mesh warps
    stab.getFrameWarp();

    // Apply updated mesh warps & save the results
    stab.generateStabilizedVideo(cap);
}

}
