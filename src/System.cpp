#include "System.h"

namespace meshflow {

System::System(const std::string& filename) : filename{filename} {}

void System::run() {
    cv::VideoCapture cap(filename);

    // Propagate motion vectors and generate vertex profiles
    Stabilizer stab;
    stab.readVideo(cap);

    // Stabilize the  vertex profiles
    stab.stabilize(false);

    // Get updated mesh warps
    stab.getFrameWarp();

    // Apply updated mesh warps & save the results
    stab.generateStabilizedVideo(cap);
}

}
