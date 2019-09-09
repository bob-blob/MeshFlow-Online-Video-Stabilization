#pragma once

#include <string>

#include "OpticalFlow.h"
#include "Stabilizer.h"

namespace meshflow {

class System {
public:
    System(const std::string& filename);
    void run();

private:
    std::string filename;
};

}
