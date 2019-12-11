#ifndef EVALUATION_H
#define EVALUATION_H

#include <iostream>
#include <opencv2/core/core.hpp>

namespace meshflow {

typedef double EVALUATION_SCORE;

class Evaluation
{
public:
    Evaluation();
    EVALUATION_SCORE distortion(const std::string& originalVideoFilename,
                                const std::string& stabilizedVideoFilename) const;
};

}   // namespace meshflow

#endif // EVALUATION_H
