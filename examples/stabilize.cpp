#include "System.h"
#include <iostream>

int main(int argc, char *argv[])
{
    std::cout << "Hello, this is MeshFlow algorithm implentation!" << std::endl;
    if (argc < 2) {
        std::cout << "Usage: ./stabilize PATH" << std::endl;
        exit(1);
    }

    meshflow::System sys(argv[1], "../configs/test_config.yml");

    sys.stabilize();

    return 0;
}
