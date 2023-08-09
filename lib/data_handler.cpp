#include "..\include\data_handler.h"
#include <random>
#include <chrono>
#include <algorithm>

std::random_device rd;
std::mt19937 gen(rd());

// Generate random number inside desired range.
int randomNum(int low, int high) {
    std::uniform_int_distribution<> dist(low, high);
    return dist(gen);
}

// Object that hold data and splits.
data_handler::data_handler() {
    data_array = new std::vector<data *>;
    test_data = new std::vector<data *>;
    training_data = new std::vector<data *>;
    validation_data = new std::vector<data *>;
}

data_handler::~data_handler() {/* Free dynamically allocated memory.*/}

int data_handler::get_class_counts() {return num_classes;}

void data_handler::read_feature_vector(std::string path) {
    uint32_t header[4]; // Magic // Num Images // Row Size // Col Size
    unsigned char bytes[4];

    std::cout << "Data handler: read feature vector." << std::endl;

    FILE *f = fopen(path.c_str(), "rb");
    if(f) {
        for(int i = 0; i < 4; i++) {
            if(fread(bytes, sizeof(bytes), 1, f)) {
                header[i] = convert_to_little_endian(bytes);
            }
        }

        std::cout << "Done getting input file header." << std::endl;
        
        int image_size = header[2] * header[3];
        // Iterate in all images
        for(int i = 0; i < header[1]; i++) {
            data *d = new data();
            uint8_t element[1];

            for(int j = 0; j < image_size; j++) {
                if(fread(element, sizeof(element), 1, f)) {
                    d->append_to_feature_vector(element[0]);
                } else
                {
                    std::cout << "Error reading from file." << j << std::endl;
                    exit(1);
                }
            }
            data_array->push_back(d);
        }
      std::cout << "Successfully read and stored " <<
        data_array->size() << " feature vectors." << std::endl;  
    } else
    {
        std::cout << "Could not find the file." << std::endl;
        exit(2);
    }
}

void data_handler::read_feature_labels(std::string path)
{
    uint32_t header[2]; // Magic // Images 
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "rb");
    std::cout << "Data handler: read label vector" << std::endl;

    if(f)
    {
        for(int i = 0; i < 2; i++)
        {
            if(fread(bytes, sizeof(bytes), 1, f))
            {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        std::cout << "Done getting label file header." << std::endl;
        
        for(int i = 0; i < header[1]; i++)
        {
            uint8_t element[1];

            if(fread(element, sizeof(element), 1, f))
            {
                data_array->at(i)->set_label(element[0]);
                    
            } else
            {
                std::cout << "Error reading from file. " << i << std::endl;
                exit(3);
            }
        }
      std::cout << "Successfully read and stored " << data_array->size() << " feature labels." << std::endl;  
    } else
    {
        std::cout << "Could not find the file." << std::endl;
        exit(4);
    }
}

// Split dataset in training, validation and test
void data_handler::split_data() {
    // Amount of samples in each split
    int train_size = data_array->size() * TRAIN_SET_PERCENT;
    int test_size = data_array->size() * TEST_SET_PERCENT;
    int validation_size = data_array->size() * VALIDATION_SET_PERCENT;
    // Shuffle dataset
    unsigned num = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(data_array->begin(), data_array->end(), std::default_random_engine(num));
    // Split data
    for (int i = 0; i < data_array -> size(); i++) {
        if (i <= train_size - 1) {
            training_data->push_back(data_array->at(i));
        } else if (i <= test_size + train_size - 1) {
            test_data->push_back(data_array->at(i));
        } else {
            validation_data->push_back(data_array->at(i));
        }
    }

    std::cout << "Training data size: " << training_data->size() << "." << std::endl;
    std::cout << "Test data size: " << test_data->size() << "." << std::endl;
    std::cout << "Validation data size: " << validation_data->size() << "." << std::endl;
}

void data_handler::count_classes() {
    int count{0};
    for (unsigned i = 0; i < data_array->size(); i++) {
        if (class_map.find(data_array->at(i)->get_label()) == class_map.end()) {
            class_map[data_array->at(i)->get_label()] = count;
            data_array->at(i)->set_enumerated_label(count);
            count++;
        }
    }
    num_classes = count;
    std::cout << "Successfully extracted " << num_classes
        << " unique classes." << std::endl;
}

uint32_t data_handler::convert_to_little_endian(const unsigned char* bytes) {
    return (uint32_t) ((bytes[0] << 24) |
                       (bytes[1] << 16) |
                       (bytes[2] << 8) |
                       (bytes[3]));
}

std::vector<data *> * data_handler::get_training_data()
{
    return training_data;
}
std::vector<data *> * data_handler::get_test_data()
{
    return test_data;
}
std::vector<data *> * data_handler::get_validation_data()
{
    return validation_data;
}