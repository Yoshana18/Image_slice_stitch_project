// These are the include headers to run this cpp file.
#include <vips/vips8>
#include <string>
#include <vector>
#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <cstdio>     // For popen(), pclose(), FILE*
#include <map>        // For std::map
#include <cstring>    // For strchr(), strncpy()
#include <fstream>    // For std::ofstream
#include <cctype>     // For isspace()

extern "C" { // This is calling the darknet module that is installed in this PC environment 
    #include <darknet.h>
}

// OpenCV path for image processing
#include <opencv4/opencv2/opencv.hpp>

// For logs saving
#include <nlohmann/json.hpp>

// Using some namespaces - shorterns the codes.
using namespace std;
using namespace vips;
namespace fs = std::filesystem;
using namespace cv;

// Hardcode for the class names of the coco dataset. 
const std::vector<std::string> coco_names = {
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "cup"
};

// Struct to hold detection results - Important 
struct Detection {
    int cls;  //class_id
    float score;  // confidence score
    float x1, y1, x2, y2; // Bounding box coordinates - the 4 corners 
};

// Helper Function 
// This is to parse a standard output from the darknet with an -ext_output 
// Uses coco_map to convert class names (string value) to a class IDs (int value)
std::vector<Detection> parse_darknet_output(FILE* pipe, const std::map<std::string, int>& coco_map) {
    std::vector<Detection> detections;
    char buffer[1024];
    
    // Read Darknet output line by line 
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        // This will print the raw output from darknet as our program sees it
        // cout << "[Darknet RAW]: " << buffer;

        // Example detection line from Darknet: person   c=93.695190%    x=68    y=95    w=56    h=120
        char class_name_str[100];
        float score;
        int left, top, width, height;

        // Try to find the start of the detection data, which looks like " c="
        // We use " c=" to be more specific and avoid "compute_capability"
        char* data_ptr = strstr(buffer, " c=");
        
        // If " c=" isn't found, try just "c="
        if (!data_ptr) {
            data_ptr = strstr(buffer, "c=");
            // But make sure it's not "compute_capability"
            if (data_ptr == strstr(buffer, "compute_capability")) {
                data_ptr = nullptr; // Ignore this line
            }
        }

        // If we found a valid "c=" marker, try to parse from there
        if (data_ptr) {
            // Extract the class name
            // The class name is everything from the start of the buffer up to data_ptr
            int class_name_len = data_ptr - buffer; // Get length of class name
            if (class_name_len > 99) class_name_len = 99; // Prevent buffer overflow
            
            strncpy(class_name_str, buffer, class_name_len);
            class_name_str[class_name_len] = '\0';
            
            // Trim trailing whitespace from the class name (e.g., "person   " -> "person")
            char* end = class_name_str + class_name_len - 1;
            while (end > class_name_str && isspace(static_cast<unsigned char>(*end))) {
                *end = '\0';
                end--;
            }

            // Look up the class ID from the trimmed name
            if (coco_map.find(class_name_str) == coco_map.end()) {
                cerr << "Unknown class: '" << class_name_str << "'" << endl; // Optional log
                continue; // Skip if class name not in our list
            }
            int class_id = coco_map.at(class_name_str);

            // Parse the numeric data
            // Format is: " c=%f%% x=%d y=%d w=%d h=%d"
            int parsed_count = sscanf(data_ptr, " c=%f%% x=%d y=%d w=%d h=%d",
                                      &score, &left, &top, &width, &height);

            if (parsed_count == 5) {
                // Successfully parsed!
                Detection det;
                det.cls = class_id;
                det.score = score; // Note: score is already 0-100
                det.x1 = static_cast<float>(left);
                det.y1 = static_cast<float>(top);
                det.x2 = static_cast<float>(left + width);
                det.y2 = static_cast<float>(top + height);
                detections.push_back(det);
            }
        }
    }
    return detections;
}

// Saves a vector of detections to a JSON file
// The vector of Detection structs
// The full path for the output .json file 
void save_metadata(const std::vector<Detection>& detections, const std::string& output_path) {
    nlohmann::json j_dets = nlohmann::json::array();
    
    for (const auto& det : detections) {
        nlohmann::json j_det;
        j_det["bbox"] = { det.x1, det.y1, det.x2, det.y2 };
        j_det["class_id"] = det.cls;
        j_det["class_name"] = coco_names[det.cls]; // Assumes coco_names is global const
        j_det["score"] = det.score;
        j_dets.push_back(j_det);
    }

    // Write JSON to file
    std::ofstream meta_out(output_path);
    if (!meta_out) {
        cerr << "Failed to open metadata file for writing: " << output_path << "\n";
    } else {
        meta_out << j_dets.dump(4); // Pretty print with indent=4 spaces
        meta_out.close();
        cout << "Saved metadata: " << output_path << endl;
    }
}

// Main function to run the code (this is the execution function)
int main(int argc, char **argv) {
    // Initialise Livips
    if (VIPS_INIT(argv[0])) {
        vips_error_exit(nullptr);
        return 1;
    }

    // Darknet config paths - using absolute path to define darknet functions.
    const string darknet_exec  = "/home/src/darknet/build/src-cli/darknet";  // actual location of darknet model
    const string data_file     = "/home/src/darknet/cfg/coco.data";
    const string cfg_file      = "/home/src/darknet/cfg/yolov7-tiny.cfg"; // Model Configuration file
    const string weights_file  = "/home/src/darknet/cfg/yolov7-tiny.weights"; // Model Weights 
    
    // Defining a Desired tile size - its the input size of the darknet-yolov7 pixel size.
    const int tile_size = 416;
    
    // This checks for the input image in the terminal - need to tag the image after running ./main <input image.jpg>
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_image>" << endl;
        return 1;
    }
    
    // Reading image path / either in binary or something.
    std::string input_path = argv[1];
    std::cout << "Reading: " << input_path << std::endl;

    // Using OpenCV to read the image from the input path
    cv::Mat mat = cv::imread(input_path);
    if (mat.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return 1;
    }

    if (!mat.isContinuous()) mat = mat.clone();

    // Convert BGR -> RGB before passing to libvips (optional)
    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);

    // creating VImage, from memory - input image using cv::mat
    VImage image = VImage::new_from_memory(
        mat.data,
        mat.total() * mat.elemSize(),
        mat.cols, mat.rows, mat.channels(),
        VIPS_FORMAT_UCHAR
    );

    // Create the name-to-ID map for COCO classes - changes can be made here when changing the class name 
    std::map<std::string, int> coco_map;
    for (int i = 0; i < coco_names.size(); ++i) {
        coco_map[coco_names[i]] = i;
    }

// --------------------------------------- ORIGINAL IMAGE - MODEL INFERENCE --------------------------------------------
    // Original Image Processing
    cout << "\nRunning Darknet Detection on the Original Image";
    cout << "\n[Darknet] Processing " << input_path << endl;

    // Construct Darknet command for the original image
    string original_command = darknet_exec + " detector test " +
                              data_file + " " + cfg_file + " " + weights_file + " " +
                              input_path + " -dont_show -ext_output < /dev/null 2>/dev/null"; // Redirect stderr

    std::vector<Detection> original_dets;
    
    // Print the command being executed
    // cout << "Executing: " << original_command << endl;
    
    // Opens the pipe for reading the output from the terminal 
    FILE* orig_pipe = popen(original_command.c_str(), "r");
    
    if (!orig_pipe) {
        cerr << "popen() failed for original image command!" << endl;
    } else {
        // Parse the output
        original_dets = parse_darknet_output(orig_pipe, coco_map);
        // Close the pipe and get return status
        int orig_ret = pclose(orig_pipe);
        int orig_exit_status = WEXITSTATUS(orig_ret);
        
        if (orig_exit_status != 0) {
            cerr << "Darknet failed on the original image (code " << orig_exit_status << ")\n";
            // DEBUG: Add specific error for code 127
            if (orig_exit_status == 127) {
                cerr << "ERROR 127: Command not found." << endl;
                cerr << "Please check the 'darknet_exec' variable path: " << darknet_exec << " ---" << endl;
            }
        }
    }

    // Move/rename Darknet output (predictions.jpg → original_annotated.png)
    string prediction_file = "predictions.jpg";
    string annotated_original = "../original_annotated.png";

    if (fs::exists(prediction_file)) {
        fs::rename(prediction_file, annotated_original);
        cout << "Annotated original image saved"<< endl;
    } else {
        cerr << "No predictions.jpg found for original image!\n";
    }
    
    // Save metadata for original image
    string original_metadata_file = "../original_metadata.json";
    save_metadata(original_dets, original_metadata_file);

 // ----------------------------------------- PREP FOR IMAGE SLICING ------------------------------------------------------

    // Here comes the adaptive tiling for image slicing 

    // Get the image's width and height to define the adaptive tiling.
    int width = image.width();
    int height = image.height();

    // Calculate padded dimensions.
    // This is used to create a squared figure so that the image can be sliced into all equal sizes
    int padded_width = ((width + tile_size - 1) / tile_size) * tile_size;
    int padded_height = ((height + tile_size - 1) / tile_size) * tile_size;

    // Just to display the before and after Image process (visualisation / reference)
    cout << "\nOriginal image size: " << width << "x" << height << endl;
    cout << "Padded image size:   " << padded_width << "x" << padded_height << endl;

    // Pad image - optional when to force the padding.
    VImage padded = image;
    if (padded_width != width || padded_height != height) {
        padded = image.embed(0, 0, padded_width, padded_height,
                             VImage::option()->set("extend", "black"));
    }

    // To count the number of tiles in X & Y axis.
    int tiles_x = padded_width / tile_size;
    int tiles_y = padded_height / tile_size;

    cout << "Creating " << tiles_x << "x" << tiles_y << " tiles (" << tiles_x * tiles_y << " total)\n";

    // Hardcoding save directories to for darknet to reference the tiles from
    string sliced_dir = "../saved_images";
    string annotated_dir = "../annotated_tiles";
    string output_dir = "../output_metadata";
    fs::create_directories(sliced_dir);
    fs::create_directories(annotated_dir);
    fs::create_directories(output_dir);

// ------------------------------------- IMAGE SLICING -----------------------------------------------------------
    // Slicing the image into tiles
    for (int y = 0; y < tiles_y; ++y) {
        for (int x = 0; x < tiles_x; ++x) {
            int left = x * tile_size;
            int top = y * tile_size;
            VImage tile = padded.extract_area(left, top, tile_size, tile_size);
            
            // to name the individual tiles and save them into the folder.
            string filename = sliced_dir + "/tile_" + to_string(x) + "_" + to_string(y) + ".png";
            tile.write_to_file(filename.c_str());
            cout << "Saved: " << filename << endl; // for visualisation in the terminal.
        }
    }

    ofstream logfile("../detection_log.txt");
    logfile << "Tile Detection Log\n";
    logfile << "=====================\n\n";

// ------------------------------- TILES PROCESSING - MODEL INFERENCE -------------------------------------------
    // Process each tile with Darknet
    for (int y = 0; y < tiles_y; ++y) {
        for (int x = 0; x < tiles_x; ++x) {
            // passing the tiles through darknet and saving the tiles as annotated.
            string tile_name = "tile_" + to_string(x) + "_" + to_string(y) + ".png";
            string tile_path = sliced_dir + "/" + tile_name;
            string annotated_tile = annotated_dir + "/annotated_" + tile_name;

            cout << "\n[Darknet] Processing " << tile_path << endl;

            // Construct command to call Darknet
            string command = darknet_exec + " detector test " +
                             data_file + " " + cfg_file + " " + weights_file + " " +
                             tile_path + " -dont_show -ext_output < /dev/null 2>/dev/null"; // Redirect stderr
            
            std::vector<Detection> final_dets;
            
            // Print the command being executed
            // cout << "Executing: " << command << endl;
            
            FILE* tile_pipe = popen(command.c_str(), "r");
            
            if (!tile_pipe) {
                cerr << "popen() failed for " << tile_name << endl;
                continue; // Skip this tile
            }

            final_dets = parse_darknet_output(tile_pipe, coco_map);
            int ret = pclose(tile_pipe);
            int exit_status = WEXITSTATUS(ret);

            if (exit_status != 0) {
                cerr << "Darknet exited with code " << exit_status << " for " << tile_name << "\n";
                // DEBUG: Add specific error for code 127
                if (exit_status == 127) {
                    cerr << "ERROR 127: Command not found." << endl;
                    cerr << "Please check the 'darknet_exec' variable path: " << darknet_exec << " ---" << endl;
                }
            }

            // Move/rename Darknet output (predictions.jpg → annotated_tile)
            string tile_prediction_file = "predictions.jpg"; // Darknet always outputs this
            if (fs::exists(tile_prediction_file)) {
                fs::rename(tile_prediction_file, annotated_tile);
                cout << "Annotated tile saved: " << annotated_tile << endl;
            } else {
                cerr << "No predictions.jpg found for " << tile_name << endl;
            }

            // Inside your tile loop, after getting final_dets:
            logfile << "annotated_tile_" << x << "_" << y << "\n";
            logfile << "Detections: " << final_dets.size() << "\n";

            for (const auto& det : final_dets) {
                float w = det.x2 - det.x1;
                float h = det.y2 - det.y1;

                logfile << "  - cls=" << det.cls
                        << ", class_name=" << coco_names[det.cls]
                        << ", score=" << det.score
                        << ", x1=" << det.x1
                        << ", y1=" << det.y1
                        << ", w=" << w
                        << ", h=" << h
                        << "\n";
            }

            logfile << std::endl;

            // Save JSON metadata for the tile 
            string metadata_file = output_dir + "/annotated_" + tile_name;
            size_t ext_pos = metadata_file.rfind(".png");
            if (ext_pos != string::npos) {
                metadata_file.replace(ext_pos, 4, ".json"); // replace ".png" with ".json"
            } else {
                // fallback if no .png extension (unlikely)
                metadata_file += ".json";
            }

            // Call the helper function to save the JSON
            save_metadata(final_dets, metadata_file);
        }
    }

    // Moved logfile.close() outside the loops
    logfile.close();
    cout << "Detection log saved to ../detection_log.txt" << endl;

    cout << "\nAll tiles processed. Reconstructing annotated image...\n";

// ------------------------------------------- IMAGE STITCHING ----------------------------------------------------
    // Stitch annotated tiles back together
    vector<VImage> row_images;
    for (int y = 0; y < tiles_y; ++y) {
        vector<VImage> row;
        for (int x = 0; x < tiles_x; ++x) {
            // The annotated tiles are called from its safed location and stitched back together.
            string filename = annotated_dir + "/annotated_tile_" + to_string(x) + "_" + to_string(y) + ".png";
            if (!fs::exists(filename)) {
                cerr << "Missing annotated tile: " << filename << endl;
                
                // If annotated tile is missing, use the *original* tile to prevent crash
                string original_tile_filename = sliced_dir + "/tile_" + to_string(x) + "_" + to_string(y) + ".png";
                if (fs::exists(original_tile_filename)) {
                    cerr << "Using original tile as fallback: " << original_tile_filename << endl;
                    row.push_back(VImage::new_from_file(original_tile_filename.c_str()));
                } else {
                    cerr << "ERROR: Original tile also missing! " << original_tile_filename << endl;
                    continue;
                }
            } else {
                 row.push_back(VImage::new_from_file(filename.c_str()));
            }
        }

        if (row.empty()) continue;

        // Joining the images together in the veritical manner
        VImage row_joined = VImage::arrayjoin(
            row,
            VImage::option()->set("across", static_cast<int>(row.size()))
        );
        row_images.push_back(row_joined);
    }

    // A method of troubleshooting if there are any errors.
    if (row_images.empty()) {
        cerr << "No rows were joined! Check annotated tile folder.\n";
        vips_shutdown();
        return 1;
    }
    // Using Libvips - arrayjoin to stitch the images back together in horizontal manner
    VImage stitched = VImage::arrayjoin(
        row_images,
        VImage::option()->set("across", 1)
    );

    // Crop back to original image size - remain data integrity
    stitched = stitched.extract_area(0, 0, width, height);

    // Saving the final annotated image for reference
    stitched.write_to_file("../annotated_stitched.png");
    cout << "Annotated stitched image saved as annotated_stitched.png\n";

    // Clean up the system
    vips_shutdown();
    cout << "\nDone!\n";
    return 0; 
}

