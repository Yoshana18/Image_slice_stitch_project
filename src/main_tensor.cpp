// The code i used to create the nanodet.engine file
// /usr/local/bin/trtexec --onnx=model/nanodet.onnx --saveEngine=model/nanodet.engine --fp16

#include <vips/vips8>
#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <cmath>
#include <fstream>
#include <memory>
#include <opencv4/opencv2/opencv.hpp> //opencv lib

// TensorRT - Nanodet - Cuda functions
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <cuda_runtime.h>

// For logs saving
#include <nlohmann/json.hpp>

// Using some namespaces - shorterns the codes.
using namespace std;
using namespace vips;
namespace fs = std::filesystem;
using namespace cv;
using namespace nvinfer1;

// Constants used for the codes
const float score_threshold = 0.3f; // the min score to reach so that it can store the annotation output after object detection
const float nms_threshold = 0.5f; // Non-Maximum Suppression - Hyperparameter in OD to eliminate redundant, overlapping bounding boxes for the same object
const int default_num_classes = 80; // The number of classes in the coco dataset

// Class names that nanodet is trained on (so hardcoded)
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

// Important Function (Required for annotation)
// A detection structure - to store the results of detection
// Common use in majority of the functions 
struct Detection {
    float x1, y1, x2, y2; // Bounding box coordinates - the 4 corners 
    float score;  // confidence score
    int cls;  //class_id
};

// Helper Function
// Logger for internal messages and for debugging
// This is a custom logger for TensorRT - it can print warnings, errors and debug messages when building or running the enine
class TRTLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress info-level messages
        if (severity <= Severity::kINFO)
            return;
        std::cerr << "[TRT] " << msg << std::endl;
    }
} gLogger; // to globalise the instance of the logger

// Helper function
// The nanodet.onnx file is converted into nanodet.engine file
// TensorRT inference works with this engine file and cuda stuff to make the inference running at high efficiency
// Used in TRTWrapper function - to read engine file into vector<char>
vector<char> readFile(const string &path) {
    ifstream file(path, ios::binary);
    if (!file) throw runtime_error("Failed to open engine file: " + path); // debug when engine is missing
    file.seekg(0, ios::end);     // Goes to the end
    size_t size = file.tellg();  // Get size
    vector<char> buf(size);      // Create buffer
    file.seekg(0, ios::beg);     // Go back to the beginning
    file.read(buf.data(), size); // This is to read the data correctly
    return buf; // return obj that would be used at other instances.
}

// Helper Function
// Compute volume from Dims (Dimensions)
// Used in Class TRTWrapper 
size_t volume(const Dims &d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= (d.d[i] <= 0 ? 1 : d.d[i]);
    return v;
}

// Helper Function
// Intersect Over Union (IOU) between two bounding boxes (Custom Calculation)
// Use in nms_per_clas function (below)
static float iou(const Detection &a, const Detection &b) {
    float x1 = max(a.x1, b.x1), y1 = max(a.y1, b.y1); // Intersection top-left X & Y
    float x2 = min(a.x2, b.x2), y2 = min(a.y2, b.y2); // Intersection of bottom-right X & Y
    // To compute intersecttion width/height
    float w = max(0.0f, x2 - x1);
    float h = max(0.0f, y2 - y1);

    float inter = w * h; // Intersection Area
    // To compute area of each bounding box
    float areaA = max(0.0f, a.x2 - a.x1) * max(0.0f, a.y2 - a.y1);
    float areaB = max(0.0f, b.x2 - b.x1) * max(0.0f, b.y2 - b.y1);
    return inter / (areaA + areaB - inter + 1e-6f); // IOU = Intersection / Union(with epsilon that avoids division by zero)
}

// Helper Function
// Per-class Non-Maximum Suppression (NMS)
// To remove duplicate overlapping bounding boxes.
// Used for run_postprocess function - nms to clear the excess bounding boxes 
vector<Detection> nms_per_class(const vector<Detection> &dets, float iou_th) {
    //Group detections by class ID
    unordered_map<int, vector<Detection>> bycls;
    for (auto &d : dets) bycls[d.cls].push_back(d);

    vector<Detection> out;
    // Process each class independently (correct behavior for multi-class NMS)
    for (auto &kv : bycls) {
        auto v = kv.second; // Extract detections that belongs to the specific Class
        // Sorting detections by decending confidence scores 
        sort(v.begin(), v.end(), [](const Detection &a, const Detection &b){ return a.score > b.score; });  
        vector<char> removed(v.size(), 0); // To track which detections are removed
        for (size_t i = 0; i < v.size(); ++i) { // Perfoms NMS
            if (removed[i]) continue; // to skip the bounding boxes that are already removed
            out.push_back(v[i]); // Compares with the remaining detections
            for (size_t j = i+1; j < v.size(); ++j) {
                if (removed[j]) continue;
                if (iou(v[i], v[j]) > iou_th) removed[j] = 1; // removes box j with high iou
            }
        }
    }
    // Returns all selected boxes across the differnt classes
    return out;
}

// Helper Function
// Softmax + integral decode for distributional regression
// To decode a distribution into a single continuous value using Integral (DFL) decoding
// "dist" - array of logits for each bin,"bins" - number of bins
// Used in run_postprocess function
float integral_decode(const float* dist, int bins) {
    // To find the max value for numerical stability in softmax
    float maxv = dist[0];
    for (int i = 1; i < bins; ++i) if (dist[i] > maxv) maxv = dist[i];
    // Computes exponentials of shifted logits and them add them together (softmax denominator)
    float sum = 0.f;
    vector<float> exps(bins);
    for (int i = 0; i < bins; ++i) {
        exps[i] = expf(dist[i] - maxv);
        sum += exps[i];
    }
    // To compute expected value numerator
    float exp_acc = 0.f;
    for (int i = 0; i < bins; ++i) exp_acc += exps[i] * i;
    // Final vlaue = expected value - sum[i*softmax(i) / sum(softmax(i))] 
    return exp_acc / (sum + 1e-12f);
}

// Helper Function
// To draw bounding boxes with annotation on the detected objects
// Draw detections on image (expects BGR(Libvips) mat) using OpenCv
// Used in the main funtion - for drawing annotation on the original image & the tiles 
void draw_detections(cv::Mat &bgr, const vector<Detection> &dets, const vector<string> &class_names) {
    for (const auto &d : dets) {
        // To draw a common color bounding box (green)
        cv::rectangle(bgr, cv::Point((int)d.x1, (int)d.y1), cv::Point((int)d.x2, (int)d.y2), cv::Scalar(0,255,0), 2);
        // string label = to_string(d.cls) + ":" + to_string(int(d.score*100)); // prints the score only (old code)
        
        // To include the classes of defect and the score (eg. Person:0.38)
        string label;
        if (d.cls >= 0 && d.cls < class_names.size()) {
            label = class_names[d.cls] + ":" + to_string(int(d.score*100)); // valid class index
        }
        else { // fallback for out-of-range class indices
            label = "ID(" + to_string(d.cls) + "):" + to_string(int(d.score*100));
        }
        // Computes text size for background padding
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        // To compute the top-left corner of the label box
        int x = max((int)d.x1, 0);
        int y = max((int)d.y1 - ts.height - 4, ts.height+4);
        // To draw a filled rectangle as the background for the text
        cv::rectangle(bgr, cv::Rect(x, y - ts.height - 2, ts.width + 6, ts.height + 4), cv::Scalar(0,255,0), -1);
        // cv::rectangle(img, cv::Point(det.x1, det.y1), cv::Point(det.x2, det.y2), cv::Scalar(0, 255, 0), 2);

        // To draw the label text in black colour
        cv::putText(bgr, label, cv::Point(x+3, y-3), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,0,0), 1);
    }
}

// Helper Function
// To save a vector of detections and some metadata to a json file 
// Used to save the metadata for all the individual tiles and the original image 
void save_metadata_json(const std::vector<Detection>& detections, const std::string& output_path) {
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
        cout << "Saved JSON metadata: " << output_path << endl;
    }
}

// Apparently this is a tensorRT Wrapper that is necessary to run the program - it also works as a debugger
// This wrapper is important as it works at the image ingestion part - from reading the engine to reading the received image.
// loads the engine file -> creates tensorRT runtime, engine, execution context -> query bindings -> allocates gpu buffers -> copy input data to gpu -> run inference -> copy output data back to gpu
// TensorRT wrapper that exposes binding info & inference
class TRTWrapper {
public:
    // Constructor : loads engine, allocates buffer, prepares context  
    TRTWrapper(const string &engine_path) {
        std::cout << "Attempting to read engine: " << engine_path << std::endl;
        vector<char> data = readFile(engine_path); // load engine file into memory
        if (data.empty()) {
            throw std::runtime_error("readFile failed or file is empty. Check path: " + engine_path);
        }
        std::cout << "Successfully read " << data.size() << " bytes from engine file." << std::endl;
        // Creates TensorRT Runtime
        runtime_.reset(createInferRuntime(gLogger));
        if (!runtime_) throw runtime_error("createInferRuntime failed");
        // Deserialise engine from file buffer
        engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
        if (!engine_) throw runtime_error("deserializeCudaEngine failed");
        // Creates execution context used for inference
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw runtime_error("createExecutionContext failed");

        nbBindings_ = engine_->getNbIOTensors(); // query number of i/o tensors
        // Storage structures for the binding metadata
        bindingNames_.resize(nbBindings_);
        bindingDims_.resize(nbBindings_);
        bindingSizes_.resize(nbBindings_);
        bindingIsInput_.resize(nbBindings_);

        // Loop through all bindings and extract metadata
        for (int b = 0; b < nbBindings_; ++b) {
            const char* name = engine_->getIOTensorName(b); // binding names
            bindingNames_[b] = std::string(name);
            bindingDims_[b] = engine_->getTensorShape(name); // tensor dimensions
            bindingIsInput_[b] = (engine_->getTensorIOMode(bindingNames_[b].c_str()) == nvinfer1::TensorIOMode::kINPUT); // checks whether the binding is i/o

            // compute size in bytes (float assumed)
            size_t vol = volume(bindingDims_[b]);
            bindingSizes_[b] = vol * sizeof(float);
        }
        // To allocate GPU memory for each bindings / tensor
        buffers_.resize(nbBindings_, nullptr);
        for (int i = 0; i < nbBindings_; ++i) {
            cudaError_t e = cudaMalloc(&buffers_[i], bindingSizes_[i]);
            if (e != cudaSuccess) {
                for (int j = 0; j < i; ++j) if (buffers_[j]) cudaFree(buffers_[j]);
                throw runtime_error("cudaMalloc failed for binding " + to_string(i));
            }
        }
        // Creates a CUDA stream for async operation
        cudaStreamCreate(&stream_);
        // Assign GPU buffers to execution context
        for (int i = 0; i < nbBindings_; ++i) {
            context_->setTensorAddress(bindingNames_[i].c_str(), buffers_[i]);
        }
    }

    // Destructor: free GPU memory and destroy CUDA Streams
    ~TRTWrapper() {
        for (auto &b : buffers_) if (b) cudaFree(b);
        cudaStreamDestroy(stream_);
    }
    // These are the accessor functions for binding metadata
    int nbBindings() const { return nbBindings_; }
    const nvinfer1::Dims& getBindingDims(int idx) const { return bindingDims_[idx]; }
    bool isInput(int idx) const { return bindingIsInput_[idx]; }
    size_t getBindingSizeBytes(int idx) const { return bindingSizes_[idx]; }
    ICudaEngine* engine() const { return engine_.get(); }

    // Runs inference. Copy input + GPU -> Run Engine -> Stream Sync
    bool infer(const vector<float> &hostInput, int inputBindingIndex) {
        if (inputBindingIndex < 0 || inputBindingIndex >= nbBindings_) return false; // validate the binding index
        // Number of input bytes
        size_t inputBytes = hostInput.size() * sizeof(float);
        size_t expected = getBindingSizeBytes(inputBindingIndex);
        // Warns if user provided more floats then required - troubleshoot
        if (inputBytes > expected) {
            cerr << "Warning: hostInput larger than binding bytes. Copying truncated." << endl;
        }
        // Copy CPU -> GPU Asynchronously
        cudaMemcpyAsync(buffers_[inputBindingIndex], hostInput.data(), min(inputBytes, expected), cudaMemcpyHostToDevice, stream_);
        bool ok = context_->enqueueV3(stream_);
        if (!ok) {
            cerr << "Inference enqueueV3 failed\n";
            return false;
        }
        // Wait for GPU to finish
        cudaStreamSynchronize(stream_);
        return true;
    }
    // Copy output tensor from GPU -> CPU
    bool copyBindingToHost(int bindingIndex, vector<float> &out) {
        if (bindingIndex < 0 || bindingIndex >= nbBindings_) return false;
        size_t bytes = getBindingSizeBytes(bindingIndex);
        size_t floats = bytes / sizeof(float);
        out.resize(floats); // Resize host output buffer
        // Copy GPU -> CPU
        cudaMemcpyAsync(out.data(), buffers_[bindingIndex], bytes, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_); // Ensures copy finished
        return true;
    }
    // The binding names and dimensions are stored in public
    vector<string> bindingNames_;
    vector<nvinfer1::Dims> bindingDims_;

    // Other tensorRT objects stored in private
private:
    unique_ptr<nvinfer1::IRuntime> runtime_;
    unique_ptr<nvinfer1::ICudaEngine> engine_;
    unique_ptr<nvinfer1::IExecutionContext> context_;
    int nbBindings_{0}; // number of i/o tensors
    vector<size_t> bindingSizes_; // size in bytes for each bindings
    vector<void*> buffers_; // GPU buffer pointers
    vector<char> bindingIsInput_; // whether bindings is i/o
    cudaStream_t stream_; // CUDA stream for async operations
};

// Helper Function 
// Preprocess image (BGR) into NCHW float (RGB normalized to [0,1])
// NCHW -> Number of Samples, Channel, Height, Width
// Used in the main function to preprocess the original image and the individual tiles 
void preprocess_to_nchw(const cv::Mat &img_bgr, vector<float> &out, int model_w, int model_h) {
    cv::Mat rgb, resized;
    cv::cvtColor(img_bgr, rgb, cv::COLOR_BGR2RGB);
    
    if (img_bgr.cols != model_w || img_bgr.rows != model_h)
        cv::resize(rgb, resized, cv::Size(model_w, model_h));
    else
        resized = rgb;

    cv::Mat floatm;
    resized.convertTo(floatm, CV_32F, 1.0f / 255.0f);
    vector<cv::Mat> ch(3);
    cv::split(floatm, ch);
    out.clear();
    out.reserve(3 * model_w * model_h);
    
    // ORDER: R, G, B
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < model_h; ++r) {
            const float* ptr = ch[c].ptr<float>(r);
            for (int x = 0; x < model_w; ++x) out.push_back(ptr[x]);
        }
    }
}

// Helper Function
// To print binding dims (debug)
// Used in the main function to query the bindings 
string dimToString(const Dims &d) {
    string s;
    for (int i = 0; i < d.nbDims; ++i) {
        if (i) s += "x";
        s += to_string(d.d[i]);
    }
    return s;
}

// Helper Function
// Structure for binding info
// Used for generate_grid_points and run_postprocess
struct gridPoint {
    float cx, cy, stride;
}; 

// Generates the 3598 grid points for a 416x416 model
vector<gridPoint> generate_grid_points(int model_w, int model_h) {
    vector<gridPoint> grids;
    for (int stride : {8, 16, 32}) {
        int grid_w = model_w / stride;
        int grid_h = model_h / stride;
        for (int y = 0; y < grid_h; ++y) {
            for (int x = 0; x < grid_w; ++x) {
                grids.push_back({(x + 0.5f) * stride, (y + 0.5f) * stride, (float)stride});
            }
        }
    }
    cout << "Generated " << grids.size() << " grid points." << endl;
    return grids;
}

// Helper Function
// Scales detections from model coordinates to original image coordinates
// Used for original image detection only - in the main function 
void scale_detections(vector<Detection> &dets, float scale_x, float scale_y) {
    for (auto &d : dets) {
        d.x1 *= scale_x;
        d.y1 *= scale_y;
        d.x2 *= scale_x;
        d.y2 *= scale_y;
    }
}

// converts the raw TensorRT model output tensor into final object detections by selecting the best class,
// decoding bounding box offsets using grid points, filtering by score, and applying NMS to return clean bounding boxes.
// Post-process function for a single [1x3598x112] output
// Used in the main function - original images and tiles images detections 
vector<Detection> run_postprocess(TRTWrapper &trt,
                                  int outputBindingIndex,
                                  int num_classes, int bins,
                                  int model_w, int model_h,
                                  const vector<gridPoint> &grids)
{
    vector<Detection> all_detections;
    
    // Copy the single, huge output tensor from GPU to CPU
    vector<float> host_output;
    if (!trt.copyBindingToHost(outputBindingIndex, host_output)) {
        cerr << "Failed to copy output binding to host." << endl;
        return {};
    }

    size_t num_grids = grids.size(); // Should be 3549
    size_t num_outputs = host_output.size() / 112; // Should be 3598

    // Use the smaller of the two sizes to avoid crashes
    size_t num_detections_to_process = min(num_grids, num_outputs);

    // This model combines score and class (no 'centerness')
    // The 112 channels are: 80 class scores + 32 regression bins
    const int num_classes_in_model = 80; // Hard-coded based on model
    const int num_bins_in_model = 32;    // Hard-coded based on model
    const int output_stride = num_classes_in_model + num_bins_in_model; // 112

    for (size_t i = 0; i < num_detections_to_process; ++i) {
        const float* data_ptr = &host_output[i * output_stride];
        
        // Find the best class score
        const float* scores_ptr = data_ptr;
        int best_cls = -1;
        float best_score = 0.f;
        
        for (int c = 0; c < num_classes_in_model; ++c) {
            if (scores_ptr[c] > best_score) {
                best_score = scores_ptr[c];
                best_cls = c;
            }
        }

        // Check score against threshold
        if (best_score < score_threshold) {
            continue;
        }

        // Decode regression bins
        const float* bins_ptr = data_ptr + num_classes_in_model;
        
        float dist_l[32], dist_t[32], dist_r[32], dist_b[32];
        for (int bidx = 0; bidx < bins; ++bidx) {
            // Re-order bins: [l,l,l... t,t,t... r,r,r... b,b,b...]
            dist_l[bidx] = bins_ptr[bidx];
            dist_t[bidx] = bins_ptr[bidx + bins];
            dist_r[bidx] = bins_ptr[bidx + bins * 2];
            dist_b[bidx] = bins_ptr[bidx + bins * 3];
        }

        float l = integral_decode(dist_l, bins);
        float t = integral_decode(dist_t, bins);
        float r = integral_decode(dist_r, bins);
        float breg = integral_decode(dist_b, bins);

        // Get grid point
        const auto& grid = grids[i];
        
        // Calculate final box
        float x0 = grid.cx - l * grid.stride;
        float y0 = grid.cy - t * grid.stride;
        float x1 = grid.cx + r * grid.stride;
        float y1 = grid.cy + breg * grid.stride;

        // Clamp to model size
        x0 = max(0.f, min((float)model_w, x0));
        y0 = max(0.f, min((float)model_h, y0));
        x1 = max(0.f, min((float)model_w, x1));
        y1 = max(0.f, min((float)model_h, y1));

        if (x1 <= x0 || y1 <= y0) continue;

        all_detections.push_back({ x0, y0, x1, y1, best_score, best_cls });
    }

    auto final_dets = nms_per_class(all_detections, nms_threshold);
    return final_dets;
}

// Main function to run the code (this is the execution function)
int main(int argc, char **argv) {
    // Initialise Livips
    if (VIPS_INIT(argv[0])) {
        vips_error_exit(nullptr);
        return 1;
    } 
    // This checks for the input image in the terminal - need to tag the image, engine and classes 
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " input.png [engine.engine] [num_classes] [bins]\n";
        vips_shutdown();
        return 1;
    }
    // To read the paths from the terminal & to double check 
    string input_path = argv[1];
    string engine_path = (argc >= 3) ? argv[2] : "nanodet.engine";
    int num_classes = (argc >= 4) ? stoi(argv[3]) : default_num_classes;
    int forced_bins = (argc >= 5) ? stoi(argv[4]) : 0; // 0 = auto-detect

    cout << "Input: " << input_path << "\nEngine: " << engine_path << "\nNum classes: " << num_classes << "\nBins (0=auto): " << forced_bins << endl;

    // Load image with OpenCV (used for both workflows)
    cv::Mat original_bgr = cv::imread(input_path);
    if (original_bgr.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        vips_shutdown();
        return 1;
    }
    int original_w = original_bgr.cols;
    int original_h = original_bgr.rows;
    cout << "Original image size: " << original_w << "x" << original_h << endl;

    // Create TRT wrapper
    TRTWrapper trt(engine_path); // loads the nanodet engine 
    // Query how many bindings i/o the engine has. Nanodet Engine: 1 input + 1 output = 2 bindings 
    int nb = trt.nbBindings(); // number of bindings 
    cout << "Engine has " << nb << " bindings\n";

    // Identify input binding, output bindings, and model size
    int inputBinding = -1; // stores the index of the model input 
    vector<int> outBindings; // list of output binding indices 
    int model_w = 0, model_h = 0; // model input width and height 

    for (int i = 0; i < nb; ++i) {
        // query the shape (dimensions) of this binding 
        auto dims = trt.getBindingDims(i);
        cout << "Binding " << i << " (" << trt.bindingNames_[i] << ") dims: " << dimToString(dims) << " isInput=" << trt.isInput(i) << "\n";
        if (trt.isInput(i)) { // if the binding is the input tensor 
            if (inputBinding < 0) { // Store the input binding index 
                inputBinding = i;
                // Assuming NCHW format [N, C, H, W]
                if (dims.nbDims == 4) {
                    model_h = dims.d[2];
                    model_w = dims.d[3];
                } else {
                    cerr << "Warning: Input binding is not 4D (NCHW). Tiling logic might fail." << endl;
                }
            }
        } else {
            outBindings.push_back(i); // Otherwise, this is an output binding 
        }
    }

    if (inputBinding < 0) {
        cerr << "No input binding found in engine\n";
        vips_shutdown();
        return 1;
    }
    if (model_w == 0 || model_h == 0) {
        cerr << "Could not detect model input W/H from binding. Aborting." << endl;
        vips_shutdown();
        return 1;
    }
    cout << "Using input binding: " << inputBinding << endl;
    cout << "Detected Model Input Size: " << model_w << "x" << model_h << endl;
    
    // Find the single output binding
    int outputBinding = -1;
    for (int i = 0; i < nb; ++i) {
        if (!trt.isInput(i)) {
            outputBinding = i; // take the first non-input as the output 
            break;
        }
    }
    if (outputBinding == -1) {
        cerr << "No output binding found!" << endl;
        vips_shutdown();
        return 1;
    }
    cout << "Using output binding: " << outputBinding << endl;

    // Generate grid points
    // These grids correspond to each feature map point 
    auto grids = generate_grid_points(model_w, model_h);
    
    // Auto-detect bins from model output shape (112)
    int detected_bins = (trt.bindingDims_[outputBinding].d[2] - num_classes) / 4;
    // If the user eplicitly sets a bin value, override auto-detection 
    if (forced_bins != 0) {
        detected_bins = forced_bins;
    // If auto-detection fails, use default 
    } else if (detected_bins <= 0) {
        cout << "Could not auto-detect bins, defaulting to 8." << endl;
        detected_bins = 8;
    }
    cout << "Using bins = " << detected_bins << "\n";

// --------------------------------------- ORIGINAL IMAGE - MODEL INFERENCE --------------------------------------------
    // INFERENCE ON ORIGINAL (RESIZED) IMAGE
    cout << "\n Running inference on original (resized) image" << endl;
    vector<float> input_tensor_orig;
    preprocess_to_nchw(original_bgr, input_tensor_orig, model_w, model_h);

    if (!trt.infer(input_tensor_orig, inputBinding)) {
        cerr << "Inference failed for original image." << endl;
    } else {
        vector<Detection> dets_orig = run_postprocess(trt, outputBinding, num_classes, detected_bins, model_w, model_h, grids);
        cout << "Found " << dets_orig.size() << " detections (before scaling)." << endl;

        // Scale detections from model space (e.g: 416x416) to original space (e.g., 2749x3879)
        float scale_x = (float)original_w / model_w;
        float scale_y = (float)original_h / model_h;
        scale_detections(dets_orig, scale_x, scale_y);

        cv::Mat annotated_orig_bgr = original_bgr.clone();
        draw_detections(annotated_orig_bgr, dets_orig, coco_names);
        cv::imwrite("../original_annotated.png", annotated_orig_bgr);
        cout << "Saved: original_annotated.png" << endl;

        // Save JSON metadata for original image
        save_metadata_json(dets_orig, "../original_metadata.json");
    }

    ofstream logfile("../detection_log.txt");
    logfile << "Tile Detection Log\n";
    logfile << "=====================\n\n";

 // ----------------------------------------- PREP FOR IMAGE SLICING ------------------------------------------------------
    // TILED INFERENCE
    cout << "\nStarting tiled inference workflow" << endl;

    // Convert OpenCV Mat to VImage for slicing
    cv::Mat rgb_mat;
    cv::cvtColor(original_bgr, rgb_mat, cv::COLOR_BGR2RGB);
    if (!rgb_mat.isContinuous()) rgb_mat = rgb_mat.clone();
    
    VImage image = VImage::new_from_memory(
        rgb_mat.data,
        rgb_mat.total() * rgb_mat.elemSize(),
        rgb_mat.cols, rgb_mat.rows, rgb_mat.channels(),
        VIPS_FORMAT_UCHAR
    );

    int padded_width = ((original_w + model_w - 1) / model_w) * model_w;
    int padded_height = ((original_h + model_h - 1) / model_h) * model_h;
    cout << "Padded image size:   " << padded_width << "x" << padded_height << endl;

    VImage padded = image;
    if (padded_width != original_w || padded_height != original_h) {
        padded = image.embed(0, 0, padded_width, padded_height,
                             VImage::option()->set("extend", "black"));
    }

    int tiles_x = padded_width / model_w;
    int tiles_y = padded_height / model_h;
    cout << "Creating " << tiles_x << "x" << tiles_y << " tiles (" << tiles_x * tiles_y << " total)\n";

    string annotated_dir = "../annotated_tiles";
    string output_dir = "../output_metadata"; // For JSON files
    fs::create_directories(annotated_dir);
    fs::create_directories(output_dir); // Create metadata directory

// ---------------------- IMAGE SLICING + MODEL INFERENCE PROCESSING -----------------
    // Slicing and Processing Loop
    for (int y = 0; y < tiles_y; ++y) {
        for (int x = 0; x < tiles_x; ++x) {
            int left = x * model_w;
            int top = y * model_h;
            VImage tile_vips = padded.extract_area(left, top, model_w, model_h);
            
            string tile_name = "tile_" + to_string(x) + "_" + to_string(y) + ".png";
            string annotated_tile = annotated_dir + "/annotated_" + tile_name;

            // Fast way: Vips -> memory -> OpenCV
            size_t tile_data_size;
            void* tile_data = tile_vips.write_to_memory(&tile_data_size);
            cv::Mat tile_rgb(model_h, model_w, CV_8UC3, tile_data);
            cv::Mat tile_bgr;
            cv::cvtColor(tile_rgb, tile_bgr, cv::COLOR_RGB2BGR);
            free(tile_data); // Don't forget to free libvips memory!
            
            // Preprocess to NCHW floats
            vector<float> inputTensor;
            preprocess_to_nchw(tile_bgr, inputTensor, model_w, model_h);
            
            if (!trt.infer(inputTensor, inputBinding)) {
                cerr << "Inference failed for tile " << x << "," << y << "\n";
                continue;
            }

            // Post-process this tile
            vector<Detection> final_dets = run_postprocess(trt, outputBinding, num_classes, detected_bins, model_w, model_h, grids);
            
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

            // Draw on BGR tile and save
            draw_detections(tile_bgr, final_dets, coco_names);
            imwrite(annotated_tile, tile_bgr);
            cout << "Saved: " << annotated_tile << " dets=" << final_dets.size() << "\n";

            // Save JSON metadata for the tile
            string metadata_file = output_dir + "/annotated_" + tile_name;
            size_t ext_pos = metadata_file.rfind(".png");
            if (ext_pos != string::npos) {
                metadata_file.replace(ext_pos, 4, ".json"); // replace ".png" with ".json"
            } else {
                metadata_file += ".json";
            }
            save_metadata_json(final_dets, metadata_file);
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
            string fname = annotated_dir + "/annotated_tile_" + to_string(x) + "_" + to_string(y) + ".png";
            if (!fs::exists(fname)) {
                cerr << "Missing annotated tile: " << fname << " (skipping)\n";
                VImage black = VImage::black(model_w, model_h).copy(VImage::option());
                row.push_back(black);
                continue;
            }
            row.push_back(VImage::new_from_file(fname.c_str()));
        }
        // Joining the images together in the veritical manner
        VImage row_joined = VImage::arrayjoin(row, VImage::option()->set("across", (int)row.size()));
        row_images.push_back(row_joined);
    }

    // A method of troubleshooting if there are any errors.
    if (row_images.empty()) {
        cerr << "No rows were joined! Check annotated tile folder.\n";
        vips_shutdown();
        return 1;
    }

    // Using Libvips - arrayjoin to stitch the images back together in horizontal manner
    VImage stitched = VImage::arrayjoin(row_images, VImage::option()->set("across", 1));

    // Crop back to original image size
    stitched = stitched.extract_area(0, 0, original_w, original_h);

    // Saving the final annotated image for reference
    stitched.write_to_file("../annotated_stitched.png");
    cout << "Annotated stitched image saved as annotated_stitched.png\n";

    // Clean up the system 
    vips_shutdown();
    cout << "\nDone!\n";
    return 0;
}