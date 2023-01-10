#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "ccl.h"

#include <torch/script.h> // One-stop header.
#include "torch/torch.h"
#include "torch/jit.h"
#include <ATen/cuda/CUDAContext.h>

#define MAX_INPUT_SIZE 1500
#define MIN_INPUT_SIZE 128
#define OPT_INPUT_W 640
#define OPT_INPUT_H 640
#define DEVICE 0
#define POOLING_SIZE 9
#define BLOCK_ROWS 16
#define BLOCK_COLS 16
#define USE_FP16
#define BATCH_SIZE 32
#define SPEED_TEST_W 672
#define SPEED_TEST_H 448


#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

// FAST-B
//const std::map<std::string, std::vector<std::vector<int>>> kernel_sizes = {
//    {"stage1", {{3, 3}, {3, 3}, {3, 1}, {3, 3}, {3, 1}, {3, 3}, {3, 3}, {1, 3}, {3, 3}, {3, 3}}},
//    {"stage2", {{3, 3}, {1, 3}, {3, 3}, {3, 1}, {3, 3}, {3, 3}, {3, 1}, {3, 1}, {3, 3}, {3, 3}}},
//    {"stage3", {{3, 3}, {3, 3}, {3, 3}, {1, 3}, {3, 3}, {3, 1}, {3, 3}, {3, 1}}},
//    {"stage4", {{3, 3}, {1, 3}, {3, 1}, {3, 1}, {1, 3}}}
//};

// FAST-S
//const std::map<std::string, std::vector<std::vector<int>>> kernel_sizes = {
//    {"stage1", {{3, 3}, {3, 3}}},
//    {"stage2", {{3, 3}, {1, 3}, {3, 3}, {3, 1}, {3, 3}, {3, 1}, {1, 3}, {3, 3}}},
//    {"stage3", {{3, 3}, {3, 3}, {1, 3}, {3, 1}, {3, 3}, {1, 3}, {3, 1}, {3, 3}}},
//    {"stage4", {{3, 3}, {3, 1}, {1, 3}, {1, 3}, {3, 1}}}
//};

// FAST-T
const std::map<std::string, std::vector<std::vector<int>>> kernel_sizes = {
    {"stage1", {{3, 3}, {3, 3}, {3, 3}}},
    {"stage2", {{3, 3}, {1, 3}, {3, 3}, {3, 1}}},
    {"stage3", {{3, 3}, {3, 3}, {3, 1}, {1, 3}}},
    {"stage4", {{3, 3}, {3, 1}, {1, 3}, {3, 3}}}
};

static const int SHORT_INPUT = 800;

const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;
namespace F = torch::nn::functional;
static Logger gLogger;


// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }
    std::cout << "Finished Load weights: " << file << std::endl;
    return weightMap;
}

IActivationLayer *FirstBlock(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int inch, int outch, int stride, int groups, std::string lname)
{
    IConvolutionLayer *conv = network->addConvolutionNd(input, outch, DimsHW{3, 3}, weightMap[lname + "conv.weight"], weightMap[lname + "conv.bias"]);
    conv->setStrideNd(DimsHW{stride, stride});
    conv->setPaddingNd(DimsHW{1, 1});
    conv->setNbGroups(groups);
    assert(conv);
    IActivationLayer *relu = network->addActivation(*conv->getOutput(0), ActivationType::kRELU);
    assert(relu);
    return relu;
}

IActivationLayer *ConvBlock(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int inch, int outch, int kh, int kw, int stride, int groups, std::string lname)
{
    std::cout << lname << " | " << "[" << inch << ", " << outch << ", (" << kh << ", " << kw << "), " << stride << ", " << groups << "]" << std::endl;
    IConvolutionLayer *conv = network->addConvolutionNd(input, outch, DimsHW{kh, kw}, weightMap[lname + "fused_conv.weight"], weightMap[lname + "fused_conv.bias"]);
    conv->setStrideNd(DimsHW{stride, stride});
    if (kh == 3 && kw == 1) {
        conv->setPaddingNd(DimsHW{1, 0});
    } else if (kh == 1 && kw == 3) {
        conv->setPaddingNd(DimsHW{0, 1});
    } else {
        conv->setPaddingNd(DimsHW{1, 1});
    }
    conv->setNbGroups(groups);
    assert(conv);
    IActivationLayer *relu = network->addActivation(*conv->getOutput(0), ActivationType::kRELU);
    assert(relu);
    return relu;
}

IActivationLayer *makeStage(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int inch, int outch, int stride, int blocks, std::string lname, std::vector<std::vector<int>> kernels)
{
    IActivationLayer *layer;
    for (int i = 0; i < blocks; ++i)
    {
        std::vector<int> kernel = kernels.at(i);
        int kh = kernel.at(0);
        int kw = kernel.at(1);
        if (lname.compare("backbone.stage1.") == 0) // equal
        {
            if (i == 0)
                layer = ConvBlock(network, weightMap, input, inch, outch, kh, kw, 1, 1, lname + std::to_string(i) + ".");
            else if (i == 1)
                layer = ConvBlock(network, weightMap, *layer->getOutput(0), outch, outch, kh, kw, 2, 1, lname + std::to_string(i) + ".");
            else
                layer = ConvBlock(network, weightMap, *layer->getOutput(0), outch, outch, kh, kw, 1, 1, lname + std::to_string(i) + ".");
        }
        else
        {
            if (i == 0)
                layer = ConvBlock(network, weightMap, input, inch, outch, kh, kw, 2, 1, lname + std::to_string(i) + ".");
            else
                layer = ConvBlock(network, weightMap, *layer->getOutput(0), outch, outch, kh, kw, 1, 1, lname + std::to_string(i) + ".");
        }
    }
    return layer;
}

// Creat the engine using only the API and not any parser.
ICudaEngine *createEngine(std::string netName, IBuilder *builder, IBuilderConfig *config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(1U);

    // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims4{BATCH_SIZE, 3, SPEED_TEST_H, SPEED_TEST_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../" + netName + ".wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    auto stage0 = FirstBlock(network, weightMap, *data, 3, 64, 2, 1, "backbone.first_conv.");
    assert(stage0);

    std::vector<std::vector<int>> kernels = kernel_sizes.at("stage1");
    auto stage1 = makeStage(network, weightMap, *stage0->getOutput(0), 64, 64, 2, kernels.size(), "backbone.stage1.", kernels);
    assert(stage1);
    kernels = kernel_sizes.at("stage2");
    auto stage2 = makeStage(network, weightMap, *stage1->getOutput(0), 64, 128, 2, kernels.size(), "backbone.stage2.", kernels);
    assert(stage2);
    kernels = kernel_sizes.at("stage3");
    auto stage3 = makeStage(network, weightMap, *stage2->getOutput(0), 128, 256, 2, kernels.size(), "backbone.stage3.", kernels);
    assert(stage3);
    kernels = kernel_sizes.at("stage4");
    auto stage4 = makeStage(network, weightMap, *stage3->getOutput(0), 256, 512, 2, kernels.size(), "backbone.stage4.", kernels);
    assert(stage4);

    auto reduce1 = ConvBlock(network, weightMap, *stage1->getOutput(0), 64, 128, 3, 3, 1, 1, "neck.reduce_layer1.");
    assert(reduce1);
    auto reduce1_shape = network->addShape(*reduce1->getOutput(0))->getOutput(0);
    assert(reduce1_shape);

    auto reduce2 = ConvBlock(network, weightMap, *stage2->getOutput(0), 128, 128, 3, 3, 1, 1, "neck.reduce_layer2.");
    assert(reduce2);
    IResizeLayer* reduce2_resize = network->addResize(*reduce2->getOutput(0));
    reduce2_resize->setInput(1, *reduce1_shape);
    reduce2_resize->setResizeMode(ResizeMode::kNEAREST);
    reduce2_resize->setAlignCorners(false);
    assert(reduce2_resize);

    auto reduce3 = ConvBlock(network, weightMap, *stage3->getOutput(0), 256, 128, 3, 3, 1, 1, "neck.reduce_layer3.");
    assert(reduce3);
    IResizeLayer* reduce3_resize = network->addResize(*reduce3->getOutput(0));
    reduce3_resize->setInput(1, *reduce1_shape);
    reduce3_resize->setResizeMode(ResizeMode::kNEAREST);
    reduce3_resize->setAlignCorners(false);
    assert(reduce3_resize);

    auto reduce4 = ConvBlock(network, weightMap, *stage4->getOutput(0), 512, 128, 3, 3, 1, 1, "neck.reduce_layer4.");
    assert(reduce4);
    IResizeLayer* reduce4_resize = network->addResize(*reduce4->getOutput(0));
    reduce4_resize->setInput(1, *reduce1_shape);
    reduce4_resize->setResizeMode(ResizeMode::kNEAREST);
    reduce4_resize->setAlignCorners(false);
    assert(reduce4_resize);

    ITensor* inputTensors[] = {reduce1->getOutput(0), reduce2_resize->getOutput(0), reduce3_resize->getOutput(0), reduce4_resize->getOutput(0)};
    IConcatenationLayer* neck_cat = network->addConcatenation(inputTensors, 4);
    assert(neck_cat);

    auto head_conv = ConvBlock(network, weightMap, *neck_cat->getOutput(0), 512, 128, 3, 3, 1, 1, "det_head.conv.");
    assert(head_conv);

    auto head_final = network->addConvolutionNd(*head_conv->getOutput(0), 1, DimsHW{1, 1}, weightMap["det_head.final.conv.weight"], emptywts);
    head_final->setStrideNd(DimsHW{1, 1});
    head_final->setPaddingNd(DimsHW{0, 0});
    assert(head_final);

    head_final->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "Set name out" << std::endl;
    network->markOutput(*head_final->getOutput(0));

    // Set profile
//    IOptimizationProfile* profile = builder->createOptimizationProfile();
//    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMIN, Dims4(BATCH_SIZE, 3, MIN_INPUT_SIZE, MIN_INPUT_SIZE));
//    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kOPT, Dims4(BATCH_SIZE, 3, OPT_INPUT_H, OPT_INPUT_W));
//    profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMAX, Dims4(BATCH_SIZE, 3, MAX_INPUT_SIZE, MAX_INPUT_SIZE));
//    config->addOptimizationProfile(profile);

    // Build engine
    builder->setMaxBatchSize(BATCH_SIZE);
    config->setMaxWorkspaceSize(1 << 34); // 1G

#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << builder->platformHasFastInt8() << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto &mem : weightMap)
    {
        free((void *)(mem.second.values));
    }
    return engine;
}

void serializeEngine(std::string netName, IHostMemory **modelStream)
{
    // Create builder
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = createEngine(netName, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

auto ToCvImage(at::Tensor tensor)
{
    int width = tensor.sizes()[0];
    int height = tensor.sizes()[1];
    try
    {
        cv::Mat output_mat(cv::Size{ height, width }, CV_8UC1, tensor.data_ptr<uchar>());
        cv::imwrite("temp.png", output_mat);
        std::cout << "converted image from tensor" << std::endl;
        return output_mat.clone();
    }
    catch (const c10::Error& e)
    {
        std::cout << "an error has occured : " << e.msg() << std::endl;
    }
    return cv::Mat(height, width, CV_8UC1);
}


void doInference(IExecutionContext &context, cudaStream_t& stream, void **buffers, float *input, float *output, int input_h, int input_w)
{
    int input_size = BATCH_SIZE * 3 * input_h * input_w * sizeof(float);
    int output_size = BATCH_SIZE * input_h * input_w / 16 * sizeof(float);

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[0], input, input_size, cudaMemcpyHostToDevice, stream));
    context.enqueueV2(buffers, stream, nullptr);
    cudaDeviceSynchronize();

    torch::Tensor out = torch::from_blob(buffers[1], {BATCH_SIZE, 1, int(input_h/4), int(input_w/4)}).cuda().toType(torch::kFloat64);
    torch::Tensor texts = F::interpolate(out, F::InterpolateFuncOptions().size(std::vector<int64_t>({int(input_h/2), int(input_w/2)})).mode(torch::kNearest)); // B*1*320*320
    texts = torch::max_pool2d(texts, POOLING_SIZE/2+1, 1, POOLING_SIZE/2/2, 1, false);

    torch::Tensor score_maps = torch::sigmoid(texts);
    score_maps = F::interpolate(score_maps, F::InterpolateFuncOptions().size(std::vector<int64_t>({input_h, input_w})).mode(torch::kNearest)); // B*1*640*640
    score_maps = score_maps.squeeze(1);

    torch::Tensor kernels = (out.squeeze(1) > 0).toType(torch::kUInt8);  // B*160*160
    torch::Tensor labels_ = connected_componnets_labeling_2d_batch(kernels);  // B*160*160
    torch::Tensor labels = labels_.unsqueeze(1).toType(torch::kFloat32);
    labels = F::interpolate(labels, F::InterpolateFuncOptions().size(std::vector<int64_t>({input_h/2, input_w/2})).mode(torch::kNearest)); // B*1*320*320
    labels = torch::max_pool2d(labels, POOLING_SIZE/2+1, 1, POOLING_SIZE/2/2, 1, false);
    labels = F::interpolate(labels, F::InterpolateFuncOptions().size(std::vector<int64_t>({input_h, input_w})).mode(torch::kNearest)); // B*1*640*640
    labels = labels.squeeze(1).toType(torch::kInt32); // B*640*640

    cudaStreamSynchronize(stream);
    // Release stream and buffers
//    cudaStreamDestroy(stream);
//    CHECK(cudaFree(buffers[inputIndex]));
//    CHECK(cudaFree(buffers[outputIndex]));
}

float* preProcess(cv::Mat image, int& resize_h, int& resize_w, float& ratio_h, float& ratio_w)
{
    cv::Mat imageRGB;
    cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
    cv::Mat imageProcessed;
    int h = imageRGB.size().height;
    int w = imageRGB.size().width;

    int min_side = h < w ? h : w;
    float scale = SHORT_INPUT * 1.0 / float(min_side);

//    resize_h = int(h * scale + 0.5);
//    resize_w = int(w * scale + 0.5);
    resize_h = SPEED_TEST_H;
    resize_w = SPEED_TEST_W;

    if (resize_h % 32 != 0)
        resize_h = resize_h + (32 - resize_h % 32);
    if (resize_w % 32 != 0)
        resize_w = resize_w + (32 - resize_w % 32);
    ratio_h = resize_h / float(h);
    ratio_w = resize_w / float(w);

    cv::resize(imageRGB, imageProcessed, cv::Size(resize_w, resize_h));
    float* input = new float[BATCH_SIZE * 3 * resize_h * resize_w];
    cv::Mat imgFloat;

    imageProcessed.convertTo(imgFloat, CV_32FC3);
    cv::divide(imgFloat, cv::Scalar(255.0, 255.0, 255.0), imgFloat);
    cv::subtract(imgFloat, cv::Scalar(0.485, 0.456, 0.406), imgFloat, cv::noArray(), -1);
    cv::divide(imgFloat, cv::Scalar(0.229, 0.224, 0.225), imgFloat);

    std::vector<cv::Mat> chw;
    for (auto i = 0; i < 3; ++i)
        chw.emplace_back(cv::Mat(cv::Size(resize_w, resize_h), CV_32FC1, input + i * resize_w * resize_h));
    cv::split(imgFloat, chw);

//    for (int i = 0; i < 100; i++)
//        std::cout << input[i] << " ";
//    std::cout << std::endl;
    return input;
}


int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./fast -s a2 // serialize model to plan file" << std::endl;
        std::cerr << "./fast -d a2 // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s")
    {
        std::string netName = std::string(argv[2]);
        IHostMemory *modelStream{nullptr};
        serializeEngine(netName, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p(netName + ".engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    }
    else if (std::string(argv[1]) == "-d")
    {
        std::string netName = std::string(argv[2]);
        std::ifstream file(netName + ".engine", std::ios::binary | std::ios::in);

        if (file.good())
        {
            std::cout << "open engine file successfully" << std::endl;
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    }
    else
    {
        return -1;
    }

    cv::Mat image = cv::imread("../img266.jpg");
    int resize_h, resize_w;
    float ratio_h, ratio_w;

    float* input = preProcess(image, resize_h, resize_w, ratio_h, ratio_w);
    float* output = new float[BATCH_SIZE * resize_h * resize_w / 4];
    std::cout << resize_h << " " << resize_w << std::endl;

    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;


//    const ICudaEngine &engine = context->getEngine();
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine->getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

    context->setBindingDimensions(inputIndex, Dims4(BATCH_SIZE, 3, resize_h, resize_w));

    // Create GPU buffers on device
    int input_size = BATCH_SIZE * 3 * resize_h * resize_w * sizeof(float);
    int output_size = BATCH_SIZE * resize_h * resize_w / 16 * sizeof(float);
    CHECK(cudaMalloc(&buffers[inputIndex], input_size));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < 1000; ++ i) {
        doInference(*context, stream, buffers, input, output, resize_h, resize_w);
    }
    auto end = std::chrono::system_clock::now();
    std::cout << BATCH_SIZE*1000/(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/1000.0) << "FPS" << std::endl;

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
