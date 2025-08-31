#pragma once
#include <optional>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_vulkan.h>
#include "structs/structs.hpp"
#include "networks/networkManager.hpp"
#define MAX_FRAMES_IN_FLIGHT 3



struct AppState {
public:
    // GLFW
    GLFWwindow* window = nullptr;
    int windowWidth = 1280;
    int windowHeight = 720;
    bool framebufferResized = false;

    // Vulkan
    vk::raii::Context                           vulkanContext;
#ifdef _DEBUG
    vk::raii::DebugReportCallbackEXT            debugReport = VK_NULL_HANDLE;
#endif
    vk::raii::Instance                          instance = VK_NULL_HANDLE;
    vk::raii::SurfaceKHR                        surface = VK_NULL_HANDLE;
    vk::raii::PhysicalDevice                    physicalDevice = VK_NULL_HANDLE;
    vk::raii::Device                            device = VK_NULL_HANDLE;
    vk::raii::Queue                             graphicsQueue = VK_NULL_HANDLE;
    vk::raii::Queue                             presentQueue = VK_NULL_HANDLE;
    vk::raii::SwapchainKHR                      swapChain = VK_NULL_HANDLE;
    std::vector<vk::Image>                      swapChainImages;
    std::vector<vk::raii::ImageView>            swapChainImageViews;
    vk::raii::DeviceMemory                      depthImageMemory = VK_NULL_HANDLE;
    vk::raii::Image                             depthImage = VK_NULL_HANDLE;
    vk::raii::ImageView                         depthImageView = VK_NULL_HANDLE;
    vk::Format                                  swapChainImageFormat;
    vk::Extent2D                                swapChainExtent;
    vk::raii::RenderPass                        renderPass = VK_NULL_HANDLE;
    vk::raii::DescriptorSetLayout               descriptorSetLayout = VK_NULL_HANDLE;
    vk::raii::PipelineLayout                    pipelineLayout = VK_NULL_HANDLE;
    vk::raii::Pipeline                          graphicsPipeline = VK_NULL_HANDLE;
    vk::raii::PipelineCache                     pipelineCache = VK_NULL_HANDLE;
    std::vector<vk::raii::Framebuffer>          swapChainFramebuffers;
    vk::raii::DeviceMemory                      vertexBufferMemory = VK_NULL_HANDLE;
    vk::raii::Buffer                            vertexBuffer = VK_NULL_HANDLE;
    vk::raii::DeviceMemory                      indexBufferMemory = VK_NULL_HANDLE;
    vk::raii::Buffer                            indexBuffer = VK_NULL_HANDLE;
    std::vector<vk::raii::DeviceMemory>         uniformBuffersMemory;
    std::vector<vk::raii::Buffer>               uniformBuffers;
    std::vector<void*> uniformBuffersMapped;
    vk::raii::CommandPool                       commandPool = VK_NULL_HANDLE;
    vk::raii::DescriptorPool                    descriptorPool = VK_NULL_HANDLE;
    std::vector<vk::raii::DescriptorSet>        descriptorSets;
    std::vector<vk::raii::CommandBuffer>        commandBuffers;
    std::vector<vk::raii::Semaphore>            imageAvailableSemaphores;
    std::vector<vk::raii::Semaphore>            renderFinishedSemaphores;
    std::vector<vk::raii::Fence>                inFlightFences;
    std::vector<bool>                           imagesInFlight;
    UniformBufferObject                         ubo;
    BufferManager                               bufferM;
    uint32_t imageCount = 0;
    size_t currentFrame = 0;
    struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete() const {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };
    // ImGui
    vk::raii::DescriptorPool imguiPool = VK_NULL_HANDLE;
    ImGui_ImplVulkanH_Window MainWindowData;
    uint32_t imguiSubpass = 0;
    bool SwapChainRebuild = false;


    glm::mat4 projectionMatrix;

    // Performance
    float frameTime = 0.0f;
    float fps = 0.0f;
    float frameCount = 0;
    double lastTime = 0.0;

    //scene
    Scene                                       main_scene;
    Scene                                       render_scene;
    Role                                        role;
    guiFlags                                    guiFlags;
    bool                                        isInGame = true;
    bool                                        isFocus = false;
    bool                                        isFrame = false;


    //network
    bool                                        isRunNet   = false;
    networkManager                              net;
    NeuronGroupPreset                           neuronGroupPreset;
    NeuronBlockPreset                           neuronBlockPreset;
};

extern AppState app;

void initApp();
void cleanupApp();
