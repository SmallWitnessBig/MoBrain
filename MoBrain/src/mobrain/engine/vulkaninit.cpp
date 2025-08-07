#include "vulkaninit.hpp"

using std::vector;

// 初始化Vulkan环境
// 该函数按顺序初始化Vulkan的所有必要组件，包括实例、表面、设备、交换链、渲染流程等
void initVulkan() {

    //vulkan instance
    createInstance();

	//vulkan surface
    createSurface();

    //vulkan device
    pickPhysicalDevice();
    createLogicalDevice();

    //swapchain
    createSwapChain();
    createImageViews();
    createRenderPass();
    
    //depth
    createDepthResources();
    
    //framebuffers should be created after depth resources
    createFramebuffers();
    
    createDescriptorSetLayout();
    createGraphicsPipeline();

    //command
    createCommandPool();
    createCommandBuffers();

    //gpu and cpu synchronization
    createSyncObjects();
    createUniformBuffers();

    //descriptor
    createDescriptorPool();
    createDescriptorSets();

    app.bufferM.init();
}


// 检查指定的扩展是否可用
// @param properties 扩展属性列表
// @param extension 要检查的扩展名称
// @return 如果扩展可用则返回true，否则返回false
static bool IsExtensionAvailable(const vector<vk::ExtensionProperties>& properties, const char* extension)
{
    for (const vk::ExtensionProperties& p : properties)
        if (strcmp(p.extensionName, extension) == 0)
            return true;
    return false;
}

// 设备所需扩展列表
const vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

// 获取Vulkan实例所需的扩展列表
// 包括GLFW扩展以及调试、可移植性等相关扩展（如果可用）
static std::vector<const char*> getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    extensions.emplace_back(vk::KHRPortabilityEnumerationExtensionName);
    auto properties = vk::enumerateInstanceExtensionProperties();
    if (IsExtensionAvailable(properties, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
        extensions.emplace_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
    if (IsExtensionAvailable(properties, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME))
    {
        extensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }
#endif
#ifdef _DEBUG
    extensions.emplace_back(vk::EXTDebugUtilsExtensionName);
#endif // _DEBUG
    return extensions;
}

// 调试消息回调函数
// 当启用验证层时，此函数会被Vulkan调用以输出调试信息
// @param messageSeverity 消息严重程度
// @param messageTypes 消息类型
// @param pCallbackData 包含详细消息信息的结构体
// @param pUserData 用户自定义数据指针
// @return 返回false表示不中止触发该消息的Vulkan函数调用
static VKAPI_ATTR uint32_t VKAPI_CALL debugMessageFunc(
    vk::DebugUtilsMessageSeverityFlagBitsEXT       messageSeverity,
    vk::DebugUtilsMessageTypeFlagsEXT              messageTypes,
    vk::DebugUtilsMessengerCallbackDataEXT const* pCallbackData,
    void* pUserData
) {
    std::cerr<< "validation layer: {}"<< pCallbackData->pMessage;
    return false;
}

// 构造调试信使创建信息结构体
// 用于配置调试消息的严重程度和类型过滤
static constexpr vk::DebugUtilsMessengerCreateInfoEXT populateDebugMessengerCreateInfo() {
    constexpr vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
    );
    constexpr vk::DebugUtilsMessageTypeFlagsEXT    messageTypeFlags(
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
    );
    return { {}, severityFlags, messageTypeFlags, &debugMessageFunc };
}

// 创建Vulkan实例
// 配置应用程序信息、所需扩展和验证层，并创建Vulkan实例
void createInstance() {

    // 设置应用程序信息
    vk::ApplicationInfo appInfo{
        "MoBrain",
        1,
        "MoBrain Engine",
        1,
        VK_API_VERSION_1_4
    };

    vk::InstanceCreateInfo createInfo{};
    createInfo.setPApplicationInfo(&appInfo);

    auto requiredExtensions = getRequiredExtensions();
    createInfo.setPEnabledExtensionNames(requiredExtensions);
    createInfo.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;

    constexpr std::array<const char*, 1> REQUIRED_LAYERS{
    "VK_LAYER_KHRONOS_validation"
    };

#ifndef _DEBUG
    constexpr bool ENABLE_VALIDATION_LAYER = false;
#else
    constexpr bool ENABLE_VALIDATION_LAYER = true;
    // 检查所需的验证层是否可用
    if (![REQUIRED_LAYERS] {
        const auto layers = app.vcxt.enumerateInstanceLayerProperties();
        std::set<std::string> requiredLayers(REQUIRED_LAYERS.begin(), REQUIRED_LAYERS.end());
        for (const auto& layer : layers) {
            requiredLayers.erase(layer.layerName);
        }
        return requiredLayers.empty();
        }())
    {
        throw std::runtime_error("validation layers requested, but not available!");
    }
    constexpr auto DMcreateInfo = populateDebugMessengerCreateInfo();
    createInfo.pNext = &DMcreateInfo;
    createInfo.setPEnabledLayerNames(REQUIRED_LAYERS);
#endif

    app.instance = app.vcxt.createInstance(createInfo);
#ifdef _DEBUG
    app.instance.createDebugUtilsMessengerEXT(DMcreateInfo);
#endif // _DEBUG
};

// 创建窗口表面
// 使用GLFW创建一个与当前窗口关联的Vulkan表面，用于渲染输出
void createSurface() {
    VkSurfaceKHR cSurface;
    if (glfwCreateWindowSurface(*app.instance, app.window, nullptr, &cSurface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
    app.surface = vk::raii::SurfaceKHR(app.instance, cSurface);
}

void initInstanceData() {
    InstanceData i;


};