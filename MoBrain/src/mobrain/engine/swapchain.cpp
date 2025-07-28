#include "swapchain.hpp"

/**
 * @brief 选择交换链表面格式
 * 
 * 遍历所有可用的表面格式，优先选择 VK_FORMAT_B8G8R8A8_SRGB 格式和 VK_COLOR_SPACE_SRGB_NONLINEAR_KHR 颜色空间。
 * 如果找不到匹配项，则返回第一个可用格式。
 * 
 * @param availableFormats 可用的表面格式列表
 * @return 选中的表面格式
 */
static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
            availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return availableFormat;
        }
    }
    return availableFormats.at(0);
}

/**
 * @brief 选择交换链呈现模式
 * 
 * 遍历所有可用的呈现模式，优先选择 VK_PRESENT_MODE_MAILBOX_KHR（三重缓冲）。
 * 如果找不到，则默认使用 VK_PRESENT_MODE_FIFO_KHR（垂直同步）。
 * 
 * @param availablePresentModes 可用的呈现模式列表
 * @return 选中的呈现模式
 */
static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
            return availablePresentMode;
        }
    }
    return vk::PresentModeKHR::eFifo;
}

/**
 * @brief 选择交换链图像范围（分辨率）
 * 
 * 如果当前范围有效，则直接使用；否则根据窗口大小和设备能力计算合适的范围。
 * 
 * @param capabilities 表面能力结构体
 * @return 交换链图像范围
 */
vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }

    int width, height;
    glfwGetFramebufferSize(app.window, &width, &height);

    vk::Extent2D actualExtent(
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height)
    );

    actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    return actualExtent;
}

/**
 * @brief 查询交换链支持详情
 * 
 * 获取物理设备对指定表面的支持信息，包括表面能力、格式和呈现模式。
 * 
 * @param physicalDevice 物理设备对象
 * @return 包含交换链支持信息的结构体
 */
SwapChainSupportDetails querySwapChainSupport(const vk::raii::PhysicalDevice& physicalDevice) {
    SwapChainSupportDetails details;

    details.capabilities = physicalDevice.getSurfaceCapabilitiesKHR(app.surface);
    details.formats = physicalDevice.getSurfaceFormatsKHR(app.surface);
    details.presentModes = physicalDevice.getSurfacePresentModesKHR(app.surface);
    
    return details;
}

/**
 * @brief 创建交换链
 * 
 * 根据查询到的交换链支持信息，选择合适的格式、呈现模式和图像范围，
 * 然后创建 Vulkan 交换链资源，并初始化相关图像和视图数据。
 */
void createSwapChain() {
    SwapChainSupportDetails details = querySwapChainSupport(app.physicalDevice);

    const auto surfaceFormat = chooseSwapSurfaceFormat(details.formats);
    const auto presentMode = chooseSwapPresentMode(details.presentModes);
    const auto extent = chooseSwapExtent(details.capabilities);

    std::cout << "Swapchain extent: " << extent.width << "x" << extent.height << std::endl;

    // 计算交换链图像数量
    app.imageCount = details.capabilities.minImageCount + 1;
    // 确保图像数量不超过最大限制
    if (details.capabilities.maxImageCount > 0 && app.imageCount > details.capabilities.maxImageCount) {
        app.imageCount = details.capabilities.maxImageCount;
    }

    // 创建交换链创建信息对象
    vk::SwapchainCreateInfoKHR ScreateInfo{};
    // 配置交换链创建信息
    ScreateInfo
        .setSurface(app.surface)
        .setMinImageCount(app.imageCount)
        .setImageFormat(surfaceFormat.format)
        .setImageColorSpace(surfaceFormat.colorSpace)
        .setImageExtent(extent)
        .setImageArrayLayers(1)
        .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
        .setClipped(true)
        .setPresentMode(presentMode)
        .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
        .setPreTransform(details.capabilities.currentTransform)
        .setOldSwapchain(nullptr);

    // 获取队列族索引
    QueueFamilyIndices QF = findQueueFamilies(app.physicalDevice);
    
    // 准备队列族索引列表
    std::vector<uint32_t> queueFamilyIndices{ QF.graphicsFamily.value(), QF.presentFamily.value() };
    
    // 处理队列族共享情况
    if (QF.graphicsFamily != QF.presentFamily) {
        ScreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;
        ScreateInfo.setQueueFamilyIndices(queueFamilyIndices);
    } else {
        ScreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
        ScreateInfo.setPQueueFamilyIndices(nullptr);
    }

    try {
        // 创建交换链
        app.swapChain = app.device.createSwapchainKHR(ScreateInfo);
    } catch (const vk::SystemError& err) {
        // 捕获并输出 vk::SystemError 错误信息
        std::cerr << "vk::SystemError - code: " << err.code().message() << std::endl;
        std::cerr << "vk::SystemError - what: " << err.what() << std::endl;
    } catch (const std::exception& e) {
        // 捕获并输出其他异常信息
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    // 获取交换链图像
    app.swapChainImages = app.swapChain.getImages();
    // 设置交换链图像格式
    app.swapChainImageFormat = surfaceFormat.format;
    // 设置交换链范围
    app.swapChainExtent = extent;
}

/**
 * @brief 重新创建交换链
 * 
 * 当窗口大小发生变化时，等待窗口大小有效，
 * 清空旧的交换链资源，重新创建交换链、图像视图和帧缓冲，
 * 最后标记帧缓冲大小变化已处理。
 */
void recreateSwapChain() {
    // 获取当前窗口帧缓冲大小
    int width = 0, height = 0;
    glfwGetFramebufferSize(app.window, &width, &height);
    
    // 等待窗口大小有效
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(app.window, &width, &height);
        glfwWaitEvents();
    }

    // 等待设备空闲，确保资源可以安全释放
    app.device.waitIdle();
    
    // 清空旧的交换链资源
    app.swapChainImages.clear();
    app.swapChainImageViews.clear();
    app.swapChainFramebuffers.clear();
    app.swapChain = nullptr;
    app.device.waitIdle();  // 确保所有操作完成
    
    // 清理所有交换链相关资源
    app.swapChainFramebuffers.clear();
    app.swapChainImageViews.clear();
    app.swapChain = nullptr;
    
    // 重新创建关键资源
    createSwapChain();
    createImageViews();
    createDepthResources();
    createFramebuffers();
    std::cout << "Recreate swap chain end" << std::endl;
    app.framebufferResized = false;
}
