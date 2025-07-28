#include "devices.hpp"

/**
 * @brief 挑选一个支持所需特性和扩展的物理设备。
 * 
 * 该函数会枚举所有支持 Vulkan 的物理设备，然后从中挑选出第一个支持 VK_KHR_SWAPCHAIN 扩展的设备，
 * 并将其设置为应用程序的物理设备。如果没有找到支持 Vulkan 的 GPU，会抛出运行时错误。
 */
void pickPhysicalDevice() {
    // 枚举所有支持 Vulkan 的物理设备
    const auto physicalDevices = app.instance.enumeratePhysicalDevices();
    
    // 检查是否有可用的物理设备
    if (physicalDevices.empty()) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    // 遍历所有物理设备
    for (const auto& devices : physicalDevices) {
        // 枚举当前物理设备支持的所有扩展
        const std::vector<vk::ExtensionProperties> availableExtensions = devices.enumerateDeviceExtensionProperties();
        
        // 查找是否支持 VK_KHR_SWAPCHAIN 扩展
        auto it = std::find_if(availableExtensions.begin(), availableExtensions.end(),
            [](const vk::ExtensionProperties& ext) { return std::strcmp(ext.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0; });
        
        // 如果找到支持的扩展，则将该设备设置为应用程序的物理设备并跳出循环
        if (it != availableExtensions.end()) {
            app.physicalDevice = devices;
            break;
        }
    }
}

/**
 * @brief 查找指定物理设备的图形队列族和呈现队列族索引。
 * 
 * 该函数会遍历物理设备的所有队列族，查找支持图形操作和呈现操作的队列族，
 * 并将对应的索引记录在 QueueFamilyIndices 结构体中。当找到所有需要的队列时，停止查找。
 * 
 * @param physicalDevice 要查找队列族的物理设备
 * @return QueueFamilyIndices 包含图形队列时和呈现队列时索引的结构体
 */
QueueFamilyIndices findQueueFamilies(const vk::raii::PhysicalDevice& physicalDevice) {
    QueueFamilyIndices indices;

    // 获取物理设备的所有队列族属性
    const auto queueFamilies = physicalDevice.getQueueFamilyProperties();
    
    // 遍历所有队列族
    for (int i = 0; const auto& queueFamily : queueFamilies) {
        // 检查当前队列族是否支持图形操作
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
            indices.graphicsFamily = i;
        }

        // 检查当前队列族是否支持呈现操作
        if (physicalDevice.getSurfaceSupportKHR(i, app.surface)) {
            indices.presentFamily = i;
        }

        // 如果已经找到所有需要的队列时，则停止查找
        if (indices.isComplete())  break;

        ++i;
    }

    return indices;
}

/**
 * @brief 创建逻辑设备并初始化图形队列和呈现队列。
 * 
 * 该函数会先调用 findQueueFamilies 函数查找所需的队列族，然后根据这些队列族创建队列创建信息，
 * 接着配置设备特性和扩展，最后创建逻辑设备并获取图形队列和呈现队列。
 * 如果队列时索引不完整，会抛出运行时错误。
 */
void createLogicalDevice() {
    // 查找物理设备的图形队列族和呈现队列族索引
    auto indices = findQueueFamilies(app.physicalDevice);

    // 检查队列族索引是否完整
    if (!indices.isComplete()) {
        throw std::runtime_error("Queue family indices are not complete!");
    }

    // 获取唯一的队列族索引
    std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    constexpr float queuePriority = 1.0f;

    // 为每个唯一的队列族创建队列创建信息
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        vk::DeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.queueFamilyIndex = queueFamily;   // 队列时索引
        queueCreateInfo.setQueuePriorities(queuePriority); // 队列优先级
        queueCreateInfos.push_back(queueCreateInfo);
    }

    // 配置设备特性，仅在支持时启用各向异性采样
    vk::PhysicalDeviceFeatures deviceFeatures = app.physicalDevice.getFeatures();
    deviceFeatures.samplerAnisotropy = deviceFeatures.samplerAnisotropy ? VK_TRUE : VK_FALSE;

    // 配置设备创建信息
    vk::DeviceCreateInfo createInfo{};
    createInfo.setPEnabledFeatures(&deviceFeatures).setQueueCreateInfos(queueCreateInfos);
    createInfo.setPEnabledExtensionNames(vk::KHRSwapchainExtensionName);
    createInfo.setEnabledExtensionCount(1);
    createInfo.setEnabledLayerCount(0);

    // 创建逻辑设备
    app.device = app.physicalDevice.createDevice(createInfo);

    // 获取图形队列和呈现队列
    app.graphicsQueue = app.device.getQueue(indices.graphicsFamily.value(), 0);
    app.presentQueue = app.device.getQueue(indices.presentFamily.value(), 0);
}
