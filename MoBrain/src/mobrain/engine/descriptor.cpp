#include "descriptor.hpp"

/**
 * @brief 创建描述符集布局。
 * 
 * 此函数创建一个统一缓冲区的描述符集布局绑定，然后使用该绑定创建描述符集布局，
 * 并将结果存储在 app.descriptorSetLayout 中。
 */
void createDescriptorSetLayout() {
    // 创建一个统一缓冲区的描述符集布局绑定
    vk::DescriptorSetLayoutBinding uboLayoutBinding;
    uboLayoutBinding
        .setBinding(0)  // 设置绑定点为 0
        .setDescriptorType(vk::DescriptorType::eUniformBuffer)  // 设置描述符类型为统一缓冲区
        .setDescriptorCount(1)  // 设置描述符数量为 1
        .setStageFlags(vk::ShaderStageFlagBits::eVertex);  // 设置在顶点着色器阶段使用

    // 创建描述符集布局创建信息
    vk::DescriptorSetLayoutCreateInfo layoutInfo;
    layoutInfo
        .setBindings(uboLayoutBinding);  // 设置绑定信息

    // 创建描述符集布局并存储结果
    app.descriptorSetLayout = app.device.createDescriptorSetLayout(layoutInfo);
}

/**
 * @brief 创建描述符集。
 * 
 * 此函数为每个飞行中的帧分配描述符集，然后为每个描述符集更新统一缓冲区信息。
 */
void createDescriptorSets() {
    // 为每个飞行中的帧创建描述符集布局向量
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *app.descriptorSetLayout); 
    
    // 创建描述符集分配信息
    vk::DescriptorSetAllocateInfo allocInfo;
    allocInfo.descriptorPool = app.descriptorPool;  // 设置描述符池
    allocInfo.setSetLayouts( layouts );  // 设置描述符集布局

    // 分配描述符集并存储结果
    app.descriptorSets = app.device.allocateDescriptorSets(allocInfo);

    // 为每个飞行中的帧更新描述符集
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        // 创建统一缓冲区信息
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo
            .setBuffer(app.uniformBuffers[i])
            .setOffset(0)
            .setRange(sizeof(glm::mat4));
        // 创建描述符写入信息
        vk::WriteDescriptorSet descriptorWrite;
        descriptorWrite
            .setBufferInfo(bufferInfo)  // 设置缓冲区信息
            .setDescriptorType(vk::DescriptorType::eUniformBuffer)  // 设置描述符类型为统一缓冲区
            .setDstBinding(0)  // 设置目标绑定点为 0
            .setDstArrayElement(0)  // 设置目标数组元素为 0
            .setDstSet(app.descriptorSets[i]);  // 设置目标描述符集

        // 更新描述符集
        app.device.updateDescriptorSets(descriptorWrite, nullptr);
    }
}   

/**
 * @brief 创建描述符池。
 * 
 * 此函数创建一个包含多种描述符类型的描述符池，每种描述符类型的数量为 1000，
 * 并设置描述符池可自由释放描述符集的标志，最后将结果存储在 app.descriptorPool 中。
 */
void createDescriptorPool() {
    // 定义各种描述符类型及其数量
    std::vector<vk::DescriptorPoolSize> poolsizes = {
        { vk::DescriptorType::eSampler, 30 },  // 采样器
        { vk::DescriptorType::eCombinedImageSampler, 30 },  // 组合图像采样器
        { vk::DescriptorType::eSampledImage, 30 },  // 采样图像
        { vk::DescriptorType::eUniformBuffer, 30+static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)},  // 统一缓冲区
        { vk::DescriptorType::eUniformTexelBuffer, 30 },  // 统一纹素缓冲区
        { vk::DescriptorType::eStorageBufferDynamic, 30 },  // 动态存储缓冲区
        { vk::DescriptorType::eUniformBufferDynamic, 30 },  // 动态统一缓冲区
        { vk::DescriptorType::eInputAttachment, 30 }  // 输入附件
    };

    // 计算最大描述符集数量
    uint32_t maxSets = 0;
    for (const auto& poolsize : poolsizes) {
        maxSets += poolsize.descriptorCount;
    }

    // 创建描述符池创建信息
    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;  // 设置可自由释放描述符集的标志
    poolInfo.setPoolSizes( poolsizes );  // 设置描述符池大小数组
    poolInfo.maxSets = maxSets;  // 设置最大描述符集数量

    // 创建描述符池并存储结果
    app.descriptorPool = app.device.createDescriptorPool(poolInfo);
}
