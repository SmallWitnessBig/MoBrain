#include "commandpool.hpp"
// 将命令记录到指定的命令缓冲区中
// 参数:
// - commandbuffer: 要记录命令的命令缓冲区
// - image_index: 交换链图像的索引
void recordCommandBuffer(const vk::raii::CommandBuffer& commandbuffer, uint32_t image_index) {
    // 创建一个默认的命令缓冲区开始信息
    constexpr vk::CommandBufferBeginInfo beginInfo;
    // 开始记录命令到命令缓冲区
    commandbuffer.begin(beginInfo);

    // 定义渲染区域，从 (0, 0) 开始，大小为交换链图像的尺寸
    const vk::Rect2D renderArea = {
        {0, 0},
        {app.swapChainExtent.width, app.swapChainExtent.height}
    };

    std::array<vk::ClearValue,2> clearValues ;
    clearValues[0].color = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f};
    clearValues[1].depthStencil = vk::ClearDepthStencilValue{1.0f, 0};

    // 定义视口，覆盖整个交换链图像区域
    const vk::Viewport viewport(
        0.0f, 0.0f, // x, y 坐标
        static_cast<float>(app.swapChainExtent.width),    // 视口宽度
        static_cast<float>(app.swapChainExtent.height),   // 视口高度
        0.0f, 1.0f  // 最小深度和最大深度
    );

    // 定义裁剪区域，与交换链图像区域一致
    const vk::Rect2D scissor(
        vk::Offset2D{ 0, 0 }, // 裁剪区域偏移量
        app.swapChainExtent   // 裁剪区域大小
    );

    // 创建渲染通道开始信息
    vk::RenderPassBeginInfo RPinfo;
    // 设置渲染通道
    RPinfo
        .setRenderPass(app.renderPass)
        // 设置帧缓冲区
        .setFramebuffer(app.swapChainFramebuffers[image_index])
        // 设置渲染区域
        .setRenderArea(renderArea)
        // 设置清除值
        .setClearValues(clearValues);
    
    // 开始渲染通道，使用内联命令
    commandbuffer.beginRenderPass(RPinfo, vk::SubpassContents::eInline);
    // 绑定图形管线
    commandbuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, app.graphicsPipeline);
    // 设置视口
    commandbuffer.setViewport(0, viewport);
    // 设置裁剪区域
    commandbuffer.setScissor(0, scissor);
    // 绑定描述符集
    commandbuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        app.pipelineLayout,
        0,
        *app.descriptorSets[app.currentFrame],
        nullptr
    );

    auto& vertexBlocks = app.bufferM.vertexBlocks;
    auto& indexBlocks = app.bufferM.indexBlocks;
    auto& instanceBlocks = app.bufferM.instanceBlocks;
    auto& objectLocations = app.bufferM.objectLocations;
    auto& instanceLocations = app.bufferM.instanceLocations;
    auto& instanceFirstLocations = app.bufferM.instanceFirstLocations;

    // Render Cube
    {
        auto Loc = instanceFirstLocations[InstanceType::CUBE];

        auto& vertexBuffer = vertexBlocks[Loc.vertexBlockIndex].buffer;
        auto& indexBuffer = indexBlocks[Loc.indexBlockIndex];


        for (auto& i : instanceBlocks) {
            std::array<vk::Buffer,2> buffers = {*vertexBuffer,*i.buffer};
            std::array<vk::DeviceSize,2> offsets = {Loc.vertexOffset,0};
            commandbuffer.bindVertexBuffers(0, buffers, offsets);
            commandbuffer.bindIndexBuffer(*indexBuffer.buffer, 0, vk::IndexType::eUint32);
            commandbuffer.drawIndexed(
                36,
                i.used/sizeof(InstanceData),
                Loc.indexOffset,
                Loc.vertexOffset,
                0
            );

        }
        
    }



    // 使用 ImGui 渲染绘制数据
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *commandbuffer);

    // 结束渲染通道
    commandbuffer.endRenderPass();
    // 结束命令记录
    commandbuffer.end();
}

// 命令池用于分配命令缓冲区
// 创建命令池
void createCommandPool() {
    // 获取图形队列时序和呈现队列时序索引
    const auto [graphicsFamily, presentFamily] = findQueueFamilies(app.physicalDevice);

    // 创建命令池创建信息
    vk::CommandPoolCreateInfo poolInfo;
    // 设置命令池标志，允许重置命令缓冲区
    poolInfo
        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
        // 设置命令池所属的队列时序索引
        .setQueueFamilyIndex(graphicsFamily.value());
    // 创建命令池
    app.commandPool = app.device.createCommandPool(poolInfo);
}

// 创建命令缓冲区
void createCommandBuffers() {
    // 创建命令缓冲区分配信息
    vk::CommandBufferAllocateInfo allocInfo;
    // 设置命令池
    allocInfo
        .setCommandPool(app.commandPool)
        // 设置要分配的命令缓冲区数量
        .setCommandBufferCount(MAX_FRAMES_IN_FLIGHT)
        // 设置命令缓冲区级别为主要级别
        .setLevel(vk::CommandBufferLevel::ePrimary);

    // 分配命令缓冲区
    app.commandBuffers = app.device.allocateCommandBuffers(allocInfo);
}
