#include "drawer.hpp"

/**
 * @brief 创建同步对象，包括信号量和围栏，用于控制渲染流程的同步。
 * 
 * 该函数会为每一帧创建图像可用信号量、渲染完成信号量和围栏，
 * 这些同步对象的数量由 MAX_FRAMES_IN_FLIGHT 决定。
 */
void createSyncObjects() {
    // 定义信号量创建信息，使用默认配置
    constexpr vk::SemaphoreCreateInfo semaphoreInfo;
    // 定义围栏创建信息，初始状态为已发出信号
    constexpr vk::FenceCreateInfo fenceInfo{
        vk::FenceCreateFlagBits::eSignaled
    };
    
    // 为每一帧创建对应的同步对象
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        app.imageAvailableSemaphores.emplace_back(app.device, semaphoreInfo);
        app.renderFinishedSemaphores.emplace_back(app.device, semaphoreInfo);
        app.inFlightFences.emplace_back(app.device, fenceInfo);
    }
}

/**
 * @brief 执行一帧的渲染流程，包括获取图像、更新缓冲区、提交命令和呈现图像。
 * 
 * 该函数会处理帧渲染的完整流程，包括等待上一帧完成、获取交换链图像、
 * 更新统一缓冲区、记录并提交命令缓冲区，最后将图像呈现到交换链。
 * 如果在过程中遇到交换链过期或次优的情况，会重新创建交换链。
 */
void drawFrame() {
    // 用于存储从交换链获取的图像索引
	uint32_t imageIndex;
    // 等待当前帧的围栏，确保上一帧渲染完成
    if(app.device.waitForFences(*app.inFlightFences[app.currentFrame], true, std::numeric_limits<uint64_t>::max())
        != vk::Result::eSuccess) {
        throw std::runtime_error{ "waitForFences in drawFrame was failed" };
    }
    
    // 尝试从交换链获取下一帧图像
    try
    {
        const auto [res, idx] = app.swapChain.acquireNextImage(UINT64_MAX, *app.imageAvailableSemaphores[app.currentFrame]);
        // 如果获取结果为次优，重新创建交换链
        if (res == vk::Result::eSuboptimalKHR) {
            recreateSwapChain();
            return;
        }
        imageIndex = idx;
    }
    catch (const vk::OutOfDateKHRError&)
    {
        // 如果交换链过期，重新创建交换链
        recreateSwapChain();
        return;
    }

    // 重置当前帧的围栏，准备新的渲染
    app.device.resetFences(*app.inFlightFences[app.currentFrame]);
    
    // 重置当前帧的命令缓冲区
    app.commandBuffers[app.currentFrame].reset();

    // 更新当前帧的统一缓冲区
    updateUniformBuffer(app.currentFrame);
    
    // 记录绘制命令到命令缓冲区
    recordCommandBuffer(app.commandBuffers[app.currentFrame], imageIndex);

    // 创建提交信息，用于提交命令缓冲区到图形队列
    vk::SubmitInfo submitInfo;

    // 设置等待的信号量为图像可用信号量
    submitInfo.setWaitSemaphores(*app.imageAvailableSemaphores[app.currentFrame]);
    // 设置等待的管线阶段为颜色附件输出阶段
    std::array<vk::PipelineStageFlags, 1> waitStages = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
    submitInfo.setWaitDstStageMask(waitStages);
    // 设置要提交的命令缓冲区
    submitInfo.setCommandBuffers(*app.commandBuffers[app.currentFrame]);
    // 设置渲染完成后要发出信号的信号量
    submitInfo.setSignalSemaphores(*app.renderFinishedSemaphores[app.currentFrame]);
    // 提交命令缓冲区到图形队列，并关联当前帧的围栏
    app.graphicsQueue.submit(submitInfo, app.inFlightFences[app.currentFrame]);

    // 创建呈现信息，用于将图像呈现到交换链
    vk::PresentInfoKHR presentInfo;
    presentInfo
        .setWaitSemaphores(*app.renderFinishedSemaphores[app.currentFrame])
        .setSwapchains(*app.swapChain)
        .setPImageIndices(&imageIndex);

    // 执行图像呈现操作
    try{
        auto res = app.presentQueue.presentKHR(presentInfo);
        // 如果呈现结果为次优或交换链过期，重新创建交换链
        if (res == vk::Result::eSuboptimalKHR || res == vk::Result::eErrorOutOfDateKHR) {
            recreateSwapChain();
        } else if (res != vk::Result::eSuccess) {
            // 呈现失败，抛出异常
            throw std::runtime_error{ "Failed to present swap chain image" };
        }
    } catch (const vk::OutOfDateKHRError&) {
        // 如果交换链过期，重新创建交换链
        recreateSwapChain();
    }
    // 如果帧缓冲区大小被调整，重新创建交换链并重置标记
    if (app.framebufferResized) {
        recreateSwapChain();
        app.framebufferResized = false;  // 重置窗口尺寸标记
        return; // 重新创建交换链后立即返回，避免继续执行当前帧
    }
    
    // 更新当前帧索引，循环使用帧资源
    app.currentFrame = (app.currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

/**
 * @brief 更新指定索引的统一缓冲区，设置模型、视图和投影矩阵。
 * 
 * 该函数会根据当前时间计算模型矩阵，固定设置视图和投影矩阵，
 * 最后将这些矩阵数据复制到统一缓冲区中。
 * 
 * @param currentImage 当前要更新的统一缓冲区的索引
 */
void updateUniformBuffer(const uint32_t currentImage) {
    // 记录程序开始时间
    static auto startTime = std::chrono::high_resolution_clock::now();
    // 获取当前时间
    const auto currentTime = std::chrono::high_resolution_clock::now();
    // 计算从程序开始到现在经过的时间
    const float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
    // 创建统一缓冲区对象
    

    // 设置模型矩阵，让物体绕 Z 轴旋转
    app.ubo.model =  glm::mat4(1.0f);

    // 使用 camera 命名空间中的参数设置视图矩阵
    glm::vec3 cameraPos = camera::pos;
    glm::vec3 cameraFront = camera::front;
    glm::vec3 cameraUp = camera::up;

    app.ubo.view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);

    // 设置投影矩阵，并使用 camera 中的 fov 变量
    app.ubo.proj = glm::perspective(
        glm::radians(camera::fov),
        static_cast<float>(app.swapChainExtent.width) / static_cast<float>(app.swapChainExtent.height),
        0.1f,
        1000.0f
    );
    // 翻转 Y 轴，适应 Vulkan 的坐标系
    app.ubo.proj[1][1] *= -1;

    // 计算最终的MVP矩阵
    glm::mat4 mvp = app.ubo.proj * app.ubo.view * app.ubo.model;
    // 将MVP矩阵的数据复制到对应的内存映射区
    memcpy(app.uniformBuffersMapped[currentImage], &mvp, sizeof(mvp));
}

/**
 * @brief 帧缓冲区大小调整回调函数，用于标记帧缓冲区已被调整。
 * 
 * 当 GLFW 检测到窗口帧缓冲区大小发生变化时，会调用此函数。
 * 该函数会设置 app.framebufferResized 标记为 true，以便后续处理。
 * 
 * @param window 发生大小变化的 GLFW 窗口指针
 * @param width 新的帧缓冲区宽度
 * @param height 新的帧缓冲区高度
 */
void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    app.framebufferResized = true;
}
