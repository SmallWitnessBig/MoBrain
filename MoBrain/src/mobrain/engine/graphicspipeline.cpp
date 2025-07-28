#include "graphicspipeline.hpp"
#include <filesystem>

/**
 * @brief 创建着色器模块
 * 
 * @param code 着色器代码的二进制数据
 * @return vk::raii::ShaderModule 创建的着色器模块对象
 */
vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) {
    vk::ShaderModuleCreateInfo createinfo;
    createinfo
        .setPCode(reinterpret_cast<const uint32_t*>(code.data()))
        .setCodeSize(code.size());
    
    return app.device.createShaderModule(createinfo);
}

// 定义默认视口和裁剪区域
static vk::Viewport viewport{
    0.0f, 0.0f,
    static_cast<float>(app.swapChainExtent.width), static_cast<float>(app.swapChainExtent.height),
    0.0f, 1.0f
};

static vk::Rect2D scissor{ 
    {0, 0},
    {app.swapChainExtent.width, app.swapChainExtent.height}
};

/**
 * @brief 创建图形管线
 * 
 * 该函数封装了 Vulkan 图形管线的创建流程，包括加载顶点和片段着色器、配置各种管线状态等。
 */
void createGraphicsPipeline() {
    try {
        // 获取当前工作目录
        std::filesystem::path baseDir = std::filesystem::current_path();
        // 拼接顶点着色器文件路径
        auto vertDir = baseDir / "shaders" / "graphics.vert.spv";
        // 拼接片段着色器文件路径
        auto fragDir = baseDir / "shaders" / "graphics.frag.spv";
        // 读取顶点着色器二进制文件内容
        const std::vector<char> vertShaderCode = atebinaryFile(vertDir.string());
        // 读取片段着色器二进制文件内容
        const std::vector<char> fragShaderCode = atebinaryFile(fragDir.string());
        // 创建顶点着色器模块
        vk::raii::ShaderModule vertex = createShaderModule(vertShaderCode);
        // 创建片段着色器模块
        vk::raii::ShaderModule fragment = createShaderModule(fragShaderCode);

        // 配置顶点着色器阶段信息
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo;
        vertShaderStageInfo
            .setModule(vertex)  // 设置顶点着色器模块
            .setStage(vk::ShaderStageFlagBits::eVertex)  // 设置为顶点着色器阶段
            .setPName("main");  // 设置入口函数名为 "main"

        // 配置片段着色器阶段信息
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo;
        fragShaderStageInfo
            .setModule(fragment)  // 设置片段着色器模块
            .setStage(vk::ShaderStageFlagBits::eFragment)  // 设置为片段着色器阶段
            .setPName("main");  // 设置入口函数名为 "main"

        // 将顶点和片段着色器阶段信息存储在数组中
        const auto shaderStages = std::array<vk::PipelineShaderStageCreateInfo, 2>{
            vertShaderStageInfo, fragShaderStageInfo
        };

        // 定义动态状态，这里使用视口和裁剪区域动态状态
        const vk::ArrayProxy<vk::DynamicState> dynamicStates = {
            vk::DynamicState::eScissor,
            vk::DynamicState::eViewport
        };
        // 创建动态状态创建信息对象
        vk::PipelineDynamicStateCreateInfo dynamicstate;
        dynamicstate.setDynamicStates(dynamicStates);
        // 创建顶点输入状态创建信息对象
        vk::PipelineVertexInputStateCreateInfo vertexInputinfo;
        // 获取顶点绑定描述
        const auto bindingDescription = Vertex::getBindingDescription();
        // 获取顶点属性描述
        const auto attributeDescriptions = Vertex::getAttributeDescriptions();
        vertexInputinfo
            .setVertexAttributeDescriptions(attributeDescriptions)  // 设置顶点属性描述
            .setVertexBindingDescriptions(bindingDescription);  // 设置顶点绑定描述

        // 创建输入装配状态创建信息对象
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
        inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;  // 设置图元拓扑为三角形列表
        inputAssembly.primitiveRestartEnable = false;  // 禁用图元重启

        // 创建视口状态创建信息对象
        vk::PipelineViewportStateCreateInfo viewportState;
        viewportState
            .setScissors(scissor)  // 设置裁剪区域
            .setViewports(viewport)  // 设置视口
        ;
        // 创建光栅化状态创建信息对象
        vk::PipelineRasterizationStateCreateInfo rasterizer;
        rasterizer
            .setRasterizerDiscardEnable(false)  // 禁用光栅化丢弃
            .setDepthClampEnable(false)  // 禁用深度钳制
            .setPolygonMode(vk::PolygonMode::eFill)  // 设置多边形填充模式
            .setLineWidth(1.0f)  // 设置线宽
            .setCullMode(vk::CullModeFlagBits::eBack)  // 设置背面剔除
            .setFrontFace(vk::FrontFace::eCounterClockwise)  // 设置正面为逆时针方向
            .setDepthBiasEnable(false);  // 禁用深度偏移

        // 创建多重采样状态创建信息对象
        vk::PipelineMultisampleStateCreateInfo multisampling;
        multisampling
            .setSampleShadingEnable(false)  // 禁用采样着色
            .setRasterizationSamples(vk::SampleCountFlagBits::e1);  // 设置采样数量为 1

        vk::PipelineDepthStencilStateCreateInfo depthStencil;
        depthStencil
            .setDepthTestEnable(true)
            .setDepthWriteEnable(true)
            .setDepthCompareOp(vk::CompareOp::eLess);

        // 创建颜色混合附件状态对象
        vk::PipelineColorBlendAttachmentState colorBlendAttachment;
        colorBlendAttachment.blendEnable = false; // 默认禁用颜色混合
        colorBlendAttachment.colorWriteMask = (
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        );  // 设置颜色写入掩码

        // 创建颜色混合状态创建信息对象
        vk::PipelineColorBlendStateCreateInfo CBinfo;
        CBinfo
            .setAttachments(colorBlendAttachment)  // 设置颜色混合附件
            .setLogicOp(vk::LogicOp::eCopy)  // 设置逻辑操作
            .setLogicOpEnable(false);  // 禁用逻辑操作

        // 创建管线布局创建信息对象
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
        pipelineLayoutInfo
            .setSetLayouts(*app.descriptorSetLayout); // 设置描述符集布局指针为 descriptorSetLayout
        // 创建管线布局
        app.pipelineLayout = app.device.createPipelineLayout(pipelineLayoutInfo);

        // 创建图形管线创建信息对象
        vk::GraphicsPipelineCreateInfo pipelineinfo;
        pipelineinfo
            .setStages(shaderStages)  // 设置着色器阶段
            .setLayout(*app.pipelineLayout)  // 设置管线布局
            .setRenderPass(*app.renderPass)  // 设置渲染通道
            .setPViewportState(&viewportState)  // 设置视口状态
            .setPColorBlendState(&CBinfo)  // 设置颜色混合状态
            .setPVertexInputState(&vertexInputinfo)  // 设置顶点输入状态
            .setPMultisampleState(&multisampling)  // 设置多重采样状态
            .setPInputAssemblyState(&inputAssembly)  // 设置输入装配状态
            .setPDepthStencilState(&depthStencil)  // 设置深度模板状态
            .setPDynamicState(&dynamicstate)  // 设置动态状态
            .setPRasterizationState(&rasterizer)  // 设置光栅化状态
            .setSubpass(0);  // 设置子通道索引为 0

        // 创建图形管线
        app.graphicsPipeline = app.device.createGraphicsPipeline(nullptr, pipelineinfo);
    }
    // 捕获 Vulkan 系统错误
    catch (const vk::SystemError& e) {
        std::cerr << "failed to create pipeline, reason is " << e.what() << std::endl;
    }
    // 捕获标准运行时错误
    catch (const std::runtime_error& e) {
        std::cerr << "failed to create pipeline, reason is " << e.what() << std::endl;
    }
};
