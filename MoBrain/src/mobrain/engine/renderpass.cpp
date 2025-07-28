#include "renderpass.hpp"
#include "buffers.hpp"
void createRenderPass() {
	// 创建颜色附件描述
	vk::AttachmentDescription colorAttachment;
	colorAttachment
		// 设置颜色附件格式与交换链图像格式一致
		.setFormat(app.swapChainImageFormat)
		// 设置多重采样为1个样本（不使用多重采样）
		.setSamples(vk::SampleCountFlagBits::e1)
		// 在渲染开始时清除帧缓冲区内容
		.setLoadOp(vk::AttachmentLoadOp::eClear)
		// 渲染结束后存储帧缓冲区内容
		.setStoreOp(vk::AttachmentStoreOp::eStore)
		// 模板缓冲区加载操作不关心（因为我们没有使用模板测试）
		.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
		// 模板缓冲区存储操作不关心
		.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
		// 初始布局未定义（因为我们将在渲染过程中转换它）
		.setInitialLayout(vk::ImageLayout::eUndefined)
		// 最终布局为呈现源（用于显示到屏幕）
		.setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

	// 创建颜色附件引用，指定附件索引和布局
	vk::AttachmentReference colorAttachmentRef{
		0, // 附件索引为0
		vk::ImageLayout::eColorAttachmentOptimal // 使用颜色附件最优布局
	};
	// 创建深度附件描述
    vk::AttachmentDescription depthAttachment;
    depthAttachment
		.setFormat(findDepthFormat({vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint}))
        .setSamples(vk::SampleCountFlagBits::e1)
        .setLoadOp (vk::AttachmentLoadOp::eClear)
        .setStoreOp (vk::AttachmentStoreOp::eDontCare)
        .setStencilLoadOp (vk::AttachmentLoadOp::eDontCare)
        .setStencilStoreOp (vk::AttachmentStoreOp::eDontCare)
        .setInitialLayout (vk::ImageLayout::eUndefined)
        .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

	vk::AttachmentReference depthAttachmentRef{
		1, // 附件索引为1
		vk::ImageLayout::eDepthStencilAttachmentOptimal // 使用深度附件最优布局
	};
	// 创建子通道描述
	vk::SubpassDescription subpass;
	subpass
		// 设置图形管线绑定点
		.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
		// 设置颜色附件引用
		.setColorAttachments(colorAttachmentRef)
		.setPDepthStencilAttachment(&depthAttachmentRef);
	// 创建子通道依赖关系
	vk::SubpassDependency dependency;
	dependency
		// 源子通道为外部（即来自之前渲染操作）
		.setSrcSubpass(vk::SubpassExternal)
		// 源阶段为颜色附件输出阶段
		.setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests)
		// 源访问掩码为空（因为之前没有访问）
		.setSrcAccessMask({})
		// 目标子通道为当前子通道（索引为0）
		.setDstSubpass(0)
		// 目标访问掩码为颜色附件写入
		.setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite)
		// 目标阶段为颜色附件输出阶段
		.setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests);

	const auto attachments = { colorAttachment, depthAttachment };

	// 创建渲染通道创建信息
	vk::RenderPassCreateInfo renderPassInfo;
	renderPassInfo
		// 设置子通道描述
		.setSubpasses(subpass)
		// 设置附件描述
		.setAttachments(attachments)
		// 设置依赖关系
		.setDependencies(dependency);

	// 尝试创建渲染通道
	try{
		app.renderPass = app.device.createRenderPass(renderPassInfo);
	}
	catch (const vk::SystemError& e) {
		// 如果创建失败，抛出运行时错误
		throw std::runtime_error("failed to create render pass: " + std::string(e.what()));
	}
};

