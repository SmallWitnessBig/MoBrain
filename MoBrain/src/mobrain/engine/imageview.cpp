#include "imageview.hpp"
/*
交换链图像本质：

交换链图像是二维纹理，用于在屏幕上显示渲染结果
它们本质上是 2D 数组，每个像素有一个特定的 (x, y) 坐标
即使你渲染的是 3D 场景，最终输出也是投影到 2D 屏幕上的结果
渲染管线流程：

3D 渲染是指在顶点着色器和片段着色器中处理 3D 几何体
但是最终的渲染目标（framebuffer）仍然是 2D 图像
交换链图像就是这些 2D 渲染目标
*/
void createImageViews() {
	app.swapChainImageViews.reserve(app.swapChainImages.size());
	for (auto& image: app.swapChainImages)
	{
		app.swapChainImageViews.emplace_back(createImageView(image, app.swapChainImageFormat,vk::ImageAspectFlagBits::eColor));
	}
};
vk::raii::ImageView createImageView(
	const vk::Image image,
	const vk::Format format,
	const vk::ImageAspectFlags aspectFlags
)  {
    vk::ImageViewCreateInfo createInfo;
	createInfo
		.setImage(image)
		.setViewType(vk::ImageViewType::e2D)
		.setFormat(format) 
		.setSubresourceRange({ aspectFlags, 0, 1, 0, 1 });	
    return app.device.createImageView(createInfo);
}